import numpy as np
import pandas as pd
import scipy.integrate as integrate
import scipy.stats as stats

## TODO: The ANC likelihood implementation is based on Jeff Eaton's anclik package,
## which converts data stored in matrices to lists of arrays, then operates on those
## lists. As a result, the current implementation converts from long dataframes to
## matrices, then matrices to lists of arrays. It would probably be more efficient
## to convert directly from long dataframes to lists of arrays.

## TODO: write_csv
## TODO: set up unit test with Botswana urban ANC-SS data
## TODO: streamline read_csv and __prepare_anc_data to eliminate the conversion from dataframe to matrices
## TODO: add methods to get and set ANC-SS and ANC-RT bias and variance inflation parameters
## TODO: accept ANC-RT site-level data
## TODO: accept ANC-RT census-level data
## TODO: error handling for malformatted csv files
## TODO: more indicative variable names than "W", "V", etc.

class ancprev:
    def __init__(self):
        self.type_prev = np.float64
        self.type_size = np.int32
        self.type_used = np.bool_
        self.base_year = 1970

        # parameters of the inverse gamma prior on the variance in site random effects
        self.invgamma_shape = 0.58
        self.invgamma_rate  = 93.0

        self.ancss_bias = 0.2637549
        self.ancss_var_inflation = 0.0

        self.quad_lower = 1e-15
        self.quad_upper = 0.3

    def read_csv(self, csv_name):
        """ Read ANC prevalence data from a CSV file
        csv_name -- CSV file name

        Suppose A is an instance of ancprev. After calling A.read_csv(...), several
        member variables of A are populated:
        A.years -- array of years spanning the earliest to latest ANC prevalence datapoints. This may include some years without surveillance.
        A.sites -- array of site names.
        A.prev -- sites-by-years matrix of prevalence estimates, -1 if missing.
        A.size -- sites-by-years matrix of sample sizes (N sampled for sentinel surveillance, N ascertained for routine testing), -1 if missing.
        A.used -- sites-by-years matrix of indicator variables. Datapoints are included in likelihood evaluation if True and excluded otherwise.
        A.W -- posterior ANC prevalence estimates on probit scale, subject to a Beta(0.5, 0.5) prior
        A.v -- approximate variance in ANC prevalence estimates
        A.i -- time index of observations relative to self.base_year (e.g., if A.base_year = 1970, i=21 for an observation in 1990)

        The notation W and v is from Alkema et al (doi:10.1214/07-aoas111). W, v, and i are lists of arrays. These lists have one element
        for each site that has some non-excluded (A.used=True) datapoints. The arrays correspond to the non-excluded datapoints for that
        site.
        """
        anc_data = pd.read_csv(csv_name)

        self.years = np.array(range(anc_data.Year.min(), anc_data.Year.max() + 1))
        self.sites = anc_data.Site.unique()

        ns = len(self.sites)
        ny = len(self.years)
        prev = np.full((ns, ny), fill_value=-1,    dtype = self.type_prev)
        size = np.full((ns, ny), fill_value=-1,    dtype = self.type_size)
        used = np.full((ns, ny), fill_value=False, dtype = self.type_used)

        # long-to-wide transform
        prev_wide = anc_data.pivot_table(index="Site", columns="Year", values="Prevalence",   fill_value=-1)
        size_wide = anc_data.pivot_table(index="Site", columns="Year", values="N",            fill_value=-1)
        used_wide = anc_data.pivot_table(index="Site", columns="Year", values="UseDataInFit", fill_value=False)

        # map from rows and columns of prev_wide (some years missing, site ordering may differ) to prev (all years needed)
        sidx = [prev_wide.index.get_loc(name) for name in self.sites]
        yidx = [np.argwhere(self.years==year)[0][0] for year in prev_wide.columns]

        # Fill in the full arrays for the years with data. Sites may still be out-of-order
        prev[:,yidx] = prev_wide.to_numpy(dtype = self.type_prev)
        size[:,yidx] = size_wide.to_numpy(dtype = self.type_size)
        used[:,yidx] = used_wide.to_numpy(dtype = self.type_used)

        self.prev = prev[sidx,:]
        self.size = size[sidx,:]
        self.used = used[sidx,:]

        self.W, self.v, self.i = self.__prepare_anc_data()

    def write_csv(self):
        pass

    def likelihood(self, prevalence):
        """ Calculate the ANC log-likelihood given a time series of pregnant women HIV prevalence estimates """
        probit_prev = stats.norm.ppf(prevalence) + self.ancss_bias
        dlst = [w - probit_prev[i] for (w, i) in zip(self.W, self.i)]
        vlst = [v + self.ancss_var_inflation for v in self.v]
        return self.__anc_resid_likelihood(dlst, vlst)

    def __prepare_anc_data(self):
        """ Reformat ANC data for likelihood calculation """

        ns = len(self.sites)
        nt = len(self.years)

        # We may mutate some prevalence and sample size entries, so we create copies here.
        prev = self.prev.copy()
        size = self.size.copy()

        # drop observations that are not used
        prev[np.logical_not(self.used)] = -1
        size[np.logical_not(self.used)] = -1

        # drop observations that are missing a prevalence or a sample size
        prev[size<0] = -1
        size[prev<0] = -1

        # Drop sites that have no data used
        site_used = (size >= 0).any((1))
        prev_used = prev[site_used,:]
        size_used = size[site_used,:]
        nu = sum(site_used)

        # get a list with one element per site that consists of an array of indices of non-missing observations for that site
        indlist = [np.argwhere(prev_used[i,:] > 0) for i in range(nu)] 

        # extract the non-missing observations. np.concatenate converts from an nx1 matrix to a 1-d array
        prev_list = [np.concatenate(prev_used[i,indlist[i]]) for i in range(nu)]
        size_list = [np.concatenate(size_used[i,indlist[i]]) for i in range(nu)]
        year_list = [np.concatenate(self.years[indlist[i]] - self.base_year) for i in range(nu)] # convert from years to indices into the model prevalence estimates

        # apply data transforms used in likelihood calculation. The terms (x, W, v) follow variable naming
        # conventions in doi:10.1214/07-aoas111
        x_list = [(p * n + 0.5) / (n + 1.0) for (p, n) in zip(prev_list, size_list)] # posterior prevalence given ANC data and Jeffreys prior
        W_list = [stats.norm.ppf(x) for x in x_list]                                 # probit-transformed posterior prevalence
        v_list = [2 * np.pi * np.exp(W * W) * x * (1.0 - x) / n for (W, x, n) in zip(W_list, x_list, size_list)] # approximate variance

        return W_list, v_list, year_list

    def __anc_integrand(self, s2, dlst, Vlst):
        return np.exp(sum([stats.multivariate_normal.logpdf(d, cov = v + s2) for (d, v) in zip(dlst, Vlst)])) * s2**(-self.invgamma_shape - 1.0) * np.exp(-1.0 / (s2 * self.invgamma_rate))

    def __anc_resid_likelihood(self, dlst, vlst):
        Vlst = [np.diag(v) for v in vlst]
        return integrate.quad(self.__anc_integrand, self.quad_lower, self.quad_upper, args=(dlst, Vlst))[0]

if __name__ == "__main__":
    model_prev = np.array([0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00111, 0.00123, 0.00136, 0.00150, 0.00166,
                           0.00183, 0.00200, 0.00239, 0.00309, 0.00432, 0.00641, 0.00988, 0.01552, 0.02432, 0.03736,
                           0.05551, 0.07899, 0.10703, 0.13798, 0.16956, 0.19942, 0.22556, 0.24661, 0.26194, 0.27150,
                           0.27566, 0.27503, 0.27045, 0.26344, 0.25587, 0.24928, 0.24303, 0.23681, 0.23020, 0.22382,
                           0.21841, 0.21386, 0.21015, 0.20715])

    anc = ancprev()
    anc.read_csv("tests/bwa-urban-anc.csv")
    print(anc.likelihood(model_prev))
