import numpy as np
import pandas as pd
import scipy.integrate as integrate
import scipy.stats as stats

## TODO: write_csv
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
        A.anc_data -- unmodified data from CSV file
        A.prev -- list of prevalence estimates by site
        A.size -- list of sample sizes by site (numbers sampled in sentinel surveillance or ascertained in routine testing)
        A.yidx -- list of year indices for obdservations, shifted so that A.base_year has index 0
        A.W -- posterior ANC prevalence estimates on probit scale, subject to a Beta(0.5, 0.5) prior
        A.v -- approximate variance in ANC prevalence estimates

        The notation W and v is from Alkema et al (doi:10.1214/07-aoas111). prev, size, yidx, W, and v are lists of numpy arrays,
        with array per site. Arrays elements correspond to non-excluded (records marked UseDataInFit=True) datapoint for that site.
        """
        self.anc_data = pd.read_csv(csv_name)
        anc_data_used = self.anc_data[self.anc_data['UseDataInFit']] # drop unused observations

        # split anc_data_used into a list with one dataframe per site. sort=False preserves the site ordering
        # in the input CSV. Note that reordering sites in the input file can change the calculated likelihood
        # due to loss of precision in the likelihood calculations (that is, changing sort=False to sort=True)
        # is enough to fail unit tests based on a ground-truth likelihood value).
        site_list = [site for _, site in anc_data_used.groupby('Site', sort=False)]

        # split anc_data_used by site, then extract prev, size, and year indices relative to self.base_year
        self.prev = [df['Prevalence'].to_numpy() for df in site_list]
        self.size = [df['N'].to_numpy() for df in site_list]
        self.yidx = [df['Year'].to_numpy() - self.base_year for df in site_list]

        self.W, self.v = self.__prepare_anc_data()

    def write_csv(self):
        pass

    def likelihood(self, prevalence):
        """ Calculate the ANC log-likelihood given a time series of pregnant women HIV prevalence estimates """
        probit_prev = stats.norm.ppf(prevalence) + self.ancss_bias
        dlst = [w - probit_prev[i] for (w, i) in zip(self.W, self.yidx)]
        vlst = [v + self.ancss_var_inflation for v in self.v]
        return self.__anc_resid_likelihood(dlst, vlst)

    def __prepare_anc_data(self):
        """ Reformat ANC data for likelihood calculation """
        # apply data transforms used in likelihood calculation. The terms (x, W, v) follow variable naming
        # conventions in doi:10.1214/07-aoas111
        x_list = [(p * n + 0.5) / (n + 1.0) for (p, n) in zip(self.prev, self.size)] # posterior prevalence given ANC data and Jeffreys prior
        W_list = [stats.norm.ppf(x) for x in x_list]                                 # probit-transformed posterior prevalence
        v_list = [2 * np.pi * np.exp(W * W) * x * (1.0 - x) / n for (W, x, n) in zip(W_list, x_list, self.size)] # approximate variance
        return W_list, v_list

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
