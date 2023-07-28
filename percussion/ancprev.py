import numpy as np
import pandas as pd
import scipy.integrate as integrate
import scipy.stats as stats

## TODO: add methods to get and set ANC-SS and ANC-RT bias and variance inflation parameters
## TODO: more indicative variable names than "W", "V", etc.
## TODO: handle error case where model prevalence does not span all years of data

## TODO: error handling for malformatted csv files
## - Type must be either RT or SS
## - Use (region, site) composite identifiers to allow same-name sites in different regions

class ancprev:
    def __init__(self, proj_first_year):
        """ ancprev constructor
        The likelihood of ANC prevalence data is evaluated in the context of model projections of
        pregnant women projection. All model projections evaluated using an ancprev instance are
        required to start in the same year (e.g., projections from 1970 onward).

        proj_first_year -- the first year of HIV prevalence projections
        """
        self.type_prev = np.float64
        self.type_size = np.int32
        self.type_used = np.bool_
        self.base_year = proj_first_year

        # parameters of the inverse gamma prior on the variance in site random effects
        self.invgamma_shape = 0.58
        self.invgamma_rate  = 93.0

        # Parameters of the beta prior on site and census observations. Given
        # Y HIV+ pregnant women found among N whose HIV status was ascertained,
        # the posterior prevalence estimate is (Y + beta_mean) / (N + beta_size)
        self.beta_mean = 0.5
        self.beta_size = 1.0

        self.quad_lower = 1e-15
        self.quad_upper = 0.3

        # Likelihood model parameters
        self.__ancss_bias = 0.2637549
        self.__ancrt_bias = 0.0         # ANC-RT site calibration parameter (see Sheng 2017 AIDS, doi:10.1097/QAD.0000000000001428)
        self.__var_inflate_site   = 0.0 # ANC site variance inflation (see Eaton 2017 AIDS, doi:10.1097/QAD.0000000000001419)
        self.__var_inflate_census = 0.0

    def read_csv(self, csv_name):
        """ Read ANC prevalence data from a CSV file
        csv_name -- CSV file name

        Suppose A is an instance of ancprev. After calling A.read_csv(...), several
        member variables of A are populated:
        A.anc_data -- unmodified data from CSV file

        Site-level variables, stored as lists of arrays (one array per ANC site, one array element per observation at that site)
        A.site_prev -- observed prevalence
        A.site_size -- sample sizes, either numbers sampled in sentinel surveillance or women whose HIV status was ascertained in routine testing
        A.site_yidx -- year indices of observations relative to the first year of projection
        A.site_type -- indicator variables, 1 for routine testing, 0 for sentinel surveillance
        A.site_W -- posterior prevalence estimates on probit scale
        A.site_v -- variances in prevalence estimates

        ANC-RT census variables, stored as arrays
        A.census_prev -- prevalence estimates
        A.census_size -- numbers of women with HIV status ascertained
        A.census_yidx -- year indices relative to the first year of projection
        A.census_W -- posterior prevalence estimates on probit scale
        A.census_v -- variances in prevalence estimates

        The notation W and v is from Alkema et al (doi:10.1214/07-aoas111). prev,
        size, yidx, type, W, and v are lists of numpy arrays, with one array per
        site. Each array element corresponds to a non-excluded datapoint
        (datapoints marked UseDataInFit=True) for that site.
        """
        self.anc_data = pd.read_csv(csv_name)

        # Site data
        anc_data_used = self.anc_data[self.anc_data['UseDataInFit']] # drop unused observations
        anc_site_data = anc_data_used[anc_data_used['Site'] != 'Census']

        anc_site_data.loc[:,'Type'] = anc_site_data['Type'].map(dict(SS=0, RT=1))

        # split anc_data_used into a list with one dataframe per site. sort=False preserves the site ordering
        # in the input CSV. Note that reordering sites in the input file can change the calculated likelihood
        # due to loss of precision in the likelihood calculations (that is, changing sort=False to sort=True)
        # is enough to fail unit tests based on a ground-truth likelihood value).
        site_list = [site for _, site in anc_site_data.groupby('Site', sort=False)]

        # split anc_data_used by site, then extract prev, size, and year indices relative to self.base_year
        self.site_prev = [df['Prevalence'].to_numpy() for df in site_list]
        self.site_size = [df['N'].to_numpy() for df in site_list]
        self.site_yidx = [df['Year'].to_numpy() - self.base_year for df in site_list]
        self.site_type = [df['Type'].to_numpy() for df in site_list]

        self.site_W, self.site_v = self.__prepare_data_site() # TODO: could pass site_prev and site_size, then discard after read_csv?

        anc_census_data = anc_data_used[anc_data_used['Site'] == 'Census']
        self.census_prev = anc_census_data['Prevalence'].to_numpy()
        self.census_size = anc_census_data['N'].to_numpy()
        self.census_yidx = anc_census_data['Year'].to_numpy() - self.base_year

        self.census_W, self.census_v = self.__prepare_data_census() # TODO: could pass census_prev and census_size, then discard after read_csv?

    def likelihood(self, proj_prev):
        """ Calculate the log-likelihood of ANC prevalence data given projected pregnant women HIV prevalence estimates
        proj_prev -- array of annual HIV prevalence esimates. proj_prev[0] must be the prevalence at ancprev.base_year
        """
        return self.likelihood_site(proj_prev) + self.likelihood_census(proj_prev)

    def likelihood_site(self, proj_prev):
        """ Calculate the log-likelihood of site-level ANC sentinel surveillance and routine testing HIV prevalence data
        proj_prev -- array of annual HIV prevalence esimates. proj_prev[0] must be the prevalence at ancprev.base_year
        """
        probit_prev = stats.norm.ppf(proj_prev) + self.__ancss_bias # TODO: consider having separate series for SS and RT = probit(prev) shifted by appropriate bias/calibration terms
        dlst = [w - probit_prev[i] - r * self.__ancrt_bias for (w, i, r) in zip(self.site_W, self.site_yidx, self.site_type)]
        vlst = [v + self.__var_inflate_site for v in self.site_v]
        return self.__site_resid_likelihood(dlst, vlst)
    
    def likelihood_census(self, proj_prev):
        """ Calculate the log-likelihood of census-level ANC routine testing HIV prevalence data
        proj_prev -- array of annual HIV prevalence esimates. proj_prev[0] must be the prevalence at ancprev.base_year
        """
        probit_prev = stats.norm.ppf(proj_prev)
        return stats.norm.logpdf(self.census_W, probit_prev[self.census_yidx], np.sqrt(self.census_v + self.__var_inflate_census)).sum()

    def __prepare_data_site(self):
        """ Reformat site-level ANC data for likelihood calculation """
        # apply data transforms used in likelihood calculation. The terms (x, W, v) follow variable naming
        # conventions in doi:10.1214/07-aoas111
        x_list = [self.__posterior_prevalence(p * n, n) for (p, n) in zip(self.site_prev, self.site_size)]
        W_list = [stats.norm.ppf(x) for x in x_list] # probit-transformed posterior prevalence
        v_list = [2.0 * np.pi * np.exp(W**2) * x * (1.0 - x) / n for (W, x, n) in zip(W_list, x_list, self.site_size)] # approximate variance
        return W_list, v_list
    
    def __prepare_data_census(self):
        """ Reformat census-level ANC-RT data for likelihood calculation """
        x = self.__posterior_prevalence(self.census_prev * self.census_size, self.census_size)
        W = stats.norm.ppf(x)
        v = 2.0 * np.pi * np.exp(W**2) * x * (1.0 - x) / self.census_size
        return W, v

    def __posterior_prevalence(self, num_hivpos, num_ascertained):
        return (num_hivpos + self.beta_mean) / (num_ascertained + self.beta_size)

    def __site_integrand(self, s2, dlst, Vlst):
        term1 = sum([stats.multivariate_normal.logpdf(d, cov = v + s2) for (d, v) in zip(dlst, Vlst)])
        term2 = np.log(s2) * (-self.invgamma_shape - 1.0)
        term3 = -1.0 / (s2 * self.invgamma_rate)
        return np.exp(term1 + term2 + term3)

    def __site_resid_likelihood(self, dlst, vlst):
        Vlst = [np.diag(v) for v in vlst]
        return np.log(integrate.quad(self.__site_integrand, self.quad_lower, self.quad_upper, args=(dlst, Vlst))[0])

if __name__ == "__main__":
    # model_prev = np.array([0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00111, 0.00123, 0.00136, 0.00150, 0.00166,
    #                        0.00183, 0.00200, 0.00239, 0.00309, 0.00432, 0.00641, 0.00988, 0.01552, 0.02432, 0.03736,
    #                        0.05551, 0.07899, 0.10703, 0.13798, 0.16956, 0.19942, 0.22556, 0.24661, 0.26194, 0.27150,
    #                        0.27566, 0.27503, 0.27045, 0.26344, 0.25587, 0.24928, 0.24303, 0.23681, 0.23020, 0.22382,
    #                        0.21841, 0.21386, 0.21015, 0.20715])
    
    coast_prev = np.array([0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00006, 0.00016, 0.00030, 0.00055,
                           0.00099, 0.00175, 0.00302, 0.00510, 0.00843, 0.01354, 0.02089, 0.03084, 0.04351, 0.05842,
                           0.07439, 0.08936, 0.10141, 0.11015, 0.11538, 0.11711, 0.11577, 0.11198, 0.10660, 0.10036,
                           0.09369, 0.08681, 0.07997, 0.07347, 0.06773, 0.06277, 0.05856, 0.05504, 0.05194, 0.04944,
                           0.04741, 0.04573, 0.04423, 0.04253, 0.04105, 0.03985, 0.03811, 0.03614, 0.03434, 0.03232,
                           0.03021, 0.02871, 0.02751, 0.02587, 0.02393, 0.02205, 0.02033, 0.01876, 0.01736, 0.01610,
                           0.01496])

    anc = ancprev(1970)
    # anc.read_csv("tests/bwa-urban-anc.csv")
    # anc.read_csv("tests/ken-coast-ancss.csv") # expect -19.470508610987594
    # anc.read_csv("tests/ken-coast-anc-site.csv") # expect -24.213141355812205
    anc.read_csv("tests/ken-coast-anc.csv")
    print(anc.likelihood(coast_prev))
    print(anc.likelihood_site(coast_prev))
    print(anc.likelihood_census(coast_prev))
