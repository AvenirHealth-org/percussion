import numpy as np
import pandas as pd
import scipy.stats as stats

## The hivprev class allows prevalence data from arbitrary population,
## sex, and age combinations. Aligning model outputs with the data may
## have avoidable overheads (say, from matching on sex, etc.). One way
## to avoid this is to create multiple hivprev instances for specific
## subpopulations. For example, if you have data on the 15-49 population
## overall and in specific high-risk populations, you might use separate
## instances for the overall population and each high-risk population.

## TODO: If speed is an issue, create a direct index map from hivprev::hiv_data_used
## to hivprev::template during hivprev::__prepare_data.
## TODO: Develop some unit tests

class hivprev:
    """ Class for storing and evaluating the likelihood of HIV prevalence data

    HIV prevalence in pregnant women attending antenatal care (ANC) should use
    the ancprev module instead
    """

    def __init__(self, proj_first_year):
        self.base_year = proj_first_year
        self.hiv_data = None
        self.cols_template = ['Population', 'Gender', 'AgeMin', 'AgeMax', 'Year']

    def read_csv(self, csv_name):
        """ Read a CSV file storing HIV prevalence data
        
        The CSV file should have one datapoint per row, and include the following columns:
        Indicator    -- The indicator stored. Only rows labelled "HIV prevalence" will be stored.
        Population   -- The population prevalence was measured in. This is free-form.
        Gender       -- The gender prevalence was measured in. This is free-form.
        AgeMin       -- Minimum (integer) age of the population prevalence was measured in, inclusive.
        AgeMax       -- Maximum (integer) age of the population prevalence was measured in, inclusive.
        Value        -- Measured HIV prevalence.
        NumTested    -- The number of people whose HIV status was ascertained.
        Year         -- Year that prevalence was measured.
        UseDataInFit -- TRUE/FALSE indicator. Data points with UseDataInFit=FALSE are not evaluated in the likelihood.

        You may include additional columns in the CSV (e.g., to document the data
        source). Any additional columns will be ignored during likelihood calculation.

        The Population and gender columns can use whatever levels you choose, but it
        is your responsibility to align the populations you have data for with the
        populations your model makes estimates for when evaluating the likelihood.
        The projection template produced with hivprev.projection_template() is
        designed to assist with this alignment.

        The HIV prevalence likelihood assumes datapoints are conditionally independent
        given the modeled estimate. Therefore, you should not input datapoints via the
        CSV that violate this assumption. For example, do not include data for 15-24,
        24-49 and 15-49 for the same population; either exclude 15-49 or exclude 15-24
        and 25-49. Similarly, do not include male, female, and male+female estimates for
        the same age range; use either the sex-specific or sex-aggregated estimates but
        not both. For this reason, we recommend using the ancprev module to evaluate the
        likelihood of ANC data, as site-level observations in particular are not assumed
        conditionally independent from year-to-year.

        HIV prevalence data often comes from national household surveys in some settings.
        These surveys use weights to adjust for survey design effects and other
        non-sampling errors. With these data, we recommend setting NumTested to an
        effective sample size that accounts for weighting, rather than the raw numbers of
        participants tested, to avoid overstating the precision of survey-based estimates.

        HIV prevalence in key and vulnerable populations measured through focused
        bio-behavioral surveys are subject to similar issues due to the effects of
        network sampling methods like respondent-driven sampling. Effective sample
        sizes should be used instead of raw numbers of respondents for these surveys
        if available.
        """
        self.hiv_data = pd.read_csv(csv_name)
        self.__prepare_data()

    def likelihood(self, proj_prev):
        """ Calculate the log-likelihood of prevalence data"""
        ## Inefficient but reliable baseline implementation. Inefficiency is from using a join operation
        ## to align model-based estimates with data. One way to optimize would be to create an index mapping
        ## from proj_prev to hiv_data_used during __prepare_data.
        df = self.hiv_data_used.merge(proj_prev, how='left', on=self.cols_template)
        return sum(stats.beta.logpdf(df['Value'], df['NumTested'] * df['Prevalence'] + 1.0, df['NumTested'] * (1.0 - df['Prevalence']) + 1.0))

    def projection_template(self):
        """ Request a template for prevalence data

        Return a template for collecting prevalence estimates for likelihood
        evaluation. The template is a pandas dataframe with columns 'Population',
        'Gender', 'AgeMin', 'AgeMax', 'Year', 'Prevalence'. Each row corresponds to
        one or more prevalence estimates needed to evaluate the likelihood,
        based on prevalence data loaded into the hivprev instance.
        
        The 'Prevalence' column is intended to store prevalence estimates, and will
        be empty on return from hivprev.projection_template. The user is responsible
        for filling in the Prevalence column for each row. Once completed, it can be passed
        to hivprev.likelihood(...) for likelihood evaluation.
        """
        return self.template.copy()

    def __prepare_data(self):
        cols_data = ['Population', 'Gender', 'AgeMin', 'AgeMax', 'Year', 'Value', 'NumTested']
        rows_data = (self.hiv_data['Indicator'] == "HIV prevalence") & (self.hiv_data['UseDataInFit'] == True)
        self.hiv_data_used = self.hiv_data.loc[rows_data, cols_data]
        self.template = self.hiv_data_used.groupby(self.cols_template).size().reset_index(name='Prevalence')
        self.template.loc[:,'Prevalence'] = np.nan

if __name__ == "__main__":
    hiv = hivprev(1970)
    hiv.read_csv("tests/mwi-hiv-prev-data.csv")
    template = hiv.projection_template()
    template['Prevalence'] = 0.1
    print(hiv.likelihood(template))
