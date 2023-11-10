import numpy as np
import pandas as pd
import scipy.stats as stats

class all_cause_deaths:
    def __init__(self, proj_first_year):
        self.base_year = proj_first_year
        self.cols_template = ['Year', 'Gender', 'AgeMin', 'AgeMax']

        # Initialize the likelihood model parameter with a default value. This
        # value is loosely based on the value used by Thembisa 4.6 (April 2023)
        # manual. This value is specific to Thembisa in the context of South Africa.
        self.death_stddev = 0.25

    def read_csv(self, csv_name):
        """ Read a CSV file storing all-cause deaths data

        The CSV file should have one datapoint per row, and include the following columns:
        Year         -- The year when reported deaths occurred.
        Gender       -- The gender that deaths were reported in. This is free-form.
        AgeMin       -- Minimum (integer) age of the population that deaths were reported in, inclusive.
        AgeMax       -- Maximum (integer) age of the population that deaths were reported in, inclusive.
        Value        -- Reported number of deaths.
        UseDataInFit -- TRUE/FALSE indicator. Data points with UseDataInFit=FALSE are not evaluated in the likelihood.

        You may include additional columns in the CSV (e.g., to document data
        sources or reasons why any data points were excluded). Any additional
        columns will be ignored during likelihood evaluation.

        The Gender column can use whatever levels you choose, but it is your
        responsibility to align the levels in this file with the groups your
        model makes estimates for when evaluating the likelihood. The projection
        template produced with all_cause_deaths.projection_template() is
        designed to assist with this alignment.
        
        The likelihood assumes datapoints are conditionally independent given
        the model estimate. Therefore, you should not input datapoints via the
        CSV that violate this assumption. For example, do not include deaths for
        15-19, 20-24, and 15-24 for the same gender, since this will double
        the influence of deaths in these age groups. Use the "UseDataInFit"
        column to exclude either the 15-24 datapoint or both 15-24 and 20-24
        datapoints.
        """
        self.death_data = pd.read_csv(csv_name)
        self.__prepare_data()

    def set_parameters(self, death_stddev):
        """ Helper function to initialize likelihood model parameters

        death_stddev -- Standard deviation in modeled deaths

        If D is an instance of all_cause_deaths, then the standard deviation
        can also be set directly via D.death_stddev = <value>.
        """
        self.death_stddev = death_stddev

    def likelihood(self, proj_deaths):
        df = self.death_data_used.merge(proj_deaths, how='left', on=self.cols_template)
        return sum(stats.norm.logpdf(np.log(df['Value']), np.log(df['Deaths']), self.death_stddev))
    
    def projection_template(self):
        """ Request a template for all-cause deaths estimates

        Return a template for collecting deaths estimates for likelihood
        evaluation. The template is a pandas dataframe with columns 'Year',
        'Gender', 'AgeMin', 'AgeMax', and 'Deaths'. Each row corresponds to one
        or more estimates needed to evaluate the likelihood, based on deaths
        data loaded into the all_cause_deaths instance.

        The 'Deaths' column is intended to store deaths estimates, and will
        be empty on return from all_cause_deaths.projection_template. The user
        is responsible for filling in the Value column for each row. Once
        completed, it can be passed to all_cause_deaths.likelihood(...) for
        likelihood evaluation.
        """
        return self.template.copy()
    
    def __prepare_data(self):
        cols_data = ['Year', 'Gender', 'AgeMin', 'AgeMax', 'Value']
        rows_data = (self.death_data['UseDataInFit'] == True)
        self.death_data_used = self.death_data.loc[rows_data, cols_data]
        self.template = self.death_data_used.groupby(self.cols_template).size().reset_index(name='Deaths')
        self.template.loc[:,'Deaths'] = np.nan ## TODO: check that the template is correct (test setting UseDataInFit=FALSE for some groups)

if __name__ == "__main__":
    deaths = all_cause_deaths(1970)
    deaths.read_csv("tests/deaths-data-synthetic.csv")
    template = deaths.projection_template()
    template['Deaths'] = 3200
    print(deaths.likelihood(template))
