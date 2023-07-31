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

class hivprev:
    def __init__(self, proj_first_year):
        self.base_year = proj_first_year
        self.hiv_data = None
        self.cols_template = ['Population', 'Gender', 'AgeMin', 'AgeMax', 'Year']

    def read_csv(self, csv_name):
        self.hiv_data = pd.read_csv(csv_name)
        self.__prepare_data()

    def likelihood(self, proj_prev):
        """ Calculate the log-likelihood of prevalence data"""
        ## Inefficient but reliable baseline implementation. Inefficiency is from using a join operation
        ## to align model-based estimates with data. One way to optimize would be to create an index mapping
        ## from proj_prev to hiv_data_used during __prepare_data.
        df = self.hiv_data_used.merge(proj_prev, how='left', on=self.cols_template)
        return sum(stats.beta.logpdf(df['Value'], df['NumTested'] * df['Prevalence'], df['NumTested'] * (1.0 - df['Prevalence'])))

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

        Population, Gender, AgeMin, AgeMax, and Year specify who HIV prevalence was
        measured in and when. Population can be any population included in the HIV
        prevalence data entered via hivprev.read_csv(). Gender is either 'Female',
        'Male', or 'All'. AgeMin and AgeMax are integers specifying the age
        range (inclusive).
        """
        return self.template.copy()

    def __prepare_data(self):
        cols_data = ['Population', 'Gender', 'AgeMin', 'AgeMax', 'Year', 'Value', 'NumTested']
        self.hiv_data_used = self.hiv_data.loc[self.hiv_data['UseDataInFit'] == True, cols_data]
        self.template = self.hiv_data_used.groupby(self.cols_template).size().reset_index(name='Prevalence')
        self.template.loc[:,'Prevalence'] = np.nan

if __name__ == "__main__":
    hiv = hivprev(1970)
    hiv.read_csv("tests/mwi-hiv-prev-data.csv")
    template = hiv.projection_template()
    template['Prevalence'] = 0.1
    print(hiv.likelihood(template))
