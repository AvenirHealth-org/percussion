import pandas as pd
import unittest
import percussion.all_cause_deaths as all_cause_deaths

class Test_TestAllCauseDeaths(unittest.TestCase):
    def test_deaths_template(self):
        deaths = all_cause_deaths.all_cause_deaths(1970)
        deaths.read_csv("tests/deaths-data-synthetic.csv")
        template_test = deaths.projection_template()
        template_base = pd.read_csv("tests/deaths-projected.csv")
        keycols = ['Year', 'Gender', 'AgeMin', 'AgeMax']
        self.assertTrue(template_test[keycols].equals(template_base[keycols]))

    def test_deaths_lnlhood(self):
        deaths = all_cause_deaths.all_cause_deaths(1970)
        deaths.read_csv("tests/deaths-data-synthetic.csv")
        proj_mort = pd.read_csv("tests/deaths-projected.csv")
        lnlhood = deaths.likelihood(proj_mort)
        ## TODO: Reimplement the likelihood in R using the example data and verify that
        ## the log-likelihood matches this value
        self.assertEqual(lnlhood, 8.609445243532194)

if __name__ == "__main__":
    unittest.main()
