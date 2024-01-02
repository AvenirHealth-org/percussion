import pandas as pd
import unittest
import percussion.all_cause_deaths as all_cause_deaths

class Test_TestAllCauseDeaths(unittest.TestCase):
    def test_deaths_lnlhood(self):
        deaths = all_cause_deaths.all_cause_deaths(1970)
        deaths.read_csv("tests/deaths-data-synthetic.csv")
        template = deaths.projection_template()
        template['Deaths'] = 3200
        lnlhood = deaths.likelihood(template)
        self.assertEqual(lnlhood, -93.10678896101508)

if __name__ == "__main__":
    unittest.main()
