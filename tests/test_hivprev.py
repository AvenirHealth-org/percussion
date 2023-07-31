import pandas as pd
import unittest
import percussion.hivprev as hivprev

class Test_TestHIVPrev(unittest.TestCase):
    def test_hivprev_template(self):
        hiv = hivprev.hivprev(1970)
        hiv.read_csv("tests/mwi-hiv-prev-data.csv")
        template_test = hiv.projection_template()
        template_base = pd.read_csv("tests/mwi-hiv-prev-projected.csv")
        keycols = ['Population', 'Gender', 'AgeMin', 'AgeMax', 'Year']
        self.assertTrue(template_test[keycols].equals(template_base[keycols]))

    def test_hivprev_lnlhood(self):
        hiv = hivprev.hivprev(1970)
        hiv.read_csv("tests/mwi-hiv-prev-data.csv")
        proj_prev = pd.read_csv("tests/mwi-hiv-prev-projected.csv")
        lnlhood = hiv.likelihood(proj_prev)
        self.assertEqual(lnlhood, -112.72247309230656)

if __name__ == "__main__":
    unittest.main()
