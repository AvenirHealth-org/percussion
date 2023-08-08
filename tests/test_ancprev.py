import numpy as np
import unittest
import percussion.ancprev as ancprev

# run from percussion toplevel folder: python -m tests.test_ancprev

class Test_TestANCPrev(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.bwa_urban_prev = np.array([
            0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00111, 0.00123, 0.00136, 0.00150, 0.00166,
            0.00183, 0.00200, 0.00239, 0.00309, 0.00432, 0.00641, 0.00988, 0.01552, 0.02432, 0.03736,
            0.05551, 0.07899, 0.10703, 0.13798, 0.16956, 0.19942, 0.22556, 0.24661, 0.26194, 0.27150,
            0.27566, 0.27503, 0.27045, 0.26344, 0.25587, 0.24928, 0.24303, 0.23681, 0.23020, 0.22382,
            0.21841, 0.21386, 0.21015, 0.20715])
        
        self.ken_coast_prev = np.array([
            0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00006, 0.00016, 0.00030, 0.00055,
            0.00099, 0.00175, 0.00302, 0.00510, 0.00843, 0.01354, 0.02089, 0.03084, 0.04351, 0.05842,
            0.07439, 0.08936, 0.10141, 0.11015, 0.11538, 0.11711, 0.11577, 0.11198, 0.10660, 0.10036,
            0.09369, 0.08681, 0.07997, 0.07347, 0.06773, 0.06277, 0.05856, 0.05504, 0.05194, 0.04944,
            0.04741, 0.04573, 0.04423, 0.04253, 0.04105, 0.03985, 0.03811, 0.03614, 0.03434, 0.03232,
            0.03021, 0.02871, 0.02751, 0.02587, 0.02393, 0.02205, 0.02033, 0.01876, 0.01736, 0.01610,
            0.01496])

    def test_ancprev_lnlhood_ancss(self):
        target = np.log(6.224725671139364e+20) # value from anclik package demo dataset
        anc = ancprev.ancprev(1970)
        anc.read_csv("tests/bwa-urban-anc.csv")
        lnlhood = anc.likelihood(self.bwa_urban_prev)
        self.assertTrue(np.isclose(lnlhood, target, rtol=1e-10, atol=1e-10))

    def test_ancprev_lnlhood_site(self):
        target = -24.213141355812205
        anc = ancprev.ancprev(1970)
        anc.read_csv("tests/ken-coast-anc.csv")
        lnlhood = anc.likelihood_site(self.ken_coast_prev)
        self.assertTrue(np.isclose(lnlhood, target, rtol=1e-10, atol=1e-10))

    def test_ancprev_lnlhood_census(self):
        target = -26.37767756638095
        anc = ancprev.ancprev(1970)
        anc.read_csv("tests/ken-coast-anc.csv")
        lnlhood = anc.likelihood_census(self.ken_coast_prev)
        self.assertTrue(np.isclose(lnlhood, target, rtol=1e-10, atol=1e-10))

if __name__ == "__main__":
    unittest.main()
