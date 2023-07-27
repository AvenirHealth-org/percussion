import numpy as np
import unittest
import percussion.ancprev as ancprev

# run from percussion toplevel folder: python -m tests.test_ancprev

class Test_TestANCPrev(unittest.TestCase):
    def test_ancprev_site_lnlhood(self):
        model_prev = np.array([0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00111, 0.00123, 0.00136, 0.00150, 0.00166,
                               0.00183, 0.00200, 0.00239, 0.00309, 0.00432, 0.00641, 0.00988, 0.01552, 0.02432, 0.03736,
                               0.05551, 0.07899, 0.10703, 0.13798, 0.16956, 0.19942, 0.22556, 0.24661, 0.26194, 0.27150,
                               0.27566, 0.27503, 0.27045, 0.26344, 0.25587, 0.24928, 0.24303, 0.23681, 0.23020, 0.22382,
                               0.21841, 0.21386, 0.21015, 0.20715])
        anc = ancprev.ancprev()
        anc.read_csv("tests/bwa-urban-anc.csv")
        lnlhood = anc.likelihood(model_prev)
        self.assertEqual(lnlhood, 47.88023123236735) # ln(6.224725671139364e+20)

if __name__ == "__main__":
    unittest.main()
