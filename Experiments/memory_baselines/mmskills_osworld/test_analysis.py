import unittest
from analyze_results import exact_mcnemar, holm, percent


class StatisticalChecks(unittest.TestCase):
    def test_exact_paired_binary_test(self):
        self.assertEqual(exact_mcnemar(0,0),1)
        self.assertEqual(exact_mcnemar(8,0),.0078125)
        self.assertEqual(exact_mcnemar(1,3),.625)
        self.assertEqual(exact_mcnemar(3,1),.625)

    def test_holm_and_rounding(self):
        self.assertEqual(holm([.01,.04,.03]),[.03,.06,.06])
        self.assertEqual(percent(1/32),'3.13%')
        self.assertEqual(percent(5/32),'15.63%')


if __name__=='__main__':
    unittest.main()
