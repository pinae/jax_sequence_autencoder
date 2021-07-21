import unittest
from nn_activations import relu, sigmoid, apr1


class ReluTestCase(unittest.TestCase):
    def test_relu_negative(self):
        self.assertAlmostEqual(0.0, relu(-1.0), places=6)

    def test_relu_zero(self):
        self.assertAlmostEqual(0.0, relu(0.0), places=6)

    def test_relu_pointfive(self):
        self.assertAlmostEqual(0.5, relu(0.5), places=6)

    def test_relu_one(self):
        self.assertAlmostEqual(1.0, relu(1.0), places=6)


class SigmoidTestCase(unittest.TestCase):
    def test_sigmoid_negative(self):
        self.assertAlmostEqual(0.2689414, sigmoid(-1.0), places=6)

    def test_sigmoid_zero(self):
        self.assertAlmostEqual(0.5, sigmoid(0.0), places=6)

    def test_sigmoid_pointfive(self):
        self.assertAlmostEqual(0.62245935, sigmoid(0.5), places=6)

    def test_sigmoid_one(self):
        self.assertAlmostEqual(1-0.2689414, sigmoid(1.0), places=6)


class Apr1TestCase(unittest.TestCase):
    def test_apr1_negative(self):
        self.assertAlmostEqual(0.0, apr1(-1.0), places=6)

    def test_apr1_zero(self):
        self.assertAlmostEqual(0.0, apr1(0.0), places=6)

    def test_apr1_pointfive(self):
        self.assertAlmostEqual(0.333333, apr1(0.5), places=6)

    def test_apr1_one(self):
        self.assertAlmostEqual(0.5, apr1(1.0), places=6)

    def test_apr1_thousand(self):
        self.assertLessEqual(apr1(1000.0), 1.0)


if __name__ == '__main__':
    unittest.main()
