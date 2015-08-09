from code.experiments import Experiment
import unittest


class TestExperiment(unittest.TestCase):
	def setUp(self):
		pass

	def test_my_second_bloody_test(self):
		self.assertEqual(4, 4)

if __name__ == "__main__":
	unittest.main()
