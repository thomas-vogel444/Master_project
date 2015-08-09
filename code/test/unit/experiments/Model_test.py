from code.experiments import Model
import unittest


class TestModel(unittest.TestCase):
	def setUp(self):
		pass

	def test_my_first_bloody_test(self):
		self.assertEqual(4, 4)

if __name__ == "__main__":
	unittest.main()
