from experiments.Experiment import Experiment
from experiments.Model import Model

import unittest
from mock import Mock, MagicMock, patch, create_autospec

@patch('experiments.Experiment.os.path.join')
class TestExperiment(unittest.TestCase):

	def setUp(self):
		# Mock the model and set up the experiment object to test
		self.mock_model = create_autospec(Model)
		self.mock_model.training_parameters = MagicMock()
		self.mock_model.model_parameters = MagicMock()

		# Set up the model parameters to be used
		training_parameter_values 	= {"savingDirectory": "saving_directory"}
		model_parameter_values 		= {
				"modelFilePath"			: "model_file_path",
				"modelTemplateDirectory": "model_template_directory",
				"modelTemplate"			: "model_template"
			}
		
		def training_parameter_side_effects(key):
			return training_parameter_values[key]

		def model_parameter_side_effects(key):
			return model_parameter_values[key]

		self.mock_model.training_parameters.__getitem__.side_effect = training_parameter_side_effects
		self.mock_model.model_parameters.__getitem__.side_effect = model_parameter_side_effects

		self.experiment = Experiment(self.mock_model)

	def test_run_experiment_calls_model_train(self, mock_os_path_join):
		# Mock the other methods of the experiment object
		self.experiment.generate_model_source_file 	= Mock()
		self.experiment.save_as_json 				= Mock()
		self.experiment.save_model 					= Mock()

		self.experiment.run_experiment()

		self.assertTrue(self.mock_model.train.called)

	@patch('__builtin__.open')
	@patch('experiments.Experiment.json.dump')
	def test_save_as_json_calls_json_dumps(self, mock_open, mock_os_path_join, mock_json_dump):
		"""
			Tests that json.dump is called.
		"""
		self.experiment.save_as_json(Mock(), Mock())
		self.assertTrue(mock_json_dump.called)

	@patch('experiments.Experiment.shutil.move')
	@patch('experiments.Experiment.os.remove')
	@patch('experiments.Experiment.os.path.isfile')
	@patch('experiments.Experiment.os.path.basename')
	def test_save_model(self, mock_os_path_basename, mock_os_path_isfile, mock_os_remove, mock_shutil_move, mock_os_path_join):
		"""
			Tests Experiment.save_model.
			There are 4 modules to mock:
				- os.path.basename
				- os.path.isfile
				- os.remove
				- shutil.move
		"""	

		# Mock os.path.basename to return its argument for simplicity
		mock_os_path_basename.side_effect = lambda name: name

		# Scenario 1: filename doesn't exit
		mock_os_path_isfile.return_value = False
		self.experiment.save_model()
		self.assertFalse(mock_os_remove.called)
		self.assertTrue(mock_shutil_move.called)
		mock_shutil_move.reset_mock()

		# Scenario 2: filename does exist and so should call os.remove
		mock_os_path_isfile.return_value = True
		self.experiment.save_model()
		self.assertTrue(mock_os_remove.called) # Doesn't work as expected. Not sure why...
		mock_shutil_move.called

	@patch('experiments.Experiment.jinja2.FileSystemLoader')
	@patch('experiments.Experiment.jinja2.Environment')
	@patch('__builtin__.open', create = True)
	def test_generate_model_source_file(self, mock_open, mock_Environment, mock_FileSystemLoader, mock_os_path_join):
		mock_open.return_value = MagicMock(spec=file)
		self.experiment.generate_model_source_file()

		file_handle = mock_open.return_value.__enter__.return_value
		file_handle.write.called