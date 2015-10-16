from experiments.Model import Model
import unittest
from mock import Mock, patch


class TestModel(unittest.TestCase):
	def setUp(self):
		self.base_training_parameters = {
			"GPU_identifier"	: 1,
			"number_of_GPUs"	: 4,
			"savingDirectory"	: "",
			"presavedModelPath"	: "",
			"modelFilePath"		: "",
			"maxepoch"			: 100, 
			"learningRate"		: 0.5, 
			"batchSize"			: 1500*4, 
			"momentum"			: 0.5, 
			"training_dataset" 	: "",
			"testing_dataset" 	: ""
		}

		self.base_model_parameters = {
			"NN_code_directory"		: "",
			"modelTemplateDirectory": "",
			"modelTemplate" 		: "",
			"modelFilePath"			: "",
			"activation_function"	: "ReLU",
			"pooling_function"		: "SpatialAveragePooling",
			"nfeaturemaps"  		: [32,64,100],
			"filtsize" 	  			: 5,
			"poolsize" 	  			: [2,2],
			"featuremaps_h" 		: 5,
			"featuremaps_w" 		: 5,
			"noutputs" 	  			: 2
		}

	@patch('experiments.Model.subprocess')
	def test_call_training_routine_calls_th_main(self, mock_subprocess):		
		model = Model(	self.base_training_parameters, 
						self.base_model_parameters)

		model.call_training_routine()

		self.assertTrue(mock_subprocess.call.called)

	@patch('experiments.Model.os')
	def test_train_calls_training_routine(self, mock_os):
		model = Model(	self.base_training_parameters, 
						self.base_model_parameters)

		model.call_training_routine = Mock()
		model.train()
		self.assertTrue(model.call_training_routine.called)

	