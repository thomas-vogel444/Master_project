from string import Template
import shutil
import subprocess
import os

def train_model(seed			= 1, 
				threads			= 2, 
				identifier 		= 1, 
				dataset 		= "../../datasets/CNN_datasets.hdf5", 
				size 			= "full", 
				model 			= "convnet", 
				maxepoch		= 15, 
				save			= "../results", 
				learningRate	= 1, 
				batchSize		= 512, 
				weightDecay		= 0.0, 
				momentum		= 0.0, 
				type			= "cuda"):

	training_command = Template("th main.lua -seed $seed -threads $threads -identifier $identifier -dataset $dataset -size $size \
			-model $model -maxepoch $maxepoch -save $save -learningRate $learningRate -batchSize $batchSize -weightDecay $weightDecay \
			-momentum $momentum -type $type").substitute(
			seed			= seed, 
			threads			= threads, 
			identifier		= identifier, 
			dataset			= dataset, 
			size			= size, 
			model			= model, 
			maxepoch		= maxepoch, 
			save			= save, 
			learningRate	= learningRate, 
			batchSize		= batchSize, 
			weightDecay		= weightDecay, 
			momentum		= momentum, 
			type			= type)

	subprocess.call(training_command, shell=True)

#***************************************************************************************************************
if __name__ == "__main__":
	# Parameters for the set of experiments to be conducted
	experiment_directory = os.getcwd()
	code_directory = "../../code/CNN"
	os.chdir(code_directory)

	# A bunch of experiments
	learningRates	= (10, 1, 0.1, 0.01)
	type 			= "float"
	maxepoch 		= 1
	dataset 		= os.path.abspath("../../datasets/small_CNN_datasets.hdf5")

	for identifier, learningRate in enumerate(learningRates):
		result_directory = os.path.abspath(os.path.join(experiment_directory, str(identifier)))
		train_model(learningRate = 0.1, type = type, identifier = identifier, save = result_directory, maxepoch = maxepoch, dataset = dataset)

	os.chdir(experiment_directory)




















