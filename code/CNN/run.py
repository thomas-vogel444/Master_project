import subprocess

options = {
"seed": 1 ,
"threads": 2,
"identifier": 2,
"dataset": "../../datasets/CNN_datasets.hdf5"
"size": "full",       	# options:  small, full
"model": "convnet",		# options: mlp, convnet, convnet2
"maxepoch": 15,
"save": "results",
"learningRate": 1,
"batchSize": 512,
"weightDecay": 0.000,
"momentum": 0.000,
"type": "cuda"
}

command_template = "th main.lua -seed %(seed)i -threads %(threads)i -identifier %(identifier)i -size %(size)s -model %(model)s -maxepoch %(maxepoch)i " \
"-save %(save)s -learningRate %(learningRate)f -batchSize %(batchSize)i -weightDecay %(weightDecay)f " \
"-momentum %(momentum)f -type %(type)s"

#***************************************************************************************************************
# options["learningRate"] = 10
# subprocess.call(command_template%options, shell=True)

# options["learningRate"] = 1
# subprocess.call(command_template%options, shell=True)

options["learningRate"] = 0.1
subprocess.call(command_template%options, shell=True)

# options["learningRate"] = 0.01
# subprocess.call(command_template%options, shell=True)

