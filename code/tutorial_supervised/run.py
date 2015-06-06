import subprocess

options = {
"seed": 1 ,
"threads": 2,
"identifier": 2,
"size": "reduced",
"model": "convnet",
"maxepoch": 1,
"save": "results",
"learningRate": 0.1,
"batchSize": 64,
"weightDecay": 0.001,
"momentum": 0.001,
"type": "float"
}

command = "th main.lua -seed %(seed)i -threads %(threads)i -identifier %(identifier)i -size %(size)s -model %(model)s -maxepoch %(maxepoch)i " \
"-save %(save)s -learningRate %(learningRate)f -batchSize %(batchSize)i -weightDecay %(weightDecay)f " \
"-momentum %(momentum)f -type %(type)s" % options

subprocess.call(command, shell=True)