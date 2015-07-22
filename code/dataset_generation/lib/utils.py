import sys

def drawProgressBar(percent, bar_length = 20):
    sys.stdout.write("\r")
    progress = ""
    for i in range(bar_length):
        if i < int(bar_length * percent):
            progress += "="
        else:
            progress += " "
    sys.stdout.write("[ %s ] %.2f%%" % (progress, percent*100))
    if percent == 1:
		sys.stdout.write("\n")    	
    sys.stdout.flush()