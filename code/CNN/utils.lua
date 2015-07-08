require "torch"

function normalize(X)
	X:add(-X:mean())
	X:div(X:std())
end

function writeToFile(file, data, errorType) 
	file:write(errorType .. 'ClassificationError_' .. opt.identifier .. ": ")
	file:write(data[1])
	for i = 2, #data do
		file:write("," .. data[i])
	end
	file:write("\n")
end