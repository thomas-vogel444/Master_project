require "torch"

function normalize(X)
	X:add(-X:mean())
	X:div(X:std())
end