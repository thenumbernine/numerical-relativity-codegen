#!/usr/bin/env luajit
require 'ext'
require 'symmath'.setup{implicitVars=true, MathJax={title='3x3 inverse'}}
local xs = table{'x', 'y', 'z'}
local gamma = Matrix:lambda({3,3}, function(i,j)
	return var('\\gamma^{'..xs[i]..xs[j]..'}')
end)
printbr(gamma)
local det = gamma:determinant()
printbr('det',det)
printbr(gamma:inverse(nil, function(AInv, A, i, j) 
	printbr(i, j, AInv, A)
end))
