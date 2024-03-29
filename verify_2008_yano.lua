#!/usr/bin/env luajit
--[[
The most-complete paper I've found on Z4 ... and it's not by the authors.  This seems to be a norm in academic journals.
This gives the left and right eigenvectors, the eigenvalues, the flux jacobian, and all the flux equations.
Now to verify that the math is correct (I'd decompose it with Mathematica myself ... but I tried ... and Mathematica spent days accomplishing nothing)
...and then to use the implementation.
--]]

-- starting at the bottom and working my way up to the top
require 'ext'
require 'symmath'.setup{MathJax={title='2008 Yano based implementation of Z4'}}

-- enable to verify the left and right eigenvectors work
local verify = true
local verifyFlux = true

local f = var'f'
local m = var'm'
local alpha = var'\\alpha'

local xs = table{'x', 'y', 'z'}
local syms = table{'xx', 'xy', 'xz', 'yy', 'yz', 'zz'}

local function sym(i,j)
	if i > j then i,j = j,i end
	return xs[i]..xs[j]
end

local chart = Tensor.Chart{coords=xs:mapi(function(x) return var(x) end)}

local gUvars = syms:mapi(function(xij) return var('\\gamma^{'..xij..'}') end)
local gUxx, gUxy, gUxz, gUyy, gUyz, gUzz = gUvars:unpack()
local gU = Tensor('^ij', function(i,j) return gUvars[assert(syms:find(sym(i,j)))] end)

local gLvars = syms:mapi(function(xij) return var('\\gamma_{'..xij..'}') end)
local gLxx, gLxy, gLxz, gLyy, gLyz, gLzz = gLvars:unpack()
local gL = Tensor('_ij', function(i,j) return gLvars[assert(syms:find(sym(i,j)))] end)

local dLvars = table()
for k,xk in ipairs(xs) do
	for _,xij in ipairs(syms) do
		dLvars:insert(var('d_{'..xk..xij..'}'))
	end
end
local dL = Tensor('_kij', function(k,i,j) return dLvars[6*(k-1)+syms:find(sym(i,j))] end)

local KLvars = syms:mapi(function(xij) return var('K_{'..xij..'}') end)
local KL = Tensor('_ij', function(i,j) return KLvars[assert(syms:find(sym(i,j)))] end)

local aL = Tensor('_i', function(i) return var('a_'..xs[i]) end)
local ZL = Tensor('_i', function(i) return var('Z_'..xs[i]) end)

local Theta = var'\\Theta'

local betaU = Tensor('^i', function(i) return var('\\beta^'..xs[i]) end)
local bUL = Tensor('^l_i', function(l,i) return var('{b^'..xs[l]..'}_'..xs[i]) end)
local Q1L = Tensor('_i', function(i) return var('Q1_{'..xs[i]..'}') end)
local Q2L = Tensor('_ij', function(i,j) return var('Q2_{'..sym(i,j)..'}') end)


local g = var'\\gamma'

local evl = Matrix:zeros{31,31}
evl[1+0][1+2] = frac(1,2)
evl[1+0][1+13] = frac(1,2) * ( gUxy^2 / gUxx - gUyy )
evl[1+0][1+14] = frac(1,2) * (gUxy * gUxz / gUxx - gUyz )
evl[1+0][1+18] = frac(1,2) * (-gUxy^2 / gUxx + gUyy )
evl[1+0][1+19] = frac(1,2) * (-gUxy * gUxz / gUxx + gUyz)
evl[1+1][1+1] = frac(1,2)
evl[1+1][1+13] = frac(1,2) * (-gUxy * gUxz / gUxx + gUyz)
evl[1+1][1+14] = frac(1,2) * (-gUxz * gUxz / gUxx + gUzz)
evl[1+1][1+18] = frac(1,2) * (gUxy * gUxz / gUxx - gUyz)
evl[1+1][1+19] = frac(1,2) * (gUxz * gUxz / gUxx - gUzz)
evl[1+2][1+20] = 1
evl[1+3][1+19] = 1
evl[1+4][1+18] = 1
evl[1+5][1+17] = 1
evl[1+6][1+16] = 1
evl[1+7][1+15] = 1
evl[1+8][1+14] = 1
evl[1+9][1+13] = 1
evl[1+10][1+12] = 1
evl[1+11][1+11] = 1
evl[1+12][1+10] = 1
evl[1+13][1+9] = 1
evl[1+14][1+2] = -1 / (2 * gUxx)
evl[1+14][1+5] = 1
evl[1+14][1+7] = gUxy / gUxx
evl[1+14][1+8] = gUxz / gUxx
evl[1+14][1+11] = -gUxy / (2 * gUxx)
evl[1+14][1+13] = gUyy / (2 * gUxx)
evl[1+14][1+14] = gUyz / (2 * gUxx)
evl[1+14][1+16] = -gUxy / (2 * gUxx)
evl[1+14][1+17] = -gUxz / gUxx
evl[1+14][1+18] = -gUyy / (2 * gUxx)
evl[1+14][1+19] = -gUyz / (2 * gUxx)
evl[1+14][1+30] = 1/gUxx
evl[1+15][1+1] = -1 / (2 * gUxx)
evl[1+15][1+4] = 1
evl[1+15][1+6] = gUxy / gUxx
evl[1+15][1+7] = gUxz / gUxx
evl[1+15][1+10] = -gUxy / gUxx
evl[1+15][1+11] = -gUxz / (2 * gUxx)
evl[1+15][1+13] = -gUyz / (2 * gUxx)
evl[1+15][1+14] = -gUzz / (2 * gUxx)
evl[1+15][1+16] = -gUxz / (2 * gUxx)
evl[1+15][1+18] = gUyz / (2 * gUxx)
evl[1+15][1+19] = gUzz / (2 * gUxx)
evl[1+15][1+29] = 1/ gUxx
evl[1+16][1+0] = -1 / (f * gUxx)
evl[1+16][1+1] = (-1 + f) * gUxy / (f * gUxx^2)
evl[1+16][1+2] = (-1 + f) * gUxz / (f * gUxx^2)
evl[1+16][1+3] = 1
evl[1+16][1+6] = (gUxy^2 * (-2 + m) - gUxx * gUyy * (-1 + m)) / gUxx^2
evl[1+16][1+7] = 2 * (gUxy * gUxz * (-2 + m) - gUxx * gUyz * (-1 + m)) / gUxx^2
evl[1+16][1+8] = (gUxz^2 * (-2 + m) - gUxx * gUzz * (-1 + m)) / gUxx^2
evl[1+16][1+10] = (gUxy^2 * (2 - m) + gUxx * gUyy * (-1 + m)) / gUxx^2
evl[1+16][1+11] = (gUxy * gUxz * (2 - m) + gUxx * gUyz * (-1 + m)) / gUxx^2
evl[1+16][1+13] = (gUxz * gUyy - gUxy * gUyz) * (-2 + m) / gUxx^2
evl[1+16][1+14] = (gUxz * gUyz - gUxy * gUzz) * (-2 + m) / gUxx^2
evl[1+16][1+16] = (gUxy * gUxz * (2 - m) + gUxx * gUyz * (-1 + m)) / gUxx^2
evl[1+16][1+17] = (gUxz^2 * (2 - m) + gUxx * gUzz * (-1 + m)) / gUxx^2
evl[1+16][1+18] = -(gUxz * gUyy - gUxy * gUyz) * (-2 + m) / gUxx^2
evl[1+16][1+19] = -(gUxz * gUyz - gUxy * gUzz) * (-2 + m) / gUxx^2
evl[1+16][1+28] = m / gUxx
evl[1+16][1+29] = gUxy * (-2 + m) / gUxx^2
evl[1+16][1+30] = gUxz * (-2 + m) / gUxx^2
evl[1+17][1+2] = -frac(1,4)
evl[1+17][1+13] = frac(1,4) * (-gUxy^2 / gUxx + gUyy)
evl[1+17][1+14] = frac(1,4) * (-gUxy * gUxz / gUxx + gUyz)
evl[1+17][1+18] = frac(1,4) * (gUxy^2 / gUxx - gUyy)
evl[1+17][1+19] = frac(1,4) * (gUxy * gUxz / gUxx - gUyz)
evl[1+17][1+23] = sqrt(gUxx) / 2
evl[1+17][1+25] = gUxy / (2 * sqrt(gUxx))
evl[1+17][1+26] = gUxz / (2 * sqrt(gUxx))
evl[1+17][1+30] = frac(1,2)
evl[1+18][1+1] = -frac(1,4)
evl[1+18][1+13] = frac(1,4) * (gUxy * gUxz / gUxx - gUyz)
evl[1+18][1+14] = frac(1,4) * (gUxz^2 / gUxx - gUzz)
evl[1+18][1+18] = frac(1,4) * (-gUxy * gUxz / gUxx + gUyz)
evl[1+18][1+19] = frac(1,4) * (-gUxz^2 / gUxx + gUzz)
evl[1+18][1+22] = sqrt(gUxx) / 2
evl[1+18][1+24] = gUxy / (2 * sqrt(gUxx))
evl[1+18][1+25] = gUxz / (2 * sqrt(gUxx))
evl[1+18][1+29] = frac(1,2)
evl[1+19][1+1] = gUxy / (4 * gUxx)
evl[1+19][1+2] = gUxz / (4 * gUxx)
evl[1+19][1+13] = (-gUxz * gUyy + gUxy * gUyz) / (4 * gUxx)
evl[1+19][1+14] = (-gUxz * gUyz + gUxy * gUzz) / (4 * gUxx)
evl[1+19][1+18] = (gUxz * gUyy - gUxy * gUyz) / (4 * gUxx)
evl[1+19][1+19] = (gUxz * gUyz - gUxy * gUzz) / (4 * gUxx)
evl[1+19][1+22] = -gUxy / (2 * sqrt(gUxx))
evl[1+19][1+23] = -gUxz / (2 * sqrt(gUxx))
evl[1+19][1+24] = -gUyy / (2 * sqrt(gUxx))
evl[1+19][1+25] = -gUyz / sqrt(gUxx)
evl[1+19][1+26] = -gUzz / (2 * sqrt(gUxx))
evl[1+19][1+27] = 1 / (2 * sqrt(gUxx))
evl[1+19][1+28] = frac(1,2)
evl[1+20][1+6] = (gUxy^2 - gUxx * gUyy) / (2 * sqrt(gUxx))
evl[1+20][1+7] = (gUxy * gUxz - gUxx * gUyz) / sqrt(gUxx)
evl[1+20][1+8] = (gUxz^2 - gUxx * gUzz) / (2 * sqrt(gUxx))
evl[1+20][1+10] = (-gUxy^2 + gUxx * gUyy) / (2 * sqrt(gUxx))
evl[1+20][1+11] = (-gUxy * gUxz + gUxx * gUyz) / (2 * sqrt(gUxx))
evl[1+20][1+13] = (gUxz * gUyy - gUxy * gUyz) / (2 * sqrt(gUxx))
evl[1+20][1+14] = (gUxz * gUyz - gUxy * gUzz) / (2 * sqrt(gUxx))
evl[1+20][1+16] = (-gUxy * gUxz + gUxx * gUyz) / (2 * sqrt(gUxx))
evl[1+20][1+17] = (-gUxz^2 + gUxx * gUzz) / (2 * sqrt(gUxx))
evl[1+20][1+18] = (-gUxz * gUyy + gUxy * gUyz) / (2 * sqrt(gUxx))
evl[1+20][1+19] = (-gUxz * gUyz + gUxy * gUzz) / (2 * sqrt(gUxx))
evl[1+20][1+27] = frac(1,2)
evl[1+20][1+28] = sqrt(gUxx) / 2
evl[1+20][1+29] = gUxy / (2 * sqrt(gUxx))
evl[1+20][1+30] = gUxz / (2 * sqrt(gUxx))
evl[1+21][1+8] = -sqrt(gUxx) / 2
evl[1+21][1+14] = -gUxy / (2 * sqrt(gUxx))
evl[1+21][1+17] = sqrt(gUxx) / 2
evl[1+21][1+19] = gUxy / (2 * sqrt(gUxx))
evl[1+21][1+26] = frac(1,2)
evl[1+22][1+7] = -sqrt(gUxx) / 2
evl[1+22][1+11] = sqrt(gUxx) / 4
evl[1+22][1+13] = -gUxy / (4 * sqrt(gUxx))
evl[1+22][1+14] = gUxz / (4 * sqrt(gUxx))
evl[1+22][1+16] = sqrt(gUxx) / 4
evl[1+22][1+18] = gUxy / (4 * sqrt(gUxx))
evl[1+22][1+19] = -gUxz / (4 * sqrt(gUxx))
evl[1+22][1+25] = frac(1,2)
evl[1+23][1+2] = -frac(1,4)
evl[1+23][1+13] = frac(1,4) * (-gUxy^2 / gUxx + gUyy)
evl[1+23][1+14] = frac(1,4) * (-gUxy * gUxz / gUxx + gUyz)
evl[1+23][1+18] = frac(1,4) * (gUxy^2 / gUxx - gUyy)
evl[1+23][1+19] = frac(1,4) * (gUxy * gUxz / gUxx - gUyz)
evl[1+23][1+23] = -sqrt(gUxx) / 2
evl[1+23][1+25] = -gUxy / (2 * sqrt(gUxx))
evl[1+23][1+26] = -gUxz / (2 * sqrt(gUxx))
evl[1+23][1+30] = frac(1,2)
evl[1+24][1+1] = -frac(1,4)
evl[1+24][1+13] = frac(1,4) * (gUxy * gUxz / gUxx - gUyz)
evl[1+24][1+14] = frac(1,4) * (gUxz^2 / gUxx - gUzz)
evl[1+24][1+18] = frac(1,4) * (-gUxy * gUxz / gUxx + gUyz)
evl[1+24][1+19] = frac(1,4) * (-gUxz^2 / gUxx + gUzz)
evl[1+24][1+22] = -sqrt(gUxx) / 2
evl[1+24][1+24] = -gUxy / (2 * sqrt(gUxx))
evl[1+24][1+25] = -gUxz / (2 * sqrt(gUxx))
evl[1+24][1+29] = frac(1,2)
evl[1+25][1+1] = gUxy / (4 * gUxx)
evl[1+25][1+2] = gUxz / (4 * gUxx)
evl[1+25][1+13] = (-gUxz * gUyy + gUxy * gUyz) / (4 * gUxx)
evl[1+25][1+14] = (-gUxz * gUyz + gUxy * gUzz) / (4 * gUxx)
evl[1+25][1+18] = (gUxz * gUyy - gUxy * gUyz) / (4 * gUxx)
evl[1+25][1+19] = (gUxz * gUyz - gUxy * gUzz) / (4 * gUxx)
evl[1+25][1+22] = gUxy / (2 * sqrt(gUxx))
evl[1+25][1+23] = gUxz / (2 * sqrt(gUxx))
evl[1+25][1+24] = gUyy / (2 * sqrt(gUxx))
evl[1+25][1+25] = gUyz / sqrt(gUxx)
evl[1+25][1+26] = gUzz / (2 * sqrt(gUxx))
evl[1+25][1+27] = -1 / (2 * sqrt(gUxx))
evl[1+25][1+28] = frac(1,2)
evl[1+26][1+6] = (-gUxy^2 + gUxx * gUyy) / (2 * sqrt(gUxx))
evl[1+26][1+7] = (-gUxy * gUxz + gUxx * gUyz) / sqrt(gUxx)
evl[1+26][1+8] = (-gUxz^2 + gUxx * gUzz) / (2 * sqrt(gUxx))
evl[1+26][1+10] = (gUxy^2 - gUxx * gUyy) / (2 * sqrt(gUxx))
evl[1+26][1+11] = (gUxy * gUxz - gUxx * gUyz) / (2 * sqrt(gUxx))
evl[1+26][1+13] = (-gUxz * gUyy + gUxy * gUyz) / (2 * sqrt(gUxx))
evl[1+26][1+14] = (-gUxz * gUyz + gUxy * gUzz) / (2 * sqrt(gUxx))
evl[1+26][1+16] = (gUxy * gUxz - gUxx * gUyz) / (2 * sqrt(gUxx))
evl[1+26][1+17] = (gUxz^2 - gUxx * gUzz) / (2 * sqrt(gUxx))
evl[1+26][1+18] = (gUxz * gUyy - gUxy * gUyz) / (2 * sqrt(gUxx))
evl[1+26][1+19] = (gUxz * gUyz - gUxy * gUzz) / (2 * sqrt(gUxx))
evl[1+26][1+27] = frac(1,2)
evl[1+26][1+28] = -sqrt(gUxx) / 2
evl[1+26][1+29] = -gUxy / (2 * sqrt(gUxx))
evl[1+26][1+30] = -gUxz / (2 * sqrt(gUxx))
evl[1+27][1+8] = sqrt(gUxx) / 2
evl[1+27][1+14] = gUxy / (2 * sqrt(gUxx))
evl[1+27][1+17] = -sqrt(gUxx) / 2
evl[1+27][1+19] = -gUxy / (2 * sqrt(gUxx))
evl[1+27][1+26] = frac(1,2)
evl[1+28][1+7] = sqrt(gUxx) / 2
evl[1+28][1+11] = -sqrt(gUxx) / 4
evl[1+28][1+13] = gUxy / (4 * sqrt(gUxx))
evl[1+28][1+14] = -gUxz / (4 * sqrt(gUxx))
evl[1+28][1+16] = -sqrt(gUxx) / 4
evl[1+28][1+18] = -gUxy / (4 * sqrt(gUxx))
evl[1+28][1+19] = gUxz / (4 * sqrt(gUxx))
evl[1+28][1+25] = frac(1,2)
evl[1+29][1+0] = -1/(2 * sqrt(f * gUxx))
evl[1+29][1+1] = -gUxy / (2 * gUxx * sqrt(f * gUxx))
evl[1+29][1+2] = -gUxz / (2 * gUxx * sqrt(f * gUxx))
evl[1+29][1+6] = sqrt(f) * ((-gUxy^2 + gUxx * gUyy) * (-2 + m)) / (2 * (-1 + f) * gUxx * sqrt(gUxx))
evl[1+29][1+7] = sqrt(f) * ((-gUxy * gUxz + gUxx * gUyz) * (-2 + m )) / ((-1 + f) * gUxx * sqrt(gUxx))
evl[1+29][1+8] = sqrt(f) * ((-gUxz^2 + gUxx * gUzz) * (-2 + m)) / (2 * (-1 + f) * gUxx * sqrt(gUxx))
evl[1+29][1+10] = sqrt(f) * ((gUxy^2 - gUxx * gUyy) * (-2 + m)) / (2 * (-1 + f) * gUxx * sqrt(gUxx))
evl[1+29][1+11] = sqrt(f) * ((gUxy * gUxz - gUxx * gUyz) * (-2 + m)) / (2 * (-1 + f) * gUxx * sqrt(gUxx))
evl[1+29][1+13] = sqrt(f) * ((-gUxz * gUyy + gUxy * gUyz) * (-2 + m)) / (2 * (-1 + f) * gUxx * sqrt(gUxx))
evl[1+29][1+14] = sqrt(f) * ((-gUxz * gUyz + gUxy * gUzz) * (-2 + m)) / (2 * (-1 + f) * gUxx * sqrt(gUxx))
evl[1+29][1+16] = sqrt(f) * ((gUxy * gUxz - gUxx * gUyz) * (-2 + m)) / (2 * (-1 + f) * gUxx * sqrt(gUxx))
evl[1+29][1+17] = sqrt(f) * ((gUxz^2 - gUxx * gUzz) * (-2 + m)) / (2 * (-1 + f) * gUxx * sqrt(gUxx))
evl[1+29][1+18] = sqrt(f) * ((gUxz * gUyy - gUxy * gUyz) * (-2 + m)) / (2 * (-1 + f) * gUxx * sqrt(gUxx))
evl[1+29][1+19] = sqrt(f) * ((gUxz * gUyz - gUxy * gUzz) * (-2 + m)) / (2 * (-1 + f) * gUxx * sqrt(gUxx))
evl[1+29][1+21] = frac(1,2)
evl[1+29][1+22] = gUxy / gUxx
evl[1+29][1+23] = gUxz / gUxx
evl[1+29][1+24] = gUyy / (2 * gUxx)
evl[1+29][1+25] = gUyz / gUxx
evl[1+29][1+26] = gUzz / (2 * gUxx)
evl[1+29][1+27] = (2 - f * m) / (-2 * gUxx + 2 * f * gUxx)
evl[1+29][1+28] = -sqrt(f) * (-2 + m) / (2 * (-1 + f) * sqrt(gUxx))
evl[1+29][1+29] = -sqrt(f) * gUxy * (-2 + m) / (2 * (-1 + f) * gUxx * sqrt(gUxx))
evl[1+29][1+30] = -sqrt(f) * gUxz * (-2 + m) / (2 * (-1 + f) * gUxx * sqrt(gUxx))
evl[1+30][1+0] = 1 / (2 * sqrt(f * gUxx))
evl[1+30][1+1] = gUxy / (2 * gUxx * sqrt(f * gUxx))
evl[1+30][1+2] = gUxz / (2 * gUxx * sqrt(f * gUxx))
evl[1+30][1+6] = sqrt(f) * (gUxy^2 - gUxx * gUyy) * (-2 + m) / (2 * (-1 + f) * gUxx * sqrt(gUxx))
evl[1+30][1+7] = sqrt(f) * (gUxy * gUxz - gUxx * gUyz) * (-2 + m) / ((-1 + f) * gUxx * sqrt(gUxx))
evl[1+30][1+8] = sqrt(f) * (gUxz^2 - gUxx * gUzz) * (-2 + m) / (2 * (-1 + f) * gUxx * sqrt(gUxx))
evl[1+30][1+10] = sqrt(f) * (-gUxy^2 + gUxx * gUyy) * (-2 + m) / (2 * (-1 + f) * gUxx * sqrt(gUxx))
evl[1+30][1+11] = sqrt(f) * (-gUxy * gUxz + gUxx * gUyz) * (-2 + m) / (2 * (-1 + f) * gUxx * sqrt(gUxx))
evl[1+30][1+13] = sqrt(f) * (gUxz * gUyy - gUxy * gUyz) * (-2 + m) / (2 * (-1 + f) * gUxx * sqrt(gUxx))
evl[1+30][1+14] = sqrt(f) * (gUxz * gUyz - gUxy * gUzz) * (-2 + m) / (2 * (-1 + f) * gUxx * sqrt(gUxx))
evl[1+30][1+16] = sqrt(f) * (-gUxy * gUxz + gUxx * gUyz) * (-2 + m) / (2 * (-1 + f) * gUxx * sqrt(gUxx))
evl[1+30][1+17] = sqrt(f) * (-gUxz^2 + gUxx * gUzz) * (-2 + m) / (2 * (-1 + f) * gUxx * sqrt(gUxx))
evl[1+30][1+18] = sqrt(f) * (-gUxz * gUyy + gUxy * gUyz) * (-2 + m) / (2 * (-1 + f) * gUxx * sqrt(gUxx))
evl[1+30][1+19] = sqrt(f) * (-gUxz * gUyz + gUxy * gUzz) * (-2 + m) / (2 * (-1 + f) * gUxx * sqrt(gUxx))
evl[1+30][1+21] = frac(1,2)
evl[1+30][1+22] = gUxy / gUxx
evl[1+30][1+23] = gUxz / gUxx
evl[1+30][1+24] = gUyy / (2 * gUxx)
evl[1+30][1+25] = gUyz / gUxx
evl[1+30][1+26] = gUzz / (2 * gUxx)
evl[1+30][1+27] = (2 - f * m) / (-2 * gUxx + 2 * f * gUxx)
evl[1+30][1+28] = sqrt(f) * (-2 + m) / (2 * (-1 + f) * sqrt(gUxx))
evl[1+30][1+29] = sqrt(f) * gUxy * (-2 + m) / (2 * (-1 + f) * gUxx * sqrt(gUxx))
evl[1+30][1+30] = sqrt(f) * gUxz * (-2 + m) / (2 * (-1 + f) * gUxx * sqrt(gUxx))
evl = clone(evl)

printbr(var'L':eq(evl))	-- print L before g_ij substitution


local evr = Matrix:zeros{31,31}
evr[1+0][1+0] = -2 * gUxz / gUxx
evr[1+0][1+1] = -2 * gUxy / gUxx
evr[1+0][1+3] = (gUxz * gUyz - gUxy * gUzz) / gUxx
evr[1+0][1+4] = (gUxz * gUyy - gUxy * gUyz) / gUxx
evr[1+0][1+8] = (-gUxz * gUyz + gUxy * gUzz) / gUxx
evr[1+0][1+9] = (-gUxz * gUyy + gUxy * gUyz) / gUxx
evr[1+0][1+20] = -f * (-2 + m) / ((-1 + f) * sqrt(gUxx))
evr[1+0][1+26] = f * (-2 + m) / ((-1 + f) * sqrt(gUxx))
evr[1+0][1+29] = -sqrt(f * gUxx)
evr[1+0][1+30] = sqrt(f * gUxx)
evr[1+1][1+1] = 2
evr[1+1][1+3] = -gUxz^2 / gUxx + gUzz
evr[1+1][1+4] = -gUxy * gUxz / gUxx + gUyz
evr[1+1][1+8] = gUxz^2 / gUxx - gUzz
evr[1+1][1+9] = gUxy * gUxz / gUxx - gUyz
evr[1+2][1+0] = 2
evr[1+2][1+3] = gUxy * gUxz / gUxx - gUyz
evr[1+2][1+4] = gUxy^2 / gUxx - gUyy
evr[1+2][1+8] = -gUxy * gUxz / gUxx + gUyz
evr[1+2][1+9] = -gUxy^2 / gUxx + gUyy
evr[1+3][1+16] = 1
evr[1+3][1+17] = gUxz * gUyy / (-gUxx * gUxy^2 + gUxx^2 * gUyy)
evr[1+3][1+18] = gUxy * gUyy / (-gUxx * gUxy^2 + gUxx^2 * gUyy)
evr[1+3][1+19] = (-1 + gUxy^2 / (-gUxy^2 + gUxx * gUyy)) / gUxx
evr[1+3][1+20] = (f * gUxy^2 * (m - 2) + gUxx * gUyy * (1 + f * (1 - m))) / ((f - 1) * gUxx * sqrt(gUxx) * (-gUxy^2 + gUxx * gUyy))
evr[1+3][1+21] = (-gUxz^2 * gUyy + gUxy^2 * gUzz) / (gUxx * sqrt(gUxx) * (-gUxy^2 + gUxx * gUyy))
evr[1+3][1+22] = (2 * gUxy * (-gUxz * gUyy + gUxy * gUyz)) / (gUxx * sqrt(gUxx) * (-gUxy^2 + gUxx * gUyy))
evr[1+3][1+23] = (gUxz * gUyy) / (-gUxx * gUxy^2 + gUxx^2 * gUyy)
evr[1+3][1+24] = gUxy * gUyy / (-gUxx * gUxy^2 + gUxx^2 * gUyy)
evr[1+3][1+25] = (-1 + gUxy^2 / (-gUxy^2 + gUxx * gUyy)) / gUxx
evr[1+3][1+26] = (gUxy^2 / (-gUxy^2 + gUxx * gUyy) + (-1 + f * (-1 + m)) / (-1 + f)) / (gUxx * sqrt(gUxx))
evr[1+3][1+27] = (gUxz^2 * gUyy - gUxy^2 * gUzz) / (gUxx * sqrt(gUxx) * (-gUxy^2 + gUxx * gUyy))
evr[1+3][1+28] = (2 * gUxy * (gUxz * gUyy - gUxy * gUyz)) / (gUxx * sqrt(gUxx) * (-gUxy^2 + gUxx * gUyy))
evr[1+3][1+29] = -1 / sqrt(f * gUxx)
evr[1+3][1+30] = 1 / sqrt(f * gUxx)

evr[1+4][1+15] = 1
evr[1+4][1+17] = gUxy * gUxz / (gUxx * (gUxy^2 - gUxx * gUyy))
evr[1+4][1+18] = gUyy / (gUxy^2 - gUxx * gUyy)
evr[1+4][1+19] = gUxy / (gUxy^2 - gUxx * gUyy)
evr[1+4][1+20] = gUxy / (sqrt(gUxx) * (-gUxy^2 + gUxx * gUyy))
evr[1+4][1+21] = gUxy * (gUxz^2 - gUxx * gUzz) / (gUxx * sqrt(gUxx) * (-gUxy^2 + gUxx * gUyy))
evr[1+4][1+22] = (gUxz * (gUxy^2 + gUxx * gUyy) - 2 * (gUxx * gUxy * gUyz)) / ( gUxx * sqrt(gUxx) * (-gUxy^2 + gUxx * gUyy) )

--evr[1+4][1+23] = gUxy^2 / (gUxx * ( gUxy^2 - gUxx * gUyy) )		-- in paper
evr[1+4][1+23] = gUxy * gUxz / (gUxx * ( gUxy^2 - gUxx * gUyy) )

evr[1+4][1+24] = gUyy / (gUxy^2 - gUxx * gUyy)
evr[1+4][1+25] = gUxy / (gUxy^2 - gUxx * gUyy)
evr[1+4][1+26] = gUxy / (sqrt(gUxx) * (gUxy^2 - gUxx * gUyy))
evr[1+4][1+27] = gUxy * (-gUxz^2 + gUxx * gUzz) / (gUxx * sqrt(gUxx) * (-gUxy^2 + gUxx * gUyy))
evr[1+4][1+28] = (-gUxz * (gUxy^2 + gUxx * gUyy) + 2 * gUxx * gUxy * gUyz) / (gUxx * sqrt(gUxx) * (-gUxy^2 + gUxx * gUyy))

evr[1+5][1+14] = 1
evr[1+5][1+17] = -1/gUxx
evr[1+5][1+21] = gUxz / (gUxx * sqrt(gUxx))
evr[1+5][1+22] = gUxy / (gUxx * sqrt(gUxx))
evr[1+5][1+23] = -1/gUxx
evr[1+5][1+27] = -gUxz / (gUxx * sqrt(gUxx))
evr[1+5][1+28] = -gUxy / (gUxx * sqrt(gUxx))
evr[1+6][1+4] = -gUxz / gUxx
evr[1+6][1+9] = gUxz / gUxx
evr[1+6][1+12] = 1
evr[1+6][1+17] = gUxz / (-gUxy^2 + gUxx * gUyy)
evr[1+6][1+18] = gUxy / (-gUxy^2 + gUxx * gUyy)
evr[1+6][1+19] = gUxx / (-gUxy^2 + gUxx * gUyy)
evr[1+6][1+20] = sqrt(gUxx) / (gUxy^2 - gUxx * gUyy)
evr[1+6][1+21] = (gUxz^2 - gUxx * gUzz) / (sqrt(gUxx) * (gUxy^2 - gUxx * gUyy))
evr[1+6][1+22] = (2 * (gUxy * gUxz - gUxx * gUyz)) / (sqrt(gUxx) * (gUxy^2 - gUxx * gUyy))
evr[1+6][1+23] = gUxz / (-gUxy^2 + gUxx * gUyy)
evr[1+6][1+24] = gUxy / (-gUxy^2 + gUxx * gUyy)
evr[1+6][1+25] = gUxx / (-gUxy^2 + gUxx * gUyy)
evr[1+6][1+26] = sqrt(gUxx) / (-gUxy^2 + gUxx * gUyy)
evr[1+6][1+27] = (gUxz^2 - gUxx * gUzz) / (sqrt(gUxx) * (-gUxy^2 + gUxx * gUyy))
evr[1+6][1+28] = (-2 * (gUxy * gUxz - gUxx * gUyz)) / (sqrt(gUxx) * (gUxy^2 - gUxx * gUyy))
evr[1+7][1+3] = -gUxz / (2 * gUxx)
evr[1+7][1+4] = gUxy / (2 * gUxx)
evr[1+7][1+6] = frac(1,2)	-- 0 in the paper
evr[1+7][1+8] = gUxz / (2 * gUxx)
evr[1+7][1+9] = -gUxy / (2 * gUxx)
evr[1+7][1+11] = frac(1,2)
evr[1+7][1+22] = -1 / sqrt(gUxx)
evr[1+7][1+28] = 1 / sqrt(gUxx)

evr[1+8][1+3] = gUxy / gUxx
evr[1+8][1+5] = 1
evr[1+8][1+8] = -gUxy / gUxx
evr[1+8][1+21] = -1/sqrt(gUxx)
evr[1+8][1+27] = 1/sqrt(gUxx)

evr[1+9][1+13] = 1
evr[1+10][1+12] = 1
evr[1+11][1+11] = 1
evr[1+12][1+10] = 1
evr[1+13][1+9] = 1
evr[1+14][1+8] = 1
evr[1+15][1+7] = 1
evr[1+16][1+6] = 1
evr[1+17][1+5] = 1
evr[1+18][1+4] = 1
evr[1+19][1+3] = 1
evr[1+20][1+2] = 1
evr[1+21][1+17] = gUxz * gUyy / (sqrt(gUxx) * (gUxy^2 - gUxx * gUyy))
evr[1+21][1+18] = gUxy * gUyy / (sqrt(gUxx) * (gUxy^2 - gUxx * gUyy))
evr[1+21][1+19] = (1 + gUxy^2 / (gUxy^2 - gUxx * gUyy)) / sqrt(gUxx)
evr[1+21][1+20] = (gUxy^2 / (-gUxy^2 + gUxx * gUyy) + (-1 + f * (-1 + m)) / (-1 + f)) / gUxx
evr[1+21][1+21] = (gUxz^2 * gUyy - gUxy^2 * gUzz) / (-gUxx * gUxy^2 + gUxx^2 * gUyy)
evr[1+21][1+22] = (2 * gUxy * (gUxz * gUyy - gUxy * gUyz)) / (gUxx * (-gUxy^2 + gUxx * gUyy))
evr[1+21][1+23] = gUxz * gUyy / (sqrt(gUxx) * (-gUxy^2 + gUxx * gUyy))
evr[1+21][1+24] = gUxy * gUyy / (sqrt(gUxx) * (-gUxy^2 + gUxx * gUyy))

--evr[1+21][1+25] = (2 * (gUxy * gUxy)^2 - gUxx * gUyy) / (sqrt(gUxx) * (-gUxy^2 + gUxx * gUyy))		-- in paper
evr[1+21][1+25] = (2 * gUxy^2 - gUxx * gUyy) / (sqrt(gUxx) * (-gUxy^2 + gUxx * gUyy))

evr[1+21][1+26] = (gUxy^2 / (-gUxy^2 + gUxx * gUyy) + (-1 + f * (-1 + m)) / (-1 + f)) / gUxx
evr[1+21][1+27] = (gUxz^2 * gUyy - gUxy^2 * gUzz) / (-gUxx * gUxy^2 + gUxx^2 * gUyy)
evr[1+21][1+28] = (2 * gUxy * (gUxz * gUyy - gUxy * gUyz)) / (gUxx * (-gUxy^2 + gUxx * gUyy))
evr[1+21][1+29] = 1
evr[1+21][1+30] = 1
evr[1+22][1+17] = gUxy * gUxz / (sqrt(gUxx) * (-gUxy^2 + gUxx * gUyy))
evr[1+22][1+18] = sqrt(gUxx) * gUyy / (-gUxy^2 + gUxx * gUyy)
evr[1+22][1+19] = sqrt(gUxx) * gUxy / (-gUxy^2 + gUxx * gUyy)
evr[1+22][1+20] = gUxy / (gUxy^2 - gUxx * gUyy)
evr[1+22][1+21] = gUxy * (-gUxz^2 + gUxx * gUzz) / (gUxx * (-gUxy^2 + gUxx * gUyy))
evr[1+22][1+22] = (gUxz * (gUxy^2 + gUxx * gUyy) - 2 * gUxx * gUxy * gUyz) / (gUxx * (gUxy^2 - gUxx * gUyy))
evr[1+22][1+23] = gUxy * gUxz / (sqrt(gUxx) * (gUxy^2 - gUxx * gUyy))
evr[1+22][1+24] = sqrt(gUxx) * gUyy / (gUxy^2 - gUxx * gUyy)
evr[1+22][1+25] = sqrt(gUxx) * gUxy / (gUxy^2 - gUxx * gUyy)
evr[1+22][1+26] = gUxy / (gUxy^2 - gUxx * gUyy)
evr[1+22][1+27] = gUxy * (-gUxz^2 + gUxx * gUzz) / (gUxx * (-gUxy^2 + gUxx * gUyy))
evr[1+22][1+28] = (gUxz * (gUxy^2 + gUxx * gUyy) - 2 * gUxx * gUxy * gUyz) / (gUxx * (gUxy^2 - gUxx * gUyy))
evr[1+23][1+17] = 1/sqrt(gUxx)
evr[1+23][1+21] = -gUxz / gUxx
evr[1+23][1+22] = -gUxy / gUxx
evr[1+23][1+23] = -1/sqrt(gUxx)
evr[1+23][1+27] = -gUxz / gUxx
evr[1+23][1+28] = -gUxy / gUxx
evr[1+24][1+17] = sqrt(gUxx) * gUxz / (gUxy^2 - gUxx * gUyy)
evr[1+24][1+18] = sqrt(gUxx) * gUxy / (gUxy^2 - gUxx * gUyy)
evr[1+24][1+19] = gUxx * sqrt(gUxx) / (gUxy^2 - gUxx * gUyy)
evr[1+24][1+20] = gUxx / (-gUxy^2 + gUxx * gUyy)
evr[1+24][1+21] = (gUxz^2 - gUxx * gUzz) / (-gUxy^2 + gUxx * gUyy)
evr[1+24][1+22] = (2 * gUxy * gUxz - 2 * gUxx * gUyz) / (-gUxy^2 + gUxx * gUyy)
evr[1+24][1+23] = sqrt(gUxx) * gUxz / (-gUxy^2 + gUxx * gUyy)
evr[1+24][1+24] = sqrt(gUxx) * gUxy / (-gUxy^2 + gUxx * gUyy)
evr[1+24][1+25] = gUxx * sqrt(gUxx) / (-gUxy^2 + gUxx * gUyy)
evr[1+24][1+26] = gUxx / (-gUxy^2 + gUxx * gUyy)
evr[1+24][1+27] = (gUxz^2 - gUxx * gUzz) / (-gUxy^2 + gUxx * gUyy)
evr[1+24][1+28] = (2 * gUxy * gUxz - 2 * gUxx * gUyz) / (-gUxy^2 + gUxx * gUyy)
evr[1+25][1+22] = 1
evr[1+25][1+28] = 1
evr[1+26][1+21] = 1
evr[1+26][1+27] = 1
evr[1+27][1+20] = 1
evr[1+27][1+26] = 1
evr[1+28][1+0] = -gUxz / gUxx
evr[1+28][1+1] = -gUxy / gUxx
evr[1+28][1+19] = 1
evr[1+28][1+25] = 1
evr[1+29][1+1] = 1
evr[1+29][1+18] = 1
evr[1+29][1+24] = 1
evr[1+30][1+0] = 1
evr[1+30][1+17] = 1
evr[1+30][1+23] = 1
evr = clone(evr)

printbr(var'R':eq(evr))	-- print R before g_ij substitution


local numErrors = 0
local n = #evr	-- hope everything is square ...

if verify then
	local evr_evl = ( evr * evl )()
	-- hmm ... simplify() isn't fully doing its job ...
	evr_evl = evr_evl()
	printbr( (var'R' * var'L'):eq(evr_evl) )
	for i=1,n do
		for j=1,n do
			local expected = Constant(i == j and 1 or 0)
			if evr_evl[i][j] ~= expected then
				printbr('for '..i..','..j..' expected '..expected..' but found '..evr_evl[i][j])
				for k=1,n do
					--if evr[i][k] ~= Constant(0) and evl[k][j] ~= Constant(0) then
						printbr('...influenced by ($R_{1+'..(i-1)..',1+'..(k-1)..'} = $'..evr[i][k]..') * ($L_{1+'..(k-1)..',1+'..(j-1)..'} = $'..evl[k][j]..')')
					--end
				end
				printbr()
				numErrors = numErrors + 1
			end
		end
	end
end

if verify then
	local evl_evr = ( evl * evr )()
	-- hmm ... simplify() isn't fully doing its job ...
	evl_evr = evl_evr()
	printbr( (var'L' * var'R'):eq(evl_evr) )
	for i=1,n do
		for j=1,n do
			local expected = Constant(i == j and 1 or 0)
			if evl_evr[i][j] ~= expected then
				printbr('for '..i..','..j..' expected '..expected..' but found '..evl_evr[i][j])
				for k=1,n do
					--if evl[i][k] ~= Constant(0) and evr[k][j] ~= Constant(0) then
						printbr('...influenced by ($L_{1+'..(i-1)..',1+'..(k-1)..'} = $'..evl[i][k]..') * ($R_{1+'..(k-1)..',1+'..(j-1)..'} = $'..evr[k][j]..')')
					--end
				end
				printbr()
				numErrors = numErrors + 1
			end
		end
	end
end

--[[
local start = os.time()
io.stderr:write'inverting...\n' io.stderr:flush()

-- this takes a few seconds to invert
-- it also could stand to be simplified a bit
-- beats typing it in every time
local evrCheck = evl:inverse()
io.stderr:write('...done (took '..(os.time() - start)..' seconds)\n') io.stderr:flush()
printbr(var'R':eq(evrCheck))

printbr'difference:'
printbr((evr - evrCheck)())
--]]

local lambdaDiags = range(0,30):mapi(function(i)
	if i >= 0 and i <= 16 then return 0 end
	if i >= 17 and i <= 22 then return -alpha * sqrt(gUxx) end
	if i >= 23 and i <= 28 then return alpha * sqrt(gUxx) end
	if i == 29 then return -alpha * sqrt(f * gUxx) end
	if i == 30 then return alpha * sqrt(f * gUxx) end
end)
local Lambda = Matrix.diagonal(lambdaDiags:unpack())
printbr(var'\\Lambda':eq(Lambda))

-- A_ij / alpha
local A_alpha = Matrix:zeros{31,31}

A_alpha[1+0][1+21] = f * gUxx
A_alpha[1+0][1+22] = 2 * f * gUxy
A_alpha[1+0][1+23] = 2 * f * gUxz
A_alpha[1+0][1+24] = f * gUyy
A_alpha[1+0][1+25] = 2 * f * gUyz
A_alpha[1+0][1+26] = f * gUzz
A_alpha[1+0][1+27] = -f * m

A_alpha[1+3][1+21] = 1
A_alpha[1+4][1+22] = 1
A_alpha[1+5][1+23] = 1
A_alpha[1+6][1+24] = 1
A_alpha[1+7][1+25] = 1
A_alpha[1+8][1+26] = 1

A_alpha[1+21][1+0] = 1
A_alpha[1+21][1+6] = gUyy
A_alpha[1+21][1+7] = 2 * gUyz
A_alpha[1+21][1+8] = gUzz
A_alpha[1+21][1+10] = -gUyy
A_alpha[1+21][1+11] = -gUyz
A_alpha[1+21][1+16] = -gUyz
A_alpha[1+21][1+17] = -gUzz
A_alpha[1+21][1+28] = -2

A_alpha[1+22][1+1] = frac(1,2)
A_alpha[1+22][1+6] = -gUxy
A_alpha[1+22][1+7] = -gUxz
A_alpha[1+22][1+10] = gUxy
A_alpha[1+22][1+11] = gUxz / 2
A_alpha[1+22][1+13] = gUyz / 2
A_alpha[1+22][1+14] = gUzz / 2
A_alpha[1+22][1+16] = gUxz / 2
A_alpha[1+22][1+18] = -gUyz / 2
A_alpha[1+22][1+19] = -gUzz / 2
A_alpha[1+22][1+29] = -1
 
A_alpha[1+23][1+2] = frac(1,2)
A_alpha[1+23][1+7] = -gUxy
A_alpha[1+23][1+8] = -gUxz
A_alpha[1+23][1+11] = gUxy / 2
A_alpha[1+23][1+13] = -gUyy / 2
A_alpha[1+23][1+14] = -gUyz / 2
A_alpha[1+23][1+16] = gUxy / 2
A_alpha[1+23][1+17] = gUxz
A_alpha[1+23][1+18] = gUyy / 2
A_alpha[1+23][1+19] = gUyz / 2
A_alpha[1+23][1+30] = -1

A_alpha[1+24][1+6] = gUxx
A_alpha[1+24][1+10] = -gUxx
A_alpha[1+24][1+13] = -gUxz
A_alpha[1+24][1+18] = gUxz

A_alpha[1+25][1+7] = gUxx
A_alpha[1+25][1+11] = -gUxx / 2
A_alpha[1+25][1+13] = gUxy / 2
A_alpha[1+25][1+14] = -gUxz / 2
A_alpha[1+25][1+16] = -gUxx / 2
A_alpha[1+25][1+18] = -gUxy / 2
A_alpha[1+25][1+19] = gUxz / 2

A_alpha[1+26][1+8] = gUxx
A_alpha[1+26][1+14] = gUxy
A_alpha[1+26][1+17] = -gUxx
A_alpha[1+26][1+19] = -gUxy

A_alpha[1+27][1+6] = -gUxy^2 + gUxx * gUyy
A_alpha[1+27][1+7] = -2 * gUxy * gUxz + 2 * gUxx * gUyz
A_alpha[1+27][1+8] = -gUxz^2 + gUxx * gUzz
A_alpha[1+27][1+10] = gUxy^2 - gUxx * gUyy
A_alpha[1+27][1+11] = gUxy * gUxz - gUxx * gUyz
A_alpha[1+27][1+13] = -gUxz * gUyy + gUxy * gUyz
A_alpha[1+27][1+14] = -gUxz * gUyz + gUxy * gUzz
A_alpha[1+27][1+16] = gUxy * gUxz - gUxx * gUyz
A_alpha[1+27][1+17] = gUxz^2 - gUxx * gUzz
A_alpha[1+27][1+18] = gUxz * gUyy - gUxy * gUyz
A_alpha[1+27][1+19] = gUxz * gUyz - gUxy * gUzz
A_alpha[1+27][1+28] = -gUxx
A_alpha[1+27][1+29] = -gUxy
A_alpha[1+27][1+30] = -gUxz

A_alpha[1+28][1+22] = gUxy
A_alpha[1+28][1+23] = gUxz
A_alpha[1+28][1+24] = gUyy
A_alpha[1+28][1+25] = 2 * gUyz
A_alpha[1+28][1+26] = gUzz
A_alpha[1+28][1+27] = -1

A_alpha[1+29][1+22] = -gUxx
A_alpha[1+29][1+24] = -gUxy
A_alpha[1+29][1+25] = -gUxz

A_alpha[1+30][1+23] = -gUxx
A_alpha[1+30][1+25] = -gUxy
A_alpha[1+30][1+26] = -gUxz
A_alpha = clone(A_alpha)

local rowsplits = table{0, 3, 21, 27, 28, 31}
A_alpha.rowsplits = rowsplits
A_alpha.colsplits = rowsplits

if verify then
	local A_check = (evr * Lambda * evl)()
	printbr((var'A' / alpha):eq(A_alpha))
	printbr((var'A_{check}' / alpha):eq(A_check))
	for i=1,n do
		for j=1,n do
			local A_check_ij_alpha = (A_check[i][j] / alpha)()
			if (A_check_ij_alpha - A_alpha[i][j])() ~= Constant(0) then
				printbr('$A_{1+'..(i-1)..',1+'..(j-1)..'} / \\alpha = $'..A_check_ij_alpha..' should be '..A_alpha[i][j])
				numErrors = numErrors + 1
			end
		end
	end
end

-- A = evr Lambda evl
-- evl = (evr)^-1
-- permute: P
-- A = evr Lambda (evr)^-1
-- A = evr P P^-1 Lambda P P^-1 (evr)^-1
-- A = (evr P) (P^-1 Lambda P) (evr P)^-1

local permIndexes = table()
	:append{30}
	:append(range(18, 23))
	:append(range(1, 17):reverse())	-- TODO reverse
	:append(range(24, 29))
	:append{31}

--[[ using a matrix
local function permute(p)
	return Matrix:lambda({n,n}, function(i,j)
		return i == p[j] and 1 or 0
	end)
end
local P = permute(permIndexes)

printbr( P )
evr = (evr * P)()
evl = (P:T() * evl)()
Lambda = (P:T() * Lambda * P)()
--]]
-- [[ permuting manually
local function permuteRows(p,m)
	return Matrix:lambda({n,n}, function(i,j)
		return m[p[i]][j]
	end)
end
local function permuteCols(p,m)
	return Matrix:lambda({n,n}, function(i,j)
		return m[i][p[j]]
	end)
end
evr = permuteCols(permIndexes, evr)
evl = permuteRows(permIndexes, evl)
Lambda = permuteRows(permIndexes, permuteCols(permIndexes, Lambda))
-- printbr(var'L':eq(evl))
-- printbr(var'R':eq(evr))
-- printbr(var'\\Lambda':eq(Lambda))
--]]

if verify then
	local A_check = (evr * Lambda * evl)()
	-- printbr((var'A' / alpha):eq(A_alpha))
	-- printbr((var'A_{check}' / alpha):eq(A_check))
	for i=1,n do
		for j=1,n do
			local A_check_ij_alpha = (A_check[i][j] / alpha)()
			if (A_check_ij_alpha - A_alpha[i][j])() ~= Constant(0) then
				printbr('$A_{1+'..(i-1)..',1+'..(j-1)..'} / \\alpha = $'..A_check_ij_alpha..' should be '..A_alpha[i][j])
				numErrors = numErrors + 1
			end
		end
	end
end

local Us = table()
for i,aLi in ipairs(aL) do
	Us:insert(aLi)
end
for kij,dLkij in ipairs(dLvars) do
	Us:insert(dLkij)
end
for ij,KLij in ipairs(KLvars) do
	Us:insert(KLij)
end
Us:insert(Theta)
for i,ZLi in ipairs(ZL) do
	Us:insert(ZLi)
end
local U = Matrix(Us):transpose()
printbr(var'U':eq(U))

-- [[ verify flux in the 'r' direction
local Fv
if verifyFlux then
	local delta = var'\\delta'
	local gamma = var'\\gamma'
	local alpha = var'\\alpha'
	local beta = var'\\beta'
	local b = var'b'
	local K = var'K'
	local a = var'a'
	local d = var'd'
	local Z = var'Z'
	--local Q0 = var'Q0'
	local Q1 = var'Q1'	-- hmm ... TODO find out what this should be
	--local Q2 = var'Q2'	-- not described in the paper
	-- based on comparing the flux jacobian times state with the flux in eqns 1-6 and matching terms ...
	local Q0 = f * (K'_mn' * gamma'^mn' - Theta * m)
	local Q2 = K	-- to-be-indexed as Q2'_ij' == K'_ij'
	--local lambda = var'\\lambda'
	local xi = 0	--var'\\xi'
	local lambda_rij = gamma'^rm' * (d'_mij' - frac(1,2) * (1 + xi) * (d'_ijm' + d'_jim'))
		+ frac(1,2) * delta'^r_i' * (a'_j' + (d'_jmn' - (1 - xi) * d'_mnj') * gamma'^mn' - 2 * Z'_j')
		+ frac(1,2) * delta'^r_j' * (a'_i' + (d'_imn' - (1 - xi) * d'_mni') * gamma'^mn' - 2 * Z'_i')
	local F_alpha = 0
	local F_gamma = 0
	local F_beta = 0
	local F_ai = -beta'^r' * a'_i' + delta'^r_i' * (alpha * Q0 + beta'^m' * a'_m')
	local F_bli = -beta'^r' * b'^i_l' + delta'^r_l' * (alpha * Q1'_i' + beta'^m' * b'^i_m')
	local F_dkij = -beta'^r' * d'_kij' + delta'^r_k' * (alpha * Q2'_ij' + beta'^m' * d'_mij')
	local F_Kij = -beta'^r' * K'_ij' + alpha * lambda_rij	--lambda'^r_ij'
	local F_Theta = -beta'^r' * Theta + alpha * (gamma'^rm' * ((d'_mpq' - d'_pqm') * gamma'^pq' - Z'_m'))
	local F_Zi = -beta'^r' * Z'_i' + alpha * (-gamma'^rm' * K'_mi' + delta'^r_i' * (K'_mn' * gamma'^mn' - Theta))
	
	local F_lhsVarsIndexed = Matrix{
		var'F(\\alpha)''_r',
		var'F(\\gamma)''_rij',
		var'F(\\beta)''_r^l',
		var'F(b)''_r^l_i',
		-- these match with Us, which doesn't include any shift variables right now
		var'F(a)''_ri',
		var'F(d)''_rkij',
		var'F(K)''_rij',
		var'F(\\Theta)''_r',
		var'F(Z)''_ri',
	}:T()
	
	local F_rhsIndexed = Matrix{
		F_alpha,
		F_gamma,
		F_beta,
		F_bli,
		F_ai,
		F_dkij,
		F_Kij,
		F_Theta,
		F_Zi
	}:T()
	
	printbr(var'F_r':eq(F_lhsVarsIndexed):eq(F_rhsIndexed))

	
	local repls = table{
		delta'^i_j':eq(Tensor('^i_j', Matrix.identity(3):unpack())'^i_j'),
		gamma'_ij':eq(gL'_ij'),
		gamma'^ij':eq(gU'^ij'),
		a'_i':eq(aL'_i'),
		Z'_i':eq(ZL'_i'),
		beta'^i':eq(betaU'^i'),
		b'^l_i':eq(bUL'^l_i'),
		K'_ij':eq(KL'_ij'),
		d'_kij':eq(dL'_kij'),
		Q1'_i':eq(Q1L'_i'),
		Q2'_ij':eq(Q2L'_ij'),
	}

	local lhsDense = table()
	local rhsDense = table()
	
	local numrows = #F_rhsIndexed
	assert(numrows == #F_lhsVarsIndexed)
	for row=1,numrows do
		local lhs = F_lhsVarsIndexed[row][1]
		local rhs = F_rhsIndexed[row][1]
		for _,repl in ipairs(repls) do
			rhs = rhs:substIndex(repl)
		end
		rhs = rhs()
		local permutestr = range(2,#lhs):mapi(function(i) return tostring(lhs[i]) end):concat()
		if Constant.isValue(rhs, 0) then
			rhs = Tensor(permutestr)
		else
			rhs = rhs:permute(permutestr)
		end

		lhs = lhs:reindex{r='x'}
		rhs = rhs[1]
		
		assert(Tensor.Ref:isa(lhs))
		assert(Variable:isa(lhs[1]))
		assert(lhs[2].lower)

		-- return true if we should skip this index permutation
		-- this way we skip symmetric index duplicates
		local function skip(is)
			if #lhs == 4 and lhs[3].lower and lhs[4].lower then	-- K_ij, gamma_ij
				return is[1] > is[2]
			elseif #lhs == 5 and lhs[3].lower and lhs[4].lower and lhs[5].lower then	-- d_kij
				return is[2] > is[3]
			end
		end

		local degree = #lhs-2
		-- template for variable-nesting for-loop
		local is = range(degree):mapi(function() return 1 end)
		while true do
			-- iterator callback
			if not skip(is) then
				local thislhs = lhs
				local thisrhs = rhs
				for i,j in ipairs(is) do
					thislhs = thislhs:reindex{[lhs[i+2].symbol] = xs[j]}
					thisrhs = thisrhs[j]
				end
				lhsDense:insert(thislhs)
				rhsDense:insert(thisrhs)
			end

			if degree == 0 then break end
			local done
			for j=degree,1,-1 do
				is[j] = is[j] + 1
				if is[j] > #xs then
					is[j] = 1
					if j == 1 then
						done = true
						break
					end
				else
					break
				end
			end
			if done then break end
		end
	end
	for i=1,#lhsDense do
		printbr(lhsDense[i]:eq(rhsDense[i]))
	end
	printbr'Flux described in the paper, eqns 1-6:'
	printbr(Matrix(lhsDense):T():eq(Matrix(rhsDense):T()))
	
	local A = (A_alpha * alpha)()
	Fv = (A * U)()	-- this is only the flux if the system has the homogeneity property ... which idk if it does
	printbr'Flux from homogeneity, from flux-jacobian matrix in paper:'
	printbr((-U'_t'):eq(Fv))

	-- well, the good news is, all the differences only involve the shift
	-- this means CHECK homogeneity is true
	-- it also means that I can output the flux in code ... 
	printbr'difference:'
	for i=1,#Fv do
		printbr(i, ':', (rhsDense[i + #rhsDense - #Fv] - Fv[i][1])())
	end
	os.exit()
end
--]]

printbr('found '..numErrors..' errors')
printbr()
printbr()

evl = clone(evl)
evr = clone(evr)
A_alpha = clone(A_alpha)
--[[
local detg_gL_def= Matrix(
	{
		gUyy * gUzz - gUyz^2,
		gUxz * gUyz - gUxy * gUzz,
		gUxy * gUyz - gUxz * gUyy,
	},
	{
		gUxz * gUyz - gUxy * gUzz,
		gUxx * gUzz - gUxz^2,
		gUxy * gUxz - gUxx * gUyz,
	},
	{
		gUxy * gUyz - gUxz * gUyy,
		gUxy * gUxz - gUxx * gUyz,
		gUxx * gUyy - gUxy^2,
	}
)
for i=1,3 do
	for j=1,3 do
		assert(op.sub:isa(detg_gL_def[i][j]))
		local a,b = table.unpack(detg_gL_def[i][j])
		
		--printbr(detg_gL_def[i][j], ' => ', g * gL[i][j])
		evl = evl:replace(a - b, g * gL[i][j])
		evr = evr:replace(a - b, g * gL[i][j])
		A_alpha = A_alpha:replace(a - b, g * gL[i][j])
		
		evl = evl:replace(-b + a, g * gL[i][j])
		evr = evr:replace(-b + a, g * gL[i][j])
		A_alpha = A_alpha:replace(-b + a, g * gL[i][j])
		
		evl = evl:replace(b - a, -g * gL[i][j])
		evr = evr:replace(b - a, -g * gL[i][j])
		A_alpha = A_alpha:replace(b - a, -g * gL[i][j])
		
		evl = evl:replace(-a + b, -g * gL[i][j])
		evr = evr:replace(-a + b, -g * gL[i][j])
		A_alpha = A_alpha:replace(-a + b, -g * gL[i][j])
	end
end
--]]

-- print after g_ij substitution
-- printbr(var'L':eq(evl))
-- printbr(var'R':eq(evr))
-- printbr(var'\\Lambda':eq(Lambda))
-- printbr((var'A' / alpha):eq(A_alpha))
-- output the code to a separate file ...
print'<pre>'
local vVars = range(n):mapi(function(i)
	return var('input['..(i-1)..']')
end)
local vs = Matrix(vVars):T()
local Lv = (evl * vs)():T()[1]
local Rv = (evr * vs)():T()[1]
local Lambda_v = (Lambda * vs)():T()[1]
local A_alpha_v = (A_alpha * vs)():T()[1]
local o = assert(io.open('2008_yano.c', 'w'))
local compileVars = table()
	:append{alpha, f, g, m, Theta}
	:append(gLvars)
	:append(gUvars)
	:append(vVars)
for _,info in ipairs(table{
	{'L', Lv},
	{'R', Rv},
	{'Lambda', Lambda_v},
	{'A_alpha', A_alpha_v},
}:append{Fv and {'flux', Fv} or nil}) do
	local name, exprs = table.unpack(info)
	print(name)
--	o:write('void compute'..name..'() {\n')
--	for i=1,n do
		local s =
--[[			
			'\tresult['..(i-1)..'] = '..
--]]			
			(symmath.export.C:toCode{
				output = range(n):mapi(function(i) 
					return {['result['..(i-1)..']'] = exprs[i]}
				end),
				notmp = true,	-- tmpvars goes really slow
				assignOnly = true,
				--input = compileVars
			})
--[[	
				
				--:match('{ return (.*); }')
				:gsub('\\gamma%^{(..)}', function(ij)
					return 'gamma_uu.'..ij
				end)
				:gsub('\\gamma%_{(..)}', function(ij)
					return 'gamma_ll.'..ij
				end)
			..';\n'
		local function fixname(name)
			return name
				:gsub('a_(.)', 'a_l.%1')
				:gsub('d_{(.)(..)}', 'd_lll.%1.%2')
				:gsub('K_{(..)}', 'K_ll.%1')
				:gsub('\\Theta', 'Theta')
				:gsub('Z_(.)', 'Z_l.%1')
		end
		if name == 'L' then 	-- replace input[] with the state variables
			s = s:gsub('input%[(%d+)%]', function(i)
				return fixname(Us[1+i].name)
			end)
		elseif name == 'R' then	-- replace result[] with the state variables
			s = s:gsub('result%[(%d+)%]', function(i)
				return fixname(Us[1+i].name)
			end)
		end
		s = s:gsub('[%+%-]', '\n\t\t%0')
--]]	
		print(s)
--	end
--	o:write'}\n'
--	o:write'\n'
	print()
end
print'</pre>'
