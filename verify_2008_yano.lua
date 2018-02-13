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

local f = var'f'
local m = var'm'
local alpha = var'\\alpha'

local xs = table{'x', 'y', 'z'}
local syms = table{'xx', 'xy', 'xz', 'yy', 'yz', 'zz'}
local gUxx, gUxy, gUxz, gUyy, gUyz, gUzz = syms:map(function(xij)
	return var('\\gamma^{'..xij..'}')
end):unpack()

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
printbr(var'L':eq(evl))

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
printbr(var'R':eq(evr))

local numErrors = 0
local n = #evr	-- hope everything is square ...

local evr_evl = ( evr * evl )()
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

local evl_evr = ( evl * evr )()
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

printbr('found '..numErrors..' errors')

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

local lambdaDiags = range(0,30):map(function(i)
	if i >= 0 and i <= 16 then return 0 end
	if i >= 17 and i <= 22 then return -alpha * sqrt(gUxx) end
	if i >= 23 and i <= 28 then return alpha * sqrt(gUxx) end
	if i == 29 then return -alpha * sqrt(f * gUxx) end
	if i == 30 then return alpha * sqrt(f * gUxx) end
	error'here'
end)
local Lambda = Matrix.diagonal(lambdaDiags:unpack())
printbr(var'\\Lambda':eq(Lambda))

local A = (evr * Lambda * evl)()
printbr(var'A':eq(A))
for i=1,n do
	for j=1,n do
		if A[i][j] ~= Constant(0) then
			printbr('$A_{1+'..(i-1)..',1+'..(j-1)..'} / \\alpha = $'..(A[i][j] / alpha)())
		end
	end
end

local Us = table()
for i,xi in ipairs(xs) do
	Us:insert(var('a_'..xi))
end
for k,xk in ipairs(xs) do
	for ij,xij in ipairs(syms) do
		Us:insert(var('d_{'..xk..xij..'}'))
	end
end
for ij,xij in ipairs(syms) do
	Us:insert(var('K_{'..xij..'}'))
end
Us:insert(var'\\Theta')
for i,xi in ipairs(xs) do
	Us:insert(var('Z_'..xi))
end
local U = Matrix(Us):transpose()
printbr(var'U':eq(U))
os.exit()

local F = (A * U)()
printbr(var'F':eq(F))

--[[
RL[8,17] = R_8,k L_k,17
so the 8th row of R and/or the 17th col of L have something wrong

LR[i,7] = L_ik R_k,7
so the 7th col of R

so R_8,7 is likely to have an error ...
--]]
