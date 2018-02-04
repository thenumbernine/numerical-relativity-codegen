-- TODO remind me where again which paper this came from

local class = require 'ext.class'
local table = require 'ext.table'
local range = require 'ext.range'

local symmath = require 'symmath'
local Tensor = symmath.Tensor

local Z4 = class()

function Z4:init(nrCodeGen, useShift)
	self.nrCodeGen = nrCodeGen

	local xNames = nrCodeGen.xNames
	local symNames = nrCodeGen.symNames
	local from3x3to6 = nrCodeGen.from3x3to6
	local var = nrCodeGen.var

	local alpha = var('\\alpha')
	local betas = xNames:map(function(xi) return var('\\beta^'..xi) end)
	local As = xNames:map(function(xi) return var('A_'..xi) end)
	local gammaLLSym = symNames:map(function(xij) return var('\\gamma_{'..xij..'}') end)
	local DSym = xNames:map(function(xi)
		return symNames:map(function(xjk) return var('D_{'..xi..xjk..'}') end)
	end)
	local DFlattened = table():append(DSym:unpack())

	local KSym = symNames:map(function(xij) return var('K_{'..xij..'}') end)
	local Vs = xNames:map(function(xi) return var('V_'..xi) end)

	-- other vars based on state vars
	local gammaUUSym = symNames:map(function(xij) return var('\\gamma^{'..xij..'}') end)
	
	-- tensors of variables:
	local beta = Tensor('^i', function(i) return self.useShift and betas[i] or 0 end)
	local gammaUU = Tensor('^ij', function(i,j) return gammaUUSym[from3x3to6(i,j)] end)
	local gammaLL = Tensor('_ij', function(i,j) return gammaLLSym[from3x3to6(i,j)] end)
	local A = Tensor('_i', function(i) return As[i] end)
	local B = Tensor('_i^j', function(i,j) return self.useShift and Bs[i][j] or 0 end)
	local D = Tensor('_ijk', function(i,j,k) return DSym[i][from3x3to6(j,k)] end)
	local K = Tensor('_ij', function(i,j) return KSym[from3x3to6(i,j)] end)

	local Theta = var'\\Theta'
	local Zs = xNames:map(function(xi) return var('Z_'..xi) end)
	local Z = Tensor('_i', function(i) return Zs[i] end)

	Tensor.metric(gammaLL, gammaUU)

	local timeVars = table()
	timeVars:insert{alpha}
	timeVars:insert(gammaLLSym)

	local fieldVars = table()
	fieldVars:insert(A)
	fieldVars:append{DFlattened, KSym}
	fieldVars:insert{Theta}
	fieldVars:insert(Zs)

	self.alpha = alpha
	self.betas = betas
	self.gammaLLSym = gammaLLSym
	self.gammaUUSym = gammaUUSym
	self.gammaUU = gammaUU
	self.gammaLL = gammaLL
	self.As = As
	self.DSym = DSym
	self.DFlattened = DFlattened
	self.Vs = Vs
	self.Theta = Theta
	self.Zs = Zs
	self.beta = beta
	self.A = A
	self.B = B
	self.D = D
	self.K = K
	self.Z = Z
	self.timeVars = timeVars
	self.fieldVars = fieldVars
	
	self:getFlattenedVars()
end

function Z4:getFlattenedVars()
	local timeVars = self.timeVars
	local fieldVars = self.fieldVars

	-- variables flattened and combined into one table
	local timeVarsFlattened = table()
	local fieldVarsFlattened = table()
	for _,info in ipairs{
		{timeVars, timeVarsFlattened},
		{fieldVars, fieldVarsFlattened},
	} do
		local infoVars, infoVarsFlattened = table.unpack(info)
		infoVarsFlattened:append(table.unpack(infoVars))
	end

	local varsFlattened = table()
	if self.includeTimeVars then varsFlattened:append(timeVarsFlattened) end
	varsFlattened:append(fieldVarsFlattened)
	
	local expectedNumVars = 31
	assert(#varsFlattened == expectedNumVars, "expected "..expectedNumVars.." but found "..#varsFlattened)

	self.timeVarsFlattened = timeVarsFlattened
	self.fieldVarsFlattened = fieldVarsFlattened
	self.varsFlattened = varsFlattened

	self:getCompileVars()
end

function Z4:getCompileVars()
	local nrCodeGen = self.nrCodeGen
	local f = nrCodeGen.f
	local gammaUUSym = self.gammaUUSym
	local varsFlattened = self.varsFlattened
	
	local compileVars = table():append(varsFlattened, {f}, gammaUUSym)
	if self.useShift then compileVars:append(betas, BFlattened) end
	if not self.includeTimeVars then compileVars:append(self.timeVarsFlattened) end

	self.compileVars = compileVars
end

-- TODO UPDATE THIS
function Z4:getSourceTerms()
	local nrCodeGen = self.nrCodeGen
	
	local var = nrCodeGen.var
	local xNames = nrCodeGen.xNames
	local symNames = nrCodeGen.symNames
	local f = nrCodeGen.f
	
	local from3x3to6 = nrCodeGen.from3x3to6
	local from6to3x3 = nrCodeGen.from6to3x3
	local comment = nrCodeGen.comment
	local def = nrCodeGen.def
	local I = nrCodeGen.I
	local outputCode = nrCodeGen.outputCode
	local outputMethod = nrCodeGen.outputMethod

	local gammaUU = self.gammaUU
	local gammaLL = self.gammaLL
	local alpha = self.alpha
	local As = self.As
	local DFlattened = self.DFlattened
	local Vs = self.Vs
	local beta = self.beta
	local A = self.A
	local B = self.B
	local D = self.D
	local K = self.K
	local Z = self.Z

	local compileVars = self.compileVars
	local varsFlattened = self.varsFlattened

	local ToStringLua = require 'symmath.tostring.Lua'

	local V = (D'_im^m' - D'^m_mi' - Z'_i')()

	local Ssym = symNames:map(function(xij) return var('S_{'..xij..'}') end)
	local S = Tensor('_ij', function(i,j) return Ssym[from3x3to6(i,j)] end)
	local P = Tensor('_i', function(i) return var('P_'..xNames[i]) end)

	--]=]

	local s = (B'_ij' + B'_ji')/alpha
	local trK = K'^i_i'()

	local Q = f * trK		-- lapse function 
	local QU = Tensor('^i')	-- shift function

	local alphaSource = (-alpha^2 * f * trK + 2 * beta'^k' * A'_k')()
	local betaUSource = (-2 * alpha^2 * QU + 2 * B'_k^i' * beta'^k')()
	local gammaLLSource = (-2 * alpha * (K'_ij' - s'_ij') + 2 * beta'^k' * D'_kij')()

	-- not much use for this at the moment
	-- going to display it alongside the matrix
	local sourceTerms = symmath.Matrix(
	table():append(self.includeTimeVars and {
		-- alpha: -alpha^2 Q + alpha beta^k A_k
		alphaSource,
		-- gamma_ij: -2 alpha (K_ij - s_ij) + 2 beta^r D_rij
		gammaLLSource[1][1],
		gammaLLSource[1][2],
		gammaLLSource[1][3],
		gammaLLSource[2][2],
		gammaLLSource[2][3],
		gammaLLSource[3][3],
	} or nil):append{
		-- A_k: 1995 says ... (2 B_k^r - alpha s^k_k delta^r_k) A_r ... though 1997 says 0
		0, 0, 0,
		-- D_kij: 1995 says a lot ... but 1997 says 0
		0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0,
		-- K_ij = alpha S_ij
		alpha * S[1][1],
		alpha * S[1][2],
		alpha * S[1][3],
		alpha * S[2][2],
		alpha * S[2][3],
		alpha * S[3][3],
		-- V_k = alpha P_k
		alpha * P[1],
		alpha * P[2],
		alpha * P[3],
	}:map(function(x) return {x} end):unpack())

	return sourceTerms
end

function Z4:getEigenfields(dir)
	local nrCodeGen = self.nrCodeGen
	local from6to3x3 = nrCodeGen.from6to3x3
	local f = nrCodeGen.f

	local alpha = self.alpha
	local betas = self.betas
	local DSym = self.DSym
	local gammaLLSym = self.gammaLLSym
	local beta = self.beta
	local gammaUU = self.gammaUU
	local gammaLL = self.gammaLL
	local A = self.A
	local B = self.B
	local D = self.D
	local K = self.K
	local Theta = self.Theta
	local Z = self.Z

	-- helpful variables
	local trK = K'^i_i'()
	local DULL = D'^k_ij'()
	local D1L = D'_km^m'()
	--local delta3 = symmath.Matrix.identity(3)
	local delta3 = Tensor('^i_j', function(i,j) return i == j and 1 or 0 end)

	local AU = A'^i'()

	local V = (D'_im^m' - D'^m_mi' - Z'_i')()
	local VU = V'^i'()

	local LambdaULL = Tensor'^k_ij'
	LambdaULL['^k_ij'] = (DULL'^k_ij' + (delta3'^k_i' * (A'_j' + 2 * V'_j' - D1L'_j') + delta3'^k_j' * (A'_i' + 2 * V'_i' - D1L'_i')) / 2)()
io.write'warning -- LambdaULL is only true for zeta=-1\n'

	local Lambda1U = (LambdaULL'^k_ij' * gammaUU'^ij')()

	-- x's other than the current dir
	local oxIndexes = range(3)
	oxIndexes:remove(dir)

	-- symmetric, with 2nd index matching dir removed 
	local osymIndexes = range(6):filter(function(ij)
		local i,j = from6to3x3(ij)
		return i ~= dir or j ~= dir
	end)


	local eigenfields = table()
	
	-- timelike:
	--[[	
	the alpha and gamma don't have to be part of the flux, but i'm lazy so i'm adding them in with the rest of the lambda=0 eigenfields
	however in doing so, it makes factoring the linear system a pain, because you're not supposed to factor out alphas or gammas
	...except with the alpha and gamma eigenfields when you have to ...
	--]]
	if self.includeTimeVars then
			-- alpha
		eigenfields:insert{w=alpha, lambda=-beta[dir]}
			-- gamma_ij
		eigenfields:append(gammaLLSym:map(function(gamma_ij,ij) return {w=gamma_ij, lambda=-beta[dir]} end))
	end
		-- 2, A^x', x' != dir
	eigenfields:append(oxIndexes:map(function(p) return {w=A[p], lambda=-beta[dir]} end))
		-- 12, D_x'ij, x' != dir
	eigenfields:append(oxIndexes:map(function(p)
		return DSym[p]:map(function(D_pij)
			return {w=D_pij, lambda=-beta[dir]}
		end)
	end):unpack())
		-- 3, A_k - f D_km^m + V_k
	eigenfields:append(range(3):map(function(i) return {w=A[i] - f * D1L[i] - V[i], lambda=-beta[dir]} end))

	local sqrt = symmath.sqrt
	for sign=-1,1,2 do
		-- light cone -+
			-- K_ix' +- lambda^x_ix' ... ? I'm not sure about this
		local loc = sign == -1 and 1 or #eigenfields+1
		for _,ij in ipairs(osymIndexes) do
			local i,j = from6to3x3(ij)
			if j == dir then i,j = j,i end
			assert(j ~= dir)
			eigenfields:insert(loc, {
				w = K[i][j] + sign * LambdaULL[dir][i][j],
				lambda = -beta[dir] + sign * alpha * sqrt(gammaUU[dir][dir]),
			})
			loc=loc+1
		end
		-- energy -+
		local loc = sign == -1 and 1 or #eigenfields+1
		eigenfields:insert(loc, {
			w = Theta + sign * VU[dir],
			lambda = -beta[dir] + sign * alpha * sqrt(gammaUU[dir][dir]),
		})
		-- gauge -+
		
		local lambdaParam1 = symmath.var'\\lambda_1' -- equal to (2 - lambda) / (f - 1)
		local lambdaParam2 = symmath.var'\\lambda_2' -- equal to (2 * f - lambda) / (f - 1)
		
		local loc = sign == -1 and 1 or #eigenfields+1
		eigenfields:insert(loc, {
			w = sqrt(f) * (trK + lambdaParam1 * Theta) + sign * (AU[dir] + lambdaParam2 *  VU[dir]),
			lambda = -beta[dir] + sign * alpha * sqrt(f * gammaUU[dir][dir]),
		})
	end

	return eigenfields
end

return Z4
