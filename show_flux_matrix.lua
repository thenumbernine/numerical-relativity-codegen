#!/usr/bin/env luajit

io.stdout:setvbuf'no'

-- idk where to put this, or what it should do
-- I just wanted a script to create a flux jacobian matrix from tensor index equations
require 'ext'
require 'symmath'.setup()	-- TODO setup within _env ?
op = symmath.op	--override ext.op
local TensorRef = require 'symmath.tensor.TensorRef'

-- hmm, why did I push this? I forgot
symmath.TensorRef:pushRule'Prune/replacePartial'

-- [[ one of these is running slow. maybe all?
-- I was testing this in 'symmath/tests/BSSN - index'

-- I think this is getting stuck
-- but with only this pushed I'm still getting simplification loops
-- hmm, gotta find a better fix
--symmath.op.div:pushRule'Prune/conjOfSqrtInDenom'	-- I redid this to not use Wildcard, so hopefully now it is fast enough.
--symmath.op.div:pushRule'Factor/polydiv'			-- though I bet this will be helpful with the eigen solver.
--symmath.op.pow.wildcardMatches = nil
--symmath.matchMulUnknownSubstitution = false
--]]

-- [[ this will lose some memory, but it will be handy in knowing what isn't simplifying correctly
-- so enable this if you see 'simplification loop' warnings.
symmath.simplify.debugLoops = true
symmath.simplify.maxIter = 20
--]]

--local outputType = 'txt'				-- this will output a txt
local outputType = 'html'				-- this will output a html file
--local outputType = 'tex'				-- this will output a pdf file
local outputMathematica = false			-- this will output the flux as mathematica and exit

local keepSourceTerms = false			-- this goes slow with 3D
local outputCodeForSourceTerms = false	-- this goes really really slow.  exclusive with 'keepSourceTerms'

local use1D = false						-- consider spatially x instead of xyz
local removeZeroRows = true			-- whether to keep variables whose dt rows are entirely zero.  only really useful when shift is disabled. TODO don't allow derivs in the source, so for those, only introduce those zero rows?
local useShift = false					-- whether to include beta^i_,t.  TODO this is set to hyperbolic gamma driver .. which still needs evaluation of Gamma^i_jk,t
local useLowerShift = false				-- (TODO still) works with useShift.  uses beta_i,t instead of beta^i_,t.  not really that useful, since beta^i is more often paired with state vars. just make sure not to mix gamma_ij's and gamma^ij's in the flux.
local useConnInsteadOfD = false			-- use conn^k_ij instead of d_kij = 1/2 g_ij,k = conn_(ij)k

-- these are all exclusive with one another:
local useV = false						-- ADM Bona-Masso with V constraint.  Not needed with use1D
local useGamma = false					-- ADM Bona-Masso with Gamma^i_,t . Exclusive to useV ...
local useZ4 = false						-- Z4

local showEigenfields = false			-- my attempt at using eigenfields to deduce the left eigenvectors
local forceRemakeHeader = true

-- TODO a same thing for Q as a whole
-- this is specific to Bona-Masso slicing, where Q = f (K - m Theta)
--local bakeF = 'harmonic'
local bakeF = '1+log'						-- set this to true to bake f's value into the expression.  otherwise 'f' is a free variable (function of alpha), and who knows how that affects the flux determinant, inverse and eigensystem.


local t,x,y,z = vars('t','x','y','z')
local xs = use1D and table{x} or table{x,y,z}


local fluxdir = 1
--local fluxdir = 2
--local fluxdir = 3
local fluxdirvar = xs[fluxdir]
local depvars = table{t,fluxdirvar}



local ToString
if outputType == 'html' then -- [[ mathjax output
	symmath.export.MathJax.useCommaDerivative = true
	ToString = symmath.export.MathJax
elseif outputType == 'txt' then --]]
	--[[ text output - breaking
	function var(s)
		if symmath.tostring.fixImplicitName then
			s = symmath.tostring:fixImplicitName(s)
			if s:sub(1,1) == '\\' then s = s:sub(2) end
		end
		return Variable(s)
	end
	--]]
elseif outputType == 'tex' then
	ToString = symmath.export.LaTeX
end
if ToString then
	symmath.tostring = ToString
	ToString.showDivConstAsMulFrac = true
	ToString.usePartialLHSForDerivative = true
end

-- kronecher delta
local delta = Tensor:deltaSymbol()	-- for simpliyMetrics
local gamma = var'\\gamma'
Tensor.metricVariable = gamma	-- override for simplifyMetrics()

-- adm metric
local alpha = var('\\alpha', depvars)
local beta = var'\\beta'
-- pi
local pi = var'\\pi'
-- stress-energy T^ab n_a n_b
local rho = var'\\rho'
-- extrinsic curvature
local K = var'K'
local A = var'A'	-- trace-free
-- first-order
local a = var'a'
local d = var'd'
-- Ricci
local R = var'R'
-- projected stress-energy
local S = var'S'
-- lapse function
local f = var'f'
local df = var"f'"	-- TODO chain rule!
-- shift
local b = var'b'	-- b^i_j = beta^i_,j
local B = var'B'	-- B^i = beta^i_,t
-- adm variants
local V = var'V'
local Gamma = var'\\Gamma'	-- Gamma^k_ij or Gamma^k
-- z4
local Z = var'Z'
local Theta = var('\\Theta', depvars)
local m = var'm'
local Q = var'Q'

Tensor.coords{
	{variables=xs},
	{variables={t}, symbols='t'},
	{variables={x}, symbols='x'},
	{variables={y}, symbols='y'},
	{variables={z}, symbols='z'},

	--[[
	{variables={y,z}, symbols=
		range('a':byte(), 'z':byte()):map(function(i)
			return '\\bar{'..string.char(i)..'}'
		end)
	},
	--]]
}

-- looking like the 'betterSimplify' found in a few other projects
local function simplify(expr)
	expr = expr():factorDivision()
	if op.add:isa(expr) then
		for i=1,#expr do expr[i] = expr[i]() end
	end
	return expr
end

local outputSuffix = (useZ4 and 'z4' or 'adm')
	..(keepSourceTerms and '_withSource' or '')
	..(useConnInsteadOfD and '_useConn' or '')
	..(useV and '_useV' or '')
	..(useGamma and '_useGamma' or '')
	..(useShift and (useLowerShift and '_useLowerShift' or '_useShift') or '')
	..(removeZeroRows and '_noZeroRows' or '')
	..(use1D and '_1D' or '')
	..(bakeF and ('_'..bakeF) or '')


local outputNameBase = 'flux_matrix_output/flux_matrix.'..outputSuffix

local lineEnding = ({
	txt = '\n',
	html = '<br>\n',
	tex = ' \\\\\n',
})[outputType] or '\n'

local printbr, outputFiles
local closeFile
do
	local filename = outputNameBase..'.'..outputType
	print('writing to '..filename)
	outputFiles = table{assert(file(filename):open'w')}
	for _,f in ipairs(outputFiles) do
		f:setvbuf'no'
		if ToString then f:write(tostring(ToString.header)) end
	end
	printbr = function(...)
		assert(#outputFiles > 0)
		for _,f in ipairs(outputFiles) do
			local n = select('#', ...)
			for i=1,n do
				f:write(tostring(select(i, ...)))
				if i<n then f:write'\t' end
			end
			f:write(lineEnding)
			-- TODO why not just setvbuf?
			f:flush()
		end
	end
	closeFile = function()
		for _,f in ipairs(outputFiles) do
			f:write[[
	</body>
</html>
]]
			f:close()
		end
	end
end


local betaVars
if not useLowerShift then
	betaVars = Tensor('^i', function(i)
		return var('\\beta^'..xs[i].name)
	end)
end

local gammaUVars = Tensor('^ij', function(i,j)
	if i > j then i,j = j,i end
	return var('\\gamma^{'..xs[i].name..xs[j].name..'}', depvars)
end)

local gammaLVars = Tensor('_ij', function(i,j)
	if i > j then i,j = j,i end
	return var('\\gamma_{'..xs[i].name..xs[j].name..'}', depvars)
end)

local betaLVars
if useShift and useLowerShift then
	betaLVars = Tensor('_i', function(i) return var('\\beta_'..xs[i].name) end)
	betaVars = (gammaUVars'^ij' * betaLVars'_j')()
end

local aVars = Tensor('_k', function(k)
	return var('a_'..xs[k].name, depvars)
end)
local bVars = Tensor('^i_j', function(i,j)
	return var('{b^'..xs[i].name..'}_'..xs[j].name, depvars)
end)
local BVars = Tensor('^i', function(i)
	return var('B^'..xs[i].name, depvars)
end)

local dVars = Tensor('_kij', function(k,i,j)
	if i > j then i,j = j,i end
	return var('d_{'..xs[k].name..xs[i].name..xs[j].name..'}', depvars)
end)
-- TODO reorder from [k][i][j] to [i][j][k]

local connVars = Tensor('^k_ij', function(k,i,j)
	if i > j then i,j = j,i end
	return var('{\\Gamma^'
		..xs[k].name..'}'
		..'_{'..xs[i].name
		..' '..xs[j].name..'}'
		, depvars)
end)
local KVars = Tensor('_ij', function(i,j)
	if i > j then i,j = j,i end
	return var('K_{'..xs[i].name..xs[j].name..'}', depvars)
end)
local VVars = Tensor('_k', function(k)
	return var('V_'..xs[k].name, depvars)
end)
local GammaVars = Tensor('_k', function(k)
	return var('\\Gamma_'..xs[k].name, depvars)
end)
local ZVars = Tensor('_k', function(k)
	return var('Z_'..xs[k].name, depvars)
end)

local SLVars = Tensor('_j', function(i)
	return var('S_{'..xs[i].name..'}', depvars)
end)

local SLLVars = Tensor('_ij', function(i,j)
	if i > j then i,j = j,i end
	return var('S_{'..xs[i].name..xs[j].name..'}', depvars)
end)

--[[
NOTICE this is only necessary if your flux has gamma_ij and gamma^ij

gamma^ij = inv(gamma_kl)^ij = 1/det(gamma_kl) adj(gamma_kl)^ij
gamma_ij = inv(gamma^kl)_ij = 1/det(gamma^kl) adj(gamma^kl)_ij = det(gamma_kl) adj(gamma^kl)_ij
TODO get 3x3 inverse to work automatically.
i.e. through adjacency matrices
i.e. through the delta definition of inverses
--]]
local det_gamma_times_gammaUInv = not use1D and Matrix(
	{
		gammaUVars[2][2] * gammaUVars[3][3] - gammaUVars[2][3]^2,
		gammaUVars[1][3] * gammaUVars[2][3] - gammaUVars[1][2] * gammaUVars[3][3],
		gammaUVars[1][2] * gammaUVars[2][3] - gammaUVars[1][3] * gammaUVars[2][2],
	},
	{
		gammaUVars[1][3] * gammaUVars[2][3] - gammaUVars[1][2] * gammaUVars[3][3],
		gammaUVars[1][1] * gammaUVars[3][3] - gammaUVars[1][3]^2,
		gammaUVars[1][2] * gammaUVars[1][3] - gammaUVars[1][1] * gammaUVars[2][3],
	},
	{
		gammaUVars[1][2] * gammaUVars[2][3] - gammaUVars[1][3] * gammaUVars[2][2],
		gammaUVars[1][2] * gammaUVars[1][3] - gammaUVars[1][1] * gammaUVars[2][3],
		gammaUVars[1][1] * gammaUVars[2][2] - gammaUVars[1][2]^2,
	}
) or nil


-- by this point we're going to switch to expanded variables
-- so defining a metric is safe
Tensor.metric(gammaLVars, gammaUVars)


-- in-place substitution
local gammaLU = (gammaLVars'_ik' * gammaUVars'^kj')()
local gammaLL = (gammaLVars'_ik' * gammaLU'_j^k')()
local gammaUU = (gammaUVars'^ik' * gammaLU'_k^j')()

-- g delta^k_l = g (g^kx g_lx + g^ky g_ly + g^kz g_lz)
-- g delta^k_l = g (g^ka g_la + g^kb g_lb + g^kc g_lc)
-- g (delta^k_l - g^ka g_la) = g (g^kb g_lb + g^kc g_lc)
local someMoreRules = table()
do
-- [[
	if det_gamma_times_gammaUInv then
		for k=1,3 do
			someMoreRules[k] = table()
			for l=1,3 do
				local delta_kl = k == l and 1 or 0
				someMoreRules[k][l] = table()
				for a=1,3 do
					local b = a%3+1
					local c = b%3+1
					local find = (gammaUVars[k][b] * det_gamma_times_gammaUInv[l][b]
								+ gammaUVars[k][c] * det_gamma_times_gammaUInv[l][c])()
					
					local sign = 1
					if op.unm:isa(find) then find = find[1] sign = -1 end
					local repl = (sign * gamma * (delta_kl - gammaUVars[k][a] * gammaLVars[l][a]))()
					
--					printbr(k,',',l,',',find:eq(repl))
					
					someMoreRules[k][l]:insert{find, repl}
				end
			end
		end
	end
--]]
--[[
	printbr()
	for k=1,3 do
		for l=1,3 do
			printbr(k,',',l,',',(gamma * gammaLVars[k][l]):eq( det_gamma_times_gammaUInv[k][l] ))
		end
	end
--]]
--[[
	printbr()
	for k=1,3 do
		for l=1,3 do
			printbr(k,',',l,',',gammaLVars[k][l]:eq(gammaLL[k][l]))
		end
	end
	printbr()
	for k=1,3 do
		for l=1,3 do
			printbr(k,',',l,',',gammaUVars[k][l]:eq(gammaUU[k][l]))
		end
	end
	printbr()
	for k=1,3 do
		for l=1,3 do
			printbr(k,',',l,',',(Constant(k == l and 1 or 0)):eq(gammaLU[k][l]))
		end
	end
--]]
end

local function fixJacobianCell(fluxJacobian,i,j)
	if not det_gamma_times_gammaUInv then return end
	for k=1,3 do
		for l=1,3 do
-- hmm ... the (1-f)'s are messing me up... it can't factor them out ...
			
-- [[
			local delta_kl = k == l and 1 or 0
			fluxJacobian[i][j] = fluxJacobian[i][j]()
			fluxJacobian[i][j] = fluxJacobian[i][j]:replace(gammaLL[k][l], gammaLVars[k][l])()
			fluxJacobian[i][j] = fluxJacobian[i][j]:replace(gammaUU[k][l], gammaUVars[k][l])()
			fluxJacobian[i][j] = fluxJacobian[i][j]:replace(gammaLU[k][l], delta_kl)()
--]]

-- [[
			local expr = det_gamma_times_gammaUInv[k][l]
			local expr_eq = gamma * gammaLVars[k][l]
			fluxJacobian[i][j] = fluxJacobian[i][j]:replace( expr(), expr_eq)()
			
			assert(op.sub:isa(expr) and #expr == 2)
			local neg = expr[2] - expr[1]
			local neg_eq = -gamma * gammaLVars[k][l]
			fluxJacobian[i][j] = fluxJacobian[i][j]:replace( neg, neg_eq)()
--]]
-- [[ this is causing an explosion of terms ...
			for _,rule in ipairs(someMoreRules[k][l]) do
				fluxJacobian[i][j] = fluxJacobian[i][j]:replace(rule[1], rule[2])
			end
--]]
		end
	end
end
local function fixFluxJacobian(fluxJacobian, i, j, k, reason)
do return end
-- NOTICE you have to do everything for useV useShift noZeroRows
--	if not reason then -- just do everything
		print('fixing everything in fluxJacobian...')
		for i=1,#fluxJacobian do
			for j=1,#fluxJacobian[1] do
				fixJacobianCell(fluxJacobian,i,j)
			end
		end
--[[
	else
		if reason == 'scale' then 	-- scale row i
			for x=1,#fluxJacobian do
				fixJacobianCell(fluxJacobian,i,x)
			end
		elseif reason == 'row' then	-- modified row j
			for x=1,#fluxJacobian do
				fixJacobianCell(fluxJacobian,j,x)
			end
		end
	end
--]]
end


local function replaceBakedF(expr)
	if not bakeF then return expr end

	expr = clone(expr)
	
	local fixes = {
		harmonic = function()
			expr = expr:replace(f, 1)()
		end,
		['1+log'] = function()
			expr = expr:replace(f, 2 / alpha)()
		end,
	}
	local fix = assert(fixes[bakeF], "couldn't figure out how to bake in that f(alpha) value for bakeF="..require 'ext.tolua'(bakeF))
	fix()
	return expr
end


-- if modify time of show_flux_matrix.lua is newer than the symmath A cache then rebuild
-- otherwise use the cached prefix
-- (TODO store the prefix in a separate file)
local symmathJacobianFilename = 'flux_matrix_output/symmath.'..outputSuffix..'.lua'
local headerExpressionFilename = 'flux_matrix_output/header.'..outputSuffix..'.'..outputType

local fluxJacobian

if not forceRemakeHeader
and (file(headerExpressionFilename):exists()
	or file(symmathJacobianFilename):exists())
then
	if not (file(headerExpressionFilename):exists()
		and file(symmathJacobianFilename):exists())
	then
		error("you need both "..headerExpressionFilename.." and "..symmathJacobianFilename..", but you only have one")
	end

	-- load the cached Jacobian
	fluxJacobian = Matrix( table.unpack (
		assert(load([[
local alpha, f, df, betaVars, gammaLVars, gammaUVars, gamma = ...
return ]] .. file(symmathJacobianFilename):read()))(
			alpha, f, df, betaVars, gammaLVars, gammaUVars, gamma
		)
	))
else
	outputFiles:insert(assert(file(headerExpressionFilename):open'w'))

--[[
	-- TODO start with EFE, apply Gauss-Codazzi-Ricci, then automatically recast all higher order derivatives as new variables of 1st derivatives
	printbr'Einstein Field Equations in spacetime:'
	
	local Gamma_def = Gamma'_abc':eq(g'_ab,c' + g'_ac,b' - g'_bc,a')
	printbr(Gamma_def)
	
	-- Riemann def, with coordinate basis, no commutation, no torsion
	local Riemann_def = R'^a_bcd':eq(Gamma'^a_bd,c' - Gamma'^a_bc,d' + Gamma'^a_uc' * Gamma'^u_bd' - Gamma'^a_ud' * Gamma'^u_bc')
	printbr(Riemann_def)
	
	local Ricci_def = R'_ab':eq(R'^u_aub')
	printbr(Ricci_def)
	
	local Einstein_def = G'_ab':eq(R'_ab' - frac(1,2) * R * g'_ab')
	printbr(Einstein_def)
	
	local EFEdef = G'_ab':eq(8 * pi * T'_ab')
	printbr(EFEdef)
--]]

	local K_trK_term = gamma'^kl' * K'_kl'
	if useZ4 then
		K_trK_term = K_trK_term - 2 * Theta
	end

	printbr[[gauge vars]]

	--[[
	2005 Bona, Lehner, Palenzuela-Luque:
	alpha_,t = -alpha^2 Q
	Algebraic gauge condition (eqn 10)
	Q = -beta^k alpha_,k / alpha^2 + K
	alpha_,t = beta^k alpha_,k - alpha^2 K
	Semialgebraic gauge condition (eqn 11)
	Q = -beta^k alpha_,k / alpha^2 + f (K - 2 Theta)
	alpha_,t = beta^k alpha_,k - alpha^2 f (K - 2 Theta)

	2009 Alic, Bona, Bona-Casas:
	alpha_,t = -alpha^2 Q
	Q = f (K - m Theta)
	alpha_,t = -alpha^2 f (K - m Theta)
	
	so to include the Z4 term, and to neglect the Lie derivative, use:
	Q = f (K - 2 Theta)
	
	TODO fully derive the Z4 so you know where the 'm' coefficient comes about
	2009 Alic section 4 says choosing m=2 makes Z4 coincide with BSSN
	--]]
	local Q_def = Q:eq(alpha * f * K_trK_term:reindex{k='i'})

	printbr(Q_def)

	local Qu_def
	if useShift then
		Qu_def = Q'^i':eq( -1/alpha * beta'^k' * b'^i_k' - alpha * gamma'^ki' * (gamma'_jk,l' * gamma'^jl' - Gamma'^j_kj' - a'_k'))
		printbr(Qu_def)
	end

	printbr[[primitive $\partial_t$ defs]]

	local dt_alpha_def = alpha'_,t':eq(
		-alpha * Q
		-- Lie derivative terms.   Note that the 2005 Bona et al and 2009 Alic et al combine this into the Q function
		+ alpha'_,i' * beta'^i'
	)
	printbr(dt_alpha_def)

	dt_alpha_def = dt_alpha_def:substIndex(Q_def)()
	printbr(dt_alpha_def)

	if bakeF then
		dt_alpha_def = replaceBakedF(dt_alpha_def)
		printbr'substituting f value...'
		dt_alpha_def = dt_alpha_def:substIndex(Q_def)()
		printbr(dt_alpha_def)
	end

	local dt_beta_def, dt_B_def
	if useShift then
		-- hyperbolic Gamma driver
		-- 2008 Alcubierre eqns 4.3.31 & 4.3.32
		-- beta^i_,t = B^i
		-- so what should be used for beta^i_,j ? Bona&Masso use B for that in their papers ...
		-- I'll use b for the spatial derivative and B for the time derivative
		dt_beta_def = beta'^k_,t':eq(
			B'^k'
			-- advection term.  (not Lie derivative.)
			+ beta'^i' * beta'^k_,i'
		)
		printbr(dt_beta_def)
		local xi = frac(3,4)
		
		--[[ is actually the MDE driver that i
		dt_B_def = B'^i_,t':eq(
			-- Lie derivative
			beta'^k' * B'^i_,k'
			-- eq
			+ alpha^2 * xi * (
				-- 2nd derivs
				gamma'^jk' * beta'^i_,jk'
				+ frac(1,3) * gamma'^ik' * beta'^j_,jk'
				-- beta derivs
				+ 2 * gamma'^jk' * Gamma'^i_lj' * beta'^l_,k'
				+ frac(1,3) * gamma'^ik' * Gamma'^j_lj' * beta'^l_,k'
				- Gamma'^l' * beta'^i_,l'
				-- 1st order derivs
				+ Gamma'^i_lj,k' * gamma'^jk' * beta'^l'
				+ frac(1,3) * Gamma'^j_lj,k' * gamma'^ik' * beta'^l'
				- 2 * alpha * (
					gamma'^ik' * gamma'^lj'
					- frac(1,3) * gamma'^ij' * gamma'^kl') * K'_kl,j'
				
				-- source terms
				+ beta'^l' * Gamma'^m_lj' * Gamma'^i_mk' * gamma'^jk'
				- beta'^l' * Gamma'^i_lm' * Gamma'^m_jk' * gamma'^jk'
				+ beta'^j' * R'_jk' * gamma'^ki'
				
				- 2 * alpha * a'_j' * A'^ij'
				- 2 * alpha * Gamma'^i_jk' * A'^jk'
				- 2 * alpha * Gamma'^j_jk' * A'^ik'
			)
		)
		--]]
		-- [[ hyperbolic gamma driver
		local eta = frac(3,4)
		dt_B_def = B'^i_,t':eq(
			alpha^2 * xi * (
				Gamma'^i_jk,t' * gamma'^jk'
				- Gamma'^ijk' * gamma'_jk,t'
				- beta'^l' * Gamma'^i_jk,l' * gamma'^jk'
				+ beta'^l' * Gamma'^ijk' * (Gamma'_jkl' + Gamma'_kjl')
			)
			- eta * B'^i'
			-- advection term.  (not Lie derivative.)
			+ beta'^k' * B'^i_,k'
		)
		--]]
		printbr(dt_B_def)
	end

	local dt_gamma_def = gamma'_ij,t':eq(
		-2 * alpha * K'_ij'
		-- Lie derivative terms
		+ gamma'_ij,k' * beta'^k'
		+ gamma'_kj' * beta'^k_,i'
		+ gamma'_ik' * beta'^k_,j'
	)
	printbr(dt_gamma_def)

	local K_R_term = R'_ij'
	if useZ4 then
		K_R_term = K_R_term + Z'_j,i' - Gamma'^k_ji' * Z'_k' + Z'_i,j' - Gamma'^k_ij' * Z'_k'
--		if not useConnInsteadOfD then
-- TODO don't use gamma
--			K_R_term = K_R_term:replaceIndex
--		end
	end

	-- K_ij,t def
	local dt_K_def = K'_ij,t':eq(
		
		- alpha'_,ij'				-- \__ -alpha_;ij
		+ Gamma'^k_ij' * alpha'_,k'	-- /
		-- random fact:
		-- in 1995 Bona, Masso, Seidel, Stela "new formalism for numerical relativity" eqn 2
		-- in 2005 Bona, Lehner, Palenzuela-Luque "geometrically motivated ... " eqn 20
		-- in 2007 Alic, Bona, Bona-Casas, Masso "efficient implementation ..." eqn 33
		-- in 2009 Alic, Bona, Bona-Casas, "towards a gauge ..." eqn 4
		-- in 2012 Alic, Bona-Casas, Bona, Rezzolla, Palenzuela "conformal and covariant ..." eqn 7
		--  ...
		-- -alpha_;ij is written as -alpha_i;j or -nabla_i alpha_j ... even though alpha is a scalar.  should be nabla_i nabla_j alpha

		+ alpha * (
			K_R_term
			+ K_trK_term * K'_ij'
			- 2 * K'_ik' * gamma'^kl' * K'_jl'
		)
		-- Lie derivative terms
		+ K'_ij,k' * beta'^k'
		+ K'_ki' * beta'^k_,j'
		+ K'_kj' * beta'^k_,i'
		-- stress-energy terms
		+ 4 * pi * alpha * (gamma'_ij' * (S - rho) - 2 * S'_ij')
	)
	printbr(dt_K_def)

	local dt_Theta_def, dt_Z_def
	if useZ4 then
		dt_Theta_def = Theta'_,t':eq(
			-- Lie derivative
			beta'^k' * Theta'_,k'
			-- 2005 Bona et al eqn A.3 of S(Theta)
			- (alpha * (d'^kj_j' - d'_j^jk' - Z'^k'))'_,k'
			+ alpha/2 * (
				2 * a'_k' * (d'^kj_j' - d'_j^jk' - 2 * Z'^k')
				+ d'_k^rs' * Gamma'^k_rs'
				- d'^kj_j' * (d'_kl^l' - 2 * Z'_k')
				- K'_kl' * gamma'^km' * gamma'^ln' * K'_mn'
				+ gamma'^kl' * K'_kl' * K_trK_term:reindex{kl='mn'}
			)
			-- stress-energy terms
			- 8 * pi * alpha * rho
		)
		printbr(dt_Theta_def)

		dt_Z_def = Z'_i,t':eq(
			(beta'^k' * Z'_i')'_,k'
			+ (alpha * gamma'^kl' * K'_li')'_,k'
			- (alpha * (gamma'^kl' * K'_kl' - Theta))'_,i'
			-- 2005 Bona et al eqn A.2 S(Z_i)
			- Z'_i' * b'^k_k'
			+ Z'_k' * b'^k_i'
			+ alpha * (
				a'_i' * (gamma'^kl' * K'_kl' - 2 * Theta)
				- a'_k' * gamma'^kl' * K'_li'
				- gamma'^kl' * K'_lr' * Gamma'^r_ki'
				+ gamma'^kl' * K'_li' * (d'_km^m' - 2 * Z'_k')
			)
			- 8 * pi * alpha * S'_i'
		)
		printbr(dt_Z_def)
	end

	printbr[[lapse vars]]

	-- TODO functions, dependent variables, and total derivatives
	local df_def = f'_,k':eq(df * alpha * a'_k')
	printbr(df_def)

	printbr[[hyperbolic state variables]]

	local a_def = a'_k':eq(log(alpha)'_,k')
	printbr(a_def)

	local a_def = a_def()
	printbr(a_def)

	local dalpha_for_a = (a_def * alpha)():switch()
	printbr(dalpha_for_a)

	-- I'm flipping the order from Bona et al - they like partials, I like comma derivatives
	-- I think I've seen this with a 1/2 in front of it in some papers ...
	-- or should I call this 'b_for_dbeta' ?
	local b_def
	local dbeta_for_b

	if useShift then
		b_def = b'^i_j':eq(beta'^i_,j')
		printbr(b_def)

		dbeta_for_b = b_def:solve(beta'^i_,j')
		printbr(dbeta_for_b)
	end

	local d_def, dgamma_for_d
	if not useConnInsteadOfD then
		d_def = d'_kij':eq(frac(1,2) * gamma'_ij,k')
		printbr(d_def)

		dgamma_for_d = (d_def * 2)():switch()
		printbr(dgamma_for_d)
	end

	printbr[[connections wrt aux vars]]
	local conn_def = Gamma'^k_ij':eq(
		frac(1,2) * gamma'^kl' * (
			gamma'_li,j' + gamma'_lj,i' - gamma'_ij,l'
		)
	)
	printbr(conn_def)
	
	local connL_def = Gamma'_ijk':eq(frac(1,2) * (gamma'_ij,k' + gamma'_ik,j' - gamma'_jk,i'))
	printbr(connL_def)

	local dgamma_for_connL = gamma'_ij,k':eq(Gamma'_ijk' + Gamma'_jik')
	printbr(dgamma_for_connL)


	local dgamma_for_conn = dgamma_for_connL
		:substIndex(Gamma'_ijk':eq(gamma'_il' * Gamma'^l_jk'))
	printbr(dgamma_for_conn)

	local connL_for_d, conn_for_d
	if not useConnInsteadOfD then
		connL_for_d = connL_def
			:substIndex(dgamma_for_d)
			:simplify()
		printbr(connL_for_d)

		-- [[ just raise Gamma, keep d and gamma separate
		conn_for_d = (gamma'^il' * connL_for_d:reindex{ijk='ljk'})()
			:replace(gamma'^il' * Gamma'_ljk', Gamma'^i_jk')
		--]]
		--[[ expand() is adding a -1 somewhere that makes the last replace() choke
		conn_for_d = (gamma'^il' * connL_for_d:reindex{ijk='ljk'})()
			:expand()
			:replace(gamma'^il' * Gamma'_ljk', Gamma'^i_jk')
			:replace(gamma'^il' * d'_jlk', d'_jk^i')
			:replace(gamma'^il' * d'_klj', d'_kj^i')
			:replace((-gamma'^il' * d'_ljk')():expand(), -d'^i_jk')
		--]]
		--[[ TODO raise expression / equation? this works only for dense tensors
		conn_for_d = connL_for_d'^i_jk'()
		--]]
		--[[ if you want to raise d's indexes
		conn_for_d = connL_for_d:map(function(expr)
			if TensorRef:isa(expr) then
				for i=2,#expr do
					if expr[i].symbol == 'i' then expr[i].lower = false end
				end
			end
		end)
		--]]
		printbr(conn_for_d)
	end

	printbr[[${\gamma^{ij}}_{,k}$ wrt aux vars]]

	local dgammaU_def = gamma'^ij_,k':eq(-gamma'^il' * gamma'_lm,k' * gamma'^mj')
	printbr(dgammaU_def)

	local dgammaU_for_d
	if not useConnInsteadOfD then
		dgammaU_for_d = dgammaU_def:substIndex(dgamma_for_d)()
		printbr(dgammaU_for_d)
	end
	
	local dgammaU_for_conn
	if useConnInsteadOfD then
		dgammaU_for_conn = dgammaU_def:substIndex(dgamma_for_connL)()
		printbr(dgammaU_for_conn)
	end
	
	printbr[[Ricci wrt aux vars]]

	local R_def = R'_ij':eq(Gamma'^k_ij,k' - Gamma'^k_ik,j' + Gamma'^k_lk' * Gamma'^l_ij' - Gamma'^k_lj' * Gamma'^l_ik')
	printbr(R_def)

	local R_for_d
	if not useConnInsteadOfD then
		R_for_d = R_def:splitOffDerivIndexes():substIndex(conn_for_d)
		printbr(R_for_d)

		R_for_d = R_for_d()
		printbr(R_for_d)

		R_for_d = R_for_d:substIndex(dgammaU_for_d)()
		printbr(R_for_d)

		printbr'symmetrizing'
		R_for_d = R_for_d
			:symmetrizeIndexes(gamma, {1,2})()
			:symmetrizeIndexes(d, {2,3})()		-- g_ab,cd = 1/2 d_cabd ... so d is symmetric on terms 23 and on terms 14
			:symmetrizeIndexes(d, {1,4}, true)()
		printbr(R_for_d)

		R_for_d = R_for_d:tidyIndexes()()
			:symmetrizeIndexes(gamma, {1,2})()
			:symmetrizeIndexes(d, {2,3})()
			:symmetrizeIndexes(d, {1,4}, true)()
		printbr(R_for_d)
	
		-- [[ there's still 3 terms that could cancel, but aren't cancelling ...
		-- to do this automatically, add support for simplifying d_a^a => d^a_a.  I think right now tidyIndexes() will simplify g^ab d_ba => g^ab d_ab ... but not traces
		local rest = R_for_d
		rest = rest:replaceIndex(d'_ijk' * gamma'^jk', d'_i')
		printbr(rest)
		rest = rest:replaceIndex(d'_jki' * gamma'^jk', e'_i')
		printbr(rest)
		rest = rest
			:simplifyMetrics()()
			:tidyIndexes()()
			:symmetrizeIndexes(gamma, {1,2})()
			:symmetrizeIndexes(d, {2,3})()
			:symmetrizeIndexes(d, {1,4}, true)()	-- 'true' means 'break the rules', aka 'symmetrize across comma derivatives'
		printbr(rest)
		local e = var'e'
		rest = rest
			:replace(d'^a_ab', e'_b')
			:replace(d'^b_ab', e'_a')
			:replace(d'_a^b_b', d'_a')()
			:tidyIndexes()()
		printbr(rest)
		rest = rest
			:replace(d'_a^b_i' * d'_b^a_j', d'^b_ai' * d'^a_bj')
			:replace(d'_a^b_i' * d'_j^a_b', d'^a_bi' * d'_ja^b')
			:replace(d'_a^b_j' * d'^a_bi', d'^a_bj' * d'_a^b_i')
			:replace(d'_a^b_j' * d'_i^a_b', d'^a_bj' * d'_ia^b')
			:replace(d'_ia^b' * d'_j^a_b', d'_i^a_b' * d'_ja^b')
			:simplify()
		printbr(rest)
		rest = rest
			:replace(d'_a', d'_acd' * gamma'^cd')
			:replace(e'_a', d'_cda' * gamma'^cd')
			:replaceIndex(d'^a_bc', d'_dbc' * gamma'^da')
			:replaceIndex(d'_a^b_c', d'_adc' * gamma'^db')
			:replaceIndex(d'_ab^c', d'_abd' * gamma'^db')
			:tidyIndexes()
			:simplify()
		printbr(rest)
		--]]
	end


	printbr[[time derivative of $\alpha_{,t}$]]

	-- don't subst alpha,t ..
	dt_alpha_def = dt_alpha_def:subst(dalpha_for_a:reindex{k='i'})
	printbr(dt_alpha_def)

	printbr[[time derivative of $\gamma_{ij,t}$]]

	printbr(dt_gamma_def)

	if not useConnInsteadOfD then
		-- don't use substIndex to preserve gamma_ij,t
		--dt_gamma_def = dt_gamma_def:substIndex(dgamma_for_d)
		dt_gamma_def = dt_gamma_def:subst(dgamma_for_d)
	end
	-- if we are using conn instead of d then don't substitute out conn for gamma_ij,k just yet ...
	printbr(dt_gamma_def)
	
	if useShift then
		dt_gamma_def = dt_gamma_def:substIndex(dbeta_for_b)()
		printbr(dt_gamma_def)
	end

	printbr[[time derivative of $a_{k,t}$]]

	-- TODO splitDerivs
	local dt_a_def = a_def'_,t'()
	printbr(dt_a_def)

	dt_a_def = dt_a_def
		:replace(alpha'_,kt', alpha'_,t''_,k')
		:subst(dt_alpha_def)
	printbr(dt_a_def)

	dt_a_def = dt_a_def()
	printbr(dt_a_def)

	dt_a_def = dt_a_def
		:replace(alpha'_,ik', frac(1,2) * ( alpha'_,i''_,k' + alpha',k'',i' ))
		--:replace(K'^i_i,k', (gamma'^ij' * K'_ij')'_,k')
	printbr(dt_a_def)

	dt_a_def = dt_a_def:substIndex(df_def, dalpha_for_a)
	printbr(dt_a_def)

	dt_a_def = dt_a_def()
		:substIndex(dalpha_for_a)
		:symmetrizeIndexes(a, {1,2}, true)
		:simplify()
	printbr(dt_a_def)

	if not useConnInsteadOfD then
		dt_a_def = dt_a_def:substIndex(dgammaU_for_d)()
		printbr(dt_a_def)
	
		dt_a_def = dt_a_def
			:replace(
				-- TODO replace sub-portions of commutative operators like mul() add() etc
				-- TODO don't require that simplify() on the find() portion of replace() -- instead simplify automatically?  i experimented with this once ...
				-- TODO simplify gammas automatically ... define a tensor expression metric variable?
				(2 * alpha * f * gamma'^im' * d'_kml' * gamma'^lj' * K'_ij')(),
				2 * alpha * f * d'_k^ij' * K'_ij'
			)
		printbr(dt_a_def)
	else
		dt_a_def = dt_a_def:substIndex(dgammaU_for_conn)()
		printbr(dt_a_def)
	end

	if useShift then
		dt_a_def = dt_a_def:substIndex(dbeta_for_b)
		printbr(dt_a_def)
	end

	local dt_b_def
	if useShift then

		printbr[[time derivative of ${\beta^i}_{,t}$]]

		printbr(dt_beta_def)

		-- can't use substIndex or it'll pick up the ,t
		--dt_beta_def = dt_beta_def:substIndex(dbeta_for_b)
		dt_beta_def = dt_beta_def:subst(dbeta_for_b:reindex{ij='ki'})
		printbr(dt_beta_def)


		printbr[[time derivative of ${b^i}_{j,t}$]]

		dt_b_def = dt_beta_def'_,j'():reindex{ij='ji'}
		printbr(dt_b_def)

		dt_b_def = dt_b_def
			:replace(beta'^k_,ti', beta'^k_,i''_,t')
			:substIndex(dbeta_for_b)
			-- hmm, without this, we have problems down when splitting pdes off
			:simplify()
		printbr(dt_b_def)
		
		printbr[[aux var $A^{ij}$]]
		
		local A_for_K_uu = A'^ij':eq( K'^ij' - frac(1,3) * gamma'^ij' * gamma'^kl' * K'_kl')
		printbr(A_for_K_uu)

		-- TODO substIndex handle this -- but skip any upper/lower changes inside of comma derivatives
		local A_for_K_ll = A'_ij':eq( K'_ij' - frac(1,3) * gamma'_ij' * gamma'^kl' * K'_kl')

		printbr[[time derivative of ${B^i}_{,t}$]]

		printbr(dt_B_def)

		dt_B_def = dt_B_def()
		printbr(dt_B_def)
		
		dt_B_def = dt_B_def:substIndex(Gamma'^i':eq(Gamma'^i_jk' * gamma'^jk'))
		
		if not useConnInsteadOfD then
			dt_B_def = dt_B_def:substIndex(conn_for_d)
		end

		dt_B_def = dt_B_def
			:substIndex(A_for_K_uu)
			:substIndex(A_for_K_ll)
			:substIndex(dbeta_for_b)
			:substIndex(dbeta_for_b'_,k'())
		
		if not useConnInsteadOfD then
			dt_B_def = dt_B_def:substIndex(R_for_d)
		else
			dt_B_def = dt_B_def:substIndex(R_def)
		end

		dt_B_def = dt_B_def:simplify()

		printbr(dt_B_def)

		-- TODO automatic relabel indexes
		-- TODO prevent substIndex from using indexes already reserved for other coordinate sets
		dt_B_def = dt_B_def
			:symmetrizeIndexes(gamma, {1,2})()
		
		if not useConnInsteadOfD then
			dt_B_def = dt_B_def
				:symmetrizeIndexes(d, {2,3})()
				:symmetrizeIndexes(d, {1,4}, true)()
		end
		
		printbr(dt_B_def)
	end

	local dt_d_def
	if not useConnInsteadOfD then
		printbr[[time derivative of $d_{kij,t}$]]

		dt_d_def = d_def'_,t'()
		printbr(dt_d_def)

		dt_d_def = dt_d_def
			:replace(gamma'_ij,k,t', gamma'_ij,t''_,k')
			-- TODO automatically relabel the sum indexes
			-- ... this would require knowledge of the entire dt_d_def expression, to know what indexes are available
			:subst(dt_gamma_def:reindex{ijk='ijl'})
		printbr(dt_d_def)

		dt_d_def = dt_d_def()
		printbr(dt_d_def)

		dt_d_def = dt_d_def
			:replace(gamma'_ij,l,k', gamma'_ij,l''_,k')
			:subst(dgamma_for_d:reindex{ijk='ijl'})
			:subst(dgamma_for_d:reindex{ijk='ilk'})
			:subst(dgamma_for_d:reindex{ijk='ljk'})
			:subst(dalpha_for_a)
		if useShift then
			dt_d_def = dt_d_def:substIndex(dbeta_for_b)
		end
		dt_d_def = dt_d_def:simplify()
		printbr(dt_d_def)
	end

	local dt_conn_def
	if useConnInsteadOfD then
		printbr[[time derivative of $\Gamma_{ijk,t}$]]
	
		local dt_connL_def = connL_def'_,t'()
		printbr(dt_connL_def)
		
		dt_connL_def = dt_connL_def
			:replace(gamma'_ij,k,t', gamma'_ij,t''_,k')
			:replace(gamma'_ik,j,t', gamma'_ik,t''_,j')
			:replace(gamma'_jk,i,t', gamma'_jk,t''_,i')
			:substIndex(
				dt_gamma_def
					:reindex{k='a'}
					:replaceIndex(b'^a_b', beta'^a_,b')
			)
		printbr(dt_connL_def)
	
		dt_connL_def = dt_connL_def()
			:substIndex(dgamma_for_connL)()
			:symmetrizeIndexes(gamma, {1,2})()
			:symmetrizeIndexes(gamma, {3,4})()
			:symmetrizeIndexes(K, {1,2})()
			:symmetrizeIndexes(Gamma, {2,3})()
			:symmetrizeIndexes(beta, {2,3})()
		printbr(dt_connL_def)

		local tmp = (connL_def'_,a'() * beta'^a')()
			:symmetrizeIndexes(gamma, {3,4})
		printbr(tmp)

		dt_connL_def[2] = (dt_connL_def[2] - tmp:rhs() + tmp:lhs())()
		printbr(dt_connL_def)

		printbr[[time derivative of ${\Gamma^k}_{ij,t}$]]
	
		dt_conn_def = Gamma'^i_jk,t':eq(Gamma'^i_jk''_,t')
		dt_conn_def[2] = dt_conn_def[2]:substIndex(
			Gamma'^i_jk':eq(gamma'^il' * Gamma'_ljk')
		)()
		printbr(dt_conn_def)

		dt_conn_def = dt_conn_def
			:subst(dt_connL_def:reindex{ia='ab'})
		printbr(dt_conn_def)
		
		dt_conn_def = dt_conn_def
			:substIndex(dgammaU_def)
			
			-- hmm, substindex does make sure to not use previous sum indexes
			-- but it doesn't replace sum-indexes of expressions it's inserting
			:substIndex(dt_gamma_def:reindex{k='e'})
			
			:simplify()
		printbr(dt_conn_def)

		local tmp = Gamma'^i_jk':eq(gamma'^ia' * Gamma'_ajk')
		tmp = tmp'_,b'()
		tmp = ((tmp - gamma'^ia_,b' * Gamma'_ajk') * beta'^b' )():switch()
		printbr(tmp)
		
		dt_conn_def = dt_conn_def:substIndex(tmp)()
		
		dt_conn_def = dt_conn_def()
			:substIndex(dgammaU_def)()
			:substIndex(dgamma_for_conn)()
			:symmetrizeIndexes(gamma, {1,2})()
			:symmetrizeIndexes(gamma, {3,4})()
			:symmetrizeIndexes(K, {1,2})()
			:symmetrizeIndexes(Gamma, {2,3})()
			:symmetrizeIndexes(beta, {2,3})()
		printbr(dt_conn_def)

		dt_conn_def = dt_conn_def:replaceIndex(b'^a_b', beta'^a_,b')
		dt_conn_def = dt_conn_def:simplifyMetrics()()
		printbr(dt_conn_def)

		dt_conn_def  = dt_conn_def:tidyIndexes()()
		printbr(dt_conn_def)
	
		dt_conn_def = dt_conn_def:substIndex(dalpha_for_a)()
		dt_conn_def = dt_conn_def:replaceIndex(beta'^i_,j', b'^i_j')()
		dt_conn_def = dt_conn_def:replaceIndex(beta'^i_,jk', b'^i_jk')()
		printbr(dt_conn_def)
	end

	if useConnInsteadOfD then
		printbr[[time derivative of $\gamma_{ij}$]]
		
		dt_gamma_def[2] = dt_gamma_def[2]:substIndex(dgamma_for_connL)()
		printbr(dt_gamma_def)

		if useShift then
			printbr[[time derivative of $B^i$]]
			
			printbr(dt_B_def)

			dt_B_def = dt_B_def
				:substIndex(dt_gamma_def:reindex{ijk='abc'})
				:simplify()
			printbr(dt_B_def)

			dt_B_def = dt_B_def
				:subst(dt_conn_def)
				:simplify()
			printbr(dt_B_def)
		end
	end

	printbr[[$K_{ij,t}$ with hyperbolic terms]]

	printbr(dt_K_def)

	dt_K_def = dt_K_def
		:replace(alpha'_,ij', frac(1,2) * (alpha'_,i''_,j' + alpha'_,j''_,i'))
		:subst(dalpha_for_a:reindex{k='i'})
		:subst(dalpha_for_a:reindex{k='j'})
		:subst(dalpha_for_a)
	printbr(dt_K_def)
		
	dt_K_def = dt_K_def:simplify()
	printbr(dt_K_def)

	dt_K_def = dt_K_def
		:subst(dalpha_for_a:reindex{k='j'})
		:subst(dalpha_for_a:reindex{k='i'})
		:simplify()
	printbr(dt_K_def)

	if not useConnInsteadOfD then
		if useZ4 then
			dt_K_def = dt_K_def:substIndex(conn_for_d)
		else
			-- this seems safe enough for adm ... will substIndex work?
			dt_K_def = dt_K_def:subst(conn_for_d:reindex{ijk='kij'})
		end
		printbr(dt_K_def)

		dt_K_def = dt_K_def:subst(R_for_d)
		printbr(dt_K_def)
	else
		dt_K_def = dt_K_def:subst(R_def)
		dt_K_def = dt_K_def:replace(
			Gamma'^k_ik,j',
			frac(1,2) * (Gamma'^l_jl,i' + Gamma'^l_il,j')
		)
		printbr(dt_K_def)
	end

	dt_K_def = dt_K_def()
	printbr(dt_K_def)

	--[=[
	local dsym_def = d'_ijk,l':eq(frac(1,2) * (d'_ijk,l' + d'_ljk,i'))
	--[[ substIndex works ... but replaces the replaced ...
	dt_K_def = dt_K_def
		:substIndex(dsym_def:reindex{ijkl='ijlk'})
		:substIndex(dsym_def:reindex{ijkl='iklj'})
		:substIndex(dsym_def:reindex{ijkl='jilk'})
		:substIndex(dsym_def:reindex{ijkl='kijl'})
	--]]
	--[[
	dt_K_def = dt_K_def
		:subst(dsym_def:reindex{ijkl='ijmk'})
		:subst(dsym_def:reindex{ijkl='ikmj'})
		:subst(dsym_def:reindex{ijkl='jimk'})
		:subst(dsym_def:reindex{ijkl='kijm'})
	--]]
	--]=]
	if useShift then
		dt_K_def = dt_K_def:substIndex(dbeta_for_b)
	end
	dt_K_def = dt_K_def:tidyIndexes()()
		:symmetrizeIndexes(gamma, {1,2})()
	if not useConnInsteadOfD then
		dt_K_def = dt_K_def:tidyIndexes()()
			:symmetrizeIndexes(d, {2,3})()
			:symmetrizeIndexes(d, {1,4}, true)()
	end
	printbr(dt_K_def)


	local defs = table()

	-- TODO make compat with conn^k_ij state var
	-- TODO is Theta = Z^t or Z_t? Geom. etc paper doesn't say, Yano is missing alpha for its def, Alcubierre is missing beta for his (shiftless fwiw) def
	if useZ4 then
		
		-- I'm taking this from 2008 Yano et al Flux-Vector-Splitting method for Z4 formalism and its numerical analysis
		-- ... and from its source paper, 2005 Bona et al "Geometrically motivated hyperbolic coordinate condions for numerical relativity- Analysis, issues and implementation"

		printbr'Z4 terms'
	
		if useShift then
		
			local Qu_def = Q'^i':eq( -1/alpha * beta'^k' * b'^i_k' - alpha * gamma'^ki' * (gamma'_jk,l' * gamma'^jl' - Gamma'^j_kj' - a'_k'))
			printbr(Qu_def)

			Qu_def = Qu_def:substIndex(dgamma_for_d, conn_for_d)
			printbr(Qu_def)

			local dt_b_def = b'^i_k,t':eq(
				-(
					-(beta'^j' * b'^i_k')'_,j'
					+ (
						alpha * Q'^i'
						+ beta'^j' * b'^i_j'
					)'_,k'
				)
				+ b'^i_j' * b'^j_k'
				- b'^j_j' * b'^i_k'
			)
			printbr(dt_b_def)

			dt_b_def = dt_b_def()
			printbr(dt_b_def)

			-- TODO automatically split off deriv indexes before substIndex -- on the find and the expr?
			dt_b_def = dt_b_def:splitOffDerivIndexes():substIndex(Qu_def)
			printbr(dt_b_def)
			
			dt_b_def = dt_b_def()
			printbr(dt_b_def)

			dt_b_def = dt_b_def
				:substIndex(dgammaU_for_d)
				:substIndex(dalpha_for_a)
			
			if useShift then
				dt_b_def = dt_b_def
					:substIndex(dbeta_for_b)
			end
			
			dt_b_def = dt_b_def
				:substIndex(dgamma_for_d)
				
				-- TODO relabel
				:replace(b'^l_k' * b'^i_l', b'^i_j' * b'^j_k')
				:replace(beta'^l' * b'^i_l,k', beta'^j' * b'^i_j,k')
				:symmetrizeIndexes(gamma, {1,2})
				
				:simplify()
			printbr(dt_b_def)
			
		end

		dt_Theta_def = dt_Theta_def
			:replaceIndex(Z'^i', gamma'^ij' * Z'_j')
			:replaceIndex(d'^i_jk', gamma'^il' * d'_ljk')
			:replaceIndex(d'^ij_k', gamma'^il' * gamma'^jm' * d'_lmk')
			:replaceIndex(d'_i^jk', d'_ilm' * gamma'^lj' * gamma'^mk')
			:replaceIndex(d'_ij^k', d'_ijl' * gamma'^lk')

		dt_Theta_def = dt_Theta_def()
		printbr(dt_Theta_def)

		dt_Theta_def = dt_Theta_def
			:simplify()
			:substIndex(dgammaU_for_d)
			:substIndex(dgamma_for_d)
			:substIndex(conn_for_d)
			:substIndex(dalpha_for_a)

		if useShift then
			dt_Theta_def = dt_Theta_def
				:substIndex(dbeta_for_b)
		end

		dt_Theta_def = dt_Theta_def
			:substIndex(conn_for_d)
			:simplify()
		printbr(dt_Theta_def)


		printbr(dt_Z_def)
		
		dt_Z_def = dt_Z_def
			--:replaceIndex(K'^i_j', gamma'^ik' * K'_kj')
			:replaceIndex(d'^i_jk', gamma'^il' * d'_ljk')
			:replaceIndex(d'_ij^k', d'_ijl' * gamma'^lk')

		dt_Z_def = dt_Z_def()
		printbr(dt_Z_def)

		dt_Z_def = dt_Z_def
			:simplify()
			:substIndex(dgammaU_for_d)
			:substIndex(dgamma_for_d)
			:substIndex(conn_for_d)
			:substIndex(dalpha_for_a)

		if useShift then
			dt_Z_def = dt_Z_def
				:substIndex(dbeta_for_b)
		end

		dt_Z_def = dt_Z_def
			:substIndex(conn_for_d)
			:simplify()
		printbr(dt_Z_def)

		defs:insert(dt_beta_def)
		defs:insert(dt_b_def)
		defs:insert(dt_B_def)
		defs:insert(dt_alpha_def)
		defs:insert(dt_gamma_def)
		defs:insert(dt_a_def)
		if useConnInsteadOfD then
			defs:insert(dt_conn_def)
		else
			defs:insert(dt_d_def)
		end
		defs:insert(dt_K_def)
		defs:insert(dt_Theta_def)
		defs:insert(dt_Z_def)

	else
		--[[ primitives first, in greek order
		defs:insert(dt_alpha_def)
		defs:insert(dt_beta_def)
		defs:insert(dt_gamma_def)
		defs:insert(dt_a_def)
		defs:insert(dt_b_def)
		defs:insert(dt_B_def)
		defs:insert(dt_d_def)
		--]]
		-- [[ shifts first
		defs:insert(dt_beta_def)
		defs:insert(dt_b_def)
		defs:insert(dt_B_def)
		defs:insert(dt_alpha_def)
		defs:insert(dt_a_def)
		defs:insert(dt_gamma_def)
		if useConnInsteadOfD then
			defs:insert(dt_conn_def)
		else
			defs:insert(dt_d_def)
		end
		--]]
		
		if useV then
			-- TODO just replace the V's in this, and don't redeclare everything
			dt_K_def = K'_ij,t':eq(
				- frac(1,2) * alpha * a'_i,j'
				- frac(1,2) * alpha * a'_j,i'
				+ alpha * (
					  frac(1,2) * gamma'^pq' * (d'_ipq,j' + d'_jpq,i')
					- frac(1,2) * gamma'^mr' * (d'_mij,r' + d'_mji,r')
				)
				
				- alpha * V'_j,i'
				- alpha * V'_i,j'
				
				+ alpha * (
					-a'_i' * a'_j'
					+ (d'_ji^k' + d'_ij^k' - d'^k_ij') * (a'_k' + V'_k' - d'^l_lk')
					
					+ 2 * (d'^kl_j' - d'^lk_j') * d'_kli'

					+ 2 * d'_i^kl' * d'_klj'
					+ 2 * d'_j^kl' * d'_kli'
					- 3 * d'_i^kl' * d'_jkl'
							
					+ gamma'^kl' * K'_kl' * K'_ij'
					- gamma'^kl' * K'_il' * K'_kj'
					- gamma'^kl' * K'_jl' * K'_ki'
				)
				+ K'_ij,k' * beta'^k'
				+ K'_kj' * beta'^k_,i'
				+ K'_ik' * beta'^k_,j'
			)
			
			if useShift then
				dt_K_def = dt_K_def:substIndex(dbeta_for_b)
			end
			defs:insert(dt_K_def)
			defs:insert( V'_k,t':eq(
				-- TODO there are some source terms that should go here.
				-- the eigenvectors that the Alcubierre 2008 book has only source terms, no first derivatives
				-- how ever my own calculations come up with K_ij,k terms ... maybe they get absorbed into some other terms?
				0
			) )
		elseif useGamma then
			
			defs:insert( K'_ij,t':eq(
				- frac(1,2) * alpha * a'_i,j'
				- frac(1,2) * alpha * a'_j,i'
				
				+ alpha * (
					frac(1,2) * gamma'^pq' * (d'_ipq,j' + d'_jpq,i')
					- frac(1,2) * gamma'^mr' * (d'_mij,r' + d'_mji,r')
					- gamma'^mp' * d'_mpi,j'
					- gamma'^mp' * d'_mpj,i'
				)
			
				+ 2 * alpha * Gamma'_i,j'
				+ 2 * alpha * Gamma'_j,i'
				
				+ alpha * (
					-a'_i' * a'_j'
					+ (d'_ji^k' + d'_ij^k' - d'^k_ij') * (a'_k' - Gamma'_k')
					
					+ 2 * (d'^kl_j' - d'^lk_j') * d'_kli'
					- 3 * d'_i^kl' * d'_jkl'

					+ 4 * d'_i^kl' * d'_klj'
					+ 4 * d'_j^kl' * d'_kli'
							
					+ gamma'^kl' * K'_kl' * K'_ij'
					- gamma'^kl' * K'_il' * K'_kj'
					- gamma'^kl' * K'_jl' * K'_ki'
				)
				+ K'_ij,k' * beta'^k'
				+ K'_kj' * beta'^k_,i'
				+ K'_ik' * beta'^k_,j'
			) )
			defs:insert( Gamma'_k,t':eq(
				-alpha * 2 * gamma'^qr' * K'_kq,r'
				+ alpha * gamma'^pq' * K'_pq,k'
				- 2 * (d'_lk^r' + d'^r_kl') * beta'^l_,r'
				- (2 * d'_n^nr' - d'^rn_n') * gamma'_kl' * beta'^l_,r'
				+ Gamma'_k,r' * beta'^r'
				+ Gamma'^r' * beta'^l_,r' * gamma'_kl'
				+ Gamma'_l' * beta'^l_,k'
				+ gamma'_kl' * beta'^l_,ij' * gamma'^ij'
				- 2 * alpha * K'_ik' * Gamma'^i'
				- 2 * alpha * a'^i' * K'_ki'
				+ alpha * a'_k' * gamma'^kl' * K'_kl'
				+ 4 * alpha * d'_i^ij' * K'_kj'
				- 2 * alpha * d'^ji_i' * K'_kj'
				+ 4 * alpha * d'^ij_k' * K'_ij'
				- 2 * alpha * d'_k^ij' * K'_ij'
			) )
		else
			defs:insert(dt_K_def)
		end
	end

	printbr('partial derivatives')
	for _,def in ipairs(defs) do
		printbr(def)
	end

	if not useShift then
		for i=1,#defs do
			local lhs, rhs = table.unpack(defs[i])
			
			rhs = rhs:map(function(expr)
				if TensorRef:isa(expr)
				and (expr[1] == beta or expr[1] == b or expr[1] == B)
				then return 0 end
			end)()
			rhs = simplify(rhs)
			
			defs[i] = lhs:eq(rhs)
		end
		printbr('neglecting shift')
		for _,def in ipairs(defs) do
			printbr(def)
		end
	end

	local sourceTerms
	if not keepSourceTerms then
		-- for all summed terms, for all coefficients,
		--	if none have derivatives then remove them
		-- remove from defs any equations that no longer have any terms
		sourceTerms = table()
		defs = defs:map(function(def,i,t)
			local lhs, rhs = table.unpack(defs[i])
			
			sourceTerms[i] = rhs:map(function(expr)
				if TensorRef:isa(expr) then
					for j=2,#expr do
						if expr[j].derivative then return 0 end
					end
				end
			end)()
			
			rhs = (rhs - sourceTerms[i])()
			rhs = simplify(rhs)
			
			do -- if rhs ~= Constant(0) then
				return lhs:eq(rhs), #t+1
			end
		end)
		printbr('neglecting source terms')
		for _,def in ipairs(defs) do
			printbr(def)
		end
		
		-- TODO here - print out the x-direction source terms in C code.  expand all indexes, neglect all _,y _,z terms, (only use _,x), and then replace all U (state) and U_,x with U variables

		printbr'...and those source terms are...'
		for i,def in ipairs(defs) do
			local lhs, rhs = table.unpack(def)
			printbr(lhs..'$ + \\dots = $'..sourceTerms[i])
		end
		
		-- TODO here - print out the source terms in C code
	end


	if showEigenfields then
		printbr'separating x from other dimensions:'
		for _,def in ipairs(defs) do
			local lhs, rhs = table.unpack(def)
			assert(TensorRef:isa(lhs))
			local indexes = table()
			for i=2,#lhs do
				if lhs[i].derivative then break end
				indexes:insert(lhs[i].symbol)
			end
			if #indexes > 0 then
				-- now cycle through all permutations of the indexes, substiting for either x or indexes not spanning x
				local matrix = require 'matrix'
				matrix(table{2}:rep(#indexes)):lambda(function(...)
					local is = {...}
					local from = ''
					local to = ''
					for i=1,#is do
						if is[i] == 1 then
							from = from .. indexes[i]	-- TODO don't use the original index ... instead use a yz spanning index (which means I need to make these up)
							to = to .. 'x'
						end
					end
					printbr(def:reindex{[from]=to})
					return 0
				end)
			end
		end
	end


	-- TODO there seems to be a problem when I make sense S * gamma'_ij'
	-- esp when S'_ij' itself is a separate dense tensor
	-- it is turning the first into S'_ij' * gamma'_ij' (esp in K_ij,t's source terms)
	local function makeTensorExpressionDense(def)
		def = def
			:map(function(expr)
				if TensorRef:isa(expr)
				and expr[1] == gamma
				then
					--for i=4,#expr do	-- expr[1] is the variable, 2,3 are the ij indexes, so start at 4 for derivatives
					--	assert(not expr[i].lower, "found a gamma_ij term: "..tostring(expr))
					--end
					-- warn if there are any gamma^ij_,k... or gamma_ij,k for that matter
					if #expr == 3 then
						assert(not expr:hasDerivIndex())
						if expr[2].lower and expr[3].lower then
							return TensorRef(gammaLVars, table.unpack(expr, 2))
						end
						if not expr[2].lower and not expr[3].lower then
							return TensorRef(gammaUVars, table.unpack(expr, 2))
						end
						error("failed on "..expr)
					elseif #expr == 4 then
						if expr[2].lower and not expr[1].derivative
						and expr[3].lower and not expr[2].derivative
						and expr[4].derivative == ','
						and expr[4].symbol == 't'
						then
							return TensorRef(gammaLVars, table.unpack(expr, 2))
						end
						
						-- if it's a gamma^i_j,k derivative then it should really be a delta^i_j,k, and that is zero
						-- TODO this should
						if expr[4].derivative
						and (not not expr[2].lower) ~= (not not expr[3].lower)
						then
							printbr('warning, looks like a ${\\gamma^i}_{j,k}$ snuck in there')
							return 0
						end
					end
					error("failed on "..expr.." inside of "..def)
					--return TensorRef(gammaUVars, table.unpack(expr, 2))
				end
			end)
			:replace(a, aVars)
		
		if not useConnInsteadOfD then
			def = def:replace(d, dVars)
		else
			-- TODO this won't work with Gamma^i state variable
			def = def:replace(Gamma, connVars)
		end

		def = def
			:replace(K, KVars)
			:replace(alpha'_,t', alpha:diff(t))
			:replace(Theta'_,t', Theta:diff(t))
		
		
		--[[ you can't just replace S directly, since SVars is for a dense degree-2 tensor
		def = def:replace(S, SVars)
		--]]
		-- [[
		def = def:map(function(x)
			if TensorRef:isa(x)
			and x[1] == S
			then
				assert(not x:hasDerivIndex())
				x = x:clone()
				if #x == 3 then			-- S'_ij'
					x[1] = SLLVars
				elseif #x == 2 then		-- S'_i'
					x[1] = SLVars
				else
					error("how did I get here with "..x)
				end
				return x
			end
		end)
		--]]

		if useShift then
			def = def
				:replace(beta, betaVars)
				:replace(b, bVars)
				:replace(B, BVars)
		end
		if useV then
			def = def:replace(V, VVars)
		end
		if useGamma then
			def = def:replace(Gamma, GammaVars)
		end
		if useZ4 then
			def = def:map(function(expr)
				if TensorRef:isa(expr)
				and expr[1] == Theta
				then
					assert(#expr == 2)	-- only Theta_,i
					return Tensor(table.sub(expr, 2), function(...)
						local is = table{...}
						return Theta:diff(is:mapi(function(i) return xs[i] end):unpack())
					end)
			--def = def:replace(Theta'_,k', Tensor('_k', function(k) return Theta:diff(xs[k]) end))
			--def = def:replace(Theta'_,i', Tensor('_i', function(k) return Theta:diff(xs[k]) end))
				end
			end)
			def = def:replace(Z, ZVars)
		end
		def = def()
		return def
	end

	
	printbr('spelled out')
	local allLhs = table()
	local allRhs = table()
	local defsForLhs = table()	-- check to make sure symmetric terms have equal rhs's.  key by the lhs
	local allSrcEqns
	if not keepSourceTerms and outputCodeForSourceTerms then
		assert(#defs == #sourceTerms)
		allSrcEqns = table()
	end
	for j,def in ipairs(defs) do
		local var = def:lhs()[1]
		
		def = makeTensorExpressionDense(def)
		
		local sourceTerm
		if not keepSourceTerms and outputCodeForSourceTerms then
			sourceTerm = makeTensorExpressionDense(sourceTerms[j])
		end

		local lhs, rhs = table.unpack(def)
		if not lhs.dim then
			-- then it's already a constant
			-- TODO maybe don't automatically convert x,t into x:diff(t) ... maybe make a separate function for that
			--[[
			printbr'failed to find lhs.dim'
			printbr(tostring(lhs))
			error'here'
			--]]
		else
			local dim = lhs:dim()
			assert(dim[#dim] == 1)	-- the ,t ...
			
			-- remove the ,t dimension
			lhs = Tensor(table.sub(lhs.variance, 1, #dim-1), function(...)
				local lhs_i = lhs[{...}][1]
				assert(Expression:isa(lhs_i), "expected an Expression here, but got "..tostring(lhs_i).." from "..tostring(lhs))
				return lhs[{...}][1]
			end)

			-- if it's a constant expression
			-- TODO put this in :unravel() ?
			if not rhs.dim then
				rhs = Tensor(lhs.variance, function() return rhs end)
			end
			
			if not keepSourceTerms and outputCodeForSourceTerms then
				if not sourceTerm.dim then
					sourceTerm = Tensor(lhs.variance, function() return sourceTerm end)
				end
			end
		end

		local fluxdefs = lhs:eq(rhs):unravel()
		local srcdefs
		if not keepSourceTerms and outputCodeForSourceTerms then
			srcdefs = lhs:eq(sourceTerm):unravel()
			assert(#fluxdefs == #srcdefs)
		end
		local foundDifference
		for k,fluxdef in ipairs(fluxdefs) do
			local lhsi, rhsi = table.unpack(fluxdef)
			local srci
			if not keepSourceTerms and outputCodeForSourceTerms then
				local _
				_, srci = table.unpack(srcdefs[k])
			end
			local lhsstr = tostring(lhsi)
			rhsi = simplify(rhsi)

			if defsForLhs[lhsstr] then
				if rhsi ~= defsForLhs[lhsstr] then
					printbr'mismatch'
					printbr(lhsi:eq(rhsi))
					printbr'difference'
					printbr(simplify(rhsi - defsForLhs[lhsstr]))
					foundDifference = true
					printbr()
				end
			else
				defsForLhs[lhsstr] = rhsi:clone()

				--printbr('adding expr:')
				--printbr('lhs:', lhsi)
				allLhs:insert(lhsi)
				--printbr('rhs:', rhsi)
				allRhs:insert(rhsi)
				if not keepSourceTerms and outputCodeForSourceTerms then
					--printbr('src:', srci)
					-- store this as an equation lhs = src in case we prune zeroes beforehand and srcs vs rhss mismatch in size
					allSrcEqns:insert(lhsi:eq(srci))
				end
			end
		end
	end
	if foundDifference then
		printbr[[
Symmetric differences can be caused from symmetries that contain partial derivatives, since the flux is only evaluated in a single direction, the other symmetric half of the derivative is discarded.<br>
Ex: for $K_{(ij),t} + 2 \alpha Z_{(i,j)} = ...$,<br>
expanded will become $K_{xy,t} + \alpha Z_{x,y} + \alpha Z_{y,x} = ...$,<br>
which will match to the flux equation $K_{xy,t} + {\textbf{F}^x}_{,x} + {\textbf{F}^y}_{,y} = ...$,<br>
for ${\textbf{F}^x}_{,x} = \alpha Z_{y,x}, {\textbf{F}^y}_{,y} = \alpha Z_{x,y}$,<br>
But this means (TODO) if you are going to rotate fluxes into normal, verify mathematically that this will work.<br>
It's an easy problem to verify for the Euler fluid equations, but idk if I have for any of these EFE decompositions.<br>
]]
	end
	assert(#allLhs == #allRhs)


	for i=1,#allLhs do
		local lhs = allLhs[i]
		local rhs = allRhs[i]
		printbr(lhs:eq(rhs))
	end


	-- TODO somewhere in here I should replace f with var'alpha_f' / alpha and f' with var'alphaSq_dalpha_f' / alpha^2.
	-- This will make both variables O(1) rather than O(1/alpha) and O(1/alpha^2) respectively.
	-- Should I do it outside of doCodegen or only inside it?
	-- This optimization might not be so compatible with the eigen-decomposition.

	-- CODEGEN
	local function doCodegen(eqns)
		-- TODO  do this before removing the rhs == 0 equations
		-- I could use the 'inputs' in codegen ...
		-- but I want the lhs ones to turn into F., and the rhs ones to turn into U. or Dx.
		local function nameToC(name)
			return (name
				:gsub('^\\pi$', 'M_PI')
				:gsub("^f'$", 'dalpha_f')
				:gsub('^\\alpha$', 'alpha')
				:gsub('^\\rho$', 'rho')
				:gsub('^\\Theta$', 'Theta')
				:gsub('^\\gamma%^{(..)}$', 'gamma_uu.%1')
				:gsub('^\\gamma_{(..)}$', 'gamma_ll.%1')
				:gsub('^S_{(.)}$', 'S_l.%1')
				:gsub('^S_{(..)}$', 'S_ll.%1')
				:gsub('^K_{(..)}$', 'K_ll.%1')
				:gsub('^d_{(.)(..)}$', 'd_lll.%1.%2')
				:gsub('^a_(.)$', 'a_l.%1')
				:gsub('^Z_(.)$', 'Z_l.%1')
			)
		end
		local codegenOutputs = table()
		for _,eqn in ipairs(eqns) do
			local lhs, rhs = table.unpack(eqn)
			if not Derivative:isa(lhs)
			or #lhs ~= 2
			or lhs[2] ~= t
			or not Variable:isa(lhs[1]) -- or TensorRef:isa(lhs))
			then
				error("expected lhs of eqn to be a time derivative, got "..lhs)
			end
			lhs = lhs[1]		-- convert var_,t to just var
			if Variable:isa(lhs) then
				lhs = Variable('F.'..nameToC(lhs.name))
			end	-- TODO when wouldn't it be a Variable?
		
			-- optimize out our reciprocal alphas for 1+log slicing
			rhs = rhs:replace(f, var'f_alpha' / alpha)
			rhs = rhs:replace(df, var'alphaSq_dalpha_f' / alpha^2)
			
			rhs = rhs:map(function(expr)
				if Derivative:isa(expr) then
					-- use the Dx prefix if you want to keep the deriv variables separate ... this would be used in my hydro-cl 'eigen_fluxTransform'
					-- omit it if you want to use the same state ... this would be in 'fluxFromCons'
					-- NOTICE if you omit it here then later in the simplification and codegen it can optimize out more common structures
					-- but in that case, you'd need to run this a separate time to get the eigen_fluxTransform code
					assert(#expr == 2)
					assert(expr[2] == x)
					expr = Variable('Dx.'..nameToC(expr[1].name))
					return expr
				end
			end)
			rhs = rhs:map(function(expr)
				if Variable:isa(expr)
				and not expr.name:match'^Dx%.'
				and expr.name ~= 'f'
				then
					local name = nameToC(expr.name)
					--[[ no worries, I copy them to local variables anyways
					if not name:match'^gamma_uu' then
						name = 'U.'..name
					end
					--]]
					return Variable(name)
				end
			end)
			codegenOutputs:insert{[lhs.name] = rhs()}
		end
		printbr()
		printbr'as C code:'
		printbr'<code>'
		printbr((symmath.export.C:toCode{
			output = codegenOutputs,
			--notmp = true,	-- don't write temp variables
		}
			:gsub('\n', '<br>\n')			-- add html newlines to our <code> block
			:gsub('double F%.', 'F.')		-- don't declare struct vars
			:gsub('double ', 'real ')		-- and for our temp vars, declare as 'real'
		))
		printbr'</code>'
	end
	
	doCodegen(range(#allLhs):mapi(function(i)
		return allLhs[i]:eq((-allRhs[i])())		-- flux terms are u_i,t + f^ijk u_j,k = s_i ... so they appear on the lhs of the eqn, so negative them
	end))


	-- VERY slow at the moment
	if not keepSourceTerms and outputCodeForSourceTerms then
		printbr'source terms:'
		doCodegen(allSrcEqns)
	end


	-- not sure if I should remove zero rows before or after codegen
	if removeZeroRows then
		local newLhs = table()
		local newRhs = table()
		for i=1,#allLhs do
			local lhs = allLhs[i]
			local rhs = allRhs[i]
			if rhs == Constant(0) then
				printbr('removing zero row '..lhs:eq(rhs))
			else
				newLhs:insert(lhs)
				newRhs:insert(rhs)
			end
		end
		allLhs, allRhs = newLhs, newRhs
	end


	local allDxs = allLhs:map(function(lhs)
		assert(diff:isa(lhs), "somehow got a non-derivative on the lhs: "..tostring(lhs))
		assert(lhs[2] == t)
		assert(#lhs == 2)
		return diff(lhs[1], fluxdirvar)
	end)
	local b
	fluxJacobian, b = factorLinearSystem(allRhs, allDxs)
	local n = #fluxJacobian

	fluxJacobian = (-fluxJacobian)()	-- change from U,t = A U,x + b into U,t + A U,x = b

	if useShift then
		fluxJacobian = (fluxJacobian + betaVars[fluxdir] * Matrix.identity(n))()		-- remove diagonals
	end



	-- simplify the flux jacobian matrix
	-- [[
	if useZ4
	or (useV and useShift)
	then-- simplify inverses
		fixFluxJacobian(fluxJacobian)
	end
	--]]
	if useShift and useV then	-- idk how this happens
		for i=1,3 do
			for j=1,3 do
				fluxJacobian = fluxJacobian:replace(gammaLL[i][j], gammaLVars[i][j])
			end
		end
	end

	local dts = Matrix(allLhs):transpose()
	local dxs = Matrix(allDxs):transpose()
	if not useShift then
		printbr((dts + fluxJacobian * dxs):eq(b))
	else
		printbr((dts + (fluxJacobian - betaVars[fluxdir] * Matrix.identity(n)) * dxs):eq(b))
	end


	-- save cached 'fluxJacobian'
	do
		-- *Vars are only as big as the metric # of indexes
		-- so using 1D means only '.x' exists
		-- soo ... do I want it fake-1D and really 3D underneath?  do I want an 'r' coord equivalent (with spherical metric ds^2 = dr^2 + r^2 dOmega^2 ?
		local n = #xs
		
		local fluxJacobian = clone(fluxJacobian)
		local function replaceAll(from, to)
			fluxJacobian = fluxJacobian:replace(from, to)
			return to
		end
		local alpha = replaceAll(alpha, var'alpha')
		
		local betaVarsClone, betaLVarsClone
		if useShift and useLowerShift then
			betaVarsClone = betaVars
			betaLVarsClone = clone(betaLVars)
		else
			betaVarsClone = clone(betaVars)
		end
		local betaVars = betaVarsClone
		local betaLVars = betaLVarsClone
		
		local gammaUVars = clone(gammaUVars)
		local gammaLVars = clone(gammaLVars)
		for i=1,n do
			if useShift and useLowerShift then
				betaLVars[i] = replaceAll(betaLVars[i], var('betaLVars['..i..']'))
			else
				betaVars[i] = replaceAll(betaVars[i], var('betaVars['..i..']'))
			end
			for j=1,n do
				local u,v
				if i < j then u,v = i,j else u,v = j,i end
				local suffix = '['..u..']['..v..']'
				gammaUVars[i][j] = replaceAll(gammaUVars[i][j], var('gammaUVars'..suffix))
				gammaLVars[i][j] = replaceAll(gammaLVars[i][j], var('gammaLVars'..suffix))
			end
		end
		local gamma = replaceAll(gamma, var'gamma')
		local df = replaceAll(df, var'df')

		local vars = table{alpha, f, df}
		for i=1,n do
			if useLowerShift then
				vars:insert(betaLVars[i])
			else
				vars:insert(betaVars[i])
			end
		end
		for i=1,n do
			for j=i,n do
				vars:insert(gammaUVars[i][j])
			end
		end
		for i=1,n do
			for j=i,n do
				vars:insert(gammaLVars[i][j])
			end
		end
		vars:insert(gamma)

		file(symmathJacobianFilename):write(symmath.export.Mathematica(fluxJacobian, vars):gsub('}, {', '},\n\t{'))
	end


	-- [[ outputting to mathematica (particularly useV useShift noZeroRows
	if outputMathematica then
		-- make variables Mathematica-friendly
		local fluxJacobian = clone(fluxJacobian)
		local function replaceAll(from, to)
			fluxJacobian = fluxJacobian:replace(from, to)
			return to
		end
		local alpha = replaceAll(alpha, var'\\[Alpha]')
		local betaVars = clone(betaVars)
		local gammaUVars = clone(gammaUVars)
		local gammaLVars = clone(gammaLVars)
		for i=1,3 do
			betaVars[i] = replaceAll(betaVars[i], var('bU'..xs[i].name))
			for j=1,3 do
				local u,v
				if i < j then u,v = i,j else u,v = j,i end
				local suffix = xs[u].name .. xs[v].name
				gammaUVars[i][j] = replaceAll(gammaUVars[i][j], var('gU'..suffix))
				gammaLVars[i][j] = replaceAll(gammaLVars[i][j], var('gL'..suffix))
			end
		end
		local gamma = replaceAll(gamma, var'g')

		local vars = table{alpha, f, df}
		for i=1,3 do
			vars:insert(betaVars[i])
		end
		for i=1,3 do
			for j=i,3 do
				vars:insert(gammaUVars[i][j])
			end
		end
		for i=1,3 do
			for j=i,3 do
				vars:insert(gammaLVars[i][j])
			end
		end
		vars:insert(gamma)

		file('flux_matrix_output/mathematica.'..outputSuffix..'.txt'):write(symmath.export.Mathematica(fluxJacobian, vars):gsub('}, {', '},\n\t{'))
	end
	--]]

	assert(#outputFiles == 2)
	outputFiles:remove():close()
end	-- done generating the header
-- now we can copy the header into the main file
--for _,f in ipairs(outputFiles) do
--	f:write(file(headerExpressionFilename):read())
--end
-- nope -already copied

local A = fluxJacobian
local n = #fluxJacobian



-- [[ I don't have poly factoring so this doesn't matter
-- it's also freezing for almost anything with useShift
-- maybe I should change my shift condition?
io.stderr:write'computing characteristic polynomial...\n'
local lambda = var'\\lambda'
local charpolymat = (A - Matrix.identity(n) * lambda)()
local charpoly  = charpolymat:determinant()
printbr'characteristic polynomial:'
printbr(charpoly)
--os.exit()
printbr('simplified...')
charpoly = charpoly()
printbr(charpoly)
--]]


-- [[ this works fast enough without simplification
A = A()
printbr'simplified:'
printbr(A)

local sofar, reduce = A:inverse()
printbr(sofar)
printbr(reduce)
--]]

--if not useV and not useGamma and not useZ4 and not use1D then
--	closeFile() os.exit()
--end

--[[
here's where I need polynomial factoring
let:
	x = lambda^2
	a = alpha^2 * g
char poly:
	x^4 - a * (3 + f) * x^3 + a^2 * (3 + 3*f) * x^2 - a^3 * (1 + 3*f) * x + a^4 * f
	= (x - a)^3 * (x - a f)
	= (lambda^2 - alpha^2 gamma^xx)^3 * (lambda^2 - f alpha^2 gamma^xx)
	= (lambda + alpha sqrt(gamma^xx))^3 (lambda - alpha sqrt(gamma^xx))^3 (lambda + alpha sqrt(f gamma^xx)) (lambda - alpha sqrt(f gamma^xx))
roots are:
	lambda = alpha sqrt(f gamma^xx)
	lambda = -alpha sqrt(f gamma^xx)
	lambda = alpha sqrt(gamma^xx) x3
	lambda = -alpha sqrt(gamma^xx) x3


same deal, for Z4, no shift, remove zero rows:
	x^7 - x^6 a (6 + f) + x^5 a^2 (6 f + 15) - x^4 a^3 (20 + 15 f) + x^3 a^4 (15 + 20 f) - x^2 a^5 (6 + 15 f) + x a^6 (1 + 6 f) - a^7 f
	= (x - a f) (x - a)^6

	lambda = +- alpha sqrt(f gamma^xx) has multiplicity 1
	lambda = +- alpha sqrt(gamma^xx) has multiplicity 6
	lambda = 0 has multiplicity 3
 
	however the eigenvectors of alpha sqrt(gamma^xx) only have dimension 5, not 6 ...
	so ... that means this is not hyperbolic, right?

--]]

-- try solving it for one particular eigenvector/value
local gammaUjj = gammaUVars[fluxdir][fluxdir]
-- the eigenvalues are the same for useV on or off
-- but the multiplicities are different:
-- with useV off we get 19, 3, 3, 1, 1
-- with useV on we get 18, 5, 5, 1, 1
local lambdas
if use1D then
	lambdas = table{
		-alpha * sqrt(f * gammaUjj),
		0,
		alpha * sqrt(f * gammaUjj),
	}
elseif useZ4 or (useV and useShift) then
	lambdas = table{
		0,
		-alpha * sqrt(gammaUjj),
		alpha * sqrt(gammaUjj),
		-alpha * sqrt(f * gammaUjj),
		alpha * sqrt(f * gammaUjj),
	}
elseif useGamma and not useShift then	-- same for both removeZeroRows and not removeZeroRows
	--[[
	x = lambda^2
	g = alpha^2 gamma^xx
	asking wolfram alpha:
	x^6 + (3-f) x^5 g - (3+f) x^4 g^2 + (5f - 11) x^3 g^3 + (f + 6) x^2 g^4 + 4 (3 - 2 f) x g^5 + 4 (f - 2) g^6
	roots:
	lambda^2 = -2 alpha^2 gamma^xx
	lambda^2 = alpha^2 gamma^xx
	lambda^2 = (f-2) alpha^2 gamma^xx
	...and we have a complex root...
	lambda = +-i alpha sqrt(2 gamma^xx)
	lambda = +- alpha sqrt(gamma^xx)
	lambda = +- alpha sqrt((f-2) gamma^xx)
	--]]
	lambdas = table{
		-alpha * sqrt((f-2) * gammaUjj),
		-alpha * sqrt(gammaUjj),
		0,
		alpha * sqrt(gammaUjj),
		alpha * sqrt((f-2) * gammaUjj),
	}
else

-- for adm noZeroRows:
-- a = alpha, g = gamma, x = lambda
-- -x^5 (3 a^4 g^2 x^4 - a^6 g^3 x^2 - 3 g a^2 x^6 - f g a^2 x^6 + x^8 + 3 f a^4 g^2 x^4 - 3 f a^6 g^3 x^2 + f a^8 g^4) = 0
-- for y = x^2, b = a^2 g
-- x^5 (y^4 - (f + 3) b y^3 + 3 b^2 (f + 1) y^2 - b^3 (3 f + 1) y + f b^4) = 0
-- x^5 (b f - y) (b - y)^3 = 0
-- x^5 = 0, (a^2 g - x^2)^3 = 0, a^2 g f - x^2 = 0
-- {x = 0} x5, {x = ±a sqrt(g)} x3, {x = ±a sqrt(f g)} x1

	lambdas = table{
		-- the more multiplicity, the easier it is to factor
		-- also, without useV, the 1-multiplicity eigenvectors take forever and the expressions get huge and take forever
		--[[
		-alpha * sqrt(f * gammaUjj),
		-alpha * sqrt(gammaUjj),
		0,
		alpha * sqrt(gammaUjj),
		alpha * sqrt(f * gammaUjj),
		--]]
		
		
		-- [[
		0,
		alpha * sqrt(gammaUjj),
		-alpha * sqrt(gammaUjj),
		alpha * sqrt(f * gammaUjj),
		-alpha * sqrt(f * gammaUjj),
		--]]
	}
end


if bakeF then
	for i=#lambdas,1,-1 do
		lambdas[i] = replaceBakedF(lambdas[i])
		for j=i+1,#lambdas do
			if lambdas[i] == lambdas[j] then
				table.remove(lambdas[j])
			end
		end
	end
end


local evs = table()
local multiplicity = table()
for _,lambda in ipairs(lambdas) do
io.stderr:write('finding eigenvector of eigenvalue '..tostring(lambda)..'\n') io.stderr:flush()
	printbr('eigenvalue:', lambda)
	
	local A_minus_lambda_I = ((A - Matrix.identity(n) * lambda))()
	printbr'reducing'
	printbr(A_minus_lambda_I)

	local ev = A_minus_lambda_I:nullspace(
		--printbr	-- to debug
	)

	if not ev then
		printbr("found no eigenvectors associated with eigenvalue",lambda_)
	else
		--[[ try to remove any inverses ...
		for i=1,3 do
			for j=1,3 do
				ev = ev:replace(det_gamma_times_gammaUInv[i][j](), gamma * gammaLVars[i][j])
				-- this doesn't simplify like I want it to ...
				ev = ev:replace((-det_gamma_times_gammaUInv[i][j])(), -gamma * gammaLVars[i][j])
			end
		end
		--]]

		multiplicity:insert(#ev[1])
		printbr('eigenvector:')
		printbr(ev)
		evs:insert(ev)
	end
end
io.stderr:write('done finding eigenvector!\n') io.stderr:flush()

local lambdaMat = Matrix.diagonal( table():append(
	lambdas:map(function(lambda,i)
		return range(multiplicity[i]):map(function() return lambda end)
	end):unpack()
):unpack() )
printbr('$\\Lambda$:')
printbr(lambdaMat)

local evRMat = Matrix(
	table():append(
		evs:map(function(ev)
			return ev:transpose()
		end):unpack()
	):unpack()
):transpose()
printbr('R:')
printbr(evRMat)

local evLMat, _, reason = evRMat:inverse()
assert(not reason, reason)	-- hmm, make Matrix.inverse more assert-compatible?

printbr('L:')
printbr(evLMat)

closeFile()
