#!/usr/bin/env luajit

-- idk where to put this, or what it should do
-- I just wanted a script to create a flux jacobian matrix from tensor index equations
require 'ext'
require 'symmath'.setup()
local TensorRef = require 'symmath.tensor.TensorRef'

TensorRef:pushRule'Prune/replacePartial'


local textOutput = false
local keepSourceTerms = false	-- this goes slow with 3D
local use1D = false
local removeZeroRows = false	-- whether to keep variables whose dt rows are entirely zero
local useShift = false			-- whether to include beta^i_,t
-- these are all exclusive
local useV = false				-- ADM Bona-Masso with V constraint.  Not needed with use1D
local useGamma = false			-- ADM Bona-Masso with Gamma^i_,t . Exclusive to useV ... 
local useZ4 = false				-- Z4.  TODO factor and substitute metric inverses better



local t,x,y,z = vars('t','x','y','z')
local xs = use1D and table{x} or table{x,y,z}


local fluxdir = 1
--local fluxdir = 2
--local fluxdir = 3
local fluxdirvar = xs[fluxdir]
local depvars = table{t,fluxdirvar}



local MathJax
if not textOutput then -- [[ mathjax output
	MathJax = require 'symmath.tostring.MathJax'
	MathJax.usePartialLHSForDerivative = true
	symmath.tostring = MathJax
else --]] 
	--[[ text output - breaking
	function var(s)
		if symmath.tostring.fixImplicitName then
			s = symmath.tostring:fixImplicitName(s)
			if s:sub(1,1) == '\\' then s = s:sub(2) end
		end
		return Variable(s)
	end
	--]]
end

-- kronecher delta
local delta = var'\\delta'
-- adm metric
local alpha = var'\\alpha'
local beta = var'\\beta'
local gamma = var'\\gamma'
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
local Gamma = var'\\Gamma'
-- z4
local Z = var'Z'
local Theta = var('\\Theta', depvars)
local m = var'm'

Tensor.coords{
	{variables=xs},
	{variables={t}, symbols='t'},
	{variables={x}, symbols='x'},
	{variables={y}, symbols='y'},
	{variables={z}, symbols='z'},
}

local function simplify(expr)
	expr = expr():factorDivision()
	if op.add.is(expr) then
		for i=1,#expr do expr[i] = expr[i]() end
	end
	return expr
end


local outputPrefix = 'flux_matrix_output/flux_matrix.'
	..(useZ4 and 'z4' or 'adm')
	..(keepSourceTerms and '_withSource' or '')
	..(useV and '_useV' or '')
	..(useGamma and '_useGamma' or '')
	..(useShift and '_useShift' or '')
	..(removeZeroRows and '_noZeroRows' or '')
	..(use1D and '_1D' or '')
local outputExt = textOutput and '.txt' or '.html'

local printbr
local closeFile
do
	local filename = outputPrefix..outputExt
	print('writing to '..filename)
	local printbr_file = assert(io.open(filename, 'w'))
	if MathJax then printbr_file:write(tostring(MathJax.header)) end
	printbr = function(...)
		assert(printbr_file)
		local n = select('#', ...)
		for i=1,n do
			printbr_file:write(tostring(select(i, ...)))
			if i<n then printbr_file:write'\t' end
		end
		printbr_file:write'<br>\n'
		printbr_file:flush()
	end
	closeFile = function()
		printbr_file:write[[
	</body>
</html>
]]
		printbr_file:close()
	end
end


-- TODO start with EFE, apply Gauss-Codazzi-Ricci, then automatically recast all higher order derivatives as new variables of 1st derivatives
printbr[[primitive $\partial_t$ defs]]

local dt_alpha_def = alpha',t':eq( 
	- alpha^2 * f * K'^i_i' 
	+ alpha'_,i' * beta'^i' 
)
printbr(dt_alpha_def)

local dt_beta_def, dt_B_def
if useShift then
	-- hyperbolic Gamma driver
	-- 2008 Alcubierre eqns 4.3.31 & 4.3.32
	-- B^i = beta^i_,t
	-- so what should be used for beta^i_,j ? Bona&Masso use B for that in their papers ...
	-- I'll use b for the spatial derivative and B for the time derivative
	dt_beta_def = beta'^k_,t':eq(
		beta'^i' * beta'^k_,i' + B'^k'
	)
	printbr(dt_beta_def)
	local xi = frac(3,4)
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
			+ (gamma'^im' * gamma'^jk' + frac(1,3) * gamma'^ik' * gamma'^jm') * (d'_jlm,k' + d'_ljm,k' - d'_mlj,k') * beta'^l'
			- alpha * (2 * gamma'^ik' * gamma'^lj' - gamma'^ij' * gamma'^kl') * K'_kl,j'
			
			-- source terms
			+ 2 * d'^jmi' * (d'_jml' + d'_lmj' - d'_mlj') * beta'^l'
			- Gamma'^i_lm' * Gamma'^m' * beta'^l'
			+ Gamma'^i_km' * gamma'^jk' * Gamma'^m_jl' * beta'^l'
			- frac(2,3) * gamma'^ik' * d'_k^jm' * Gamma'_jlm' * beta'^l'
			+ gamma'^ij' * R'_jk' * beta'^k'
			
			+ 2 * alpha * d'_j^ji' * K'^k_k'
			- 2 * alpha * d'^ijk' * K'_jk'
			+ 4 * alpha * d'^jki' * A'_jk'
			+ 4 * alpha * d'_jk^j' * A'^ik'
			
			- 2 * alpha * a'_j' * A'^ij'
			- 2 * alpha * Gamma'^i_jk' * A'^jk'
			- 2 * alpha * Gamma'^j_jk' * A'^ik'
		)
	)
	printbr(dt_B_def)
end

local dt_gamma_def = gamma'_ij,t':eq( 
	-2 * alpha * K'_ij' 
	+ gamma'_ij,k' * beta'^k' 
	+ gamma'_kj' * beta'^k_,i' 
	+ gamma'_ik' * beta'^k_,j' 
)
printbr(dt_gamma_def) 

local dt_K_def = K'_ij,t':eq(
	K'_ij,k' * beta'^k' 
	+ K'_ki' * beta'^k_,j'
	+ K'_kj' * beta'^k_,i'
	- alpha',ij'
	+ Gamma'^k_ij' * alpha',k'
	+ alpha * (R'_ij' + K'^k_k' * K'_ij' - 2 * K'_ik' * K'^k_j')
	-- stress-energy terms	
	+ 4 * pi * alpha * (gamma'_ij' * (S - rho) - 2 * S'_ij')
)
printbr(dt_K_def)

printbr[[lapse]]

-- TODO functions, dependent variables, and total derivatives 
local df_def = f',k':eq(df * alpha * a'_k')
printbr(df_def)

printbr[[auxiliary variables]]

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

local d_def = d'_kij':eq(frac(1,2) * gamma'_ij,k')
printbr(d_def)

dgamma_for_d = (d_def * 2)():switch()
printbr(dgamma_for_d)

printbr[[${\gamma^{ij}}_{,k}$ wrt aux vars]]

local dgammaU_def = gamma'^ij_,k':eq(-gamma'^il' * gamma'_lm,k' * gamma'^mj')
printbr(dgammaU_def)

local dgammaU_for_d = dgammaU_def:substIndex(dgamma_for_d)()
printbr(dgammaU_for_d)

printbr[[connections wrt aux vars]]
local connL_def = Gamma'_ijk':eq(frac(1,2) * (gamma'_ij,k' + gamma'_ik,j' - gamma'_jk,i'))
printbr(connL_def)

local connL_for_d = connL_def
	:substIndex(dgamma_for_d)
	:simplify()
printbr(connL_for_d)

-- [[ just raise Gamma, keep d and gamma separate
local conn_for_d = (gamma'^il' * connL_for_d:reindex{ljk='ijk'})()
	:replace(gamma'^il' * Gamma'_ljk', Gamma'^i_jk')
--]]
--[[ expand() is adding a -1 somewhere that makes the last replace() choke
local conn_for_d = (gamma'^il' * connL_for_d:reindex{ljk='ijk'})()
	:expand()
	:replace(gamma'^il' * Gamma'_ljk', Gamma'^i_jk')
	:replace(gamma'^il' * d'_jlk', d'_jk^i')
	:replace(gamma'^il' * d'_klj', d'_kj^i')
	:replace((-gamma'^il' * d'_ljk')():expand(), -d'^i_jk')
--]]
--[[ TODO raise expression / equation? this works only for dense tensors
local conn_for_d = connL_for_d'^i_jk'()
--]]
--[[ if you want to raise d's indexes
local conn_for_d = connL_for_d:map(function(expr)
	if TensorRef.is(expr) then
		for i=2,#expr do
			if expr[i].symbol == 'i' then expr[i].lower = false end
		end
	end
end)
--]]
printbr(conn_for_d)

printbr[[Ricci wrt aux vars]]

local R_def = R'_ij':eq(Gamma'^k_ij'',k' - Gamma'^k_ik'',j' + Gamma'^k_lk' * Gamma'^l_ij' - Gamma'^k_lj' * Gamma'^l_ik')
printbr(R_def)

local R_for_d = R_def:substIndex(conn_for_d)
printbr(R_for_d)

R_for_d = R_for_d()
printbr(R_for_d)

R_for_d = R_for_d:substIndex(dgammaU_for_d)()
printbr(R_for_d)

printbr'symmetrizing'
R_for_d = R_for_d
	:symmetrizeIndexes(gamma, {1,2})()
	:symmetrizeIndexes(d, {2,3})()
	:symmetrizeIndexes(d, {1,4})()
printbr(R_for_d)

R_for_d = R_for_d:tidyIndexes()()
	:symmetrizeIndexes(gamma, {1,2})()
	:symmetrizeIndexes(d, {2,3})()
	:symmetrizeIndexes(d, {1,4})()
printbr(R_for_d)

printbr[[time derivative of $\alpha_{,t}$]]

-- don't subst alpha,t ..
dt_alpha_def = dt_alpha_def:subst(dalpha_for_a:reindex{i='k'})
printbr(dt_alpha_def)

printbr[[time derivative of $\gamma_{ij,t}$]]

printbr(dt_gamma_def)

-- don't use substIndex to preserve gamma_ij,t
--dt_gamma_def = dt_gamma_def:substIndex(dgamma_for_d)
dt_gamma_def = dt_gamma_def
	:subst(dgamma_for_d)
if useShift then
	dt_gamma_def = dt_gamma_def
		:substIndex(dbeta_for_b)
end
printbr(dt_gamma_def)

printbr[[time derivative of $a_{k,t}$]]

-- TODO splitDerivs
local dt_a_def = a_def',t'()
printbr(dt_a_def)

dt_a_def = dt_a_def
	:replace(alpha',kt', alpha',t'',k')
	:subst(dt_alpha_def)
printbr(dt_a_def)

dt_a_def = dt_a_def() 
printbr(dt_a_def)

dt_a_def = dt_a_def:replace(alpha',ik', frac(1,2) * ( alpha',i'',k' + alpha',k'',i' )) 
	:replace(K'^i_i,k', (gamma'^ij' * K'_ij')'_,k')
printbr(dt_a_def)

dt_a_def = dt_a_def:substIndex(df_def, dalpha_for_a)
printbr(dt_a_def)

dt_a_def = dt_a_def()
	:substIndex(dalpha_for_a)
	:symmetrizeIndexes(a, {1,2})
	:simplify()
printbr(dt_a_def)

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
	dt_beta_def = dt_beta_def:subst(dbeta_for_b:reindex{ki='ij'})
	printbr(dt_beta_def)


	printbr[[time derivative of ${b^i}_{j,t}$]]

	dt_b_def = dt_beta_def',j'():reindex{ij='ji'}
	printbr(dt_b_def)

	dt_b_def = dt_b_def
		:replace(beta'^k_,ti', beta'^k_,i'',t')
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
	
	dt_B_def = dt_B_def
		:substIndex(Gamma'^i':eq(Gamma'^i_jk' * gamma'^jk'))
		:substIndex(conn_for_d)
		:substIndex(A_for_K_uu)
		:substIndex(A_for_K_ll)
		:substIndex(dbeta_for_b)
		:substIndex(dbeta_for_b',k'())
		:substIndex(R_for_d)
		:simplify()

	printbr(dt_B_def)

	-- TODO automatic relabel indexes
	-- TODO prevent substIndex from using indexes already reserved for other coordinate sets
	dt_B_def = dt_B_def
		:symmetrizeIndexes(gamma, {1,2})()
		:symmetrizeIndexes(d, {2,3})()
		:symmetrizeIndexes(d, {1,4})()
	printbr(dt_B_def)
end

printbr[[time derivative of $d_{kij,t}$]]

local dt_d_def = d_def',t'()
printbr(dt_d_def)

dt_d_def = dt_d_def
	:replace(gamma'_ij,k,t', gamma'_ij,t'',k')
	-- TODO automatically relabel the sum indexes
	-- ... this would require knowledge of the entire dt_d_def expression, to know what indexes are available
	:subst(dt_gamma_def:reindex{ijl='ijk'})
printbr(dt_d_def)

dt_d_def = dt_d_def()
printbr(dt_d_def)

dt_d_def = dt_d_def
	:replace(gamma'_ij,l,k', gamma'_ij,l'',k')
	:subst(dgamma_for_d:reindex{ijl='ijk'})
	:subst(dgamma_for_d:reindex{ilk='ijk'})
	:subst(dgamma_for_d:reindex{ljk='ijk'})
	:subst(dalpha_for_a)
if useShift then
	dt_d_def = dt_d_def:substIndex(dbeta_for_b)
end	
dt_d_def = dt_d_def:simplify()
printbr(dt_d_def)

printbr[[$K_{ij,t}$ with hyperbolic terms]]

printbr(dt_K_def)

dt_K_def = dt_K_def
	:replace(alpha',ij', frac(1,2) * (alpha',i'',j' + alpha',j'',i'))
	:subst(dalpha_for_a:reindex{i='k'})
	:subst(dalpha_for_a:reindex{j='k'})
	:subst(dalpha_for_a)
printbr(dt_K_def)
	
dt_K_def = dt_K_def:simplify()
printbr(dt_K_def)

dt_K_def = dt_K_def
	:subst(dalpha_for_a:reindex{j='k'})
	:subst(dalpha_for_a:reindex{i='k'})
	:simplify()
printbr(dt_K_def)

dt_K_def = dt_K_def:subst(conn_for_d:reindex{kij='ijk'})
printbr(dt_K_def)

dt_K_def = dt_K_def:subst(R_for_d)
printbr(dt_K_def)
dt_K_def = dt_K_def()
printbr(dt_K_def)

local dsym_def = d'_ijk,l':eq(frac(1,2) * (d'_ijk,l' + d'_ljk,i'))
--[[ substIndex works ... but replaces the replaced ...
dt_K_def = dt_K_def
	:substIndex(dsym_def:reindex{ijlk='ijkl'})
	:substIndex(dsym_def:reindex{iklj='ijkl'})
	:substIndex(dsym_def:reindex{jilk='ijkl'})
	:substIndex(dsym_def:reindex{kijl='ijkl'})
--]]
--[[
dt_K_def = dt_K_def
	:subst(dsym_def:reindex{ijmk='ijkl'})
	:subst(dsym_def:reindex{ikmj='ijkl'})
	:subst(dsym_def:reindex{jimk='ijkl'})
	:subst(dsym_def:reindex{kijm='ijkl'})
--]]
if useShift then
	dt_K_def = dt_K_def:substIndex(dbeta_for_b)
end
dt_K_def = dt_K_def:tidyIndexes()()
	:symmetrizeIndexes(gamma, {1,2})()
	:symmetrizeIndexes(d, {2,3})()
	:symmetrizeIndexes(d, {1,4})()
printbr(dt_K_def)


local defs = table()


if useZ4 then
	
	-- I'm taking this from 2008 Yano et al Flux-Vector-Splitting method for Z4 formalism and its numerical analysis
	-- ... and from its source paper, 2005 Bona et al "Geometrically motivated hyperbolic coordinate condions for numerical relativity- Analysis, issues and implementation"

	printbr'Z4 terms'

	local dt_a_def = a'_i,t':eq(
		(beta'^j' * a'_i')',j' 
		- (alpha * f * K'^j_j' + beta'^j' * a'_j')'_,i' 
		+ b'^j_i' * a'_j' 
		- b'^j_j' * a'_i'
	)
	printbr(dt_a_def)

	dt_a_def = dt_a_def():substIndex(df_def)
	printbr(dt_a_def)
	
	dt_a_def = dt_a_def
		:substIndex(dalpha_for_a)
	
	if useShift then
		dt_a_def = dt_a_def:substIndex(dbeta_for_b)
	end

	dt_a_def = dt_a_def 
		:splitOffDerivIndexes()
		:replace(K'^j_j', gamma'^jk' * K'_jk')()
		:substIndex(dgammaU_for_d)()
		-- TODO simplifyMetric(gamma) operation to do just this ...
		:replace((2 * alpha * f * gamma'^jm' * d'_iml' * gamma'^lk' * K'_jk')(), 2 * alpha * f * d'_i^jk' * K'_jk')
	printbr(dt_a_def)

	defs:insert(dt_a_def)

	local Q = var'Q'
	
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
		
		defs:insert(dt_b_def)

	end

	-- 2005 Bona et al eqn 24: gamma_ij,t = -2 alpha Q_ij ... and from eqn 3 I'm betting I've got gamma_ij,t right
	-- 2008 Yano et al doesn't give an equation for Q_ij
	local Qll_def = Q'_ij':eq(
		-1/(2*alpha) * dt_gamma_def:rhs()
	)
	printbr(Qll_def)

	dt_d_def = d'_kij,t':eq(
		(beta'^l' * d'_kij')'_,l'
		- (alpha * Q'_ij' + beta'^l' * d'_lij')'_,k'
		+ b'^l_k' * d'_lij' 
		- b'^l_l' * d'_kij'
	)
	printbr(dt_d_def)

	dt_d_def = dt_d_def:substIndex(Qll_def)	
	printbr(dt_d_def)
	
	dt_d_def = dt_d_def()
	printbr(dt_d_def)

	dt_d_def = dt_d_def
		:substIndex(dalpha_for_a)()
		
	if useShift then	
		dt_d_def = dt_d_def
			:substIndex(dbeta_for_b)()
	end

	dt_d_def = dt_d_def
		:substIndex(dgamma_for_d)()
		-- splitOffDerivIndexes isn't fully compatible with substIndex ...
	
	if useShift then
		dt_d_def = dt_d_def
			:substIndex(dbeta_for_b'_,k'())
	end

	dt_d_def = dt_d_def
		:substIndex(dgamma_for_d'_,l'())
		:symmetrizeIndexes(d, {1,4})()
	printbr(dt_d_def)
	defs:insert(dt_d_def)

	local xi = var'\\xi'
	dt_K_def = K'_ij,t':eq(
		(beta'^k' * K'_ij')'_,k'
	
		-- (alpha lambda^k_ij),k
		- (alpha * (
			d'^k_ij' 
			- frac(1,2) * (1 + xi) * (d'_ij^k' + d'_ji^k')
		))'_,k'
		- (alpha * (
			frac(1,2) * (
				a'_j'
				+ d'_jl^l'
				- (1 - xi) * d'^l_lj'
				- 2 * Z'_j'
			)
		))'_,i'
		- (alpha * (
			frac(1,2) * (
				a'_i'
				+ d'_il^l'
				- (1 - xi) * d'^l_li'
				- 2 * Z'_i'
			)
		))'_,j'
		
		-- 2005 Bona et al eqn A.1 of S(K_ij) 
		- K'_ij' * b'^k_k'
		+ K'_ik' * b'^k_j'
		+ K'_jk' * b'^k_i'
		+ alpha * (
			frac(1,2) * (1 + xi) * (
				-a'_k' * Gamma'^k_ij'
				+ frac(1,2) * (
					a'_i' * d'_jk^k'
					+ a'_j' * d'_ik^k'
				)
			)
			+ frac(1,2) * (1 - xi) * (
				a'_k' * d'^k_ij'
				- frac(1,2) * (
					a'_j' * (2 * d'^k_ki' - d'_ik^k')
					+ a'_i' * (2 * d'^k_kj' - d'_jk^k')
				)
				+ 2 * (
					d'_ir^m' * d'^r_mj'
					+ d'_jr^m' * d'^r_mi'
				)
				- 2 * d'^l_lk' * (d'_ij^k' + d'_ji^k')
			)
			+ (d'_kl^l' + a'_k' - 2 * Z'_k') * Gamma'^k_ij'
			- Gamma'^k_mj' * Gamma'^m_ki'
			- (a'_i' * Z'_j' + a'_j' * Z'_i')
			+ 2 * K'^k_i' * K'_kj'
			+ (K'^k_k' - 2 * Theta) * K'_ij'
		)
		-- stress-energy terms	
		+ 4 * pi * alpha * (gamma'_ij' * (S - rho) - 2 * S'_ij')
	)
	printbr(dt_K_def)

	-- do this before simplify, or splitOffDerivIndexes before this
	dt_K_def = dt_K_def
		:replaceIndex(d'^i_jk', gamma'^il' * d'_ljk')
		:replaceIndex(d'_ij^k', d'_ijl' * gamma'^lk')

	dt_K_def = dt_K_def()
	printbr(dt_K_def)

	dt_K_def = dt_K_def
		:replaceIndex(xi'_,k', 0)
		:simplify()
		:substIndex(dgammaU_for_d)
		:substIndex(dgamma_for_d)
		:substIndex(conn_for_d)
		:substIndex(dalpha_for_a)
	
	if useShift then
		dt_K_def = dt_K_def
			:substIndex(dbeta_for_b)
	end

	dt_K_def = dt_K_def
		:substIndex(conn_for_d)()
		:tidyIndexes()()
	printbr(dt_K_def)

	-- TODO function to simplify gammas and deltas

	defs:insert(dt_K_def)
	
	
	local dt_Theta_def = Theta'_,t':eq(
		(beta'^k' * Theta)'_,k'
		- (alpha * (d'^kj_j' - d'_j^jk' - Z'^k'))'_,k'
		-- 2005 Bona et al eqn A.3 of S(Theta)
		-Theta * b'^k_k'
		+ alpha/2 * (
			2 * a'_k' * (d'^kj_j' - d'_j^jk' - 2 * Z'^k')
			+ d'_k^rs' * Gamma'^k_rs'
			- d'^kj_j' * (d'_kl^l' - 2 * Z'_k')
			- K'^k_r' * K'^r_k'
			+ K'^k_k' * (K'^l_l' - 2 * Theta)
		)
		- 8 * pi * alpha * rho
	)
	printbr(dt_Theta_def) 
	
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

	defs:insert(dt_Theta_def)

	local dt_Z_def = Z'_i,t':eq(
		(beta'^k' * Z'_i')'_,k'
		+ (alpha * K'^k_i')'_,k'
		- (alpha * (K'^k_k' - Theta))'_,i'
		-- 2005 Bona et al eqn A.2 S(Z_i)
		- Z'_i' * b'^k_k'
		+ Z'_k' * b'^k_i'
		+ alpha * (
			a'_i' * (K'^k_k' - 2 * Theta)
			- a'_k' * K'^k_i'
			- K'^k_r' * Gamma'^r_ki'
			+ K'^k_i' * (d'kl^l' - 2 * Z'_k')
		)
		- 8 * pi * alpha * S'_i'
	)
	printbr(dt_Z_def)

	printbr(dt_Z_def) 
	
	dt_Z_def = dt_Z_def
		:replaceIndex(K'^i_j', gamma'^ik' * K'_kj')
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


	defs:insert(dt_Z_def)

else

	defs:insert(dt_alpha_def)
	defs:insert(dt_beta_def)
	defs:insert(dt_gamma_def)
	defs:insert(dt_a_def)
	defs:insert(dt_b_def)
	defs:insert(dt_B_def)
	defs:insert(dt_d_def)
	
	if useV then
		-- TODO just replace the V's in this
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
						
				+ K'^k_k' * K'_ij'
				- K'_i^k' * K'_kj'
				- K'_j^k' * K'_ki'
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
						
				+ K'^k_k' * K'_ij'
				- K'_i^k' * K'_kj'
				- K'_j^k' * K'_ki'
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
			+ alpha * a'_k' * K'^k_k'
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
			if TensorRef.is(expr) 
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

if not keepSourceTerms then
	-- for all summed terms, for all coefficients, 
	--	if none have derivatives then remove them
	-- remove from defs any equations that no longer have any terms
	local sourceTerms = table()
	defs = defs:map(function(def,i,t)
		local lhs, rhs = table.unpack(defs[i])
		
		sourceTerms[i] = rhs:map(function(expr)
			if TensorRef.is(expr) then
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
	printbr'...and those source terms are...'
	for i,def in ipairs(defs) do
		local lhs, rhs = table.unpack(def)
		printbr(lhs..'$ + \\dots = $'..sourceTerms[i])
	end
end

local betaVars = Tensor('^i', function(i)
	return var('\\beta^'..xs[i].name)
end)

local gammaVars = Tensor('_ij', function(i,j) 
	if i > j then i,j = j,i end 
	return var('\\gamma_{'..xs[i].name..xs[j].name..'}', depvars)
end)

local gammaUVars = Tensor('^ij', function(i,j) 
	if i > j then i,j = j,i end 
	return var('\\gamma^{'..xs[i].name..xs[j].name..'}', depvars)
end)

local gammaLVars = Tensor('_ij', function(i,j) 
	if i > j then i,j = j,i end 
	return var('\\gamma_{'..xs[i].name..xs[j].name..'}', depvars)
end)

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

--[[
gamma^ij = inv(gamma_kl)^ij = 1/det(gamma_kl) adj(gamma_kl)^ij
gamma_ij = inv(gamma^kl)_ij = 1/det(gamma^kl) adj(gamma^kl)_ij = det(gamma_kl) adj(gamma^kl)_ij
TODO get 3x3 inverse to work automatically.  
--]]
local det_gamma_times_gammaUInv = Matrix(
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
)


-- by this point we're going to switch to expanded variables
-- so defining a metric is safe
Tensor.metric(gammaVars, gammaUVars)

printbr('spelled out')
local allLhs = table()
local allRhs = table()
local defsForLhs = table()	-- check to make sure symmetric terms have equal rhs's.  key by the lhs
for _,def in ipairs(defs) do
	local var = def:lhs()[1]
	
	-- these should be zero anyways ...
	if var == alpha 
	or var == beta 
	or var == gamma 
	then	
		assert(def:rhs() == Constant(0), "expected zero")
	else
		def = def
			:map(function(expr)
				if TensorRef.is(expr)
				and expr[1] == gamma
				then
					-- warn if there are any gamma^ij_,k...
					for i=4,#expr do	-- expr[1] is the variable, 2,3 are the ij indexes, so start at 4 for derivatives
						assert(not expr[i].lower, "found a gamma_ij term: "..tostring(expr))
					end
					return TensorRef(gammaUVars, table.unpack(expr, 2))
				end
			end)
			:replace(a, aVars)
			:replace(d, dVars)
			:replace(K, KVars)
			:replace(Theta',t', Theta:diff(t))
		
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
			def = def:replace(Theta'_,k', Tensor('_k', function(k) return Theta:diff(xs[k]) end))
			def = def:replace(Theta'_,i', Tensor('_i', function(k) return Theta:diff(xs[k]) end))
			def = def:replace(Z, ZVars)
		end
		def = def()

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
			assert(dim[#dim].value == 1)	-- the ,t ...

			-- remove the ,t dimension
			lhs = Tensor(table.sub(lhs.variance, 1, #dim-1), function(...)
				local lhs_i = lhs[{...}][1]
				assert(Expression.is(lhs_i), "expected an Expression here, but got "..tostring(lhs_i).." from "..tostring(lhs))
				return lhs[{...}][1]
			end)

			-- if it's a constant expression
			-- TODO put this in :unravel() ?
			if not rhs.dim then
				rhs = Tensor(lhs.variance, function() return rhs end)
			end
		end

		local eqns = lhs:eq(rhs):unravel()
		for _,eqn in ipairs(eqns) do		
			local lhs, rhs = table.unpack(eqn)
			local lhsstr = tostring(lhs)
			rhs = simplify(rhs)

			if removeZeroRows and rhs == Constant(0) then
				printbr('removing zero row '..lhs:eq(rhs))
			elseif defsForLhs[lhsstr] then
				if rhs ~= defsForLhs[lhsstr] then
					printbr'mismatch'
					printbr(lhs:eq(rhs))
					printbr'difference'
					printbr(simplify(rhs - defsForLhs[lhsstr]))
					printbr()
				end
			else
				defsForLhs[lhsstr] = rhs:clone()

				--[[ exclude dt terms that are zero -- that means these dx terms belong in the source terms 
				if rhs ~= Constant(0) then
				--]]
				-- [[
				do
				--]]
					allLhs:insert(lhs)
					allRhs:insert(rhs)
					printbr(lhs:eq(rhs))
				end
			end
		end
	end
end

local allDxs = allLhs:map(function(lhs)
	assert(diff.is(lhs), "somehow got a non-derivative on the lhs: "..tostring(lhs))
	assert(lhs[2] == t)
	assert(#lhs == 2)
	return diff(lhs[1], fluxdirvar)
end)
local A, b = factorLinearSystem(allRhs, allDxs)
local n = #A

A = (-A)()	-- change from U,t = A U,x + b into U,t + A U,x = b

if useShift then
	A = (A + betaVars[fluxdir] * Matrix.identity(n))()		-- remove diagonals
end



-- in-place substitution
local gammaLU = (gammaLVars'_ik' * gammaUVars'^kj')()
local gammaLL = (gammaLVars'_ik' * gammaLU'_j^k')()
local gammaUU = (gammaUVars'^ik' * gammaLU'_k^j')()

-- g delta^k_l = g (g^kx g_lx + g^ky g_ly + g^kz g_lz)
-- g delta^k_l = g (g^ka g_la + g^kb g_lb + g^kc g_lc)
-- g (delta^k_l - g^ka g_la) = g (g^kb g_lb + g^kc g_lc)
local someMoreRules = table()
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
			if op.unm.is(find) then find = find[1] sign = -1 end
			local repl = (sign * gamma * (delta_kl - gammaUVars[k][a] * gammaLVars[l][a]))()
			
--			printbr(k,',',l,',',find:eq(repl))
			
			someMoreRules[k][l]:insert{find, repl}
		end
	end
end

local function fixCell(A,i,j)
	for k=1,3 do
		for l=k,3 do
			-- hmm ... the (1-f)'s are messing me up... it can't factor them out ...
			local delta_kl = k == l and 1 or 0			
			A[i][j] = A[i][j]:replace(gammaLL[k][l], gammaLVars[k][l])()
			A[i][j] = A[i][j]:replace(gammaUU[k][l], gammaUVars[k][l])()
			A[i][j] = A[i][j]:replace(gammaLU[k][l], delta_kl)()
			
			local expr = det_gamma_times_gammaUInv[k][l]
			local expr_eq = gamma * gammaLVars[k][l]
			A[i][j] = A[i][j]:replace( expr(), expr_eq)()
			
			assert(op.sub.is(expr) and #expr == 2)
			local neg = expr[2] - expr[1]
			local neg_eq = -gamma * gammaLVars[k][l]
			A[i][j] = A[i][j]:replace( neg, neg_eq)()
	
			
			for _,rule in ipairs(someMoreRules[k][l]) do
				A[i][j] = A[i][j]:replace(rule[1], rule[2])
			end
		end
	end
end
local function fixA(A, i, j, k, reason)
	if not reason then -- just do everything
		print('fixing everything in A...')
		for i=1,#A do
			for j=1,#A[1] do
				fixCell(A,i,j)
			end
		end
	else
		if reason == 'scale' then 	-- scale row i
			for x=1,#A do
				fixCell(A,i,x)
			end
		elseif reason == 'row' then	-- modified row j
			for x=1,#A do
				fixCell(A,j,x)
			end
		end
	end
end

-- simplify the flux jacobian matrix
-- [[
if useZ4 then-- simplify inverses
	fixA(A)
end
--]]
if useShift and useV then	-- idk how this happens
	for i=1,3 do
		for j=1,3 do
			A = A:replace(gammaLL[i][j], gammaLVars[i][j])
		end
	end
end




local dts = Matrix(allLhs):transpose()
local dxs = Matrix(allDxs):transpose()
if not useShift then
	printbr((dts + A * dxs):eq(b))
else
	printbr((dts + (A - betaVars[fluxdir] * Matrix.identity(n)) * dxs):eq(b))
end

local lambda = var'\\lambda'
local charpoly  = (A - Matrix.identity(n) * lambda):determinant()
printbr'characteristic polynomial:'
printbr(charpoly)

--[[ this works fast enough without simplification
A = A()
printbr'simplified:'
printbr(A)

local sofar, reduce = A:inverse()
printbr(sofar)
printbr(reduce)
--]]

if not useV and not useGamma and not useZ4 and not use1D then
	io.stderr:write"I'm not going to eigendecompose without useV or useGamma set\n"
	
	closeFile() os.exit()
end

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
local lambdas = use1D and table{
	-alpha * sqrt(f * gammaUjj),
	0,
	alpha * sqrt(f * gammaUjj),
} or table{
	-- the more multiplicity, the easier it is to factor
	-- also, without useVConstraints, the 1-multiplicity eigenvectors take forever and the expressions get huge and take forever
	--[[
	-alpha * sqrt(f * gammaUjj),
	-alpha * sqrt(gammaUjj),
	0,
	alpha * sqrt(gammaUjj),
	alpha * sqrt(f * gammaUjj),
	--]]
	
	
	-- [[
	0,
	-alpha * sqrt(gammaUjj),
	alpha * sqrt(gammaUjj),
	-alpha * sqrt(f * gammaUjj),
	alpha * sqrt(f * gammaUjj),
	--]]
}

local evs = table()
local multiplicity = table()
for _,lambda in ipairs(lambdas) do
io.stderr:write('finding eigenvector of eigenvalue '..tostring(lambda)..'\n') io.stderr:flush()
	printbr('eigenvalue:', lambda)
	
	local A_minus_lambda_I = ((A - Matrix.identity(n) * lambda))()
	--printbr'reducing'
	--printbr(A_minus_lambda_I)

	local sofar, reduce = A_minus_lambda_I:inverse(nil, function(AInv, A, i, j, k, reason)
		-- [[
		fixA(A, i, j, k, reason)
		fixA(AInv, i, j, k, reason)
		--]]
		--[[
		printbr('eigenvalue', lambda)	
		printbr('step', i, j, reason)
		printbr(A)
		printbr(AInv)
		--]]
		-- [[
		local f = assert(io.open(outputPrefix..'.progress'..outputExt,'w'))
		f:write(tostring(MathJax.header))
		local function printbr(...)
			for i=1,select('#', ...) do
				if i>1 then f:write'\t' end
				f:write(tostring(select(i, ...)))
			end
			f:write'<br>\n'
			f:flush()
		end
		printbr('eigenvalue', lambda)	
		printbr('step', i, j, k, reason)
		printbr(A)
		printbr(AInv)
		f:close()
-- [=[
io.write('generated '..i..', '..j..', '..k..' '..reason..' -- press enter ')
io.flush()
io.read'*l'
--]=]
		--]]
	end)
	
	--printbr('done inverting:')
	-- show the gaussian elimination results
	--printbr(sofar)
	--printbr(reduce)

	-- now find the eigenvector ...
	
	-- find all non-leading columns
	local nonLeadingCols = table()
	local j = 1
	for i=1,n do
		while reduce[i][j] == Constant(0) and j <= n do
			nonLeadingCols:insert(j)
			j=j+1
		end
		if j > n then break end
		assert(reduce[i][j] == Constant(1), "found a column that doesn't lead with 1")
		j = j + 1
	end
	nonLeadingCols:append(range(j,n))

	multiplicity:insert(#nonLeadingCols)

	-- now build the eigenvector basis for this eigenvalue
	local ev = Matrix:zeros{n, #nonLeadingCols}
	
	-- cycle through the rows
	for i=1,n do
		local k = nonLeadingCols:find(i) 
		if k then
			ev[i][k] = Constant(1)
		else
			-- j is the free param # (eigenvector col)
			-- k is the non leading column
			-- everything else in reduce[i][*] should be zero, except the leading 1
			for k,j in ipairs(nonLeadingCols) do
				ev[i][k] = ev[i][k] - reduce[i][j]
			end
		end
	end
	ev = ev()
	
	--[[ try to remove any inverses ...
	for i=1,3 do
		for j=1,3 do
			ev = ev:replace(det_gamma_times_gammaUInv[i][j](), gamma * gammaLVars[i][j])
			-- this doesn't simplify like I want it to ...
			ev = ev:replace((-det_gamma_times_gammaUInv[i][j])(), -gamma * gammaLVars[i][j])
		end
	end
	--]]

	printbr('eigenvector:')
	printbr(ev)
	evs:insert(ev)

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
