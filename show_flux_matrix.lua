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
local useV = false				-- not needed with use1D
local useGamma = false			-- exclusive to useV ... 
local useZ4 = false


local t,x,y,z = vars('t','x','y','z')
local xs = use1D and table{x} or table{x,y,z}


-- looking at all fluxes
--local depvars = table{t,x,y,z}
-- looking at the x dir only
local depvars = table{t,x}



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
}

local function simplify(expr)
	expr = expr():factorDivision()
	if op.add.is(expr) then
		for i=1,#expr do expr[i] = expr[i]() end
	end
	return expr
end


local new_printbr_file, printbr
do
	local fileindex = 0
	local printbr_file
	local ext = textOutput and '.txt' or '.html'
	new_printbr_file = function()
		fileindex = fileindex + 1
		if printbr_file then
			if MathJax then printbr_file:write(MathJax.footer) end
			printbr_file:close()
		end
		printbr_file = assert(io.open('show_flux_matrix.'..('%05d'):format(fileindex)..ext, 'w'))
		if MathJax then printbr_file:write(tostring(MathJax.header)) end
	end
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
end

new_printbr_file()


-- TODO start with EFE, apply Gauss-Codazzi-Ricci, then automatically recast all higher order derivatives as new variables of 1st derivatives
printbr[[primitive $\partial_t$ defs]]

local dt_alpha_def = alpha',t':eq( 
	- alpha^2 * f * K'^i_i' 
	+ alpha'_,i' * beta'^i' 
)
printbr(dt_alpha_def)

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
	+ 4 * pi * alpha * (gamma'_ij' * (S - rho) - 2 * S'_ij')
)
printbr(dt_K_def)

printbr[[auxiliary variables]]

local a_def = a'_k':eq(log(alpha)'_,k')
printbr(a_def)

local a_def = a_def()
printbr(a_def)

local dalpha_for_a = (a_def * alpha)():switch()
printbr(dalpha_for_a)
printbr(dalpha_for_a:reindex{i='k'})

local d_def = d'_kij':eq(frac(1,2) * gamma'_ij,k')
printbr(d_def)

dgamma_for_d = (d_def * 2)():switch()
printbr(dgamma_for_d)

printbr[[${\gamma^{ij}}_{,k}$ wrt aux vars]]

local dgammaU_def = gamma'^ij_,k':eq(-gamma'^il' * gamma'_lm,k' * gamma'^mj')
printbr(dgammaU_def)

local dgammaU_for_d = dgammaU_def:subst(dgamma_for_d:reindex{lmk='ijk'})()
printbr(dgammaU_for_d)

printbr[[connections wrt aux vars]]
local connL_def = Gamma'_ijk':eq(frac(1,2) * (gamma'_ij,k' + gamma'_ik,j' - gamma'_jk,i'))
printbr(connL_def)

local connL_for_d = connL_def
	:substIndex(dgamma_for_d)
	:simplify()
printbr(connL_for_d)

-- [[ just raise Gamma, keep d and gamma separate0
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

--[[
local orig_R_for_d = R_def
	:subst(conn_for_d:reindex{kij='ijk'})
	:subst(conn_for_d:reindex{kik='ijk'})
	:subst(conn_for_d:reindex{klkm='ijkl'})
	:subst(conn_for_d:reindex{lijn='ijkl'})
	:subst(conn_for_d:reindex{kljm='ijkl'})
	:subst(conn_for_d:reindex{likn='ijkl'})
printbr(orig_R_for_d)
--]]
local R_for_d = R_def
	:substIndex(conn_for_d)
	:reindex{llmnmn = 'abcdef'}	-- still working on the reindex automatic replace ...
printbr(R_for_d)

R_for_d = R_for_d()
printbr(R_for_d)

-- [[
local orig_R_for_d = R_for_d
	:subst(dgammaU_for_d:reindex{kljn='ijkl'})
	:subst(dgammaU_for_d:reindex{klkn='ijkl'})
	:simplify()
printbr(orig_R_for_d)
--]]
--[[
R_for_d = R_for_d:substIndex(dgammaU_for_d):simplify()
printbr(R_for_d)
os.exit()
--]]

printbr'symmetrizing'
R_for_d = R_for_d
	:splitOffDerivIndexes()
	:symmetrizeIndexes(gamma, {1,2})
	:simplify()
	:symmetrizeIndexes(d, {2,3})
	:simplify()
	:symmetrizeIndexes(d, {1,4})
	:simplify()
printbr(R_for_d)

R_for_d = R_for_d
	:replace((gamma'^km' * d'_jlm' * gamma'^ln' * d'_ikn')(), d'_ikl' * gamma'^km' * d'_jmn' * gamma'^ln')()
	:replace((gamma'^km' * d'_ljm' * gamma'^ln' * d'_ikn')(), d'_ikn' * gamma'^km' * d'_mjl' * gamma'^ln')()
	:replace((gamma'^km' * d'_ljm' * gamma'^ln' * d'_nik')(), d'_kin' * gamma'^km' * d'_mjl' * gamma'^ln')()
	:replace((gamma'^km' * d'_ljm' * gamma'^ln' * d'_kin')(), d'_nik' * gamma'^km' * d'_mjl' * gamma'^ln')()
	:simplify()
printbr(R_for_d)

printbr[[time derivative of $a_k,t$]]

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

dt_a_def = dt_a_def
	-- TODO functions, dependent variables, and total derivatives 
	:replace(f',k', df * alpha',k')

	-- TODO replace indexes automatically 
	:subst(dalpha_for_a:reindex{i='k'})
	:subst(dalpha_for_a)
	
printbr(dt_a_def)

dt_a_def = dt_a_def()
	:subst(dalpha_for_a:reindex{i='k'})
	:subst(dalpha_for_a)
	:replace(a'_i,k', a'_k,i')
	:simplify()
printbr(dt_a_def)

dt_a_def = dt_a_def:replace(gamma'^ij_,k', -gamma'^il' * gamma'_lm,k' * gamma'^mj')()
printbr(dt_a_def)

dt_a_def = dt_a_def
	:subst(dgamma_for_d:reindex{lmk='ijk'})
	:simplify()
printbr(dt_a_def)	

dt_a_def = dt_a_def
	:replace(
		-- TODO replace sub-portions of commutative operators like mul() add() etc
		-- TODO don't require that simplify() on the find() portion of replace() -- instead simplify automatically?  i experimented with this once ...
		-- TODO simplify gammas automatically ... define a tensor expression metric variable?
		(2 * alpha * f * gamma'^il' * d'_klm' * gamma'^mj' * K'_ij')(), 
		2 * alpha * f * d'_k^ij' * K'_ij'
	)
printbr(dt_a_def)

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
	:simplify()
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
dt_K_def = dt_K_def
	:subst(dsym_def:reindex{ijlk='ijkl'})
	:subst(dsym_def:reindex{iklj='ijkl'})
	:subst(dsym_def:reindex{jilk='ijkl'})
	:subst(dsym_def:reindex{kijl='ijkl'})
printbr(dt_K_def)


local defs = table()


if useZ4 then

	-- I'm omitting betas and source terms already
	
	defs:insert(a'_k,t':eq(
		-f * alpha * (gamma'^ij' * K'_ij,k' - m * Theta'_,k')
		+ a'_k' * alpha * (df * alpha + f) * (gamma'^ij' * K'_ij' - m * Theta)
	))

	defs:insert(d'_kij,t':eq(
		-alpha * (K'_ij,k' + a'_k' * K'_ij')
	))

	defs:insert(K'_ij,t':eq(
		- frac(1,2) * alpha * (a'_i,j' + a'_j,i')
		+ alpha * gamma'^kl' * (
			frac(1,2) * (d'_ilj,k' + d'_klj,i')
			- frac(1,2) * (d'_ikl,j' + d'_jkl,i')
			- frac(1,2) * (d'_kij,l' + d'_lij,k')
			+ frac(1,2) * (d'_jik,l' + d'_lik,j')
		)
		+ alpha * (Z'_j,i' + Z'_i,j')
		- alpha * a'_i' * a'_j'
	))
	
	defs:insert(Theta'_,t':eq(
		frac(1,2) * alpha * gamma'^kl' * gamma'^ij' * (d'_ilj,k' - d'_ikl,j' - d'_kij,l' + d'_jik,l')
		+ alpha * gamma'^kl' * Z'_l,k'
	))

	defs:insert(Z'_k,t':eq(
		alpha * gamma'^ij' * (K'_ik,j' - K'_ij,k')
		+ alpha * Theta'_,k'
	))

else

	defs:insert( dt_alpha_def )

	-- TODO hyp gamma driver beta in terms of B

	defs:insert(dt_gamma_def)
	defs:insert(dt_a_def)
	defs:insert(dt_d_def)
	
	if useV then
		defs:insert( K'_ij,t':eq(
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
		) )
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

for i=1,#defs do
	local lhs, rhs = table.unpack(defs[i])
	
	rhs = rhs:map(function(expr)
		if TensorRef.is(expr) and expr[1] == beta then return 0 end
	end)()
	rhs = simplify(rhs)
	
	defs[i] = lhs:eq(rhs)
end
printbr('neglecting shift')
for _,def in ipairs(defs) do
	printbr(def)
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


local gammaVars = Tensor('_ij', function(i,j) 
	if i > j then i,j = j,i end 
	return var('\\gamma_{'..xs[i].name..xs[j].name..'}', depvars)
end)

local gammaUVars = Tensor('^ij', function(i,j) 
	if i > j then i,j = j,i end 
	return var('\\gamma^{'..xs[i].name..xs[j].name..'}', depvars)
end)

local aVars = Tensor('_k', function(k)
	return var('a_'..xs[k].name, depvars)
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

-- by this ponit we're going to switch to expanded variables
-- so defining a metric is safe
Tensor.metric(gammaVars, gammaUVars)

printbr('spelled out')
local allLhs = table()
local allRhs = table()
local defsForLhs = table()	-- check to make sure symmetric terms have equal rhs's.  key by the lhs
for _,def in ipairs(defs) do
	local var = def:lhs()[1]
	if var == alpha or var == gamma then
	else
		def = def:map(function(expr)
				if TensorRef.is(expr)
				and expr[1] == gamma
				then
					for i=2,#expr do
						assert(not expr[i].lower, "found a gamma_ij term")
					end
					return TensorRef(gammaUVars, table.unpack(expr, 2))
				end
			end)
			:replace(a, aVars)
			:replace(d, dVars)
			:replace(K, KVars)
		if useV then 
			def = def:replace(V, VVars)
		end
		if useGamma then
			def = def:replace(Gamma, GammaVars)
		end
		if useZ4 then
			def = def:replace(Theta'_,k', Tensor('_k', function(k) return Theta:diff(xs[k]) end))
			def = def:replace(Z, ZVars)
		end
		def = def()

		local lhs, rhs = table.unpack(def)
		if not lhs.dim then
			-- then it's already a constant
			-- TODO maybe don't uatomatically convert x,t into x:diff(t) ... maybe make a separate function for that
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
			if defsForLhs[lhsstr] then
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
	assert(diff.is(lhs))
	assert(lhs[2] == t)
	assert(#lhs == 2)
	return diff(lhs[1], x)
end)
local A, b = factorLinearSystem(allRhs, allDxs)
A = (-A)()	-- change from U,t = A U,x + b into U,t + A U,x = b

local dts = Matrix(allLhs):transpose()
local dxs = Matrix(allDxs):transpose()
printbr((dts + A * dxs):eq(b))


local lambda = var'\\lambda'
local n = #A
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

if not useV and not useGamma and not use1D then
	io.stderr:write"I'm not going to eigendecompose without useV or useGamma set\n"
	os.exit()
end
os.exit()

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
--]]

-- try solving it for one particular eigenvector/value
local gammaUxx = gammaUVars[1][1]
-- the eigenvalues are the same for useV on or off
-- but the multiplicities are different: 
-- with useV off we get 19, 3, 3, 1, 1
-- with useV on we get 18, 5, 5, 1, 1
local lambdas = use1D and table{
	-alpha * sqrt(f * gammaUxx),
	0,
	alpha * sqrt(f * gammaUxx),
} or table{
	-- the more multiplicity, the easier it is to factor
	-- also, without useVConstraints, the 1-multiplicity eigenvectors take forever and the expressions get huge and take forever
	-alpha * sqrt(f * gammaUxx),
	-alpha * sqrt(gammaUxx),
	0,
	alpha * sqrt(gammaUxx),
	alpha * sqrt(f * gammaUxx),
}

local evs = table()
local multiplicity = table()
for _,lambda in ipairs(lambdas) do
io.stderr:write('finding eigenvector of eigenvalue '..tostring(lambda)..'\n') io.stderr:flush()
	printbr('eigenvalue:', lambda)
	
	local A_minus_lambda_I = ((A - Matrix.identity(n) * lambda))()
	--printbr'reducing'
	--printbr(A_minus_lambda_I)

	local sofar, reduce = A_minus_lambda_I:inverse(nil, function(AInv, A, i, j, reason)
		--[[
--new_printbr_file()
		printbr('eigenvalue', lambda)	
		printbr('step', i, j, reason)
		printbr(A)
		printbr(AInv)
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
