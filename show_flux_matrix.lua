#!/usr/bin/env luajit

-- idk where to put this, or what it should do
-- I just wanted a script to create a flux jacobian matrix from tensor index equations
require 'ext'
require 'symmath'.setup{MathJax={usePartialLHSForDerivative=true}}


local alpha = var'\\alpha'
local beta = var'\\beta'
local gamma = var'\\gamma'
local delta = var'\\delta'
local K = var'K'
local a = var'a'
local d = var'd'
local f = var'f'

local x,y,z = vars('x', 'y', 'z')
local t = var't'
local xs = table{x,y,z}
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


-- TODO start with EFE, apply Gauss-Codazzi-Ricci, then automatically recast all higher order derivatives as new variables of 1st derivatives

local defs = table()

defs:insert( alpha',t':eq( 
	- alpha^2 * f * gamma'^ij' * K'_ij' 
	+ alpha'_,i' * beta'^i' 
) )

-- TODO hyp gamma driver beta in terms of B

defs:insert( gamma'_ij,t':eq( 
	-2 * alpha * K'_ij' 
	+ gamma'_ij,k' * beta'^k' 
	+ gamma'_kj' * beta'^k_,i' 
	+ gamma'_ik' * beta'^k_,j' 
) )
defs:insert( a'_k,t':eq( 
	-alpha * f * gamma'^ij' * K'_ij,k' 
	- alpha^2 * f:diff(alpha) * a'_k' * gamma'^ij' * K'_ij' 
	+ 2 * alpha * f * d'_k^ij' * K'_ij'
	- alpha * a'_k' * f * gamma'^ij' * K'_ij'
	+ a'_k,i' * beta'^i' 
	+ a'_i' * beta'^i_,k' 
) )
defs:insert( d'_kij,t':eq( 
	-alpha * K'_ij,k' 
	- alpha * a'_k' * K'_ij' 
	+ d'_kij,l' * beta'^l' 
	+ d'_lij' * beta'^l_,k' 
	+ d'_kli' * beta'^l_,j' 
	+ d'_klj' * beta'^l_,i' 
	+ frac(1,2) * gamma'_li' * beta'^l_,jk' 
	+ frac(1,2) * gamma'_lj' * beta'^l_,ik'
) )
defs:insert( K'_ij,t':eq(
	- frac(1,2) * alpha * a'_i,j'
	- frac(1,2) * alpha * a'_j,i'
	+ alpha * gamma'^kl' * (
		-- gamma_ij,kl = gamma_ij,lk <=> d_kij,l = d_lij,k ... so symmetrize those ...
		frac(1,2) * (d'_ilj,k' + d'_klj,i')
		+ frac(1,2) * (d'_jik,l' + d'_lik,j')
		- frac(1,2) * (d'_ikl,j' + d'_jkl,i')
		- frac(1,2) * (d'_kij,l' + d'_lij,k')
	)
	+ alpha * (
		-a'_i' * a'_j' 
		+ a'_k' * gamma'^kl' * (d'_jli' + d'_ilj' - d'_lij')
		+ gamma'^lm' * gamma'^kn' * (d'_jli' + d'_ilj' - d'_lij') * (d'_mkn' - 2 * d'_knm')
		
		+ 2 * gamma'^lm' * gamma'^kn' * (d'_klj' - d'_lkj') * d'_nim'
		+ gamma'^lm' * gamma'^kn' * d'_ikl' * d'_jnm'
		
		+ gamma'^lm' * K'_lm' * K'_ij'
		- 2 * gamma'^lm' * K'_il' * K'_jm'
	)
	+ K'_ij,k' * beta'^k'
	+ K'_kj' * beta'^k_,i'
	+ K'_ik' * beta'^k_,j'
) )
printbr('partial derivatives')
for _,def in ipairs(defs) do
	printbr(def)
end

local TensorRef = require 'symmath.tensor.TensorRef'
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

-- for all summed terms, for all coefficients, 
--	if none have derivatives then remove them
-- remove from defs any equations that no longer have any terms
defs = defs:map(function(def,i,t)
	local lhs, rhs = table.unpack(defs[i])
	rhs = (rhs - rhs:map(function(expr)
		if TensorRef.is(expr) then
			for j=2,#expr do
				if expr[j].derivative then return 0 end
			end
		end
	end)())()
	rhs = simplify(rhs)
	
	if rhs ~= Constant(0) then
		return lhs:eq(rhs), #t+1
	end
end)
printbr('neglecting source terms')
for _,def in ipairs(defs) do
	printbr(def)
end


-- looking at all fluxes
--local depvars = table{t,x,y,z}
-- looking at the x dir only
local depvars = table{t,x}

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

printbr('spelled out')
local allLhs = table()
local allRhs = table()
local defsForLhs = table()	-- check to make sure symmetric terms have equal rhs's.  key by the lhs
for _,def in ipairs(defs) do
	local def = def
		:clone()
		:map(function(expr)
			if TensorRef.is(expr)
			and expr[1] == gamma
			then
				for i=2,#expr do
					assert(not expr[i].lower)
				end
				return TensorRef(gammaUVars, table.unpack(expr, 2))
			end
		end)
		:replace(a, aVars)
		:replace(d, dVars)
		:replace(K, KVars)
		:simplify()

	local lhs, rhs = table.unpack(def)
	local dim = lhs:dim()
	assert(dim[#dim].value == 1)	-- the ,t ...

	-- remove the ,t dimension
	lhs = Tensor(table.sub(lhs.variance, 1, #dim-1), function(...)
		return lhs[{...}][1]
	end)

	for i in lhs:innerIter() do
		print(tolua(i))
	end

	local eqns = lhs:eq(rhs):unravel()
	for _,eqn in ipairs(eqns) do		
		local lhs, rhs = table.unpack(eqn)
		local lhsstr = tostring(lhs)
		rhs = simplify(rhs)
		if defsForLhs[lhsstr] then
			if rhs ~= defsForLhs[lhsstr] then
				print'mismatch'
				print(lhs:eq(rhs))
				print'difference'
				print(simplify(rhs - defsForLhs[lhsstr]))
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

local allDxs = allLhs:map(function(lhs)
	assert(diff.is(lhs))
	assert(lhs[2] == t)
	assert(#lhs == 2)
	return diff(lhs[1], x)
end)
local A, b = factorLinearSystem(allRhs, allDxs)

local dts = Matrix(allLhs):transpose()
local dxs = Matrix(allDxs):transpose()
printbr(dts:eq(A * dxs + b))
