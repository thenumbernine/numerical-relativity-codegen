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
	+ alpha * gamma'^kl' * (d'_ilj,k' - d'_ikl,j' - d'_kij,l' + d'_jik,l')
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
	defs[i] = defs[i]:clone():map(function(expr)
		if TensorRef.is(expr) and expr[1] == beta then return 0 end
	end)()
end
printbr('neglecting shift')
for _,def in ipairs(defs) do
	printbr(def)
end

-- [[
local defs = table()
defs:insert( a'_k,t':eq( 
	-alpha * f * gamma'^ij' * K'_ij,k' 
) )
defs:insert( d'_kij,t':eq( 
	-alpha * K'_ij,k' 
) )
defs:insert( K'_ij,t':eq(
	- frac(1,2) * alpha * a'_i,j'
	- frac(1,2) * alpha * a'_j,i'
	+ alpha * gamma'^kl' * (d'_ilj,k' - d'_ikl,j' - d'_kij,l' + d'_jik,l')
) )
--]]
--[[ TODO for all summed terms, for all coefficients, if none have derivatives then remove them
--]]
printbr('neglecting source terms')
for _,def in ipairs(defs) do
	printbr(def)
end

local x,y,z = vars('x', 'y', 'z')
local t = var't'
local xs = table{x,y,z}
Tensor.coords{
	{variables=xs},
	{variables={t}, symbols='t'},
}

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

local KVars = Tensor('_ij', function(i,j)
	if i > j then i,j = j,i end 
	return var('K_{'..xs[i].name..xs[j].name..'}', depvars) 
end)

printbr('spelled out')
local defsForLhs = table()	-- key by the lhs
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

	lhs = Tensor(table.sub(lhs.variance, 1, #dim-1), function(...)
		return lhs[{...}][1]
	end)

	rhs = rhs:permute(lhs.variance)

	for i in lhs:iter() do
		local lhsstr = tostring(lhs[i])
		if defsForLhs[lhsstr] then
			if rhs[i] ~= defsForLhs[lhsstr] then
				--printbr('expected')
				--printbr(lhs[i]:eq(defsForLhs[lhsstr]))
				print('mismatch ')
				printbr(lhs[i]:eq(rhs[i]))
			end
		else
			defsForLhs[lhsstr] = rhs[i]:clone()

			if rhs[i] ~= Constant(0) then
				printbr(lhs[i]:eq(rhs[i]))
				-- TODO put into a matrix and factorLinearSystem
			end
		end
	end
end

--[[
local A = Matrix(
	{0, 0, alpha * f * gamma'^pq' * delta'^r_k'},
	{0, 0, alpha * delta'^p_i' * delta'^q_j' * delta'^r_k'},
	{
		frac(1,2) * alpha * (delta'^m_i' * delta'^r_j' + delta'^m_j' * delta'^r_i'),
		frac(1,2) * alpha * (
			gamma'^pq' * (delta'^m_i' * delta'^r_j' + delta'^m_j' * delta'^r_i')
			+ gamma'^mr' * (delta'^p_i' * delta'^q_j' + delta'^p_j' * delta'^q_i')
			- gamma'^rp' * (delta'^q_i' * delta'^m_j' + delta'^q_j' * delta'^m_i')
			- gamma'^rq' * (delta'^p_i' * delta'^m_j' + delta'^p_j' * delta'^m_i')
		), 
		0,
	}
)
printbr(A)

-- now to expand each matrix ...
-- r is fixed
for r=1,3 do
	local n = 31
	local Ar = Matrix(
		range(n):map(function(i)
			return range(n):map(function(j)
				
			
			end)
		end):unpack())
end
--]]
