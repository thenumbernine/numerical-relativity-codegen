#!/usr/bin/env luajit

-- idk where to put this, or what it should do
-- I just wanted a script to create a flux jacobian matrix from tensor index equations
require 'ext'
require 'symmath'.setup()

local useVConstraint = true

local textOutput = false

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

-- basis
local x = var'x'
local y = var'y'
local z = var'z'
local t = var't'
-- kronecher delta
local delta = var'\\delta'
-- adm metric
local alpha = var'\\alpha'
local beta = var'\\beta'
local gamma = var'\\gamma'
-- extrinsic curvature
local K = var'K'
-- first-order
local a = var'a'
local d = var'd'
local V = var'V'
-- lapse function
local f = var'f'

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
	- alpha^2 * f * gamma'^ij' * K'_ij' 
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

--[=[
printbr[[auxiliary variables]]

local a_def = a'_,k':eq(log(alpha)'_,k')
local simplified_a_def = a_def()
printbr(a_def:eq(simplified_a_def:rhs()))
a_def = simplified_a_def

local dt_a_def = a_def:lhs()',t':eq(a_def:rhs()',t')
printbr(dt_a_def)

-- TODO here -- distribute ,t -- and don't apply Derivative.visitorHandler.Prune ...
local ruleIndex = Derivative.rules.Prune:find(nil, function(kv) return next(kv) == 'otherVars' end)
local push = Derivative.rules.Prune:remove(ruleIndex)

dt_a_def = dt_a_def():replace(function(expr)
	if expr == a:diff(t) then return a'_,t' end
end)
printbr(dt_a_def)

Derivative.rules.Prune:insert(ruleIndex, push)
--]=]

local defs = table()

defs:insert( dt_alpha_def )

-- TODO hyp gamma driver beta in terms of B

defs:insert( dt_gamma_def )
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

if not useVConstraint then
	-- TODO use the V def, but assign V'_k' = gamma'^ij' * (d'_kij' - d'_ijk')
	-- but - for index expressions, you need to rename the indexes so they don't collide 
	-- and for dense tensors, you need to use the gammaUVars and dVars tensors
	--local V = (d'_kij' - d'_ijk') * gamma'^ij'
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
else
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
			+ gamma'^lm' * (d'_jli' + d'_ilj' - d'_lij') * (a'_m' + V'_m' - gamma'^kn' * d'_knm')
			
			+ gamma'^lm' * gamma'^kn' * (
				  (d'_lkj' - d'_klj') * d'_mni'
				+ (d'_lki' - d'_kli') * d'_mnj'
				
				+ 2 * d'_ilk' * d'_mnj'
				+ 2 * d'_jlk' * d'_mni'

				- 3 * d'_ilk' * d'_jmn'
			)
			
			+ gamma'^lm' * K'_lm' * K'_ij'
			- 2 * gamma'^lm' * K'_il' * K'_jm'
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
end

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
	
	do -- if rhs ~= Constant(0) then
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
local VVars = Tensor('_k', function(k)
	return var('V_'..xs[k].name, depvars)
end)

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
		if useVConstraint then 
			def = def:replace(V, VVars)
		end
		def = def()
		
		local lhs, rhs = table.unpack(def)
		if not lhs.dim then
			printbr'failed to find lhs.dim'
			printbr(tostring(lhs))
			error'here'
		end
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

if not useVConstraint then
	io.stderr:write"I'm not going to eigendecompose without useVConstraint set\n"
	os.exit()
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
--]]

-- try solving it for one particular eigenvector/value
local gammaUxx = gammaUVars[1][1]
-- the eigenvalues are the same for useVConstraint on or off
-- but the multiplicities are different: 
-- with useVConstraint off we get 19, 3, 3, 1, 1
-- with useVConstraint on we get 18, 5, 5, 1, 1
local lambdas = table{
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
