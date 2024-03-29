#!/usr/bin/env luajit
local class = require 'ext.class'
local table = require 'ext.table'
local range = require 'ext.range'
local string = require 'ext.string'
local symmath = require 'symmath'

local Tensor = symmath.Tensor

local outputMethod = ... or 'MathJax'
--local outputMethod = 'MathJax'		-- HTML
--local outputMethod = 'SingleLine'		-- pasting into Excel
--local outputMethod = 'Lua'			-- code generation
--local outputMethod = 'C'				-- code gen as well
--local outputMethod = 'GraphViz'		-- generate graphviz dot files

local MathJax
if outputMethod == 'MathJax' then
	MathJax = symmath.export.MathJax
	symmath.tostring = MathJax 
	print(MathJax.header)
elseif outputMethod == 'SingleLine' or outputMethod == 'GraphViz' then
	symmath.tostring = symmath.export.SingleLine
end

local outputCode = outputMethod == 'Lua' or outputMethod == 'C' 

local ToStringLua
if outputCode then 
	ToStringLua = symmath.export.Lua
end

-- code generation functions

local function comment(s)
	if outputMethod == 'Lua' then return '-- '..s end
	if outputMethod == 'C' then return '// '..s end
	return s
end

local function def(name, dims)
	local s = table()
	if outputMethod == 'Lua' then
		s:insert('local '..name..' = ')
	elseif outputMethod == 'C' then
		s:insert('real '..name)
		if dims then s:insert(table.map(dims,function(i) return '['..i..']' end):concat()) end
		s:insert(' = ')
	end
	if dims and #dims > 0 then s:insert('{') end
	return s:concat()
end

local function I(...)
	return table{...}:map(function(i)
		return '[' .. (outputMethod == 'Lua' and i or (i-1)) .. ']'
	end):concat()
end


local printbr
if outputCode or outputMethod == 'GraphViz' then
	printbr = print
else
	printbr = function(...)
		print(...)
		print'<br>'
		io.stdout:flush()
	end
end


local function var(name)
	if outputCode then
		name = name:gsub('[\\{}]', ''):gsub('%^', 'U')
	end
	return symmath.var(name)
end

local from3x3to6_table = {{1,2,3},{2,4,5},{3,5,6}}
local function from3x3to6(i,j) return from3x3to6_table[i][j] end

local from6to3x3_table = {{1,1},{1,2},{1,3},{2,2},{2,3},{3,3}}
local function from6to3x3(i)
	return table.unpack(from6to3x3_table[i])
end


local NRCodeGen = class()

function NRCodeGen:init()
	
	local f = var'f'
	
	-- coordinates
	local xNames = table{'x', 'y', 'z'}
	local spatialCoords = xNames:map(function(x) return var(x) end)
	Tensor.coords{{variables=spatialCoords}}

	-- symmetric indexes: xx xy xz yy yz zz
	local symNames = table()
	for i,xi in ipairs(xNames) do
		for j=i,3 do
			local xj = xNames[j]
			symNames:insert(xi..xj)
		end
	end

	self.f = f
	self.xNames = xNames
	self.symNames = symNames

	self.var = var
	self.from3x3to6 = from3x3to6
	self.from6to3x3 = from6to3x3
	self.comment = comment
	self.def = def
	self.I = I
	self.outputCode = outputCode
	self.outputMethod = outputMethod

	--self.system = require 'adm-bona-masso'(self, false)	-- ADM, no shift
	--self.system = require 'adm-bona-masso'(self, true)	-- ADM with shift
	--self.system = require 'fobssn'(self)
	self.system = require 'z4'(self)
end

local nrCodeGen = NRCodeGen()
local xNames = nrCodeGen.xNames

local system = nrCodeGen.system
local timeVarsFlattened = system.timeVarsFlattened
local fieldVarsFlattened = system.fieldVarsFlattened
local varsFlattened = system.varsFlattened

-- all variables combined into one vector
local U = symmath.Matrix(varsFlattened:map(function(Ui) return {Ui} end):unpack())

local sourceTerms = system:getSourceTerms()

local QLs = table()
local QRs = table()

--local maxdirs = 3
local maxdirs = 1	-- only x

for dir=1,maxdirs do
	local eigenfields = system:getEigenfields(dir)

	if #eigenfields ~= #varsFlattened then
		error("expected "..#varsFlattened.." eigenfields but found "..#eigenfields.."\ncontents:\n"..eigenfields:map(tostring):concat('\n'))
	end

	if not outputCode and outputMethod ~= 'GraphViz' then
		printbr('<h3>' .. ({'x','y','z'})[dir] .. ' direction</h3>')
		
		printbr()
		printbr('eigenvalues')
		printbr()
		for _,field in ipairs(eigenfields) do
			printbr(symmath.simplify(field.lambda))
		end
		printbr()
		
		printbr('eigenfields')
		printbr()
		for _,field in ipairs(eigenfields) do
			printbr(symmath.simplify(field.w))
		end
		printbr()
	end

	local linearSystems = eigenfields:map(function(field) return field.w end)

	-- now just do a matrix factor of the eigenfields on varsFlattened and viola.
	-- QL is the left eigenvector matrix
	local QL, b = symmath.factorLinearSystem(
		linearSystems,
		fieldVarsFlattened)

	-- courtesy of !useMomentumConstraints evolving the Gamma^i terms, now gamma_ij and gamma^ij terms can be present, and some simplifications can be made around them ...
	do
		-- replace (gamma^ik gamma_kj) with delta^i_j values ...
		local gammaLL = nrCodeGen.system.gammaLL
		local gammaUU = nrCodeGen.system.gammaUU
		local gammaOrtho = (gammaLL'_ik' * gammaUU'^kj')()
		for i=1,3 do
			for j=1,3 do
				QL = QL:replace( gammaOrtho[i][j], symmath.Constant(i==j and 1 or 0) )
			end
		end
		
		--[[ also represent all gamma_ij values in terms of gamma^ij, for inverse simplification's sake ...
		-- seems to add too many extra terms, and makes the inverse calculation take too long
		--[=[
		local gammaUUInv = symmath.Matrix.inverse(gammaUU)
		--]=]
		--[=[
		local gammaUUInv = symmath.Matrix({1,0,0}, {0,1,0}, {0,0,1})
		local detGamma = var'\\gamma'
		for i=1,3 do
			local i1 = i % 3 + 1
			local i2 = (i + 1) % 3 + 1
			for j=1,3 do
				local j1 = j % 3 + 1
				local j2 = (j + 1) % 3 + 1
				gammaUUInv[i][j] = symmath.Matrix.determinant(symmath.Matrix(
					{ gammaUU[j1][i1], gammaUU[j1][i2] },
					{ gammaUU[j2][i1], gammaUU[j2][i2] }
				)) * detGamma	-- divided by det(gamma^ij) means times det(gamma_ij)
			end
		end
		--]=]
		for i=1,3 do
			for j=1,3 do
				QL = QL:replace(gammaLL[i][j], gammaUUInv[i][j])
			end
		end
		--]]
	end

	if outputMethod == 'MathJax' then
		printbr('factor of eigenmodes / left eigenvector matrix')
		printbr((QL * symmath.Matrix.transpose(symmath.Matrix(fieldVarsFlattened))):eq(b))
	end

	-- now add in 0's for cols corresponding to the timelike vars (which aren't supposed to be in the linear system)
	-- [[ this asserts that the time vars go first and the field vars go second in the varsFlattened
	for i=1,#QL do
		if system.includeTimeVars then
			for j=1,#timeVarsFlattened do
				table.insert(QL[i], 1, symmath.Constant(0))
			end
		end
		assert(#QL[i] == #varsFlattened, "expected "..#varsFlattened.." cols but found "..#QL[i].."\n"..QL)
	end
	assert(#QL == #varsFlattened, "expected "..#varsFlattened.." rows but found "..#QL.."\n"..QL)
	--]]

	if system.includeTimeVars then
		-- only for the eigenfields corresponding to the time vars ...
		-- I have to pick them out of the system
		-- I *should* be not including them to begin with
		assert(#b == #eigenfields)
		for _,var in ipairs(timeVarsFlattened) do
			local j = varsFlattened:find(var) 
			for i,field in ipairs(eigenfields) do
				-- if the eigenfield is the time var then ...
				if field.w == var then
					-- ... it shouldn't have been factored out.  and there shouldn't be anything else.
					assert(b[i][1] == var, "expected "..var.." but got "..b[i].." for row "..i)
					-- so manually insert it into the eigenvector inverse 
					QL[i][j] = symmath.Constant(1)
					-- and manually remove it from the source term
					b[i][1] = symmath.Constant(0)
				end
			end
		end
	end
	
	-- make sure all source terms are gone
	for i=1,#b do
		assert(#b[i] == 1)
		assert(b[i][1] == symmath.Constant(0), "expected b["..i.."] to be 0 but found "..b[i][1])
	end

	-- get the right eigenvectors
	printbr('inverting...')
	local QR = QL:inverse()

	printbr('right eigenvector matrix:')
	printbr(QR)

	--[[ verify orthogonality
	printbr('verifying orthogonality...')
	local delta = (QL * QR)()
	for i=1,delta:dim()[1].value do
		for j=1,delta:dim()[2].value do
			local Constant = require 'symmath.Constant'
			assert(Constant:isa(delta[i][j]))
			assert(delta[i][j].value == (i == j and 1 or 0))
		end
	end
	--]]

	-- [[
	local scale = 1	-- symmath.gammaUU[1][1] ^ symmath.frac(1,2)
	printbr('reconstruction of flux Jacobian')
	printbr('(scaled by '..scale..')')
	local n = #eigenfields
	local Lambdas = symmath.Matrix(range(n):map(function(i)
		return range(n):map(function(j)
			return i==j and eigenfields[i].lambda or 0
		end)
	end):unpack())

	local A = (QR * Lambdas * QL * scale)()
	printbr( (tostring(A):gsub('0', '\\cdot')) )
	printbr()
	
	printbr('reconstruction of flux times '..scale)
	local eqns = (A * U)()
	for i=1,#eqns do
		printbr((U[i][1]:diff(symmath.var't') + eqns[i][1]:diff(symmath.var'x'))..'$ = ...$')
	end
	--]]

	printbr('...done!')

	-- save for later
	QLs:insert(QL)
	QRs:insert(QR)
end

local function processCode(code)
	code = code:gsub('v_(%d+)', function(i)
		if outputMethod == 'C' then return 'input['..(i-1)..']' end
		return 'v['..i..']'
	end)
	code = code:gsub('}, {', ',\n')
	-- replace variable names with array
	for i,var in ipairs(varsFlattened) do
		code = code:gsub(var.name, 'v['..(outputMethod == 'C' and (i-1) or i)..']')
	end
	if outputMethod == 'Lua' then
		-- separate lines
		code = code:gsub('^{{(.*)}}$', '{\n%1\n}')
		-- indent
		code = string.split(string.trim(code), '\n')
		code = code:map(function(line,i)
			if i == 1 or i == #code then
				return '\t' .. line
			else
				return '\t\t' .. line
			end
		end):concat('\n')
	elseif outputMethod == 'C' then
		code = code:match('^{{(.*)}}$')
		code = code:gsub('math%.','')
		code = code:gsub('v%[', 'input%[')
		-- add in variables
		code = code:gsub('sqrt%(f%)', 'sqrt_f')
		for _,ii in ipairs{'xx', 'yy', 'zz'} do
			code = code:gsub('sqrt%(gammaU'..ii..'%)', 'sqrt_gammaU'..ii)
			code = code:gsub('%(gammaU'..ii..' %^ %(3 / 2%)%)', 'gammaU'..ii..'_toThe_3_2')
			code = code:gsub('%(gammaU'..ii..' %^ 2%)', 'gammaU'..ii..'Sq')
		end
		-- add assignments
		code = string.split(string.trim(code), '\n'):map(function(line,i)
			line = line:gsub(',$','')..';'
			return '\t\tresults['..(i-1)..'] = '..line
		end):concat('\n')
		if code:find('sqrt%(') then error('found sqrt( at '..code) end
		if code:find('%^') then error('found ^ at '..code) end
		code = code:gsub('([^%[_])(%d+)([^%]_])', '%1%2%.f%3')
	end
	return code
end

local function processGraph(m,name)
	local f = io.open('output/adm_'..name..'.dot', 'w')
	f:write'digraph {\n'
	for i=1,#m do
		for j=1,#m[i] do
			if m[i][j] ~= symmath.Constant(0) then
				f:write('\t',('%q'):format(varsFlattened[j]), ' -> ', ('%q'):format('\\lambda_{'..i..'}'),'\n')
			end
		end
	end
	f:write'}\n'
	f:close()
end



if outputCode then 
	local compileVars = assert(system.compileVars)
	for lr,Qs in ipairs{QLs, QRs} do
		if outputMethod == 'C' then
			print('\t')
			print('\t')
		end
		-- generate the code for the linear function 
		for i,xi in ipairs(xNames) do
			print('\t'..comment((lr==1 and 'left' or 'right')..' eigenvectors in '..xi..':'))
			if outputMethod == 'C' then
				print('\t'..(i > 1 and '} else ' or '')..'if (side == '..(i-1)..') {')
				print('\t\treal sqrt_gammaUU'..xi..xi..' = sqrt(gammaUU'..xi..xi..');')
				print('\t\treal gammaUU'..xi..xi..'_toThe_3_2 = sqrt_gammaUU'..xi..xi..' * gammaUU'..xi..xi..';')
				print('\t\treal gammaUU'..xi..xi..'Sq = gammaUU'..xi..xi..' * gammaUU'..xi..xi..';')
				print('\t\t')
			end
			print(processCode(ToStringLua((Qs[i] * U)(), compileVars)))
		end
		if outputMethod == 'C' then
			print('\t}')
			print('\t')
			print('\t')
		end
	end	
else
	for dir=1,maxdirs do
		local QL = QLs[dir]
		local QR = QRs[dir]
		
		-- TODO :eq(source terms) 
		
		if outputMethod == 'GraphViz' then
			processGraph(QL,xNames[dir])
			processGraph(QR,xNames[dir]..'inv')
		else
--[[ i'm printing these as they are computed, sooo ... no need to print them here	
			printbr('inverse eigenvectors in '..xNames[dir]..' dir')
			printbr((tostring((QL * U):eq(sourceTerms)):gsub('0','\\cdot')))
			printbr()
			printbr('eigenvectors in '..xNames[dir]..' dir')
			printbr((tostring(QR * U):gsub('0','\\cdot')))
			printbr()
--]]	
		end
	end
end

if outputMethod == 'MathJax' then 
	print(MathJax.footer)
end
