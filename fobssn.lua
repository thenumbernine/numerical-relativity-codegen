-- FOBSSN
--[=[
	original desc:
... taking Bona-Masso lapse: partial_t alpha = -alpha^2 Q = -alpha^2 f K
... and no shift (hence why all the beta terms aren't there)

1: alpha,t = -alpha^2 f K
5: gammaTilde_ij,t = -2 alpha ATilde_ij
1: phi,t = -1/6 alpha K
3: a_i,t = -alpha (f' K alpha,i + f K,i)
3: Phi_i,t = -1/6 alpha K,i
15: dTilde_ijk,t = -alpha ATilde_jk,i
1: K,t = -alpha exp(-4 phi)  gammaTilde^mn a_n,m
5: ATilde_ij,t = -alpha exp(-4 phi) LambdaTilde^k_ij,k
3: Gamma^i_,t = alpha ((xi-2) ATilde^ik - 2/3 xi gammaTilde^ik K),k

total number of variables:
... 37 -- excluding those removed for traceless constraints
... 42 -- including those traceless terms

LambdaTilde^k_ij = (dTilde^k_ij + 1/2 delta^k_i (a_j - GammaTilde_j + 2 Phi_j) + 1/2 delta^k_j (a_i - GammaTilde_i + 2 Phi_i))

timelike variables:

alpha
gammaTilde_ij

flux variables (should be 30):
3: a_x, a_y, a_z
3: Phi_x, Phi_y, Phi_z
5: dTilde_xxx, dTilde_xxy, dTilde_xxz, dTilde_xyy, dTilde_xyz, dTilde_xzz: dTilde_kzz = -dTilde_kij gammaTilde^ij / gammaTilde^zz for ij all pairs except zz
5: dTilde_yxx, dTilde_yxy, dTilde_yxz, dTilde_yyy, dTilde_yyz, dTilde_yzz: same
5: dTilde_zxx, dTilde_zxy, dTilde_zxz, dTilde_zyy, dTilde_zyz, dTilde_zzz: same
1: K,
5: ATilde_xx, ATilde_xy, ATilde_xz, ATilde_yy, ATilde_yz ... ATilde_zz = -ATilde_ij gammaTilde^ij / gammaTilde^zz for ij all pairs except zz
3: GammaTilde^x, GammaTilde^y, GammaTilde^z


source terms:


eigenfields:

	timelike: (12)

1: alpha
6: gammaTilde_ij
2: dTilde_x'xx
3: dTilde_xij

	flux fields: (30)

lambda = -beta^x:
2: w_x' = a_x'
2: w_x' = Phi_x'
2*5: dTilde_x'ij, ij = xx, xy, xz, yy, yz, zz ... skip one as stated above, due to the dTilde_ijk being traceless on jk 
1: a_x - 6 f Phi_x
3: GammaTilde^i + (xi - 2) dTilde_m^mi - 4 xi gammaTilde^ik Phi_k

gauge:
lambda_+- = -beta^x +- alpha exp(-2 phi) sqrt(f gammaTilde^xx)
2: w_+- = exp(-2 phi) sqrt(f gammaTilde^xx) K -+ a^x

longitudinal:
lambda_+- = -beta^x +- alpha exp(-2 phi) sqrt(gammaTilde^xx xi/2)
2*2: w_q+- = exp(-2 phi) sqrt(gammaTilde^xx xi/2) ATilde^x_q -+ LambdaTilde^xx_q

light cones (transverse traceless):
lambda_+- = -beta^x +- alpha exp(-2 phi) sqrt(gammaTilde^xx)
8: w_+-pq = exp(2 phi) sqrt(gammaTilde^xx) (ATilde_pq + gammaTilde^pq ATilde^xx / (2 gammaTilde^xx)) -+ (LambdaTilde^x_pq + gammaTilde_pq LambdaTilde^xxx / (2 gammaTilde^xx))
...there's only supposed to be 4 of these, based on ... 5.6.16

trace:
lambda = -beta^x + alpha exp(-2 phi) sqrt(gammaTilde^xx (2 xi - 1) / 3)
2: w_+- = exp(2 phi) sqrt(gammaTilde^xx (2 xi - 1)/3) (ATilde^xx - 2/3 gammaTilde^xx K) -+ (LambdaTilde^xxx - 2/3 gammaTilde^xx gammaTilde^xm a_m)

BSSNOK <=> xi = 2

--]=]

local useBSSN = true

local alpha = var'\\alpha'
local phi = var'\\phi'
local K = var'K'
local gammaTildeUSym = symNames:map(function(xij) return var('\\tilde\\gamma^{'..xij..'}') end)
local gammaTildeLSym = symNames:map(function(xij) return var('\\tilde\\gamma_{'..xij..'}') end)
local ATildeSym = symNames:map(function(xjk) return var('\\tilde{A}_{'..xjk..'}') end)

local gammaTildeU = Tensor('^ij', function(i,j) return gammaTildeUSym[from3x3to6(i,j)] end)
local gammaTildeL = Tensor('_ij', function(i,j) return gammaTildeLSym[from3x3to6(i,j)] end)
local ATilde = Tensor('_ij', function(i,j) return ATildeSym[from3x3to6(i,j)] end)
local GammaTildeUs = xNames:map(function(xi) return var('\\Gamma^'..xi) end)
local GammaTildeU = Tensor('^i', function(i) return GammaTildeUs[i] end)
local as = xNames:map(function(xi) return var('a_'..xi) end)		-- a_i = (ln alpha),i
local a = Tensor('_i', function(i) return as[i] end)
local Phis = xNames:map(function(xi) return var('\\Phi_'..xi) end)	-- Phi,i = (ln phi),i
local dTildeSym = xNames:map(function(xi)
	return symNames:map(function(xjk) return var('\\tilde{D}_{'..xi..xjk..'}') end)
end)
local dTildeFlattened = table():append(dTildeSym:unpack())

Tensor.metric(gammaTildeL, gammaTildeU)

local timeVars = table{{alpha}, gammaTildeLSym, {phi}}
local fieldVars = table{ATildeSym, {K}, GammaTildeUs, as, Phis, dTildeFlattened}

local function getEigenfields(dir)
	-- x's other than the current dir
	local oxIndexes = range(3)
	oxIndexes:remove(dir)

	-- symmetric, with 2nd index matching dir removed 
	local osymIndexes = range(6):filter(function(ij)
		local i,j = from6to3x3(ij)
		return i ~= dir or j ~= dir
	end)

	local eigenfields = table()

		-- non-flux vars:
	eigenfields:insert{lambda=0, w=alpha}
	for ij,xij in ipairs(symNames) do
		eigenfields:insert{lambda=0, w=gammaTildeLSym[ij]}
	end
	for i,xi in ipairs(xNames) do
		local ii = from3x3to6(i,i)
		eigenfields:insert{lambda=0, w=dTildeSym[i][ii]}
	end
	-- three more

		-- timelike:
	for _,i in ipairs(oxIndexes) do
		eigenfields:insert{lambda=0, w=as[i]}
	end
	for _,i in ipairs(oxIndexes) do
		eigenfields:insert{lambda=0, w=Phis[i]}
	end
	for _,p in ipairs(oxIndexes) do
		for _,ij in ipairs(osymIndexes) do	-- skip ij == dir,dir ... use those for constraint variables?
			eigenfields:insert{lambda=0, w=dTildeSym[p][ij]}
		end
	end
	eigenfields:insert{lambda=0, w=as[dir] - 6 * f * Phi[dir]}
	local xiVar = 2 	-- var'\\xi'
	for i=1,3 do 
		eigenfields:insert{lambda=0, w=(GammaTilde'^i' + (xiVar - 2) * gammaTilde'^mn' * dTilde'_mnj' * gammaTilde'^ji' - 4 * xiVar * gammaTilde'^ik' * Phi'_k')()[i]}
	end

	local gammaL = symmath.exp(-4 * phi) * gammaL'_ij'
	local gammaU = symmath.exp(4 * phi) * gammaU'^ij'
	local ATildeUL = (gammaTilde'^ik' * ATilde'_kj')()
	local ATildeUU = (ATildeUL'^i_k' * gammaTilde'^kj')()
	--TODO local LambdaTildeULL = gammaTildeU'^kl' * dTilde'_lij' + 
	for sign=-1,1,2 do
			-- gauge
		local loc = sign == -1 and 1 or #eigenfields+1
		eigenfields:insert(loc, {
			lambda = sign * alpha * symmath.exp(-2 * phi) * symmath.sqrt(f * gammaTildeU[dir][dir]),
			w = (symmath.exp(-2 * phi) * symmath.sqrt(f * gammaTildeU[dir][dir]) * K - sign * gammaU'^xm' a'_m')(),
		})
		loc=loc+1
			-- longitudinal
		for _,q in ipairs(oxIndexes) do
			eigenfields:insert(loc, {
				lambda = sign * alhpa * symmath.exp(-2 * phi) * symmath.sqrt(gammaTildeU[dir][dir] * xiVar/2),
				w = symmath.exp(-2 * phi) * symmath.sqrt(gammaTildeU[dir][dir] * xiVar/2) * ATildeUL[dir][q] - sign * LambdaTilde3L[dir],
			})
			loc=loc+1
		end
			-- light cones
		for _,p in ipairs(oxIndexes) do
			for _,q in ipairs(oxIndexes) do
				eigenfields:insert(loc, {
					lambda = sign * alpha * symmath.exp(-2 * phi) * symmath.sqrt(gammaTildeU[dir][dir]),
					w = symmath.exp(2 * phi) * symmath.sqrt(gammaTildeU[dir][dir]) * (ATilde[p][q] + gammaTildeU[p][q] * ATildeUU[dir][dir] / (2 * gammaTildeU[dir][dir])) - sign * (LambdaTildeULL[dir][p][q] + gammaTildeU[p][q] * (LambdaTildeULL'^k_ij' * gammaTildeU'^ij')()[dir] / (2 * gammaTildeU[dir][dir])),
				})
				loc=loc+1
			end
		end

		eigenfields:insert
lambda = -beta^x + alpha exp(-2 phi) sqrt(gammaTilde^xx (2 xi - 1) / 3)
2: w_+- = exp(2 phi) sqrt(gammaTilde^xx (2 xi - 1)/3) (ATilde^xx - 2/3 gammaTilde^xx K) -+ (LambdaTilde^xxx - 2/3 gammaTilde^xx gammaTilde^xm a_m)
	end

	return eigenfields
end
