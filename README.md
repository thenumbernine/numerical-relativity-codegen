### New Code:
(`show_flux_matrix.lua`)


This is making use of symmath tensor index expressions.
1) It starts with the tensor index PDE system.  
2) From there it derives the tensor index PDE system for the specific numerical method formalism.
3) Next it represents the PDE as a linear system, showing the source terms and the flux matrix.
4) Next it factors the right eigenvectors of the flux matrix. Still working on eigenvalues. It is lacking a polynomial solver. But, with correctly provided eigenvalues, it does linearize the flux, represent it as a matrix, and calculate the right eigenvector matrices.
5) Last it inverts the right eigenvector matrix to find the left eigenvectors.

TODO with it
- (I just adding caching of the flux jacobian so I don't have to re-derive it from index expressions every single time)
- further split things up: 1 script to generate the expressions and save the flux matrix, another to process the flux matrix, another to generate eigenvectors.
- rearrange filenames so it is unique prefix, section (expressions, flux matrix, char poly, eigenvectors), ext.


[flux matrix.adm](https://thenumbernine.github.io/numrel-codegen/flux_matrix_output/flux_matrix.adm.html)

[flux matrix.adm 1D](https://thenumbernine.github.io/numrel-codegen/flux_matrix_output/flux_matrix.adm_1D.html)

[flux matrix.adm useV](https://thenumbernine.github.io/numrel-codegen/flux_matrix_output/flux_matrix.adm_useV.html)

[flux matrix.adm useV noZeroRows](https://thenumbernine.github.io/numrel-codegen/flux_matrix_output/flux_matrix.adm_useV_noZeroRows.html)

[flux matrix.adm useV useShift noZeroRows](https://thenumbernine.github.io/numrel-codegen/flux_matrix_output/flux_matrix.adm_useV_useShift_noZeroRows.html)

[flux matrix.z4](https://thenumbernine.github.io/numrel-codegen/flux_matrix_output/flux_matrix.z4.html)

[flux matrix.z4 noZeroRows](https://thenumbernine.github.io/numrel-codegen/flux_matrix_output/flux_matrix.z4_noZeroRows.html)


### Old Code:


## Code generation for numerical relativity projects.

Different formalisms can be programmed into this by providing their source
terms and eigenfields / eigenvalues.

Currently I have Bona&Masso's ADM formalism without shift.  I'm working on
adding shift, and working on FOBSSN.
Who knows what's in the future?  Plain hyperbolic ADM?  KST?

This code can be used as follows:

	./create_basis_from_waves.lua Lua		<- Generate the Lua code for the left and eigenvectors
						and source terms used in the gravitational-waves-lua
						project adm3d.lua file.

	./create_basis_from_waves.lua C			<- Generate OpenCL code for HydrodynamicsGPU project.
	./create_basis_from_waves.lua			<- Generate HTML + MathJax.
	./create_basis_from_waves.lua SingleLine
	./create_basis_from_waves.lua MultiLine <- Generate text output.  Not as readable as HTML.

This project is based on the symmath-lua project.

## Sources:

"The apeparance of coordinate shocks in hyperbolic formalisms of General
Relativity", Alcubierre, 1997.		
	This is where I get the shift-less hyperbolic Bona-Masso ADM representation.
	It also is good about specifying source terms.

"Introduction to 3+1 Numerical Relativity", Alcubierre, 2008
	This is a good source, but it has slightly different eqns.  It also gives the
	eigenfields in terms of the lapse, yet leaves out important lapse variables
	(specifically any mention of the B_k^i = 1/2 partial_k beta^i variable).  It
	also doesn't give much (any?) attention to source terms.
	It does give a good overview of hyperbolic formalisms of ADM, BM, BSSN, and
	KST, so it is a good overview.

"New Formalism for Numerical Relativity", Bona, Masso, Seidel, Stela 1995
	This is where I'm going to for verification of the ADM Bona-Masso system.
	The source terms in this paper are a bit more complex than the others,
	but are clarified in a later paper to be reducible to zero.
	This paper is also good for sharing the details of the B variable.
	There's a subtle difference in the light cone eigenvalues.

"First order hyperbolic formalism for numerical relativity", Bona, Masso,
Seidel, Stela (1997)
	This one adds some polish to the previous paper. It also has clarifications
	of the (4)R_ij and G^0_j terms (which the Alcubierre paper refers to but
	doesn't define).

"Numerical simulations with a first-order BSSN formulation of Einstein's field
equations", Brown, Deiner, Field, Hesthaven, Herrmann, Mroue, Sarbach,
Schnetter, Tiglio, Wagman (2012)
	I'm going between this and Alcubierre's book for how to implement FOBSSN.
