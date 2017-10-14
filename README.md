### New Code:
(show_flux_matrix.lua)


This is making use of symmath tensor index expressions.
It is lacking a polynomial solver.
But, with correctly provided eigenvalues, it does linearize the flux, represent it as a matrix, and calculate the right and left eigenvector matrices.

[flux matrix.adm](https://htmlpreview.github.io/?https://github.com/thenumbernine/numerical-relativity-codegen/blob/master/flux_matrix_output/flux_matrix%2eadm%2ehtml)

[flux matrix.adm 1D](https://htmlpreview.github.io/?https://github.com/thenumbernine/numerical-relativity-codegen/blob/master/flux_matrix_output/flux_matrix%2eadm_1D%2ehtml)

[flux matrix.adm useV](https://htmlpreview.github.io/?https://github.com/thenumbernine/numerical-relativity-codegen/blob/master/flux_matrix_output/flux_matrix%2eadm_useV%2ehtml)

[flux matrix.adm useV noZeroRows](https://htmlpreview.github.io/?https://github.com/thenumbernine/numerical-relativity-codegen/blob/master/flux_matrix_output/flux_matrix%2eadm_useV_noZeroRows%2ehtml)

[flux matrix.adm useV useShift noZeroRows](https://htmlpreview.github.io/?https://github.com/thenumbernine/numerical-relativity-codegen/blob/master/flux_matrix_output/flux_matrix%2eadm_useV_useShift_noZeroRows%2ehtml)

[flux matrix.z4](https://htmlpreview.github.io/?https://github.com/thenumbernine/numerical-relativity-codegen/blob/master/flux_matrix_output/flux_matrix%2ez4%2ehtml)

[flux matrix.z4 noZeroRows](https://htmlpreview.github.io/?https://github.com/thenumbernine/numerical-relativity-codegen/blob/master/flux_matrix_output/flux_matrix%2ez4_noZeroRows%2ehtml)


### Old Code:


## Code generation for numerical relativity projects.

Different formalisms can be programmed into this by providing their source
terms and eigenfields / eigenvalues.

Currently I have Bona&Masso's ADM formalism without shift.  I'm working on
adding shift, and working on FOBSSN.
Who knows what's in the future?  Plain hyperbolic ADM?  KST?

This code can be used as follows:

	./run.lua Lua		<- Generate the Lua code for the left and eigenvectors
						and source terms used in the gravitational-waves-lua
						project adm3d.lua file.

	./run.lua C			<- Generate OpenCL code for HydrodynamicsGPU project.
	./run.lua			<- Generate HTML + MathJax.
	./run.lua SingleLine
	./run.lua MultiLine <- Generate text output.  Not as readable as HTML.

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
