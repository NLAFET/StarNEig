# Expert functions

The library provides a set of configuration structures:

 - @ref starneig_hessenberg_conf : A configuration structure for Hessenberg
   reduction related expert interface functions.
 - @ref starneig_schur_conf : A configuration structure for Schur reduction
   related expert interface functions.
 - @ref starneig_reorder_conf : A configuration structure for eigenvalue
   reordering related interface functions.
 - @ref starneig_eigenvectors_conf : A configuration structure for eigenvector
   computation related interface functions.

The default parameters can generated with the following interface functions:

 - starneig_hessenberg_init_conf() : Generates default parameters for Hessenberg
   reduction related expert interface functions.
 - starneig_schur_init_conf() :  Generates default parameters for Schur
   reduction related expert interface functions.
 - starneig_reorder_init_conf() :  Generates default parameters for eigenvalue
   reordering related interface functions.
 - starneig_eigenvectors_init_conf() : Generates default parameters for
   eigenvector computation related interface functions.

A user is allowed to modify these default values before passing them to the
expert interface function.

Only certain interface functions have expert version:

 - starneig_SEP_SM_Hessenberg_expert()
 - starneig_SEP_SM_Schur_expert()
 - starneig_SEP_SM_ReorderSchur_expert()
 - starneig_SEP_SM_Eigenvectors_expert()

 - starneig_SEP_DM_Schur_expert()
 - starneig_SEP_DM_ReorderSchur_expert()

 - starneig_GEP_SM_Schur_expert()
 - starneig_GEP_SM_ReorderSchur_expert()
 - starneig_GEP_SM_Eigenvectors_expert()

 - starneig_GEP_DM_Schur_expert()
 - starneig_GEP_DM_ReorderSchur_expert()

See module @ref starneig_ex_conf for further information.
