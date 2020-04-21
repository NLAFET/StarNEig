## Authors

The following people have contributed to the StarNEig library:

 - Angelika Schwarz (angies@cs.umu.se)
    - Standard eigenvectors
 - Bo Kågström (bokg@cs.umu.se)
    - Coordinator and scientific director for the NLAFET project
    - Documentation
 - Carl Christian Kjelgaard Mikkelsen (spock@cs.umu.se)
    - Generalized eigenvectors
 - Lars Karlsson (larsk@cs.umu.se)
    - Miscellaneous user interface functions
    - Documentation
 - Mirko Myllykoski (mirkom@cs.umu.se)
    - Hessenberg reduction
    - Schur reduction (standard and generalized)
    - Eigenvalue reordering (standard and generalized)
    - Miscellaneous user interface functions
    - Test program
    - Documentation

The distributed memory Hessenberg-triangular reduction is implemented as a
ScaLAPACK wrapper and `PDGGHRD` subroutine (Lars Karlsson, Björn Adlerborn) is
included with the library. The test program includes Matrix Market I/O library
for ANSI C (http://math.nist.gov/MatrixMarket).
