# Introduction

StarNEig library aims to provide a complete task-based software stack for
solving **dense** **nonsymmetric** (generalized) eigenvalue problems. The
library is built on top of the [StarPU](http://starpu.gforge.inria.fr/)
runtime system.
The library currently supports only real arithmetic (real input and output
matrices but real and/or complex eigenvalues and eigenvectors). In addition,
some interface functions are implemented as LAPACK and ScaLAPACK wrapper
functions.

Standard eigenvalue problems:
| Component             |  Shared memory  | Distributed memory |      CUDA      |
|-----------------------|:---------------:|:------------------:|:--------------:|
| Hessenberg reduction  |  **Complete**   |      ScaLAPACK     | **Single GPU** |
| Schur reduction       |  **Complete**   |    **Complete**    | *Experimental* |
| Eigenvalue reordering |  **Complete**   |    **Complete**    | *Experimental* |
| Eigenvectors          |  **Complete**   |        ---         |      ---       |

Generalized eigenvalue problems:
| Component             |  Shared memory  | Distributed memory |      CUDA      |
|-----------------------|:---------------:|:------------------:|:--------------:|
| HT reduction          |     LAPACK      |     3rd party      |      ---       |
| Schur reduction       |  **Complete**   |    **Complete**    | *Experimental* |
| Eigenvalue reordering |  **Complete**   |    **Complete**    | *Experimental* |
| Eigenvectors          |  **Complete**   |        ---         |      ---       |

The library has been developed as a part of the NLAFET project. The project has
received funding from the European Union’s Horizon 2020 research and innovation
programme under grant agreement No. 671633. Support has also been received
from eSSENCE, a collaborative e-Science programme funded by the Swedish
Government via the Swedish Research Council (VR), and VR Grant E0485301.

The library is open source and published under BSD 3-Clause licence.

Please cite the following article when refering to StarNEig:
> Mirko Myllykoski, Carl Christian Kjelgaard Mikkelsen: *Introduction to
> StarNEig — A Task-based Library for Solving Nonsymmetric Eigenvalue Problems*,
> In Parallel Processing and Applied Mathematics, 13th International Conference,
> PPAM 2019, Bialystok, Poland, September 8–11, 2019, Revised Selected Papers,
> Part I, Lecture Notes in Computer Science, Vol. 12043, Wyrzykowski R., Deelman
> E., Dongarra J., Karczewski K. (eds), Springer International Publishing, pp.
> 70-81, 2020, doi:
> [10.1007/978-3-030-43229-4_7](https://doi.org/10.1007/978-3-030-43229-4_7)
