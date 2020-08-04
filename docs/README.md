# StarNEig User's Guide

The StarNEig User's Guide is available in both HTML and PDF formats at
https://nlafet.github.io/StarNEig. The PDF version is also available under
[releases](https://github.com/NLAFET/StarNEig/releases).

The latest version of the User's Guide can be generated independently from the
rest of the library, see instructions below.

Dependencies:

 - CMake 3.3 or newer
 - Doxygen
 - Latex installation with pdflatex

It is recommended that a user builds the documentation in a separate build
directory:
```
$ mkdir build-doc
$ cd build-doc/
$ cmake ../doc/
$ make
```

The PDF documentation is copied to `build-doc/starneig_manual.pdf` and the HTML
documentation is available under the `build-doc/html` directory.
