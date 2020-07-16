# StarNEig User's Guide

The User's Guide can be generated independently from the rest of the library.

Dependencies:

 - CMake 3.3 or newer
 - Doxygen
 - Latex + pdflatex

It is recommended that a user builds the documentation in a separate build
directory:
```
$ mkdir build-doc
$ cd build-doc/
$ cmake ../doc/
$ make
```

The PDF documentation is copied to `build-doc/starneig_manual.pdf` and the HTML
documentation is available at `build-doc/html` directory.
