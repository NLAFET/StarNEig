# StarNEig example codes

It is recommended that a user builds the example codes *together* with the
library (see the main `README.md` file and/or the User's Guide):
```
$ mkdir build
$ cd build/
$ cmake -DSTARNEIG_ENABLE_EXAMPLES=ON ../
$ make
```

If the library is already installed to the system, a user may compile
the example codes separately:
```
$ mkdir build-examples
$ cd build-examples/
$ cmake ../examples/
$ make
```
