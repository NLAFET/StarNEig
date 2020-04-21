## Contributing 

Please execute the test suite before submitting code:
```
$ mkdir build
$ cd build/
$ cmake ../
$ make
$ make test
```

If possible, consider executing the full test suite:
```
$ mkdir build
$ cd build/
$ cmake -DSTARNEIG_ENABLE_FULL_TESTS=ON ../
$ make
$ make test
```
