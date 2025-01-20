# k-NN-using-MPI
Implementation of k-NN algorithm in C programming language using MPI technology
## Example of usage
Cloning project and changing current directory:
```
https://github.com/KlimentLagrangiewicz/k-NN-using-MPI
cd k-NN-using-MPI
```
Building from source (Linux):
```
make
```
Building from source (Windows):
```
make windows
```
If building was successfully, you can find executable file in `bin` subdirectory.  
Run the program:
```
mpiexec -n 7 ./bin/mpi-knn ./datasets/iris/train.txt ./datasets/iris/test.txt 46 4 150 19 ./datasets/iris/newresult.txt ./datasets/iris/res.txt
```