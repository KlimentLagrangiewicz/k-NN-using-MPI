#ifndef KNN_H_
#define KNN_H_

#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

double getDistance(const double *x1, const double *x2, int m);
void autoscaling(double* const x, const int n, const int m);
void blockFunction1(const double* const x, double* const Ex, double* const Exx, const int m, const int perProc);
void blockFunction2(double* const x, const double* const Ex, const double* const Exx, const int m, const int perProc);
void MPI_Scalingtestdata(double* const x, const int n, const int m);
void MPI_Scalingtraindata(double* const x, const int n, const int m);
void insertionSort(double* const pr, int* const cl, const int count);
void maxHeapify(double *pr, int *cl, int heapSize, int index);
void heapSort(double *pr, int *cl, int count);
int partition(double* pr, int *cl, int left, int right);
void quickSortRecursive(double *pr, int *cl, int left, int right);
void introSort(double *pr, int *cl, int count);
int getPosMax(const int * const y, const int n);
int getClass1(const double* const x, const double *xTrain, const int *yTrain, const int nTrain, const int m, const int k, const int l);
char contain(const int *y, int s, const int val);
int* getNeighborsArray(const double *d, const int n, const int k);
int getClass2(const double* const x, const double *xTrain, const int *yTrain, const int nTrain, const int m, const int k, const int l);
void getClasses(const double* const x, int* const y, const double* const xTrain, const int* const yTrain, const int nTest, const int m, const int nTrain, const int k, const int l, int (*f)(const double* const, const double *, const int *, const int, const int, const int, const int));
void MPI_Getclasses(const double* const x, int* const y, const double* const xTrain, const int* const yTrain, const int nTest, const int m, const int nTrain, const int k, const int l, int (*f)(const double* const, const double *, const int *, const int, const int, const int, const int));
int getNumOfClass(const int* const y, const int n);
void MPI_Knn(const double* const X, int* const y, const double* const _xTrain, const int* const yTrain, const int nTest, const int m, const int nTrain, const int k);

#endif
