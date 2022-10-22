#ifndef KNN_H_
#define KNN_H_

#include <string.h>
#include <stdlib.h>
#include <math.h>

void normalization(double *x, double *y, const int n1, const int n2, const int m);
double distEv(const double *x, const double *c, const int m);
int getClass(const double *xTrain, const int *yTrain, const double *xTest, const int n, const int m, const int k, const int noc, const int i);

#endif
