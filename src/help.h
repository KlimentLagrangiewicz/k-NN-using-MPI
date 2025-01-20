#ifndef HELP_H_
#define HELP_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void fscanfTrainData(const char *fn, double *x, int *y, const int n, const int m);
void fscanfTestData(const char *fn, double* const x, const int n);
void fscanfIdealSpliting(const char *fn, int *id, const int n);
double calcAccuracy(const int *x, const int *y, const int n);
int getIntWidth(int n);
int getMax(const int *y, int n);
void fprintfResult(const char *fn, const int* const y, const int n1, const int m, const int n2, const int k, const double t);
void fprintfFullRes(const char *fn, const int* const y, const int n1, const int m, const int n2, const int k,const double a, const double t);

#endif
