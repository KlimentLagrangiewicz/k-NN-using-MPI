#include "knn.h"

void normalization(double *x, double *y, const int n1, const int n2, const int m) {
	const int buf1 = n1 * m, buf2 = n2 * m;
	double av, d, sig;
	int i, j;
	for (j = 0; j < m; j++) {
		av = sig = 0;
		for (i = j; i < buf1; i += m) {
			av += x[i];
		}
		for (i = j; i < buf2; i += m) {
			av += y[i];
		}
		av /= (n1 + n2);
		for (i = j; i < buf1; i += m) {
			d = x[i] - av;
			sig += d * d;
			x[i] = d;
		}
		for (i = j; i < buf2; i += m) {
			d = y[i] - av;
			sig += d * d;
			y[i] = d;
		}
		sig = sqrt(sig / (n1 + n2));
		for (i = j; i < buf1; i += m) {
			x[i] /= sig;
		}
        for (i = j; i < buf2; i += m) {
        	y[i] /= sig;
        }
	}
}

double distEv(const double *x, const double *c, const int m) {
	double d, r = 0;
	int i = 0;
	while (i++ < m) {
		d = *(x++) - *(c++);
		r += d * d;
	}
	return r;
}

int getClass(const double *xTrain, const int *yTrain, const double *xTest, const int n, const int m, const int k, const int noc, const int i) {
	int j, l, id = 0;
	double *d = (double*)malloc(n * sizeof(double));
	int *r = (int*)malloc(noc * sizeof(int));
	short *v = (short*)malloc(n * sizeof(short));
	memset(v, 0, n * sizeof(short));
	memset(r, 0, noc * sizeof(int));
	for (j = 0; j < n; j++) {
		d[j] = distEv(&xTest[i * m], &xTrain[j * m], m);
	}
	double min_d;
	for (j = 0; j < k; j++) {
		l = 0;
		while (v[l]) l++;
		min_d = d[l];
		id = l;
		l++;
		for (; l < n; l++) {
			if ((v[l] == 0) && (d[l] < min_d)) {
				min_d = d[l];
				id = l;
			}
		}
		v[id] = 1;
		r[yTrain[id]]++;
	}
	id = 0;
	l = 0;
	for (j = 0; j < noc; j++) {
		if (r[j] > l) {
			l = r[j];
			id = j;
		}
	}
	free(d);
	free(v);
	free(r);
	return id;
}

