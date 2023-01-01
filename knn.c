#include "knn.h"

void normalization(double *x, double *y, const int n1, const int n2, const int m) {
	const int s1 = n1 * m, s2 = n2 * m;
	double sd, Ex, Exx;
	int i, j = 0;
	while (j < m) {
		i = j;
		Ex = Exx = 0;
		while (i < s1) {
			sd = x[i];
			Ex += sd;
			Exx += sd * sd;
			i += m;
		}
		i = j;
		while (i < s2) {
			sd = y[i];
			Ex += sd;
			Exx += sd * sd;
			i += m;
		}
		Exx /= n1 + n2;
		Ex /= n1 + n2;
		sd = sqrt(Exx - Ex * Ex);
		i = j;
		while (i < s1) {
			x[i] = (x[i] - Ex) / sd;
			i += m;
		}
		i = j;
		while (i < s2) {
			y[i] = (y[i] - Ex) / sd;
			i += m;
		}
		j++;
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

int getClass(const double *xTrain, const int *yTrain, const double *xTest, const int n, const int m, const int k, const int noc, const int i,
	double *d, char *v, int *r) {
	int j, l, id = 0;
	memset(v, 0, n * sizeof(char));
	memset(r, 0, noc * sizeof(int));
	for (j = 0; j < n; j++) {
		d[j] = distEv(&xTest[i], &xTrain[j * m], m);
	}
	double min_d;
	for (j = 0; j < k; j++) {
		l = 0;
		while ((v[l]) && (l < n)) l++;
		min_d = d[l];
		id = l;
		l++;
		for (; l < n; l++) {
			if ((!v[l]) && (d[l] < min_d)) {
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
	return id;
}

