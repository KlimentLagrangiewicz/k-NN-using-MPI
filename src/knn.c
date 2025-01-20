#include "knn.h"

double getDistance(const double *x1, const double *x2, int m) {
	double d, r = 0.0;
	while (m--) {
		d = *(x1++) - *(x2++);
		r += d * d;
	}
	return sqrt(r);
}

void autoscaling(double* const x, const int n, const int m) {
	const int s = n * m;
	int j;
	for (j = 0; j < m; j++) {
		double sd, Ex = 0.0, Exx = 0.0, *ptr;
		for (ptr = x + j; ptr < x + s; ptr += m) {
			sd = *ptr;
			Ex += sd;
			Exx += sd * sd;
		}
		Exx /= n;
		Ex /= n;
		sd = sqrt(Exx - Ex * Ex);
		for (ptr = x + j; ptr < x + s; ptr += m) {
			*ptr = (*ptr - Ex) / sd;
		}
	}
}

void blockFunction1(const double* const x, double* const Ex, double* const Exx, const int m, const int perProc) {
	int i, j;
	for (i = 0; i < perProc * m; i += m) { 
		for (j = 0; j < m; j++) {
			Ex[j] += x[i + j];
			Exx[j] += x[i + j] * x[i + j];
		}
	}
}

void blockFunction2(double* const x, const double* const Ex, const double* const Exx, const int m, const int perProc) {
	int i, j;
	for (i = 0; i < perProc * m; i += m)
		for (j = 0; j < m; j++) 
			x[i + j] = (x[i + j] - Ex[j]) / Exx[j];
}

void MPI_Scalingtestdata(double* const x, const int n, const int m) {
	int numOfProc;
	MPI_Comm_size(MPI_COMM_WORLD, &numOfProc);
	int perProc = n / numOfProc;
	if (perProc == 0) {
		int pid;
		MPI_Comm_rank(MPI_COMM_WORLD, &pid);
		if (pid == 0) autoscaling(x, n, m);
	} else {
		double *localX = (double*)malloc(perProc * m * sizeof(double));
		MPI_Scatter(x, perProc * m, MPI_DOUBLE, localX, perProc * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		double *Ex = (double*)calloc(m, sizeof(double));
		double *Exx = (double*)calloc(m, sizeof(double));
		int pid;
		MPI_Comm_rank(MPI_COMM_WORLD, &pid);
		if (pid == 0 && n > perProc * numOfProc) blockFunction1(x + perProc * numOfProc * m, Ex, Exx, m, n - perProc * numOfProc);
		blockFunction1(localX, Ex, Exx, m, perProc);
		if (pid == 0) MPI_Reduce(MPI_IN_PLACE, Ex, m, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		else MPI_Reduce(Ex, Ex, m, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		if (pid == 0) MPI_Reduce(MPI_IN_PLACE, Exx, m, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		else MPI_Reduce(Exx, Exx, m, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		if (pid == 0) {
			int i;
			for (i = 0; i < m; i++) {
				Ex[i] /= n;
				Exx[i] /= n;
				Exx[i] = sqrt(Exx[i] - Ex[i] * Ex[i]);
			}
		}
		MPI_Bcast(Ex, m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(Exx, m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		if (pid == 0 && n > perProc * numOfProc) blockFunction2(x + perProc * numOfProc * m, Ex, Exx, m, n - perProc * numOfProc);
		blockFunction2(localX, Ex, Exx, m, perProc);
		MPI_Gather(localX, perProc * m, MPI_DOUBLE, x, perProc * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		free(Ex);
		free(Exx);
		free(localX);
	}
}

void MPI_Scalingtraindata(double* const x, const int n, const int m) {
	int numOfProc;
	MPI_Comm_size(MPI_COMM_WORLD, &numOfProc);
	int perProc = n / numOfProc;
	if (perProc == 0) {
		int pid;
		MPI_Comm_rank(MPI_COMM_WORLD, &pid);
		if (pid == 0) autoscaling(x, n, m);
	} else {
		double *Ex = (double*)calloc(m, sizeof(double));
		double *Exx = (double*)calloc(m, sizeof(double));
		int pid;
		MPI_Comm_rank(MPI_COMM_WORLD, &pid);
		if (pid == 0 && n > perProc * numOfProc) blockFunction1(x + perProc * numOfProc * m, Ex, Exx, m, n - perProc * numOfProc);
		blockFunction1(x + pid * perProc * m, Ex, Exx, m, perProc);
		if (pid == 0) MPI_Reduce(MPI_IN_PLACE, Ex, m, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		else MPI_Reduce(Ex, Ex, m, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		if (pid == 0) MPI_Reduce(MPI_IN_PLACE, Exx, m, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		else MPI_Reduce(Exx, Exx, m, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		if (pid == 0) {
			int i;
			for (i = 0; i < m; i++) {
				Ex[i] /= n;
				Exx[i] /= n;
				Exx[i] = sqrt(Exx[i] - Ex[i] * Ex[i]);
			}
		}
		MPI_Bcast(Ex, m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(Exx, m, MPI_DOUBLE, 0, MPI_COMM_WORLD);		
		blockFunction2(x + pid * perProc * m, Ex, Exx, m, perProc);
		MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, x, perProc * m, MPI_DOUBLE, MPI_COMM_WORLD);
		if (pid == 0 && n > perProc * numOfProc) blockFunction2(x + perProc * numOfProc * m, Ex, Exx, m, n - perProc * numOfProc);
		if (n > perProc * numOfProc) MPI_Bcast(x + perProc * numOfProc * m, n * m - perProc * numOfProc * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		free(Ex);
		free(Exx);
	}
}

void insertionSort(double* const pr, int* const cl, const int count) {
	int i;
	for (i = 1; i < count; i++) {
		double key_pr = pr[i];
		int key_cl = cl[i];
		int j = i - 1;
		while (j >= 0 && pr[j] > key_pr) {
			pr[j + 1] = pr[j];
			cl[j + 1] = cl[j];
			j--;
		}
		pr[j + 1] = key_pr;
		cl[j + 1] = key_cl;
	}
}

void maxHeapify(double *pr, int *cl, int heapSize, int index) {
	int left = 2 * index + 1;
	int right = 2 * index + 2;
	int largest = index;
	if (left < heapSize && pr[left] > pr[largest]) largest = left;
	if (right < heapSize && pr[right] > pr[largest]) largest = right;
	if (largest != index) {
		double temp_pr = pr[index];
		int temp_cl = cl[index];
		pr[index] = pr[largest];
		cl[index] = cl[largest];
		pr[largest] = temp_pr;
		cl[largest] = temp_cl;
		maxHeapify(pr, cl, heapSize, largest);
	}
}

void heapSort(double *pr, int *cl, int count) {
	int i;
	for (i = count / 2 - 1; i >= 0; i--)
		maxHeapify(pr, cl, count, i);
	for (i = count - 1; i > 0; i--) {
		double temp_pr = pr[i];
		int temp_cl = cl[i];
		pr[i] = pr[0];
		cl[i] = cl[0];
		pr[0] = temp_pr;
		cl[0] = temp_cl;
		maxHeapify(pr, cl, i, 0);
	}
}

int partition(double* pr, int *cl, int left, int right) {
	double pivot_pr = pr[right], temp_pr;
	int j, temp_cl, pivot_cl = cl[right], i = left;
	for (j = left; j < right; j++) {
		if (pr[j] <= pivot_pr) {
			temp_pr = pr[j];
			temp_cl = cl[j];
			pr[j] = pr[i];
			cl[j] = cl[i];
			pr[i] = temp_pr;
			cl[i] = temp_cl;
			i++;
		}
	}
	pr[right] = pr[i];
	cl[right] = cl[i];
	pr[i] = pivot_pr;
	cl[i] = pivot_cl;
	return i;
}

void quickSortRecursive(double *pr, int *cl, int left, int right) {
	if (left < right) {
		int q = partition(pr, cl, left, right);
		quickSortRecursive(pr, cl, left, q - 1);
		quickSortRecursive(pr, cl, q + 1, right);
	}
}

void introSort(double *pr, int *cl, int count) {
	int partitionSize = partition(pr, cl, 0, count - 1);
	if (partitionSize < 16) insertionSort(pr, cl, count);
	else if (partitionSize > (2 * log(count))) heapSort(pr, cl, count);
	else quickSortRecursive(pr, cl, 0, count - 1);	
}

int getPosMax(const int * const y, const int n) {
	int maxP = 0, maxV = *y, i;
	for (i = 1; i < n; i++) {
		if (y[i] > maxV) {
			maxP = i;
			maxV = y[i];
		}
	}
	return maxP;
}

int getClass1(const double* const x, const double *xTrain, const int *yTrain, const int nTrain, const int m, const int k, const int l) {
	int *cl = (int*)malloc(nTrain * sizeof(int));
	double *d = (double*)malloc(nTrain * sizeof(double));
	int i;
	for (i = 0; i < nTrain; i++) {
		cl[i] = yTrain[i];
		d[i] = getDistance(x, xTrain + i * m, m);
	}
	introSort(d, cl, nTrain);
	free(d);
	int *fr = (int*)calloc(l, sizeof(int));
	for (i = 0; i < k; i++)
		fr[cl[i]]++;
	i = getPosMax(fr, l);
	free(cl);
	free(fr);
	return i;
}

char contain(const int *y, int s, const int val) {
	while (s--) if (*(y++) == val) return 1;
	return 0;
}

int* getNeighborsArray(const double *d, const int n, const int k) {
	int *y = (int*)malloc(k * sizeof(int));
	int i;
	for (i = 0; i < k; i++) {
		int j = 0;
		while (j < n && contain(y, i, j)) j++;
		double min = d[j];
		int minid = j;
		j++;
		for (; j < n; j++) {
			if (d[j] < min && !contain(y, i, j)) {
				min = d[j];
				minid = j;
			}
		}
		y[i] = minid;
	}
	return y;
}

int getClass2(const double* const x, const double *xTrain, const int *yTrain, const int nTrain, const int m, const int k, const int l) {
	double *d = (double*)malloc(nTrain * sizeof(double));
	int i;
	for (i = 0; i < nTrain; i++) 
		d[i] = getDistance(x, xTrain + i * m, m);
	int *y = getNeighborsArray(d, nTrain, k);
	free(d);
	int *fr = (int*)calloc(l, sizeof(int));
	for (i = 0; i < k; i++)
		fr[yTrain[y[i]]]++;
	
	i = getPosMax(fr, l);
	free(y);
	free(fr);
	return i;
}

void getClasses(const double* const x, int* const y, const double* const xTrain, const int* const yTrain, const int nTest, const int m, const int nTrain, const int k, const int l, int (*f)(const double* const, const double *, const int *, const int, const int, const int, const int)) {
	int i;
	for (i = 0; i < nTest; i++) y[i] = f(x + i * m, xTrain, yTrain, nTrain, m, k, l);
}

void MPI_Getclasses(const double* const x, int* const y, const double* const xTrain, const int* const yTrain, const int nTest, const int m, const int nTrain, const int k, const int l, int (*f)(const double* const, const double *, const int *, const int, const int, const int, const int)) {
	int numOfProc;
	MPI_Comm_size(MPI_COMM_WORLD, &numOfProc);
	int perProc = nTest / numOfProc;
	if (perProc == 0) {
		int pid;
		MPI_Comm_rank(MPI_COMM_WORLD, &pid);
		if (pid == 0) {
			getClasses(x, y, xTrain, yTrain, nTest, m, nTrain, k, l, f);
		}
	} else {
		double *localX = (double*)malloc(perProc * m * sizeof(double));
		MPI_Scatter(x, perProc * m, MPI_DOUBLE, localX, perProc * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		int *localY = (int*)malloc(perProc * sizeof(int));
		getClasses(localX, localY, xTrain, yTrain, perProc, m, nTrain, k, l, f);
		MPI_Gather(localY, perProc, MPI_INT, y, perProc, MPI_INT, 0, MPI_COMM_WORLD);
		int pid;
		MPI_Comm_rank(MPI_COMM_WORLD, &pid);
		if (pid == 0 && nTest > perProc * numOfProc) getClasses(x + numOfProc * perProc * m, y + numOfProc * perProc, xTrain, yTrain, nTest - perProc * numOfProc, m, nTrain, k, l, f);
		free(localY);
		free(localX);
	}
}

int getNumOfClass(const int* const y, const int n) {
	int i, j, cur;
	char *v = (char*)malloc(n * sizeof(char));
	memset(v, 0, n * sizeof(char));
	for (i = 0; i < n; i++) {
		while (i < n && v[i]) i++;
		cur = y[i];
		for (j = i + 1; j < n; j++) {
			if (y[j] == cur) v[j] = 1;
		}
	}
	i = cur = 0;
	while (i < n) {
		if (v[i] == 0) cur++;
		i++;
	}
	free(v);
	return cur;
}

void MPI_Knn(const double* const X, int* const y, const double* const _xTrain, const int* const yTrain, const int nTest, const int m, const int nTrain, const int k) {
	double *x = NULL;
	int pid;	
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);
	if (pid == 0) {
		x = (double*)malloc(nTest * m * sizeof(double));
		memcpy(x, X, nTest * m * sizeof(double));
	}	
	double *xTrain = (double*)malloc(nTrain * m * sizeof(double));
	memcpy(xTrain, _xTrain, nTrain * m * sizeof(double));	
	MPI_Scalingtestdata(x, nTest, m);	
	MPI_Scalingtraindata(xTrain, nTrain, m);
	const int l = getNumOfClass(yTrain, nTrain);
	if (log(nTest) / log(2) < k) {
		MPI_Getclasses(x, y, xTrain, yTrain, nTest, m, nTrain, k, l, getClass1);
	} else {
		MPI_Getclasses(x, y, xTrain, yTrain, nTest, m, nTrain, k, l, getClass2);
	}
	if (pid == 0) free(x);
	free(xTrain);
}
