#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mpi.h"
#include "help.h"
#include "knn.h"

void getClasses(const double *xTrain, const int *yTrain, const double *xTest, int *res, const int n, const int start, const int end,
	 const int m, const int k, const int noc, double *d, char *v, int *r) {
	int i = start, j = start * m;
	while (i < end) {
		res[i] = getClass(xTrain, yTrain, xTest, n, m, k, noc, j, d, v, r);
		i++;
		j += m;
	}
}

void MPI_kNN(const double *xTrain, const int *yTrain, const double *xTest, int *res, const int n, const int n2,
	 const int m, const int k, const int noc) {
	MPI_Status status;
	double *d = (double*)malloc(n * sizeof(double));
	char *v = (char*)malloc(n * sizeof(char));
	int *r = (int*)malloc(noc * sizeof(int));
	int pid, perProc, numOfProc;
	MPI_Comm_size(MPI_COMM_WORLD, &numOfProc);
	perProc = n2 / numOfProc;
	if (perProc == 0) {
		MPI_Comm_rank(MPI_COMM_WORLD, &pid);
		if (pid == 0) {
			res[0] = getClass(xTrain, yTrain, xTest, n, m, k, noc, 0, d, v, r);
			int i;
			for (i = 1; i < n2; i++) {
				MPI_Recv(&res[i], 1, MPI_INT, i, 3, MPI_COMM_WORLD, &status);
			}
			for (i = 1; i < numOfProc; i++) {
				MPI_Send(res, n2, MPI_INT, i, 4, MPI_COMM_WORLD);
			}
		} else {
			if (pid < n2) {
				res[pid] = getClass(xTrain, yTrain, xTest, n, m, k, noc, pid * m, d, v, r);
				MPI_Send(&res[pid], 1, MPI_INT, 0, 3, MPI_COMM_WORLD);
			}
			MPI_Recv(res, n2, MPI_INT, 0, 4, MPI_COMM_WORLD, &status);
		}
	} else {
		MPI_Comm_rank(MPI_COMM_WORLD, &pid);
		if (pid == 0) {
			getClasses(xTrain, yTrain, xTest, res, n, 0, perProc, m, k, noc, d, v, r);
			getClasses(xTrain, yTrain, xTest, res, n, perProc * numOfProc, n2, m, k, noc, d, v, r);
			int i;
			for (i = 1; i < numOfProc; i++) {
				MPI_Recv(&res[i * perProc], perProc, MPI_INT, i, 1, MPI_COMM_WORLD, &status);
			}
			for (i = 1; i < numOfProc; i++) {
				MPI_Send(res, n2, MPI_INT, i, 2, MPI_COMM_WORLD);
			}
		} else {
			getClasses(xTrain, yTrain, xTest, res, n, pid * perProc, pid * perProc + perProc, m, k, noc, d, v, r);
			MPI_Send(&res[pid * perProc], perProc, MPI_INT, 0, 1, MPI_COMM_WORLD);
			MPI_Recv(res, n2, MPI_INT, 0, 2, MPI_COMM_WORLD, &status);
		}
	}
	free(d);
	free(v);
	free(r);
}

void basicNormalization(double *x, double *y, const int n1, const int n2, const int start, const int end, const int m) {
	const int s1 = n1 * m, s2 = n2 * m;
	double sd, Ex, Exx;
	int i, j;
	for (j = start; j < end; j++) {
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
	}
}

void columnsToBuffer(const double *x, double *buf, const int n, const int m, const int sm1, const int dif) {
	int i, k = 0;
	for (i = sm1; i < n * m; i += m) {
		memcpy(&buf[k], &x[i], dif * sizeof(double));
		k += dif;
	}
}

void bufferToColumns(double *x, const double *buf, const int n, const int m, const int sm1, const int dif) {
	int i, k = 0;
	for (i = sm1; i < n * m; i += m) {
		memcpy(&x[i], &buf[k], dif * sizeof(double));
		k += dif;
	}
}

void columnToBuffer(const double *x, double *buf, const int n, const int m, const int id) {
	int i, k = 0;
	for (i = id; i < n * m; i += m) {
		buf[k] = x[i];
		k++;
	}
}

void bufferToColumn(double *x, const double *buf, const int n, const int m, const int id) {
	int i, k = 0;
	for (i = id; i < n * m; i += m) {
		x[i] = buf[k];
		k++;
	}
}

void elementaryNormalization(double *x, double *y, const int n1, const int n2, const int j, const int m) {
	const int s1 = n1 * m, s2 = n2 * m;
	double sd, Ex, Exx;
	int i = j;
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
}

void MPI_Normalization(double *x, double *y, const int n1, const int n2, const int m) {
	MPI_Status status;
	int pid, perProc, numOfProc;
	MPI_Comm_size(MPI_COMM_WORLD, &numOfProc);
	perProc = m / numOfProc;
	if (perProc == 0) { /* tags = {13, 14, ..., 16} */
		MPI_Comm_rank(MPI_COMM_WORLD, &pid);
		if (pid == 0) {
			elementaryNormalization(x, y, n1, n2, 0, m);
			double *bufferX = (double*)malloc(n1 * sizeof(double));
			double *bufferY = (double*)malloc(n2 * sizeof(double));
			int i;
			for (i = 1; i < m; i++) {
				MPI_Recv(bufferX, n1, MPI_DOUBLE, i, 13, MPI_COMM_WORLD, &status);
				MPI_Recv(bufferY, n2, MPI_DOUBLE, i, 14, MPI_COMM_WORLD, &status);
				bufferToColumn(x, bufferX, n1, m, i);
				bufferToColumn(y, bufferY, n2, m, i);
			}
			free(bufferX);
			free(bufferY);
			for (i = 1; i < numOfProc; i++) {
				MPI_Send(x, n1 * m, MPI_DOUBLE, i, 15, MPI_COMM_WORLD);
				MPI_Send(y, n2 * m, MPI_DOUBLE, i, 16, MPI_COMM_WORLD);
			}
		} else {
			if (pid < m) {
				elementaryNormalization(x, y, n1, n2, pid, m);
				double *bufferX = (double*)malloc(n1 * sizeof(double));
				double *bufferY = (double*)malloc(n2 * sizeof(double));
				columnToBuffer(x, bufferX, n1, m, pid);
				columnToBuffer(y, bufferY, n2, m, pid);
				MPI_Send(bufferX, n1, MPI_DOUBLE, 0, 13, MPI_COMM_WORLD);
				MPI_Send(bufferY, n2, MPI_DOUBLE, 0, 14, MPI_COMM_WORLD);
				free(bufferX);
				free(bufferY);
			}
			MPI_Recv(x, n1 * m, MPI_DOUBLE, 0, 15, MPI_COMM_WORLD, &status);
			MPI_Recv(y, n2 * m, MPI_DOUBLE, 0, 16, MPI_COMM_WORLD, &status);
		}
	} else {
		if (perProc == 1) { /* tags = {9, 10, ..., 12} */
			MPI_Comm_rank(MPI_COMM_WORLD, &pid);
			if (pid == 0) {
				elementaryNormalization(x, y, n1, n2, 0, m);
				basicNormalization(x, y, n1, n2, perProc * numOfProc, m, m);
				double *bufferX = (double*)malloc(n1 * sizeof(double));
				double *bufferY = (double*)malloc(n2 * sizeof(double));
				int i;
				for (i = 1; i < numOfProc; i++) {
					MPI_Recv(bufferX, n1, MPI_DOUBLE, i, 9, MPI_COMM_WORLD, &status);
					MPI_Recv(bufferY, n2, MPI_DOUBLE, i, 10, MPI_COMM_WORLD, &status);
					bufferToColumn(x, bufferX, n1, m, i);
					bufferToColumn(y, bufferY, n2, m, i);
				}
				free(bufferX);
				free(bufferY);
				for (i = 1; i < numOfProc; i++) {
					MPI_Send(x, n1 * m, MPI_DOUBLE, i, 11, MPI_COMM_WORLD);
					MPI_Send(y, n2 * m, MPI_DOUBLE, i, 12, MPI_COMM_WORLD);
				}
			} else {
				elementaryNormalization(x, y, n1, n2, pid, m);
				double *bufferX = (double*)malloc(n1 * sizeof(double));
				double *bufferY = (double*)malloc(n2 * sizeof(double));
				columnToBuffer(x, bufferX, n1, m, pid);
				columnToBuffer(y, bufferY, n2, m, pid);
				MPI_Send(bufferX, n1, MPI_DOUBLE, 0, 9, MPI_COMM_WORLD);
				MPI_Send(bufferY, n2, MPI_DOUBLE, 0, 10, MPI_COMM_WORLD);
				free(bufferX);
				free(bufferY);
				MPI_Recv(x, n1 * m, MPI_DOUBLE, 0, 11, MPI_COMM_WORLD, &status);
				MPI_Recv(y, n2 * m, MPI_DOUBLE, 0, 12, MPI_COMM_WORLD, &status);
			}
		} else {
			MPI_Comm_rank(MPI_COMM_WORLD, &pid);
			if (pid == 0) {
				basicNormalization(x, y, n1, n2, 0, perProc, m);
				basicNormalization(x, y, n1, n2, perProc * numOfProc, m, m);
				double *bufferX = (double*)malloc(perProc * n1 * sizeof(double));
				double *bufferY = (double*)malloc(perProc * n2 * sizeof(double));
				int i;
				for (i = 1; i < numOfProc; i++) {
					MPI_Recv(bufferX, perProc * n1, MPI_DOUBLE, i, 5, MPI_COMM_WORLD, &status);
					MPI_Recv(bufferY, perProc * n2, MPI_DOUBLE, i, 6, MPI_COMM_WORLD, &status);
					bufferToColumns(x, bufferX, n1, m, i * perProc, perProc);
					bufferToColumns(y, bufferY, n2, m, i * perProc, perProc);
				}
				free(bufferX);
				free(bufferY);
				for (i = 1; i < numOfProc; i++) {
					MPI_Send(x, n1 * m, MPI_DOUBLE, i, 7, MPI_COMM_WORLD);
					MPI_Send(y, n2 * m, MPI_DOUBLE, i, 8, MPI_COMM_WORLD);
				}
			} else {
				double *bufferX = (double*)malloc(perProc * n1 * sizeof(double));
				double *bufferY = (double*)malloc(perProc * n2 * sizeof(double));
				basicNormalization(x, y, n1, n2, pid * perProc, pid * perProc + perProc, m);
				columnsToBuffer(x, bufferX, n1, m, pid * perProc, perProc);
				columnsToBuffer(y, bufferY, n2, m, pid * perProc, perProc);
				MPI_Send(bufferX, perProc * n1, MPI_DOUBLE, 0, 5, MPI_COMM_WORLD); /* tag 5 */
				MPI_Send(bufferY, perProc * n2, MPI_DOUBLE, 0, 6, MPI_COMM_WORLD); /* tag 6 */
				free(bufferX);
				free(bufferY);
				MPI_Recv(x, n1 * m, MPI_DOUBLE, 0, 7, MPI_COMM_WORLD, &status); /* tag 7 */
				MPI_Recv(y, n2 * m, MPI_DOUBLE, 0, 8, MPI_COMM_WORLD, &status); /* tag 8 */
			}
		}
	}
}

int main(int argc, char **argv) {
	if (argc < 8) {
		printf("Not enough parameters...\n");
		exit(1);
	}
	MPI_Init(&argc, &argv);
	const int n = atoi(argv[1]), m = atoi(argv[2]), n2 = atoi(argv[3]), k = atoi(argv[4]);
	int rank, noc;
	double *xTrain = (double*)malloc(n * m * sizeof(double));
	double *xTest = (double*)malloc(n2 * m * sizeof(double));
	int *yTrain = (int*)malloc(n * sizeof(int));
	int *yTest = (int*)malloc(n2 * sizeof(int));
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (rank == 0) {
		fscanfTrainData(xTrain, yTrain, n, m, argv[5]);
		fscanfTestData(xTest, n2 * m, argv[6]);
		noc = getNumOfClass(yTrain, n);
	}
	MPI_Bcast(&noc, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(yTrain, n, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(xTrain, n * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(xTest, n2 * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	double _time = MPI_Wtime();
	MPI_Normalization(xTrain, xTest, n, n2, m);
	MPI_kNN(xTrain, yTrain, xTest, yTest, n, n2, m, k, noc);
	_time = MPI_Wtime() - _time;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (rank == 0) {
		if (argc > 8) {
			int *id = (int*)malloc(n2 * sizeof(int));
			double a;
			fscanfIdealSpliting(id, n2, argv[8]);
			a = calcAccuracy(yTest, id, n2);
			fprintfFullRes(yTest, n2, a, _time, argv[7]);
			free(id);
			printf("Time for k-NN classification = %lf s.;\nAccuracy of k-NN classification = %lf;\nThe work of the program is completed!\n", _time, a);
		} else {
			fprintfResult(yTest, n2, _time, argv[7]);
			printf("Time for k-NN classification = %lf s.;\nThe work of the program is completed!\n", _time);
		}
	}
	free(xTrain);
	free(xTest);
	free(yTrain);
	free(yTest);
	MPI_Finalize();
	return 0;
}
