#include "help.h"

void fscanfTrainData(const char *fn, double *x, int *y, const int n, const int m) {
	FILE *fl = fopen(fn, "r");
	if (!fl) {
		printf("Error in opening %s file...\n", fn);
		exit(1);
	}
	int i, j;
	for (i = 0; i < n && !feof(fl); i++) {
		for (j = i * m; j < i * m + m && !feof(fl); j++) {
			if (fscanf(fl, "%lf", x + j) == 0) {}
		}
		if (!feof(fl)) {
			if (fscanf(fl, "%d", y + i) == 0) {}
		}
	}
	fclose(fl);
}

void fscanfTestData(const char *fn, double* const x, const int n) {
	FILE *fl = fopen(fn, "r");
	if (!fl) {
		printf("Error in opening %s file...\n", fn);
		exit(1);
	}
	int i;
	for (i = 0; i < n && !feof(fl); i++) {
		if (fscanf(fl, "%lf", x + i) == 0) {}
	}
	fclose(fl);
}

int getIntWidth(int n) {
	if (n == 0) return 1;
	int res = 0;
	if (n < 0) {
		res = 1;
		n *= -1;
	}
	while (n) {
		res++;
		n /= 10;
	}
	return res;
}

int getMax(const int *y, int n) {
	int max = *(y++);
	while (--n) {
		if (*y > max) max = *y;
		y++;
	}
	return max;
}

void fscanfIdealSpliting(const char *fn, int *id, const int n) {
	FILE *fl = fopen(fn, "r");
	if (!fl) {
		printf("Error in opening %s file...\n", fn);
		exit(1);
	}
	int i;
	for (i = 0; i < n && !feof(fl); i++) {
		if (fscanf(fl, "%d", &id[i]) == 0) {}
	}
	fclose(fl);
}

double calcAccuracy(const int *x, const int *y, const int n) {
	int i = 0, j = 0;
	while (i++ < n) if (*(x++) == *(y++)) j++;
	return (double)j / (double)n;
}

void fprintfResult(const char *fn, const int* const y, const int n1, const int m, const int n2, const int k, const double t) {
	FILE *fl = fopen(fn, "a");
	if (!fl) {
		printf("Error in opening %s result file\n", fn);
		exit(1);
	}
	fprintf(fl, "Result of k-NN classification...\nNumber of precedents\t\t= %d\nNumber of attributes\t\t= %d\nNumber of classified objects\t= %d\nNumber of neighbours\t\t= %d\nTime for classification\t\t= %lf s.\n", n1, m, n2, k, t);
	const int width1 = getIntWidth(n2), width2 = getIntWidth(getMax(y, n2));
	int i;
	for (i = 0; i < n2; i++)
		fprintf(fl, "Object[%*d]: %*d;\n", width1, i + 1, width2, y[i]);
	fputc('\n', fl);
	fclose(fl);
}

void fprintfFullRes(const char *fn, const int* const y, const int n1, const int m, const int n2, const int k,const double a, const double t) {
	FILE *fl = fopen(fn, "a");
	if (!fl) {
		printf("Error in opening %s result file\n", fn);
		exit(1);
	}
	fprintf(fl, "Result of k-NN classification...\nNumber of precedents\t\t= %d\nNumber of attributes\t\t= %d\nNumber of classified objects\t= %d\nNumber of neighbours\t\t= %d\nTime for classification\t\t= %lf s.\nAccuracy of classification\t= %lf\n", n1, m, n2, k, t, a);
	const int width1 = getIntWidth(n2), width2 = getIntWidth(getMax(y, n2));
	int i;
	for (i = 0; i < n2; i++)
		fprintf(fl, "Object[%*d]: %*d;\n", width1, i + 1, width2, y[i]);
	fputc('\n', fl);
	fclose(fl);
}
