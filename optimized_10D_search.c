#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>

#define NSEC_SEC_MUL (1.0e9)

void gridloopsearch(double kk);

struct timespec begin_grid, end_main;

// to store values of disp.txt
double a[120];

// to store values of grid.txt
double b[30];

int main() {
  int i, j;

  i = 0;
  FILE* fp = fopen("./disp.txt", "r");
  if (fp == NULL) {
    printf("Error: could not open file\n");
    return 1;
  }

  while (!feof(fp)) {
    if (!fscanf(fp, "%lf", &a[i])) {
      printf("Error: fscanf failed while reading disp.txt\n");
      exit(EXIT_FAILURE);
    }
    i++;
  }
  fclose(fp);

  // read grid file
  j = 0;
  FILE* fpq = fopen("./grid.txt", "r");
  if (fpq == NULL) {
    printf("Error: could not open file\n");
    return 1;
  }

  while (!feof(fpq)) {
    if (!fscanf(fpq, "%lf", &b[j])) {
      printf("Error: fscanf failed while reading grid.txt\n");
      exit(EXIT_FAILURE);
    }
    j++;
  }
  fclose(fpq);

  // grid value initialize
  // initialize value of kk;
  double kk = 0.3;

  clock_gettime(CLOCK_MONOTONIC_RAW, &begin_grid);
	gridloopsearch(kk);
  clock_gettime(CLOCK_MONOTONIC_RAW, &end_main);
  printf("Total time = %f seconds\n", (end_main.tv_nsec - begin_grid.tv_nsec) / NSEC_SEC_MUL +
                                          (end_main.tv_sec - begin_grid.tv_sec));

  return EXIT_SUCCESS;
}

// grid search function with loop variables

void gridloopsearch(double kk)
{
  // results values
  double x[10];

  // constraint values
  double q[10];

  // results points
  long pnts = 0;

  // re-calculated limits
  double e[10];

  // opening the "results-v0.txt" for writing he results in append mode
  FILE* fptr = fopen("./results-v0.txt", "w");
  if (fptr == NULL) {
    printf("Error in creating file !");
    exit(1);
  }

  for(int i = 0; i < 10; i++){
	  e[i] = kk * a[i*12 + 11];
  }
	int s[10];
   for(int i = 0; i < 10; i++){
       s[i] = floor((b[3*i + 1] - b[3*i]) / b[3*i + 2]);
   }


  double sum[10][10];
  // grid search starts
	for (int r1 = 0; r1 < s[0]; ++r1) {
		x[0] = b[0] + r1 * b[2];
		
		sum[0][0] = a[0*12] * x[0] - a[0*12 + 10];
		sum[0][1] = a[1*12] * x[0] - a[1*12 + 10];
		sum[0][2] = a[2*12] * x[0] - a[2*12 + 10];
		sum[0][3] = a[3*12] * x[0] - a[3*12 + 10];
		sum[0][4] = a[4*12] * x[0] - a[4*12 + 10];
		sum[0][5] = a[5*12] * x[0] - a[5*12 + 10];
		sum[0][6] = a[6*12] * x[0] - a[6*12 + 10];
		sum[0][7] = a[7*12] * x[0] - a[7*12 + 10];
		sum[0][8] = a[8*12] * x[0] - a[8*12 + 10];
		sum[0][9] = a[9*12] * x[0] - a[9*12 + 10];
		
		for (int r2 = 0; r2 < s[1]; ++r2) {
			x[1] = b[3] + r2 * b[5];
			
			sum[1][0] = sum[0][0] + a[0*12 + 1] * x[1];
			sum[1][1] = sum[0][1] + a[1*12 + 1] * x[1];
			sum[1][2] = sum[0][2] + a[2*12 + 1] * x[1];
			sum[1][3] = sum[0][3] + a[3*12 + 1] * x[1];
			sum[1][4] = sum[0][4] + a[4*12 + 1] * x[1];
			sum[1][5] = sum[0][5] + a[5*12 + 1] * x[1];
			sum[1][6] = sum[0][6] + a[6*12 + 1] * x[1];
			sum[1][7] = sum[0][7] + a[7*12 + 1] * x[1];
			sum[1][8] = sum[0][8] + a[8*12 + 1] * x[1];
			sum[1][9] = sum[0][9] + a[9*12 + 1] * x[1];
			
			for (int r3 = 0; r3 < s[2]; ++r3) {
				x[2] = b[6] + r3 * b[8];
				
				sum[2][0] = sum[1][0] + a[0*12 + 2] * x[2];
				sum[2][1] = sum[1][1] + a[1*12 + 2] * x[2];
				sum[2][2] = sum[1][2] + a[2*12 + 2] * x[2];
				sum[2][3] = sum[1][3] + a[3*12 + 2] * x[2];
				sum[2][4] = sum[1][4] + a[4*12 + 2] * x[2];
				sum[2][5] = sum[1][5] + a[5*12 + 2] * x[2];
				sum[2][6] = sum[1][6] + a[6*12 + 2] * x[2];
				sum[2][7] = sum[1][7] + a[7*12 + 2] * x[2];
				sum[2][8] = sum[1][8] + a[8*12 + 2] * x[2];
				sum[2][9] = sum[1][9] + a[9*12 + 2] * x[2];
				
				for (int r4 = 0; r4 < s[3]; ++r4) {
					x[3] = b[9] + r4 * b[11];
					
					sum[3][0] = sum[2][0] + a[0*12 + 3] * x[3];
					sum[3][1] = sum[2][1] + a[1*12 + 3] * x[3];
					sum[3][2] = sum[2][2] + a[2*12 + 3] * x[3];
					sum[3][3] = sum[2][3] + a[3*12 + 3] * x[3];
					sum[3][4] = sum[2][4] + a[4*12 + 3] * x[3];
					sum[3][5] = sum[2][5] + a[5*12 + 3] * x[3];
					sum[3][6] = sum[2][6] + a[6*12 + 3] * x[3];
					sum[3][7] = sum[2][7] + a[7*12 + 3] * x[3];
					sum[3][8] = sum[2][8] + a[8*12 + 3] * x[3];
					sum[3][9] = sum[2][9] + a[9*12 + 3] * x[3];
					
					for (int r5 = 0; r5 < s[4]; ++r5) {
						x[4] = b[12] + r5 * b[14];
						
						sum[4][0] = sum[3][0] + a[0*12 + 4] * x[4];
						sum[4][1] = sum[3][1] + a[1*12 + 4] * x[4];
						sum[4][2] = sum[3][2] + a[2*12 + 4] * x[4];
						sum[4][3] = sum[3][3] + a[3*12 + 4] * x[4];
						sum[4][4] = sum[3][4] + a[4*12 + 4] * x[4];
						sum[4][5] = sum[3][5] + a[5*12 + 4] * x[4];
						sum[4][6] = sum[3][6] + a[6*12 + 4] * x[4];
						sum[4][7] = sum[3][7] + a[7*12 + 4] * x[4];
						sum[4][8] = sum[3][8] + a[8*12 + 4] * x[4];
						sum[4][9] = sum[3][9] + a[9*12 + 4] * x[4];
						
						for (int r6 = 0; r6 < s[5]; ++r6) {
							x[5] = b[15] + r6 * b[17];
							
							sum[5][0] = sum[4][0] + a[0*12 + 5] * x[5];
							sum[5][1] = sum[4][1] + a[1*12 + 5] * x[5];
							sum[5][2] = sum[4][2] + a[2*12 + 5] * x[5];
							sum[5][3] = sum[4][3] + a[3*12 + 5] * x[5];
							sum[5][4] = sum[4][4] + a[4*12 + 5] * x[5];
							sum[5][5] = sum[4][5] + a[5*12 + 5] * x[5];
							sum[5][6] = sum[4][6] + a[6*12 + 5] * x[5];
							sum[5][7] = sum[4][7] + a[7*12 + 5] * x[5];
							sum[5][8] = sum[4][8] + a[8*12 + 5] * x[5];
							sum[5][9] = sum[4][9] + a[9*12 + 5] * x[5];
							
							for (int r7 = 0; r7 < s[6]; ++r7) {
								x[6] = b[18] + r7 * b[20];
								
								sum[6][0] = sum[5][0] + a[0*12 + 6] * x[6];
								sum[6][1] = sum[5][1] + a[1*12 + 6] * x[6];
								sum[6][2] = sum[5][2] + a[2*12 + 6] * x[6];
								sum[6][3] = sum[5][3] + a[3*12 + 6] * x[6];
								sum[6][4] = sum[5][4] + a[4*12 + 6] * x[6];
								sum[6][5] = sum[5][5] + a[5*12 + 6] * x[6];
								sum[6][6] = sum[5][6] + a[6*12 + 6] * x[6];
								sum[6][7] = sum[5][7] + a[7*12 + 6] * x[6];
								sum[6][8] = sum[5][8] + a[8*12 + 6] * x[6];
								sum[6][9] = sum[5][9] + a[9*12 + 6] * x[6];
								
								for (int r8 = 0; r8 < s[7]; ++r8) {
									x[7] = b[21] + r8 * b[23];
									
									sum[7][0] = sum[6][0] + a[0*12 + 7] * x[7];
									sum[7][1] = sum[6][1] + a[1*12 + 7] * x[7];
									sum[7][2] = sum[6][2] + a[2*12 + 7] * x[7];
									sum[7][3] = sum[6][3] + a[3*12 + 7] * x[7];
									sum[7][4] = sum[6][4] + a[4*12 + 7] * x[7];
									sum[7][5] = sum[6][5] + a[5*12 + 7] * x[7];
									sum[7][6] = sum[6][6] + a[6*12 + 7] * x[7];
									sum[7][7] = sum[6][7] + a[7*12 + 7] * x[7];
									sum[7][8] = sum[6][8] + a[8*12 + 7] * x[7];
									sum[7][9] = sum[6][9] + a[9*12 + 7] * x[7];
									
									for (int r9 = 0; r9 < s[8]; ++r9) {
										x[8] = b[24] + r9 * b[26];
										
										sum[8][0] = sum[7][0] + a[0*12 + 8] * x[8];
										sum[8][1] = sum[7][1] + a[1*12 + 8] * x[8];
										sum[8][2] = sum[7][2] + a[2*12 + 8] * x[8];
										sum[8][3] = sum[7][3] + a[3*12 + 8] * x[8];
										sum[8][4] = sum[7][4] + a[4*12 + 8] * x[8];
										sum[8][5] = sum[7][5] + a[5*12 + 8] * x[8];
										sum[8][6] = sum[7][6] + a[6*12 + 8] * x[8];
										sum[8][7] = sum[7][7] + a[7*12 + 8] * x[8];
										sum[8][8] = sum[7][8] + a[8*12 + 8] * x[8];
										sum[8][9] = sum[7][9] + a[9*12 + 8] * x[8];
										
										for (int r10 = 0; r10 < s[9]; ++r10) {
											x[9] = b[27] + r10 * b[29];
											
											q[0] = fabs(sum[8][0] + a[0*12 + 9] * x[9]);
											q[1] = fabs(sum[8][1] + a[1*12 + 9] * x[9]);
											q[2] = fabs(sum[8][2] + a[2*12 + 9] * x[9]);
											q[3] = fabs(sum[8][3] + a[3*12 + 9] * x[9]);
											q[4] = fabs(sum[8][4] + a[4*12 + 9] * x[9]);
											q[5] = fabs(sum[8][5] + a[5*12 + 9] * x[9]);
											q[6] = fabs(sum[8][6] + a[6*12 + 9] * x[9]);
											q[7] = fabs(sum[8][7] + a[7*12 + 9] * x[9]);
											q[8] = fabs(sum[8][8] + a[8*12 + 9] * x[9]);
											q[9] = fabs(sum[8][9] + a[9*12 + 9] * x[9]);
											
											if (q[0] > e[0] || q[1] > e[1] || q[2] > e[2] || q[3] > e[3] ||
												q[4] > e[4] || q[5] > e[5] || q[6] > e[6] || q[7] > e[7] ||
												q[8] > e[8] || q[9] > e[9]) {
												continue;
											}
											pnts = pnts + 1;
											fprintf(fptr, "%lf\t", x[0]);
											fprintf(fptr, "%lf\t", x[1]);
											fprintf(fptr, "%lf\t", x[2]);
											fprintf(fptr, "%lf\t", x[3]);
											fprintf(fptr, "%lf\t", x[4]);
											fprintf(fptr, "%lf\t", x[5]);
											fprintf(fptr, "%lf\t", x[6]);
											fprintf(fptr, "%lf\t", x[7]);
											fprintf(fptr, "%lf\t", x[8]);
											fprintf(fptr, "%lf\n", x[9]);
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}

  fclose(fptr);
  printf("result pnts: %ld\n", pnts);

  // end function gridloopsearch
}
