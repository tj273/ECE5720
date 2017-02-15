/* ECE 5720 hw1 problem 2
 * Name  : Tianze Jiang
 * Netid : tj273
*/

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_LENGTH 67108864
#define MIN_SIZE 1024
#define K 10
#define BILLION 1000000000L

int main()
{
  float * A;
  A = (float *)malloc(sizeof(float) * MAX_LENGTH);
  
  FILE *fp;
  fp = fopen("p2_data", "w");
  int n, s, k, i;
  float test = 0;

  double diff;
  struct timespec start, end;
  
  for (i = 0; i < MAX_LENGTH; i++)
    A[i] = ((float)rand()/(float)(RAND_MAX));
  
  for (n = MIN_SIZE; n <= MAX_LENGTH; n = n * 2)
    for (s = 1; s <= n/2; s = s * 2){
      clock_gettime(CLOCK_MONOTONIC, &start);
      for (k = 0; k < s*K; k++)
        for (i = 0; i < n; i = i + s)
          test = A[i];
      clock_gettime(CLOCK_MONOTONIC, &end);
      diff = (BILLION * ((double)end.tv_sec - (double)start.tv_sec) + (double)end.tv_nsec - (double)start.tv_nsec)/(n*K);
      printf("n = %8d, s = %8d, elapsed time = %2.4lf nanoseconds\n", n, s, diff);
      fprintf(fp, "%d, %d, %lf\n", n, s, diff);
      }
    
  free(A);
  fclose(fp);
  
  return 0;
}

