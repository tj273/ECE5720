/* ECE 5720 hw1 problem 2
 * Name  : Tianze Jiang
 * Netid : tj273
*/

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
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
  fp = fopen("p2_data.txt", "w");
  
  int n, s, k, i;
  float test = 0;
  
  uint64_t diff;
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
      diff = (BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec)/(n*K);
      printf("n = %d, s = %d, elapsed time = %llu nanoseconds\n", n, s, (long long unsigned int) diff);
      fprintf(fp, "n = %d, s = %d, elapsed time = %llu nanoseconds\n", n, s, (long long unsigned int) diff);
    }
    
  free(A);
  
  fclose(fp);
  
  return 0;
}
