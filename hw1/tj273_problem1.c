/* ECE 5720 hw1 problem 1
 * Name  : Tianze Jiang
 * Netid : tj273
*/

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#define BILLION 1000000000L
#define N 1048576

void centroid_time_test(int k);


int main(int argc, char **argv)
{
  centroid_time_test(15);
  centroid_time_test(17);
  centroid_time_test(64);
  
  return 0;
}


void centroid_time_test(int k)
{
  int i, j, t;
  
  // allocate memory space for array x
  float *x;
  x = (float *)malloc(sizeof(float) * N * k);
  
  //initialize elements in array x with random numbers
  for (i = 0; i < N; i++){
    for (j = 0; j < k; j++){
      x[i*k+j] = ((float)rand()/(float)(RAND_MAX));
    }
  }
  
  // allocate memory space for sum array c
  float *c;
  c = (float *)malloc(sizeof(float) * k);
  
  // variables used for time measurement
  uint64_t diff;
  uint64_t mean;
  struct timespec start, end;
  
  printf("test for k = %d\n", k);
  /* measure monotonic time for approach 1 */
  for (t = 0; t < 10; t++){
    
    // initialize elements of array c with zero
    for (i = 0; i < k; i++)
      c[i] = 0;
      
	  clock_gettime(CLOCK_MONOTONIC, &start);	/* mark start time */
  
    for (i = 0; i < N; i++)
      for (j = 0; j < k; j++)
        c[j] = c[j] + x[i*k+j];
  
	  clock_gettime(CLOCK_MONOTONIC, &end);	/* mark the end time */

	  diff = diff + BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
  }
  mean = diff/10;
  printf("approach 1 time = %llu nanoseconds\n", (long long unsigned int) mean);
  

  
  /* measure monotonic time for approach 2 */
  for (t = 0; t < 10; t++){
      
    // initialize elements of array c with zero
    for (i = 0; i < k; i++)
      c[i] = 0;
    
    clock_gettime(CLOCK_MONOTONIC, &start);	/* mark start time */
  
    for (j = 0; j < N; j++)
      for (i = 0; i < k; i++)
        c[i] = c[i] + x[i*k+j];
  
	  clock_gettime(CLOCK_MONOTONIC, &end);	/* mark the end time */

	  diff = diff + BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
  }
  mean = diff/10;
  printf("approach 2 time = %llu nanoseconds\n\n", (long long unsigned int) mean);
  
  free(x);
  free(c);
  
  return;
}
