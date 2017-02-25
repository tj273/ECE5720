/* ECE 5720 hw2 openmp
 * Name  : Tianze Jiang
 * Netid : tj273
 * 
 * % gcc tj273_hw2_openmp_sort.c -o tj273_openmp_sort -fopenmp
 * % ./tj273_openmp_sort
*/

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define N 4
#define NUM_THREADS 8
#define BILLION 1000000000L

void Merge(int *array,int *left_array,int leftCount,int *right_array,int rightCount, int *idx, int *left_idx, int *right_idx);
void MergeSort(int *array, int *idx, int n);
void MergeSort_omp(int *array, int *idx, int n, int ava_num_threads);
void RowSort_omp(int **A, int row);
void RowSort_sequential(int **A, int row);
void PrintArray(int **A);

uint64_t time_sorto = 0;
uint64_t time_sorts = 0;
int **A;
int **temp;

void main()
{
  int i, j;
  
  A = (int **) malloc (N * sizeof(int *));
  for ( i = 0; i < N; i++ )
    A[i] = (int *) malloc (N * sizeof(int));
    
  temp = (int **) malloc (N * sizeof(int *));
  for ( i = 0; i < N; i++ )
    temp[i] = (int *) malloc (N * sizeof(int));
    
  srand(3);
  for ( i = 0; i < N; i++ )
  	for ( j = 0; j < N; j++ )
  		A[i][j] = rand()%100 - 50;
      
  PrintArray(A);
  
  for (i = 0; i < N-1; i++ )
    RowSort_omp( A, i );
  
  PrintArray(A);
  
  printf("Time elapsed to sort all columns in entire matrix    with openmp is %15llu ns\n", (long long unsigned)time_sorto);
  
  for ( i = 0; i < N; i++ )
    for ( j = 0; j < N; j++ )
      A[i][j] = rand()%100 - 50;

  PrintArray(A);
  
  for (i = 0; i < N-1; i++ )
    RowSort_sequential( A, i );
  
  PrintArray(A);
  
  printf("Time elapsed to sort all columns in entire matrix without openmp is %15llu ns\n", (long long unsigned)time_sorts);

  for ( i = 0; i < N; i++ )
    free(A[i]);
  
  free(A);

  return;
}

void Merge(int *array,int *left_array, int leftCount, int *right_array, int rightCount, int *idx, int *left_idx, int *right_idx)
{
	int lp,rp,ap;
  
	lp = 0; // left  array pointer
  rp = 0; // right array pointer
  ap = 0; // new   array pointer


	while( (lp < leftCount) && (rp < rightCount)) {
		if(left_array[lp] < right_array[rp]) {
      array[ap] = left_array[lp];
      idx[ap] = left_idx[lp];
      ap = ap + 1;
      lp = lp + 1;
    }
		else {
      array[ap] = right_array[rp];
      idx[ap] = right_idx[rp];
      ap = ap + 1;
      rp = rp + 1;
    }
	}
  
	while(lp < leftCount) {
    array[ap] = left_array[lp];
    idx[ap] = left_idx[lp];
    ap = ap + 1;
    lp = lp + 1;
  }
	while(rp < rightCount) {
    array[ap] = right_array[rp];
    idx[ap] = right_idx[rp];
    ap = ap + 1;
    rp = rp + 1;
  }
}

void MergeSort(int *array, int *idx, int n)
{
	int mid, i;
  int *left_array, *right_array;
  int *left_idx, *right_idx;
	if(n < 2) 
    return; 

	mid = n/2;  // find the mid index. 

	left_array = (int*)malloc(mid*sizeof(int)); 
	right_array = (int*)malloc((n- mid)*sizeof(int)); 
  
  left_idx = (int*)malloc(mid*sizeof(int)); 
	right_idx = (int*)malloc((n- mid)*sizeof(int)); 
	
	for(i = 0;i<mid;i++) {
    left_array[i] = array[i]; // creating left subarray
    left_idx[i] = idx[i];
  }
  
	for(i = mid;i<n;i++) {
    right_array[i-mid] = array[i]; // creating right subarray
    right_idx[i-mid] = idx[i];
  }
  
  MergeSort(left_array, left_idx, mid);  // sorting the left subarray
  MergeSort(right_array, right_idx, n-mid);  // sorting the right subarray
  
	Merge(array, left_array, mid, right_array, n-mid, idx, left_idx, right_idx);  // Merging L and R into A as sorted list.
  
  free(left_array);
  free(right_array);
  
  free(left_idx);
  free(right_idx);
}

void MergeSort_omp(int *array, int *idx, int n, int ava_num_threads)
{ 
  if (ava_num_threads < 2) {
    MergeSort(array, idx, n);
    return;
  }
     
	int mid, i;
  int *left_array, *right_array;
  int *left_idx, *right_idx;
	if(n < 2) 
    return; 

	mid = n/2;  // find the mid index. 

	left_array = (int*)malloc(mid*sizeof(int)); 
	right_array = (int*)malloc((n- mid)*sizeof(int)); 
  
  left_idx = (int*)malloc(mid*sizeof(int)); 
	right_idx = (int*)malloc((n- mid)*sizeof(int)); 
	
	for(i = 0;i<mid;i++) {
    left_array[i] = array[i]; // creating left subarray
    left_idx[i] = idx[i];
  }
  
	for(i = mid;i<n;i++) {
    right_array[i-mid] = array[i]; // creating right subarray
    right_idx[i-mid] = idx[i];
  }
  
  omp_set_dynamic(0);
  omp_set_nested(1);
  
  #pragma omp parallel sections 
  {
    #pragma omp section
    MergeSort_omp(left_array, left_idx, mid, ava_num_threads/2);  // sorting the left subarray
    #pragma omp section
    MergeSort_omp(right_array, right_idx, n-mid, ava_num_threads-ava_num_threads/2);  // sorting the right subarray
  }
  
  Merge(array, left_array, mid, right_array, n-mid, idx, left_idx, right_idx);  // Merging L and R into A as sorted list.
  
  free(left_array);
  free(right_array);
  
  free(left_idx);
  free(right_idx);
  
  return;
}

void RowSort_omp(int **A, int row)
{
  int i, j;
  
  int idx[N-row];
  for ( i = 0; i < N-row; i++ )
    idx[i] = i + row;
  
  int array[N-row];
  for ( i = 0; i < N-row; i++ )
    array[i] = A[i+row][row];
  
  uint64_t diff;
  struct timespec start, end;
  
  int available_threads;
  #pragma omp parallel
  {
    #pragma omp master
    {
      available_threads = omp_get_num_threads();
    }
  }
    
  clock_gettime(CLOCK_MONOTONIC, &start);	/* mark start time */
  
  MergeSort_omp(array, idx, N-row, available_threads);
  
  clock_gettime(CLOCK_MONOTONIC, &end);	/* mark the end time */
  diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
  time_sorto = time_sorto + diff;
  
  for ( i = row; i < N; i++ )
  	for ( j = 0; j < N; j++ )
  		temp[i][j] = A[i][j]; 
  
  for ( i = row; i < N; i++ )
  	for ( j = 0; j < N; j++ )
  		A[i][j] = temp[idx[i-row]][j];
    
  return;
}

void RowSort_sequential(int **A, int row)
{
  int i, j;
  
  int idx[N-row];
  for ( i = 0; i < N-row; i++ )
    idx[i] = i + row;
  
  int array[N-row];
  for ( i = 0; i < N-row; i++ )
    array[i] = A[i+row][row];
  
  uint64_t diff;
  struct timespec start, end;
    
  clock_gettime(CLOCK_MONOTONIC, &start);	/* mark start time */
  
  MergeSort(array, idx, N-row);
  
  clock_gettime(CLOCK_MONOTONIC, &end);	/* mark the end time */
  diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
  time_sorts = time_sorts + diff;
  
  for ( i = row; i < N; i++ )
  	for ( j = 0; j < N; j++ )
  		temp[i][j] = A[i][j]; 
  
  for ( i = row; i < N; i++ )
  	for ( j = 0; j < N; j++ )
  		A[i][j] = temp[idx[i-row]][j];
    
  return;
}

void PrintArray(int **A)
{
  int i,j;
  
  for ( i = 0; i < N; i++ ) {
  	for ( j = 0; j < N; j++ )
  		printf ("%3d  ", A[i][j]); 
    printf("\n");
  }
  printf("\n");
  
  return;
}
