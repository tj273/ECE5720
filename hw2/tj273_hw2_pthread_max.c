/* ECE 5720 hw2 pthread
 * Name  : Tianze Jiang
 * Netid : tj273
 * 
 * % gcc tj273_hw2_pthread_max.c -O3 -o tj273_pthread_max -lpthread
 * % ./tj273_pthread_max
*/

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

#define N 32768
#define NUM_THREADS 2
#define BILLION 1000000000L

void RowSwitch_pthread(int **A, int row);
void *FindMax(void *thr_arg);
void RowSwitch_sequential(int **A, int row);
void PrintArray(int **A);

struct thread_data {
	int st;
	int ed;
  int column_num;
};
struct thread_data thread_data_array[NUM_THREADS];

uint64_t time_findmaxp = 0;
uint64_t time_findmaxs = 0;
int maxrow= 0;

pthread_t pthrd_id[NUM_THREADS]; 
pthread_mutex_t mutexcomp;

int **A;

void main()
{
	int i, j;
  
  A = (int **) malloc (N * sizeof(int *));
  for ( i = 0; i < N; i++ )
    A[i] = (int *) malloc (N * sizeof(int));
    
  srand(3);
  
  for ( i = 0; i < N; i++ )
  	for ( j = 0; j < N; j++ )
  		A[i][j] = rand()%100 - 50;
  
  //PrintArray(A);
  
  for (i = 0; i < N-1; i++ )
    RowSwitch_pthread( A, i );
  
  //PrintArray(A);
  
  printf("Time elapsed to find all pivots in entire matrix    with pthread is %15llu ns\n", (long long unsigned)time_findmaxp);
  
  for ( i = 0; i < N; i++ )
    for ( j = 0; j < N; j++ )
      A[i][j] = rand()%100 - 50;

  //PrintArray(A);
  
  for (i = 0; i < N-1; i++ )
    RowSwitch_sequential( A, i );
  
  //PrintArray(A);
  
  printf("Time elapsed to find all pivots in entire matrix without pthread is %15llu ns\n", (long long unsigned)time_findmaxs);
  
  for ( i = 0; i < N; i++ )
    free(A[i]);
  
  free(A);
  
  return;
}

void *FindMax(void *thr_arg) {
    int i, j;
    int max_row_num;                            // in thread local index of row that holds maximum absolute value in the column
    
    struct thread_data *thr_data;
    thr_data = (struct thread_data *) thr_arg;
    
    max_row_num = thr_data->st;

    for (i = thr_data->st; i <= thr_data->ed; i++)  // compare and get the index of row for max abs value
      if ( abs(A[i][thr_data->column_num]) > abs(A[max_row_num][thr_data->column_num]) )
        max_row_num = i;
    
    pthread_mutex_lock (&mutexcomp);           // lock mutex before writing shared area
    if ( abs(A[max_row_num][thr_data->column_num]) > abs(A[maxrow][thr_data->column_num]) )
      maxrow = max_row_num;
    pthread_mutex_unlock (&mutexcomp);         // unlock mutex
    
    pthread_exit(NULL);
}

void RowSwitch_pthread(int **A, int row) {
  int i;
  int thread_num = NUM_THREADS;
  int max_row_num;
  void *status;
  
  int temp;
  
  while ((N-row)/thread_num < 2)
    thread_num = thread_num-1;
  
  pthread_attr_t attr; 

  uint64_t diff;
  struct timespec start, end;
  
  maxrow = row;
  
  pthread_mutex_init(&mutexcomp, NULL);
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
  
  clock_gettime(CLOCK_MONOTONIC, &start);	/* mark start time */
  
  for ( i = 0; i < thread_num; i++ ) {
    // generate data for argument struct for thread
    thread_data_array[i].st = row + i * (N-row)/thread_num;
    thread_data_array[i].ed = thread_data_array[i].st + (N-row)/thread_num -1;
    if ( i >= thread_num - (N-row)%thread_num )
      thread_data_array[i].ed = thread_data_array[i].ed + 1;
      
    thread_data_array[i].column_num = row;
    
    int err = pthread_create(&pthrd_id[i], &attr, FindMax, (void *) &thread_data_array[i]);
    //printf("err is %d\n", err);
  }
  
  pthread_attr_destroy(&attr);
  
  // wait all thread to terminate
  for(i = 0; i < thread_num; i++)
  {
    pthread_join(pthrd_id[i], &status);
  }
  
  clock_gettime(CLOCK_MONOTONIC, &end);	/* mark the end time */
  diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
  time_findmaxp = time_findmaxp + diff;
  
  pthread_mutex_destroy(&mutexcomp);
  
  // do permutation
  if ( row != maxrow )
    for ( i = 0; i < N; i++ ) {
     	temp = A[row][i];
      A[row][i] = A[maxrow][i];
    	A[maxrow][i] = temp;
    }
  
  return;
}

void RowSwitch_sequential(int **A, int row) {
  int i = 0;
  int max_row_num = row;
  
  int temp;

  uint64_t diff;
  struct timespec start, end;
    
  clock_gettime(CLOCK_MONOTONIC, &start);	/* mark start time */

  for ( i = row; i < N; i++ )
    if ( abs(A[i][row]) > abs(A[max_row_num][row]) )
      max_row_num = i;

  clock_gettime(CLOCK_MONOTONIC, &end);	/* mark the end time */
  diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
  time_findmaxs = time_findmaxs + diff;

  if (row != max_row_num )
    for ( i = 0; i < N; i++ ) {
      temp = A[row][i];
      A[row][i] = A[max_row_num][i];
      A[max_row_num][i] = temp;
    }

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
