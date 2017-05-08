/* Name:  Tianze Jiang
 * Netid: tj273
 *
 * % mpicc -o part_I tj273_hw3_part_I.c
 * % mpirun -mca plm_rsh_no_tree_spawn 1 --hostfile my_hostfile -np N ./part_I
 */

#include <mpi.h>
#include <stdio.h>

#define ITR    50
#define LENGTH 1<<20

int main(int argc, char *argv[])
{
  int  num_tasks, my_rank;
  int  tag = 1;
  int  count;
  int  src, dest;

  double start_time, end_time, total_time;
  total_time = 0;

  //char *msgin  = (char *)malloc(LENGTH * sizeof(char));
  //char *msgout = (char *)malloc(LENGTH * sizeof(char));

  char msgin[LENGTH];
  char msgout[LENGTH];

  MPI_Status status;

  int i;
  for (i = 0; i < LENGTH; i++)
    msgin[i] = 'a';

  printf("Message array length: %d\n", LENGTH);

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  printf("num_task = %d\n", num_tasks);

  if (my_rank == 0)
  {
  //--------------------------------------------------------------------------------------------------------
  // 1 for within 1 socket
  // 18 for between sockets in one node
  // 38 for between nodes
  //--------------------------------------------------------------------------------------------------------
    dest = 38;
    src  = 38;

    for (i = 0; i < ITR; i++)
    {
      start_time = MPI_Wtime();
      MPI_Send(msgout, LENGTH, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
      MPI_Recv(msgin,  LENGTH, MPI_CHAR, src,  tag, MPI_COMM_WORLD, &status);
      end_time   = MPI_Wtime();
      printf("Task %d: run %2d: Time elapsed for ping-pong: %1.6lf s\n", my_rank, i+1, end_time-start_time);
      total_time = total_time + end_time-start_time;
    }
    printf("\nTask %d: Average time elapsed for %2d run is: %1.6lf s\n", my_rank, ITR, total_time/ITR);
  }
  //--------------------------------------------------------------------------------------------------------
  // 1 for within 1 socket
  // 18 for between sockets in one node
  // 38 for between nodes
  //--------------------------------------------------------------------------------------------------------
  else if (my_rank == 38)
  {
    dest = 0;
    src  = 0;

    for (i = 0; i < ITR; i++)
    {
      MPI_Recv(msgin,  LENGTH, MPI_CHAR, src,  tag, MPI_COMM_WORLD, &status);
      MPI_Send(msgin,  LENGTH, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
    }
  }

  MPI_Get_count(&status, MPI_CHAR, &count);
  /*printf("Task %d: Received %d char(s) from task %d with tag %d \n",
          my_rank, count, status.MPI_SOURCE, status.MPI_TAG);
  */
  MPI_Finalize();
  //free(msgin);
  //free(msgout);
  return 0;
}

              
