/* Name:  Tianze Jiang
 * Netid: tj273
 *
 * 
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define ROW 1024
#define COL 256
#define MAXSWEEP 100

typedef struct {
  double **pt;        // matrix pointer
  int row;            // row number
  int col;            // column number
} matrix_st;

matrix_st matrixInit(int row, int col);
void matrixFree(matrix_st matA);
matrix_st matrixCopy(matrix_st matA);
matrix_st matrixMult(matrix_st matA, matrix_st matB);
matrix_st matrixTrans(matrix_st matA);
matrix_st getTwoColumn(matrix_st matA, int col0, int col1);
void PrintArray(matrix_st matA);


int main(int argc, char *argv[])
{
  int i, j, k, p, q;
  int stage;
  int num_tasks, my_rank;
  int tag = 1;
  int count;
  int src, dest;

  double start, end;

  MPI_Status rec_stats[4], sen_stats[4];
  MPI_Request rec_reqs[4], sen_reqs[4];

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  int itr = 0;
  double sum, tau, t, c, s, threshold;
  double ac_sum, temp_sum;
  int sign_tau;

  matrix_st A = matrixInit(ROW, COL);
  for (i = 0; i < A.row; i++)
    for (j = 0; j < A.col; j++)
      A.pt[i][j] = i*A.row+j;

  matrix_st S, V;
  matrix_st ST, VT, S_twocol, ST_twocol, V_twocol, a_temp;
  S = matrixCopy(A);

  V = matrixInit(COL, COL);
  for (i = 0; i < V.row; i++)
  {
    for (j = 0; j < V.col; j++)
    {
      if (i == j)
        V.pt[i][j] = 1;
      else
        V.pt[i][j] = 0;
    }
  }

  int col_arr[COL/2][2];
  int temp_idx;
  int buf[2];
  double new_Scol0[ROW], new_Scol1[ROW], new_Vcol0[COL], new_Vcol1[COL];
  double Scol0[ROW], Scol1[ROW], Vcol0[COL], Vcol1[COL];

  int round = COL/2/num_tasks;

  threshold = 0.00002;
  ac_sum = 10;

  i = 0;

  start = MPI_Wtime();

  while ((i < MAXSWEEP) && (sqrt(ac_sum) > threshold))
  {
    //--------initialize parameters for each sweep------------------------ 
    ac_sum = 0;
    sum = 0;
    for (p = 0; p < COL/2; p++)
    {
      col_arr[p][0] = 2*p;
      col_arr[p][1] = 2*p+1;
    }
    //--------------------------------------------------------------------

    for (stage = 0; stage < COL - 1; stage++)
    {
      itr = 0;
      for (itr = 0; itr < round; itr++)
      {
        buf[0] = col_arr[itr*num_tasks+my_rank][0];
        buf[1] = col_arr[itr*num_tasks+my_rank][1];

        S_twocol = getTwoColumn(S, buf[0], buf[1]);
        V_twocol = getTwoColumn(V, buf[0], buf[1]);

        if (stage != 0)
        {
          if      (my_rank==0 && itr==0)
          {
            MPI_Irecv(new_Scol1, ROW, MPI_DOUBLE, my_rank+1, buf[1], MPI_COMM_WORLD, &rec_reqs[0]);
            MPI_Irecv(new_Vcol1, COL, MPI_DOUBLE, my_rank+1, buf[1], MPI_COMM_WORLD, &rec_reqs[1]);
            MPI_Waitall(2, rec_reqs, rec_stats);

            for (q = 0; q < S_twocol.row; q++)
              S_twocol.pt[q][1] = new_Scol1[q];
            for (q = 0; q < V_twocol.row; q++)
              V_twocol.pt[q][1] = new_Vcol1[q];
          }

          else if (my_rank==0 && itr!=0)
          {
            MPI_Irecv(new_Scol0, ROW, MPI_DOUBLE, num_tasks-1, buf[0], MPI_COMM_WORLD, &rec_reqs[0]);
            MPI_Irecv(new_Scol1, ROW, MPI_DOUBLE, my_rank+1,   buf[1], MPI_COMM_WORLD, &rec_reqs[1]);
            MPI_Irecv(new_Vcol0, COL, MPI_DOUBLE, num_tasks-1, buf[0], MPI_COMM_WORLD, &rec_reqs[2]);
            MPI_Irecv(new_Vcol1, COL, MPI_DOUBLE, my_rank+1,   buf[1], MPI_COMM_WORLD, &rec_reqs[3]);
            MPI_Waitall(4, rec_reqs, rec_stats);

            for (q = 0; q < S_twocol.row; q++)
              S_twocol.pt[q][0] = new_Scol0[q];
            for (q = 0; q < S_twocol.row; q++)
              S_twocol.pt[q][1] = new_Scol1[q];
            for (q = 0; q < V_twocol.row; q++)
              V_twocol.pt[q][0] = new_Vcol0[q];
            for (q = 0; q < V_twocol.row; q++)
              V_twocol.pt[q][1] = new_Vcol1[q];
          }

          else if (my_rank!=0 && my_rank!=(num_tasks-1))
          {
            MPI_Irecv(new_Scol0, ROW, MPI_DOUBLE, my_rank-1, buf[0], MPI_COMM_WORLD, &rec_reqs[0]);
            MPI_Irecv(new_Scol1, ROW, MPI_DOUBLE, my_rank+1, buf[1], MPI_COMM_WORLD, &rec_reqs[1]);
            MPI_Irecv(new_Vcol0, COL, MPI_DOUBLE, my_rank-1, buf[0], MPI_COMM_WORLD, &rec_reqs[2]);
            MPI_Irecv(new_Vcol1, COL, MPI_DOUBLE, my_rank+1, buf[1], MPI_COMM_WORLD, &rec_reqs[3]);
            MPI_Waitall(4, rec_reqs, rec_stats);

            for (q = 0; q < S_twocol.row; q++)
              S_twocol.pt[q][0] = new_Scol0[q];
            for (q = 0; q < S_twocol.row; q++)
              S_twocol.pt[q][1] = new_Scol1[q];
            for (q = 0; q < V_twocol.row; q++)
              V_twocol.pt[q][0] = new_Vcol0[q];
            for (q = 0; q < V_twocol.row; q++)
              V_twocol.pt[q][1] = new_Vcol1[q];
          }

          else if (my_rank==(num_tasks-1) && itr!=(round-1))
          {
            MPI_Irecv(new_Scol0, ROW, MPI_DOUBLE, my_rank-1, buf[0], MPI_COMM_WORLD, &rec_reqs[0]);
            MPI_Irecv(new_Scol1, ROW, MPI_DOUBLE, 0,         buf[1], MPI_COMM_WORLD, &rec_reqs[1]);
            MPI_Irecv(new_Vcol0, COL, MPI_DOUBLE, my_rank-1, buf[0], MPI_COMM_WORLD, &rec_reqs[2]);
            MPI_Irecv(new_Vcol1, COL, MPI_DOUBLE, 0,         buf[1], MPI_COMM_WORLD, &rec_reqs[3]);
            MPI_Waitall(4, rec_reqs, rec_stats);

            for (q = 0; q < S_twocol.row; q++)
              S_twocol.pt[q][0] = new_Scol0[q];
            for (q = 0; q < S_twocol.row; q++)
              S_twocol.pt[q][1] = new_Scol1[q];
            for (q = 0; q < V_twocol.row; q++)
              V_twocol.pt[q][0] = new_Vcol0[q];
            for (q = 0; q < V_twocol.row; q++)
              V_twocol.pt[q][1] = new_Vcol1[q];
          }

          else if (my_rank==(num_tasks-1) && itr==(round-1))
          {
            MPI_Irecv(new_Scol0, ROW, MPI_DOUBLE, my_rank-1, buf[0], MPI_COMM_WORLD, &rec_reqs[0]);
            MPI_Irecv(new_Vcol0, COL, MPI_DOUBLE, my_rank-1, buf[0], MPI_COMM_WORLD, &rec_reqs[1]);
            MPI_Waitall(2, rec_reqs, rec_stats);

            for (q = 0; q < S_twocol.row; q++)
              S_twocol.pt[q][1] = S_twocol.pt[q][0];
            for (q = 0; q < S_twocol.row; q++)
              S_twocol.pt[q][0] = new_Scol0[q];
            for (q = 0; q < V_twocol.row; q++)
              V_twocol.pt[q][1] = V_twocol.pt[q][0];
            for (q = 0; q < V_twocol.row; q++)
              V_twocol.pt[q][0] = new_Vcol0[q];
          }
        }

        ST_twocol = matrixTrans(S_twocol);
        a_temp = matrixMult(ST_twocol, S_twocol);

        sum = sum + 2*a_temp.pt[0][1]*a_temp.pt[0][1];
        tau = (a_temp.pt[1][1] - a_temp.pt[0][0])/(2*a_temp.pt[0][1]);
        sign_tau = (tau > 0) ? 1 : -1;

        t = 1/(tau + sign_tau*sqrt(1+tau*tau));
        c = 1/sqrt(1+t*t);
        s = c * t;

        for (p = 0; p < S.row; p++)
        {
          S.pt[p][buf[0]] = S_twocol.pt[p][0]*c - S_twocol.pt[p][1]*s;
          S.pt[p][buf[1]] = S_twocol.pt[p][0]*s + S_twocol.pt[p][1]*c;
        }

        for (p = 0; p < V.row; p++)
        {
          V.pt[p][buf[0]] = V_twocol.pt[p][0]*c - V_twocol.pt[p][1]*s;
          V.pt[p][buf[1]] = V_twocol.pt[p][0]*s + V_twocol.pt[p][1]*c;
        }
        if (stage != 0)
        {
          if ((my_rank==0 && itr==0) || (my_rank==(num_tasks-1) && itr==(round-1)))
            MPI_Waitall(2, sen_reqs, sen_stats);
          else
            MPI_Waitall(4, sen_reqs, sen_stats);
        }

        for (q = 0; q < S_twocol.row; q++)
          Scol0[q] = S_twocol.pt[q][0];
        for (q = 0; q < S_twocol.row; q++)
          Scol1[q] = S_twocol.pt[q][1];
        for (q = 0; q < V_twocol.row; q++)
          Vcol0[q] = V_twocol.pt[q][0];
        for (q = 0; q < V_twocol.row; q++)
          Vcol1[q] = V_twocol.pt[q][1];
        /*
        if (my_rank == 0)
        {
          printf("S=\n");
          PrintArray(S);
          printf("V=\n");
          PrintArray(V);
        }
        */
        matrixFree(S_twocol);
        matrixFree(V_twocol);
        matrixFree(ST_twocol);
        matrixFree(a_temp);

        if ( (my_rank!=0) && (my_rank!=num_tasks-1) )
        {
        //--------S columns------------------------------------------------
          MPI_Isend(Scol0, ROW, MPI_DOUBLE, my_rank+1, buf[0], MPI_COMM_WORLD, &sen_reqs[0]);
          MPI_Isend(Scol1, ROW, MPI_DOUBLE, my_rank-1, buf[1], MPI_COMM_WORLD, &sen_reqs[1]);
        //-----------------------------------------------------------------
        //--------V columns------------------------------------------------
          MPI_Isend(Vcol0, COL, MPI_DOUBLE, my_rank+1, buf[0], MPI_COMM_WORLD, &sen_reqs[2]);
          MPI_Isend(Vcol1, COL, MPI_DOUBLE, my_rank-1, buf[1], MPI_COMM_WORLD, &sen_reqs[3]);
        //-----------------------------------------------------------------
        }

        else if ( my_rank==0 && itr==0 )
        {
        //--------S columns------------------------------------------------
          MPI_Isend(Scol1, ROW, MPI_DOUBLE, my_rank+1, buf[1], MPI_COMM_WORLD, &sen_reqs[0]);
        //-----------------------------------------------------------------
        //--------V columns------------------------------------------------
          MPI_Isend(Vcol1, COL, MPI_DOUBLE, my_rank+1, buf[1], MPI_COMM_WORLD, &sen_reqs[1]);
        //-----------------------------------------------------------------
        }

        else if ( my_rank==0 && itr!=0 )
        {
        //--------S columns------------------------------------------------
          MPI_Isend(Scol0, ROW, MPI_DOUBLE, my_rank+1,   buf[0], MPI_COMM_WORLD, &sen_reqs[0]);
          MPI_Isend(Scol1, ROW, MPI_DOUBLE, num_tasks-1, buf[1], MPI_COMM_WORLD, &sen_reqs[1]);
        //-----------------------------------------------------------------
        //--------V columns------------------------------------------------
          MPI_Isend(Vcol1, COL, MPI_DOUBLE, my_rank+1,   buf[0], MPI_COMM_WORLD, &sen_reqs[2]);
          MPI_Isend(Vcol1, COL, MPI_DOUBLE, num_tasks-1, buf[1], MPI_COMM_WORLD, &sen_reqs[3]);
        //-----------------------------------------------------------------
        }

        else if ( my_rank==(num_tasks-1) && itr!=(round-1) )
        {
        //--------S columns------------------------------------------------
          MPI_Isend(Scol0, ROW, MPI_DOUBLE, 0,         buf[0], MPI_COMM_WORLD, &sen_reqs[0]);
          MPI_Isend(Scol1, ROW, MPI_DOUBLE, my_rank-1, buf[1], MPI_COMM_WORLD, &sen_reqs[1]);
        //-----------------------------------------------------------------
        //--------V columns------------------------------------------------
          MPI_Isend(Vcol0, COL, MPI_DOUBLE, 0,         buf[0], MPI_COMM_WORLD, &sen_reqs[2]);
          MPI_Isend(Vcol1, COL, MPI_DOUBLE, my_rank-1, buf[1], MPI_COMM_WORLD, &sen_reqs[3]);
        //-----------------------------------------------------------------
        }

        else if ( my_rank==(num_tasks-1) && itr==(round-1) )
        {
        //--------S columns------------------------------------------------
          MPI_Isend(Scol1, ROW, MPI_DOUBLE, my_rank-1, buf[1], MPI_COMM_WORLD, &sen_reqs[0]);
        //-----------------------------------------------------------------
        //--------V columns------------------------------------------------
          MPI_Isend(Vcol1, COL, MPI_DOUBLE, my_rank-1, buf[1], MPI_COMM_WORLD, &sen_reqs[1]);
        //-----------------------------------------------------------------
        }
        /*
        if (my_rank == 0)
        {
          printf("S=\n");
          PrintArray(S);
          printf("V=\n");
          PrintArray(V);
        }
        */
      }
      //------Column Rotation between stages------------------------------
      temp_idx = col_arr[0][1];
      for (j = 1; j < COL/2; j++)
        col_arr[j-1][1] = col_arr[j][1];

      col_arr[COL/2-1][1] = col_arr[COL/2-1][0];
      for (j = COL/2-1; j > 1; j--)
        col_arr[j][0] = col_arr[j-1][0];

      col_arr[1][0] = temp_idx;
      //------------------------------------------------------------------
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(&sum, &temp_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    ac_sum = ac_sum + temp_sum;

    i = i + 1;
  }

  end = MPI_Wtime();

  if (my_rank == 0)
  {
    printf("A =\n");
    PrintArray(A);
    printf("S =\n");
    PrintArray(S);
    printf("V =\n");
    PrintArray(V);
  }

  MPI_Finalize();
  printf("Time elapsed to calculate SVD for a %d * %d matrix is %lf s\n", ROW, COL, end-start);

  matrixFree(S);
  matrixFree(V);

  matrixFree(A);

  return 0;
}


matrix_st matrixInit(int row, int col)
{
  int i;
  matrix_st matA;
  matA.row = row;
  matA.col = col;

  matA.pt = (double **)malloc(matA.row * sizeof(double *));
  for (i = 0; i < matA.row; i++)
    matA.pt[i] = (double *)malloc(matA.col * sizeof(double));

  return matA;
}

void matrixFree(matrix_st matA)
{
  int i;
  for (i = 0; i < matA.row; i++)
    free(matA.pt[i]);
  free(matA.pt);
}

matrix_st matrixCopy(matrix_st matA)
{
  int i, j;
  matrix_st matCopy = matrixInit(matA.row, matA.col);
  for (i = 0; i < matCopy.row; i++)
    for (j = 0; j < matCopy.col; j++)
      matCopy.pt[i][j] = matA.pt[i][j];

  return matCopy;
}

matrix_st matrixMult(matrix_st matA, matrix_st matB)
{
  int i, j, k;
  double sum_temp = 0;
  matrix_st mult_mat = matrixInit(matA.row, matB.col);

  for (i = 0; i < mult_mat.row; i++)
  {
    for (j = 0; j < mult_mat.col; j++)
    {
      sum_temp = 0;
      for (k = 0; k < matA.col; k++)
        sum_temp = sum_temp + matA.pt[i][k]*matB.pt[k][j];
      mult_mat.pt[i][j] = sum_temp;
    }
  }

  return mult_mat;
}

matrix_st matrixTrans(matrix_st matA)
{
  int i, j;
  matrix_st matAT = matrixInit(matA.col, matA.row);

  for (i = 0; i < matAT.row; i++)
    for (j = 0; j < matAT.col; j++)
      matAT.pt[i][j] = matA.pt[j][i];

  return matAT;
}

matrix_st getTwoColumn(matrix_st matA, int col0, int col1)
{
  int i;
  matrix_st matcol = matrixInit(matA.row, 2);

  for (i = 0; i < matcol.row; i++)
      matcol.pt[i][0] = matA.pt[i][col0];

  for (i = 0; i < matcol.row; i++)
      matcol.pt[i][1] = matA.pt[i][col1];

  return matcol;
}

void PrintArray(matrix_st matA)
{
  int i, j;
  for ( i = 0; i < matA.row; i++ )
  {
    for ( j = 0; j < matA.col; j++ )
      printf ("%2.4lf  ", matA.pt[i][j]);
    printf("\n");
  }
  printf("\n");
  
  return;
}
