/* Name:  Tianze Jiang
 * Netid: tj273
 *
 * 
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include <mpi.h>

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
  int i, j, k, p;
  matrix_st A = matrixInit(ROW, COL);
  
  for (i = 0; i < A.row; i++)
    for (j = 0; j < A.col; j++)
      A.pt[i][j] = i*A.col + j;

  PrintArray(A);
  
  double sum, tau, t, c, s, threshold;
  int sign_tau;
  matrix_st S, V, a_temp, S_twocol, ST_twocol, V_twocol;

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

  threshold = 0.00002;
  sum = 10;

  i = 0;

  while ((i < MAXSWEEP) && (sum > threshold))
  {
    sum = 0;

    for (j = 0; j < COL-1; j++)
    {
      for (k = j+1; k < COL; k++)
      {
        S_twocol = getTwoColumn(S, j, k);
        V_twocol = getTwoColumn(V, j, k);
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
          S.pt[p][j] = S_twocol.pt[p][0]*c - S_twocol.pt[p][1]*s;
          S.pt[p][k] = S_twocol.pt[p][0]*s + S_twocol.pt[p][1]*c;
        }

        for (p = 0; p < V.row; p++)
        {
          V.pt[p][j] = V_twocol.pt[p][0]*c - V_twocol.pt[p][1]*s;
          V.pt[p][k] = V_twocol.pt[p][0]*s + V_twocol.pt[p][1]*c;
        }
        
        matrixFree(V_twocol);
        matrixFree(ST_twocol);
        matrixFree(S_twocol);
        matrixFree(a_temp);
      }
    }
    printf("sum[%d] = %lf\n", i, sum);
    i = i + 1;
  }
  
  printf("S = \n");
  PrintArray(S);
  //PrintArray(V);

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

