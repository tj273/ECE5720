#include <stdio.h>
#include <stdlib.h>

#define COL 8

int main()
{
  int i, j;
  int stage;
  int temp_idx;
  int col_arr[COL/2][2];

  for (i = 0; i < COL/2; i++)
  {
    col_arr[i][0] = 2*i+1;
    col_arr[i][1] = 2*i+2;
  }
  
  for (i = 0; i < COL/2; i++)
    printf("[%d %d] ", col_arr[i][0], col_arr[i][1]);
  printf("\n");
  
  for (stage = 0; stage < COL-1; stage++)
  {
    temp_idx = col_arr[0][1];    
    for (j = 1; j < COL/2; j++)
      col_arr[j-1][1] = col_arr[j][1];
        
    col_arr[COL/2-1][1] = col_arr[COL/2-1][0];
    for (j = COL/2-1; j > 1; j--)
      col_arr[j][0] = col_arr[j-1][0];
        
    col_arr[1][0] = temp_idx;
    
    for (i = 0; i < COL/2; i++)
      printf("[%d %d] ", col_arr[i][0], col_arr[i][1]);
    printf("\n");
  }
  
  return 0;
}
