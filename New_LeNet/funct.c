
void CONVOLUTE_VALID(double **input, double **output, double **weight, int size_o0, int size_o1, int size_w0, int size_w1)
{
  //int o0, o1, w0, w1;
  //for (o0 = 0; o0 < GETLENGTH(output); ++o0)
    //for (o1 = 0; o1 < GETLENGTH(*(output)); ++o1)
      //for (w0 = 0; w0 < GETLENGTH(weight); ++w0)
        //for (w1 = 0; w1 < GETLENGTH(*(weight)); ++w1)
  FOREACH(o0,size_o0)
		FOREACH(o1,size_o1)
			FOREACH(w0,size_w0)
				FOREACH(w1,size_w1)
          output[o0][o1] += input[o0+w0][o1+w1] * weight[w0][w1];
}

void CONVOLUTE_FULL(double **input, double **output, double **weight)
{
  int i0, i1, w0, w1;
  for (i0 = 0; i0 < GETLENGTH(input); ++i0)
    for (i1 = 0; i1 < GETLENGTH(*(input)); ++i1)
      for (w0 = 0; w0 < GETLENGTH(weight); ++w0)
        for (w1 = 0; w1 < GETLENGTH(*(weight)); ++w1)
          (output)[i0 + w0][i1 + w1] += (input)[i0][i1] * (weight)[w0][w1];
}

void CONVOLUTION_FORWARD(double ***input, double ***output, double ****weight, double *bias, double(*action)(double))
{
  int x, y, i, j;
  for (int x = 0; x < GETLENGTH(weight); ++x)
		for (int y = 0; y < GETLENGTH(*weight); ++y)
			CONVOLUTE_VALID(input[x], output[y], weight[x][y]);
  
  for(j = 0; j < GETLENGTH(output); ++j)
    for (i = 0; i < GETCOUNT(output[j]); ++i)
		  ((double *)output[j])[i] = action(((double *)output[j])[i] + bias[j]);
}

__global__ void CONVOLUTION_FORWARD(double ***input, double ***output, double ****weight, double *bias, double(*action)(double))
{
  __shared__ int temp[][][];
  int x, y, i, j;
  for (int x = 0; x < GETLENGTH(weight); ++x)
		for (int y = 0; y < GETLENGTH(*weight); ++y)
			CONVOLUTE_VALID(input[x], output[y], weight[x][y]);
  
  for(j = 0; j < GETLENGTH(output); ++j)
    for (i = 0; i < GETCOUNT(output[j]); ++i)
		  ((double *)output[j])[i] = action(((double *)output[j])[i] + bias[j]);
}

void CONVOLUTION_BACKWARD(double ***input, double ***inerror, double ***outerror, double ****weight, double ****wd, double *bd, double(*actiongrad)(double))
{
  int x, y, i, j;
  for (int x = 0; x < GETLENGTH(weight); ++x)
		for (int y = 0; y < GETLENGTH(*weight); ++y)
			CONVOLUTE_FULL(outerror[y], inerror[x], weight[x][y]);
	for (i = 0; i < GETCOUNT(inerror); ++i)
		((double *)inerror)[i] *= actiongrad(((double *)input)[i]);
	for (j = 0; j < GETLENGTH(outerror); ++j)
		for (i = 0; i < GETCOUNT(outerror[j]); ++i)
		  bd[j] += ((double *)outerror[j])[i];
	for (x = 0; x < GETLENGTH(weight); ++x)
		for (y = 0; y < GETLENGTH(*weight); ++y)
      CONVOLUTE_VALID(input[x], wd[x][y], outerror[y]);
}

void SUBSAMP_MAX_FORWARD(double ***input, double ***output)
{
  int i, o0, o1, l0, l1;
  const int len0 = GETLENGTH(*(input)) / GETLENGTH(*(output));
	const int len1 = GETLENGTH(**(input)) / GETLENGTH(**(output));
	for(i = 0; i < GETLENGTH(output); ++i)
	  for(o0 = 0; o0 < GETLENGTH(*(output)); ++o0)
	    for(o1 = 0; o1 < GETLENGTH(**(output)); ++o1)
	    {
		    int x0 = 0, x1 = 0, ismax;
		    for(l0 = 0; l0 < len0; ++l0)
			    for(l1 = 0; l1 < len1; ++l1)
		      {
			      ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1];
			      x0 += ismax * (l0 - x0);
			      x1 += ismax * (l1 - x1);
		      }
		    output[i][o0][o1] = input[i][o0*len0 + x0][o1*len1 + x1];
	    }
}

void SUBSAMP_MAX_BACKWARD(double ***input, double ***inerror, double ***outerror)
{
  int i, o0, o1, l0, l1;
  const int len0 = GETLENGTH(*(inerror)) / GETLENGTH(*(outerror));
	const int len1 = GETLENGTH(**(inerror)) / GETLENGTH(**(outerror));
	for (i = 0; i < GETLENGTH(outerror); ++i)
	  for(o0 = 0; o0 < GETLENGTH(*(outerror)); ++o0)
	    for(o1 = 0; o1 < GETLENGTH(**(outerror)); ++o1)
	    {
		    int x0 = 0, x1 = 0, ismax;
		    for(l0 = 0; l0 < len0; ++l0)
			    for(l1 = 0; l1 < len1; ++l1)
		      {
			      ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1];
			      x0 += ismax * (l0 - x0);
			      x1 += ismax * (l1 - x1);
		      }
		    inerror[i][o0*len0 + x0][o1*len1 + x1] = outerror[i][o0][o1];
	    }
}

void DOT_PRODUCT_FORWARD(double ***input, double *output, double **weight, double *bias, double(*action)(double))
{
  int x, y, j;
	for (x = 0; x < GETLENGTH(weight); ++x)
		for (y = 0; y < GETLENGTH(*weight); ++y)
			((double *)output)[y] += ((double *)input)[x] * weight[x][y];
	for(j = 0; j < GETLENGTH(bias); ++j)
		((double *)output)[j] = action(((double *)output)[j] + bias[j]);
}

void DOT_PRODUCT_BACKWARD(double ***input, double ***inerror, double *outerror, double **weight, double **wd, double *bd, double(*actiongrad)(double))
{
  int x, y, i, j;
	for (x = 0; x < GETLENGTH(weight); ++x)
		for (y = 0; y < GETLENGTH(*weight); ++y)
			((double *)inerror)[x] += ((double *)outerror)[y] * weight[x][y];
	for(i = 0; i < GETCOUNT(inerror); ++i)
		((double *)inerror)[i] *= actiongrad(((double *)input)[i]);
	for(j = 0; j < GETLENGTH(outerror); ++j)
		bd[j] += ((double *)outerror)[j];
	for (x = 0; x < GETLENGTH(weight); ++x)
		for (y = 0; y < GETLENGTH(*weight); ++y)
			wd[x][y] += ((double *)input)[x] * ((double *)outerror)[y];
}

//================================================================================================================================

#define CONVOLUTION_FORWARD(input,output,weight,bias,action)                                         \
{                                                                                                    \
	for (int x = 0; x < GETLENGTH(weight); ++x)                                                        \
		for (int y = 0; y < GETLENGTH(*weight); ++y)                                                     \
			CONVOLUTE_VALID(input[x], output[y], weight[x][y],                                             \
          GETLENGTH(*output),    GETLENGTH(**output),                                                \
          GETLENGTH(**weight), GETLENGTH(***weight));                                                \
	FOREACH(j, GETLENGTH(output))                                                                      \
		FOREACH(i, GETCOUNT(output[j]))                                                                  \
		((double *)output[j])[i] = action(((double *)output[j])[i] + bias[j]);                           \
}

#define CONVOLUTION_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)                         \
{                                                                                                    \
	for (int x = 0; x < GETLENGTH(weight); ++x)                                                        \
		for (int y = 0; y < GETLENGTH(*weight); ++y)                                                     \
			CONVOLUTE_FULL(outerror[y], inerror[x], weight[x][y]);                                         \
	FOREACH(i, GETCOUNT(inerror))                                                                      \
		((double *)inerror)[i] *= actiongrad(((double *)input)[i]);                                      \
	FOREACH(j, GETLENGTH(outerror))                                                                    \
		FOREACH(i, GETCOUNT(outerror[j]))                                                                \
		bd[j] += ((double *)outerror[j])[i];                                                             \
	for (int x = 0; x < GETLENGTH(weight); ++x)                                                        \
		for (int y = 0; y < GETLENGTH(*weight); ++y)                                                     \
      CONVOLUTE_VALID(input[x], wd[x][y], outerror[y],                                               \
                      GETLENGTH(**wd), GETLENGTH(***(wd)),                                           \
                      GETLENGTH(*outerror), GETLENGTH(**(outerror)));                                \
}
