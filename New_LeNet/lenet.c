#include "lenet.h"
#include <memory.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define GETLENGTH(array) (sizeof(array)/sizeof(*(array)))

#define GETCOUNT(array)  (sizeof(array)/sizeof(double))

#define FOREACH(i,count) for (int i = 0; i < count; ++i)

//#define CONVOLUTION_FORWARD(input,output,weight,bias,action)                                         \
{                                                                                                    \
	for (int x = 0; x < GETLENGTH(weight); ++x)                                                        \
		for (int y = 0; y < GETLENGTH(*weight); ++y)                                                     \
      FOREACH(o0,GETLENGTH(output[y]))                                                               \
		    FOREACH(o1,GETLENGTH(*(output[y])))                                                          \
			    FOREACH(w0,GETLENGTH(weight[x][y]))                                                        \
				    FOREACH(w1,GETLENGTH(*(weight[x][y])))                                                   \
					    output[y][o0][o1] += input[x][o0 + w0][o1 + w1] * weight[x][y][w0][w1];                \
	FOREACH(j, GETLENGTH(output))                                                                      \
		FOREACH(i, GETCOUNT(output[j]))                                                                  \
		((double *)output[j])[i] = action(((double *)output[j])[i] + bias[j]);                           \
}

//#define SUBSAMP_MAX_FORWARD(input,output)                                                            \
//{                                                                                                    \
	//const int len0 = GETLENGTH(*(input)) / GETLENGTH(*(output));                                       \
	//const int len1 = GETLENGTH(**(input)) / GETLENGTH(**(output));                                     \
	//FOREACH(i, GETLENGTH(output))                                                                      \
	//FOREACH(o0, GETLENGTH(*(output)))                                                                  \
	//FOREACH(o1, GETLENGTH(**(output)))                                                                 \
	//{                                                                                                  \
		//int x0 = 0, x1 = 0, ismax;                                                                       \
		//FOREACH(l0, len0)                                                                                \
			//FOREACH(l1, len1)                                                                              \
		//{                                                                                                \
			//ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1];           \
			//x0 += ismax * (l0 - x0);                                                                       \
			//x1 += ismax * (l1 - x1);                                                                       \
		//}                                                                                                \
		//output[i][o0][o1] = input[i][o0*len0 + x0][o1*len1 + x1];                                        \
	//}                                                                                                  \
//}

//#define DOT_PRODUCT_FORWARD(input,output,weight,bias,action)                                         \
//{                                                                                                    \
	//for (int x = 0; x < GETLENGTH(weight); ++x)                                                        \
		//for (int y = 0; y < GETLENGTH(*weight); ++y)                                                     \
			//((double *)output)[y] += ((double *)input)[x] * weight[x][y];                                  \
	//FOREACH(j, GETLENGTH(bias))                                                                        \
		//((double *)output)[j] = action(((double *)output)[j] + bias[j]);                                 \
//}

#define CONVOLUTION_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)                         \
{                                                                                                    \
	for (int x = 0; x < GETLENGTH(weight); ++x)                                                        \
		for (int y = 0; y < GETLENGTH(*weight); ++y)                                                     \
      FOREACH(i0,GETLENGTH(outerror[y]))                                                             \
		    FOREACH(i1,GETLENGTH(*(outerror[y])))                                                        \
			    FOREACH(w0,GETLENGTH(weight[x][y]))                                                        \
				    FOREACH(w1,GETLENGTH(*(weight[x][y])))                                                   \
					    inerror[x][i0 + w0][i1 + w1] += outerror[y][i0][i1] * weight[x][y][w0][w1];            \
	FOREACH(i, GETCOUNT(inerror))                                                                      \
		((double *)inerror)[i] *= actiongrad(((double *)input)[i]);                                      \
	FOREACH(j, GETLENGTH(outerror))                                                                    \
		FOREACH(i, GETCOUNT(outerror[j]))                                                                \
		bd[j] += ((double *)outerror[j])[i];                                                             \
	for (int x = 0; x < GETLENGTH(weight); ++x)                                                        \
		for (int y = 0; y < GETLENGTH(*weight); ++y)                                                     \
      FOREACH(o0,GETLENGTH(wd[x][y]))                                                                \
		    FOREACH(o1,GETLENGTH(*(wd[x][y])))                                                           \
			    FOREACH(w0,GETLENGTH(outerror[y]))                                                         \
				    FOREACH(w1,GETLENGTH(*(outerror[y])))                                                    \
					    wd[x][y][o0][o1] += input[x][o0 + w0][o1 + w1] * outerror[y][w0][w1];                  \
}

#define SUBSAMP_MAX_BACKWARD(input,inerror,outerror)                                                 \
{                                                                                                    \
	const int len0 = GETLENGTH(*(inerror)) / GETLENGTH(*(outerror));                                   \
	const int len1 = GETLENGTH(**(inerror)) / GETLENGTH(**(outerror));                                 \
	FOREACH(i, GETLENGTH(outerror))                                                                    \
	FOREACH(o0, GETLENGTH(*(outerror)))                                                                \
	FOREACH(o1, GETLENGTH(**(outerror)))                                                               \
	{                                                                                                  \
		int x0 = 0, x1 = 0, ismax;                                                                       \
		FOREACH(l0, len0)                                                                                \
			FOREACH(l1, len1)                                                                              \
		    {                                                                                            \
			    ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1];       \
			    x0 += ismax * (l0 - x0);                                                                   \
			    x1 += ismax * (l1 - x1);                                                                   \
		    }                                                                                            \
		inerror[i][o0*len0 + x0][o1*len1 + x1] = outerror[i][o0][o1];                                    \
	}                                                                                                  \
}

#define DOT_PRODUCT_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)                         \
{                                                                                                    \
	for (int x = 0; x < GETLENGTH(weight); ++x)                                                        \
		for (int y = 0; y < GETLENGTH(*weight); ++y)                                                     \
			((double *)inerror)[x] += ((double *)outerror)[y] * weight[x][y];                              \
	FOREACH(i, GETCOUNT(inerror))                                                                      \
		((double *)inerror)[i] *= actiongrad(((double *)input)[i]);                                      \
	FOREACH(j, GETLENGTH(outerror))                                                                    \
		bd[j] += ((double *)outerror)[j];                                                                \
	for (int x = 0; x < GETLENGTH(weight); ++x)                                                        \
		for (int y = 0; y < GETLENGTH(*weight); ++y)                                                     \
			wd[x][y] += ((double *)input)[x] * ((double *)outerror)[y];                                    \
}

//====================================================================================================

void CONVOLUTION_FORWARD_0_1(LeNet5 *lenet, Feature *features, double(*action)(double))
{
	for (int x = 0; x < GETLENGTH(lenet->weight0_1); ++x)
		for (int y = 0; y < GETLENGTH(*lenet->weight0_1); ++y)
      FOREACH(o0,GETLENGTH(features->layer1[y]))
		    FOREACH(o1,GETLENGTH(*(features->layer1[y])))
			    FOREACH(w0,GETLENGTH(lenet->weight0_1[x][y]))
				    FOREACH(w1,GETLENGTH(*(lenet->weight0_1[x][y])))
					    features->layer1[y][o0][o1] += features->input[x][o0 + w0][o1 + w1] * lenet->weight0_1[x][y][w0][w1];
	FOREACH(j, GETLENGTH(features->layer1))
		FOREACH(i, GETCOUNT(features->layer1[j]))
		  ((double *)features->layer1[j])[i] = action(((double *)features->layer1[j])[i] + lenet->bias0_1[j]);
}

void SUBSAMP_MAX_FORWARD_1_2(Feature *features)
{
	const int len0 = GETLENGTH(*(features->layer1)) / GETLENGTH(*(features->layer2));
	const int len1 = GETLENGTH(**(features->layer1)) / GETLENGTH(**(features->layer2));
	FOREACH(i, GETLENGTH(features->layer2))
	FOREACH(o0, GETLENGTH(*(features->layer2)))
	FOREACH(o1, GETLENGTH(**(features->layer2)))
	{
		int x0 = 0, x1 = 0, ismax;
		FOREACH(l0, len0)
			FOREACH(l1, len1)
		{
			ismax = features->layer1[i][o0*len0 + l0][o1*len1 + l1] > features->layer1[i][o0*len0 + x0][o1*len1 + x1];
			x0 += ismax * (l0 - x0);
			x1 += ismax * (l1 - x1);
		}
		features->layer2[i][o0][o1] = features->layer1[i][o0*len0 + x0][o1*len1 + x1];
	}
}

void CONVOLUTION_FORWARD_2_3(LeNet5 *lenet, Feature *features, double(*action)(double))
{
	for (int x = 0; x < GETLENGTH(lenet->weight2_3); ++x)
		for (int y = 0; y < GETLENGTH(*lenet->weight2_3); ++y)
      FOREACH(o0,GETLENGTH(features->layer3[y]))
		    FOREACH(o1,GETLENGTH(*(features->layer3[y])))
			    FOREACH(w0,GETLENGTH(lenet->weight2_3[x][y]))
				    FOREACH(w1,GETLENGTH(*(lenet->weight2_3[x][y])))
					    features->layer3[y][o0][o1] += features->layer2[x][o0 + w0][o1 + w1] * lenet->weight2_3[x][y][w0][w1];
	FOREACH(j, GETLENGTH(features->layer3))
		FOREACH(i, GETCOUNT(features->layer3[j]))
		  ((double *)features->layer3[j])[i] = action(((double *)features->layer3[j])[i] + lenet->bias2_3[j]);
}

void SUBSAMP_MAX_FORWARD_3_4(Feature *features)
{
	const int len0 = GETLENGTH(*(features->layer3)) / GETLENGTH(*(features->layer4));
	const int len1 = GETLENGTH(**(features->layer3)) / GETLENGTH(**(features->layer4));
	FOREACH(i, GETLENGTH(features->layer4))
	FOREACH(o0, GETLENGTH(*(features->layer4)))
	FOREACH(o1, GETLENGTH(**(features->layer4)))
	{
		int x0 = 0, x1 = 0, ismax;
		FOREACH(l0, len0)
			FOREACH(l1, len1)
		{
			ismax = features->layer3[i][o0*len0 + l0][o1*len1 + l1] > features->layer3[i][o0*len0 + x0][o1*len1 + x1];
			x0 += ismax * (l0 - x0);
			x1 += ismax * (l1 - x1);
		}
		features->layer4[i][o0][o1] = features->layer3[i][o0*len0 + x0][o1*len1 + x1];
	}
}

void CONVOLUTION_FORWARD_4_5(LeNet5 *lenet, Feature *features, double(*action)(double))
{
	for (int x = 0; x < GETLENGTH(lenet->weight4_5); ++x)
		for (int y = 0; y < GETLENGTH(*lenet->weight4_5); ++y)
      FOREACH(o0,GETLENGTH(features->layer5[y]))
		    FOREACH(o1,GETLENGTH(*(features->layer5[y])))
			    FOREACH(w0,GETLENGTH(lenet->weight4_5[x][y]))
				    FOREACH(w1,GETLENGTH(*(lenet->weight4_5[x][y])))
					    features->layer5[y][o0][o1] += features->layer4[x][o0 + w0][o1 + w1] * lenet->weight4_5[x][y][w0][w1];
	FOREACH(j, GETLENGTH(features->layer5))
		FOREACH(i, GETCOUNT(features->layer5[j]))
		  ((double *)features->layer5[j])[i] = action(((double *)features->layer5[j])[i] + lenet->bias4_5[j]);
}

void DOT_PRODUCT_FORWARD_5_6(LeNet5 *lenet, Feature *features, double(*action)(double))
{
	for (int x = 0; x < GETLENGTH(lenet->weight5_6); ++x)
		for (int y = 0; y < GETLENGTH(*lenet->weight5_6); ++y)
			((double *)features->output)[y] += ((double *)features->layer5)[x] * lenet->weight5_6[x][y]; 
	FOREACH(j, GETLENGTH(lenet->bias5_6))
		((double *)features->output)[j] = action(((double *)features->output)[j] + lenet->bias5_6[j]);
}

double relu(double x)
{
	return x*(x > 0);
}

double relugrad(double y)
{
	return y > 0;
}

static void forward(LeNet5 *lenet, Feature *features, double(*action)(double))
{
  CONVOLUTION_FORWARD_0_1(lenet, features, action);
	SUBSAMP_MAX_FORWARD_1_2(features);
	CONVOLUTION_FORWARD_2_3(lenet, features, action);
	SUBSAMP_MAX_FORWARD_3_4(features);
	CONVOLUTION_FORWARD_4_5(lenet, features, action);
	DOT_PRODUCT_FORWARD_5_6(lenet, features, action);
}

static void backward(LeNet5 *lenet, LeNet5 *deltas, Feature *errors, Feature *features, double(*actiongrad)(double))
{
	DOT_PRODUCT_BACKWARD(features->layer5, errors->layer5, errors->output, lenet->weight5_6, deltas->weight5_6, deltas->bias5_6, actiongrad);
	CONVOLUTION_BACKWARD(features->layer4, errors->layer4, errors->layer5, lenet->weight4_5, deltas->weight4_5, deltas->bias4_5, actiongrad);
	SUBSAMP_MAX_BACKWARD(features->layer3, errors->layer3, errors->layer4);
	CONVOLUTION_BACKWARD(features->layer2, errors->layer2, errors->layer3, lenet->weight2_3, deltas->weight2_3, deltas->bias2_3, actiongrad);
	SUBSAMP_MAX_BACKWARD(features->layer1, errors->layer1, errors->layer2);
	CONVOLUTION_BACKWARD(features->input, errors->input, errors->layer1, lenet->weight0_1, deltas->weight0_1, deltas->bias0_1, actiongrad);
}

static inline void load_input(Feature *features, image input)
{
	double (*layer0)[LENGTH_FEATURE0][LENGTH_FEATURE0] = features->input;
	const long sz = sizeof(image) / sizeof(**input);
	double mean = 0, std = 0;
	FOREACH(j, sizeof(image) / sizeof(*input))
		FOREACH(k, sizeof(*input) / sizeof(**input))
	{
		mean += input[j][k];
		std += input[j][k] * input[j][k];
	}
	mean /= sz;
	std = sqrt(std / sz - mean*mean);
	FOREACH(j, sizeof(image) / sizeof(*input))
		FOREACH(k, sizeof(*input) / sizeof(**input))
	{
		layer0[0][j + PADDING][k + PADDING] = (input[j][k] - mean) / std;
	}
}

static inline void softmax(double input[OUTPUT], double loss[OUTPUT], int label, int count)
{
	double inner = 0;
	for (int i = 0; i < count; ++i)
	{
		double res = 0;
		for (int j = 0; j < count; ++j)
		{
			res += exp(input[j] - input[i]);
		}
		loss[i] = 1. / res;
		inner -= loss[i] * loss[i];
	}
	inner += loss[label];
	for (int i = 0; i < count; ++i)
	{
		loss[i] *= (i == label) - loss[i] - inner;
	}
}

static void load_target(Feature *features, Feature *errors, int label)
{
	double *output = (double *)features->output;
	double *error = (double *)errors->output;
	softmax(output, error, label, GETCOUNT(features->output));
}

static uint8 get_result(Feature *features, uint8 count)
{
	double *output = (double *)features->output; 
	const int outlen = GETCOUNT(features->output);
	uint8 result = 0;
	double maxvalue = *output;
	for (uint8 i = 1; i < count; ++i)
	{
		if (output[i] > maxvalue)
		{
			maxvalue = output[i];
			result = i;
		}
	}
	return result;
}

static double f64rand()
{
	static int randbit = 0;
	if (!randbit)
	{
		srand((unsigned)time(0));
		for (int i = RAND_MAX; i; i >>= 1, ++randbit);
	}
	unsigned long long lvalue = 0x4000000000000000L;
	int i = 52 - randbit;
	for (; i > 0; i -= randbit)
		lvalue |= (unsigned long long)rand() << i;
	lvalue |= (unsigned long long)rand() >> -i;
	return *(double *)&lvalue - 3;
}


void TrainBatch(LeNet5 *lenet, image *inputs, uint8 *labels, int batchSize)
{
	double buffer[GETCOUNT(LeNet5)] = { 0 };
	int i = 0;
#pragma omp parallel for
	for (i = 0; i < batchSize; ++i)
	{
		Feature features = { 0 };
		Feature errors = { 0 };
		LeNet5	deltas = { 0 };
		load_input(&features, inputs[i]);
		forward(lenet, &features, relu);
		load_target(&features, &errors, labels[i]);
		backward(lenet, &deltas, &errors, &features, relugrad);
		#pragma omp critical
		{
			FOREACH(j, GETCOUNT(LeNet5))
				buffer[j] += ((double *)&deltas)[j];
		}
	}
	double k = ALPHA / batchSize;
	FOREACH(i, GETCOUNT(LeNet5))
		((double *)lenet)[i] += k * buffer[i];
}

void Train(LeNet5 *lenet, image input, uint8 label)
{
	Feature features = { 0 };
	Feature errors = { 0 };
	LeNet5 deltas = { 0 };
	load_input(&features, input);
	forward(lenet, &features, relu);
	load_target(&features, &errors, label);
	backward(lenet, &deltas, &errors, &features, relugrad);
	FOREACH(i, GETCOUNT(LeNet5))
		((double *)lenet)[i] += ALPHA * ((double *)&deltas)[i];
}

uint8 Predict(LeNet5 *lenet, image input,uint8 count)
{
	Feature features = { 0 };
	load_input(&features, input);
	forward(lenet, &features, relu);
	return get_result(&features, count);
}

void Initial(LeNet5 *lenet)
{
	for (double *pos = (double *)lenet->weight0_1; pos < (double *)lenet->bias0_1; *pos++ = f64rand());
	for (double *pos = (double *)lenet->weight0_1; pos < (double *)lenet->weight2_3; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (INPUT + LAYER1))));
	for (double *pos = (double *)lenet->weight2_3; pos < (double *)lenet->weight4_5; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER2 + LAYER3))));
	for (double *pos = (double *)lenet->weight4_5; pos < (double *)lenet->weight5_6; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER4 + LAYER5))));
	for (double *pos = (double *)lenet->weight5_6; pos < (double *)lenet->bias0_1; *pos++ *= sqrt(6.0 / (LAYER5 + OUTPUT)));
	for (int *pos = (int *)lenet->bias0_1; pos < (int *)(lenet + 1); *pos++ = 0);
}
