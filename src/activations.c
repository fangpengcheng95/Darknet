#include "activations.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *get_activation_string(ACTIVATION a)
{
    switch(a){
        case LOGISTIC:
            return "logistic";
        case LOGGY:
            return "loggy";
        case RELU:
            return "relu";
        case ELU:
            return "elu";
        case RELIE:
            return "relie";
        case RAMP:
            return "ramp";
        case LINEAR:
            return "linear";
        case TANH:
            return "tanh";
        case PLSE:
            return "plse";
        case LEAKY:
            return "leaky";
        case STAIR:
            return "stair";
        case HARDTAN:
            return "hardtan";
        case LHTAN:
            return "lhtan";
        default:
            break;
    }
    return "relu";
}

/*
**根据输入的激活函数名称，返回激活函数
**如果输入的激活函数名称未能识别，则统一使用relu，并且给出错误提示(stderr，输出到屏幕)
**
*/
ACTIVATION get_activation(char *s)
{
    if (strcmp(s, "logistic")==0) return LOGISTIC;
    if (strcmp(s, "loggy")==0) return LOGGY;
    if (strcmp(s, "relu")==0) return RELU;
    if (strcmp(s, "elu")==0) return ELU;
    if (strcmp(s, "relie")==0) return RELIE;
    if (strcmp(s, "plse")==0) return PLSE;
    if (strcmp(s, "hardtan")==0) return HARDTAN;
    if (strcmp(s, "lhtan")==0) return LHTAN;
    if (strcmp(s, "linear")==0) return LINEAR;
    if (strcmp(s, "ramp")==0) return RAMP;
    if (strcmp(s, "leaky")==0) return LEAKY;
    if (strcmp(s, "tanh")==0) return TANH;
    if (strcmp(s, "stair")==0) return STAIR;
    fprintf(stderr, "Couldn't find activation function %s, going with ReLU\n", s);
    return RELU;
}

/*
**
**根据不同的激活函数类型，调用不同的激活函数处理单个输入元素x
**输入：x
**返回 activate_function(x)
**activate_function可以是relu，tanh等等不同的激活函数，这主要取决于ACTIVATION a
**ACTIVATION是枚举类型，定义在darknet.h中
**
typedef enum{
    LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN
} ACTIVATION; 
*/
float activate(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_activate(x);
        case LOGISTIC:
            return logistic_activate(x);
        case LOGGY:
            return loggy_activate(x);
        case RELU:
            return relu_activate(x);
        case ELU:
            return elu_activate(x);
        case RELIE:
            return relie_activate(x);
        case RAMP:
            return ramp_activate(x);
        case LEAKY:
            return leaky_activate(x);
        case TANH:
            return tanh_activate(x);
        case PLSE:
            return plse_activate(x);
        case STAIR:
            return stair_activate(x);
        case HARDTAN:
            return hardtan_activate(x);
        case LHTAN:
            return lhtan_activate(x);
    }
    return 0;
}

/*
**
**activate_array:
**输入:数组x，数组x的长度n，激活函数类型
**输出：对数组x中的每个元素进行激活后输出
**
*/
void activate_array(float *x, const int n, const ACTIVATION a)
{
    int i;
    for(i = 0; i < n; ++i){
        x[i] = activate(x[i], a);
    }
}

/*
**根据不同的激活函数求取对输入的梯度(导数)
**gradient:
**输入：x，梯度函数要接收的输入值
**      a，激活函数类型
**
*/
float gradient(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_gradient(x);
        case LOGISTIC:
            return logistic_gradient(x);
        case LOGGY:
            return loggy_gradient(x);
        case RELU:
            return relu_gradient(x);
        case ELU:
            return elu_gradient(x);
        case RELIE:
            return relie_gradient(x);
        case RAMP:
            return ramp_gradient(x);
        case LEAKY:
            return leaky_gradient(x);
        case TANH:
            return tanh_gradient(x);
        case PLSE:
            return plse_gradient(x);
        case STAIR:
            return stair_gradient(x);
        case HARDTAN:
            return hardtan_gradient(x);
        case LHTAN:
            return lhtan_gradient(x);
    }
    return 0;
}

/*
**
**gradient_array:
**计算激活函数对加权输入的导数，并乘以delta，得到当前层最终的delta
**
**输入:数组x，数组x的长度n，激活函数类型，当前层的所有输出(维度为n = l.batch * l.out_c * l.out_w * l.out_h)
       n,l.output的维度,n = l.batch * l.out_c * l.out_w * l.out_h
       a 激活函数类型，ACTIVATION，枚举类型，具体可见darknet.h
       delta,当前层敏感度图(与当前层输出x维度一样)
**输出：对数组x中的每个元素进行求梯度，再乘以delta，后赋给delta后输出
**
**说明：
**1.该函数不但计算了激活函数对于加权输入的导数，还将该导数乘以了之前完成大部分计算的敏感图delta(对应元素相乘)
**  因此调用该函数后，将得到该层最终的敏感图
**
**2.这里直接利用输出值计算了激活函数关于输入的导数值是因为神经网络中所使用的绝大部分激活函数，其关于输入的导数值都可以描述为输出值的函数表达式
**  比如Sigmoid激活函数(记作f(x))，其导数值f(x)' = f(x)*(1-f(x)),因此如果给出y=f(x),那么f(x)' = y * (1 - y),只需要输出值y就可以了，不再需要输入x的值
**
**3.关于l.delta的初值，可能你有注意到在看某一类型网络层的时候，比如卷积层中的backward_convolutional_layer()函数，没有发现在此之前对l.delta赋初值的语句，
**  只是用calloc为其动态分配了内存，这样的l.delta其所有元素的值都为0，那么这里使用*=运算符得到的值恒为0
**  但是整个网络是有很多层的，且有多种类型，一般来说，不会以卷积层为最后一层，而会以COST或者REGION为最后一层，
**  这些层中，会对l.delta赋上初值，又由于l.delta是后向传播的，因此，当反向运行到某一层时，l.delta的值都不会为0
*/
void gradient_array(const float *x, const int n, const ACTIVATION a, float *delta)
{
    int i;
    for(i = 0; i < n; ++i){
        delta[i] *= gradient(x[i], a);
    }
} 

