#include "activation_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

layer make_activation_layer(int batch, int inputs, ACTIVATION activation)
{
    layer l = {0};
    // 层的类型，激活层
    l.type = ACTIVE;

    // 定义激活层的输入和输出大小
    l.inputs = inputs;
    l.outputs = inputs;
    
    // 定义激活层的批处理
    l.batch=batch;

    //给输出层分配空间
    l.output = calloc(batch*inputs, sizeof(float*));
    l.delta = calloc(batch*inputs, sizeof(float*));

    //指定前向传播和反向传播，默认是CPU版本
    l.forward = forward_activation_layer;
    l.backward = backward_activation_layer;

    // 如果定义了GPU，则使用GPU版本，GPU=1，在makefile中第一行指定
#ifdef GPU
    l.forward_gpu = forward_activation_layer_gpu;
    l.backward_gpu = backward_activation_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, inputs*batch);
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch);
#endif
    //指定激活函数，ACTIVATION是枚举类型，在darknet.h中被指定，可以是relu，tanh等多种激活函数
    l.activation = activation;
    //向屏幕输出信息
    fprintf(stderr, "Activation Layer: %d inputs\n", inputs);
    return l;
}

// CPU版本的前向传播和后向传播

// copy_cpu方法的实现在blas.c中
// void copy_cpu(int N, float *X, int INCX, float *Y, int INCY)
// {
//     int i;
//     for(i = 0; i < N; ++i) Y[i*INCY] = X[i*INCX];
// }
// net.input(X数组)，l.output(Y数组)
// 将net.input数组中的内容拷贝一份到l.output数组中

// activate_array方法的实现在activations.c中
//函数很简单对X数组的每一项进行激活操作之后在赋给X
// void activate_array(float *x, const int n, const ACTIVATION a)
// {
//     int i;
//     for(i = 0; i < n; ++i){
//         x[i] = activate(x[i], a);
//     }
// }

void forward_activation_layer(layer l, network net)
{
    copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
    activate_array(l.output, l.outputs*l.batch, l.activation);
}

void backward_activation_layer(layer l, network net)
{
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);
    copy_cpu(l.outputs*l.batch, l.delta, 1, net.delta, 1);
}

#ifdef GPU
// GPU版本的前向和后向传播
// GPU版本中的函数和意义相同，只不过是借用CUDA进行操作
//具体可查看blas_kernel.cu
void forward_activation_layer_gpu(layer l, network net)
{
    copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
}

void backward_activation_layer_gpu(layer l, network net)
{
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    copy_gpu(l.outputs*l.batch, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif
