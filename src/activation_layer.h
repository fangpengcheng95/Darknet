// if not define
#ifndef ACTIVATION_LAYER_H 
#define ACTIVATION_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"

// 创建一个激活层
// ACTIVATION的定义在darknet.h中,是一个枚举类型，可见其中的激活函数可选relu，tanh等
// typedef enum{
//     LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN
// } ACTIVATION;
layer make_activation_layer(int batch, int inputs, ACTIVATION activation);

// CPU版
// 前向传播
void forward_activation_layer(layer l, network net);
// 反向传播
void backward_activation_layer(layer l, network net);

// GPU版
#ifdef GPU
void forward_activation_layer_gpu(layer l, network net);
void backward_activation_layer_gpu(layer l, network net);
#endif

#endif

