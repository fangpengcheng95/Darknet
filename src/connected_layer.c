#include "connected_layer.h"
#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
**
**@brief构建全连接层
**@param输入：batch   该层输入中一个batch所含有的图片张数，等于net.batch
**      inputs  全连接层每张输入图片的元素个数
**      outputs 全连接层输出元素个数(由于网络配置文件指定，如果未指定，默认值为1，在parse_connected()中赋值)
**      activation 激活函数类型
**      batch_normalize 是否进行BN
**
**@param return 返回：全连接层l
*/

layer make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam)
{
    int i;
    layer l = {0};
    l.learning_rate_scale = 1;

    /*层的类型，CONNECTED，全连接层*/
    l.type = CONNECTED;

    /*全连接层输入元素个数*/
    l.inputs = inputs;

    /*全连接层输出元素个数*/
    l.outputs = outputs;

    /*一个batch中，图片张数*/
    l.batch=batch;

    /*是否进行BN*/
    l.batch_normalize = batch_normalize;

    /*全连接层的输入的高和宽均为1*/
    l.h = 1;
    l.w = 1;

    /*全连接层的输入通道数等于一张输入图片中的元素个数*/
    l.c = inputs;

    /*全连接层的输出的高和宽均为1*/
    l.out_h = 1;
    l.out_w = 1;

    /*全连接层的输出通道数等于一张输出图片的元素个数*/
    l.out_c = outputs;

    /*全连接层的所有输出(包含整个batch的)*/
    l.output = calloc(batch*outputs, sizeof(float));

    /*全连接层的所有敏感图(包含整个batch)*/
    l.delta = calloc(batch*outputs, sizeof(float));

    /*由下面forward_connected_layer()函数中调用gemm()可以看出，l.weight_updates应该理解为outputs行，inputs列*/
    l.weight_updates = calloc(inputs*outputs, sizeof(float));/*全连接层权重系数更新值个数等于一张输入图片元素个数与其对应输出元素个数之积*/

    /*全连接层偏置更新值个数等于一张输入图片的输出元素个数*/
    l.bias_updates = calloc(outputs, sizeof(float));

    /*由下面forward_connected_layer()函数中调用gemm()可以看出，l.weights应该理解为outputs行，inputs列*/
    l.weights = calloc(outputs*inputs, sizeof(float));/*全连接层权重系数个数等于一张输入图片元素个数与其对应输出元素个数之积*/
    
    /*全连接层偏置个数等于一张输入图片的输出元素个数*/
    l.biases = calloc(outputs, sizeof(float));

    /*全连接层前向、反向、更新函数*/
    l.forward = forward_connected_layer;
    l.backward = backward_connected_layer;
    l.update = update_connected_layer;

// 初始化权重：缩放因子*-1到1之间的均匀分布，缩放因子等于sqrt(2./inputs)，为什么取这个值呢？？暂时没有想清楚，
    // 注意，与卷积层make_convolutional_layer()中初始化值不同，这里是均匀分布，而卷积层中是正态分布。
    

    /*初始化权重：缩放因子scale乘以(-1,1)之间的均匀分布，缩放因子等于sqrt(2./inputs)*/
    /*与卷积层make_convolutional_layer()中初始化值不同，这里是均匀分布，而卷积层中是正态分布*/
    /*TODO：个人感觉，这里应该加一个if条件语句：if(weightfile)，因为如果导入了预训练权重文件，就没有必要这样初始化了（事实上在detector.c的train_detector()函数中，
    紧接着parse_network_cfg()函数之后，就添加了if(weightfile)语句判断是否导入权重系数文件，如果导入了权重系数文件，也许这里初始化的值也会覆盖掉，
    总之这里的权重初始化的处理方式还是值得思考的，也许更好的方式是应该设置专门的函数进行权重的初始化，同时偏置也是*/
    //float scale = 1./sqrt(inputs);
    float scale = sqrt(2./inputs);
    for(i = 0; i < outputs*inputs; ++i){
        l.weights[i] = scale*rand_uniform(-1, 1);
    }

    for(i = 0; i < outputs; ++i){
        l.biases[i] = 0;
    }

    if(adam){
        l.m = calloc(l.inputs*l.outputs, sizeof(float));
        l.v = calloc(l.inputs*l.outputs, sizeof(float));
        l.bias_m = calloc(l.outputs, sizeof(float));
        l.scale_m = calloc(l.outputs, sizeof(float));
        l.bias_v = calloc(l.outputs, sizeof(float));
        l.scale_v = calloc(l.outputs, sizeof(float));
    }
    if(batch_normalize){
        l.scales = calloc(outputs, sizeof(float));
        l.scale_updates = calloc(outputs, sizeof(float));
        for(i = 0; i < outputs; ++i){
            l.scales[i] = 1;
        }

        l.mean = calloc(outputs, sizeof(float));
        l.mean_delta = calloc(outputs, sizeof(float));
        l.variance = calloc(outputs, sizeof(float));
        l.variance_delta = calloc(outputs, sizeof(float));

        l.rolling_mean = calloc(outputs, sizeof(float));
        l.rolling_variance = calloc(outputs, sizeof(float));

        l.x = calloc(batch*outputs, sizeof(float));
        l.x_norm = calloc(batch*outputs, sizeof(float));
    }

#ifdef GPU
    l.forward_gpu = forward_connected_layer_gpu;
    l.backward_gpu = backward_connected_layer_gpu;
    l.update_gpu = update_connected_layer_gpu;

    /*由下面forward_connected_layer_gpu()函数中调用的gemm_gpu()可以看出,l.weight_gpu应该理解为outputs行,inputs列*/
    l.weights_gpu = cuda_make_array(l.weights, outputs*inputs);
    l.biases_gpu = cuda_make_array(l.biases, outputs);

    l.weight_updates_gpu = cuda_make_array(l.weight_updates, outputs*inputs);
    l.bias_updates_gpu = cuda_make_array(l.bias_updates, outputs);

    l.output_gpu = cuda_make_array(l.output, outputs*batch);
    l.delta_gpu = cuda_make_array(l.delta, outputs*batch);
    if (adam) {
        l.m_gpu =       cuda_make_array(0, inputs*outputs);
        l.v_gpu =       cuda_make_array(0, inputs*outputs);
        l.bias_m_gpu =  cuda_make_array(0, outputs);
        l.bias_v_gpu =  cuda_make_array(0, outputs);
        l.scale_m_gpu = cuda_make_array(0, outputs);
        l.scale_v_gpu = cuda_make_array(0, outputs);
    }

    if(batch_normalize){
        l.mean_gpu = cuda_make_array(l.mean, outputs);
        l.variance_gpu = cuda_make_array(l.variance, outputs);

        l.rolling_mean_gpu = cuda_make_array(l.mean, outputs);
        l.rolling_variance_gpu = cuda_make_array(l.variance, outputs);

        l.mean_delta_gpu = cuda_make_array(l.mean, outputs);
        l.variance_delta_gpu = cuda_make_array(l.variance, outputs);

        l.scales_gpu = cuda_make_array(l.scales, outputs);
        l.scale_updates_gpu = cuda_make_array(l.scale_updates, outputs);

        l.x_gpu = cuda_make_array(l.output, l.batch*outputs);
        l.x_norm_gpu = cuda_make_array(l.output, l.batch*outputs);
#ifdef CUDNN
        cudnnCreateTensorDescriptor(&l.normTensorDesc);
        cudnnCreateTensorDescriptor(&l.dstTensorDesc);
        cudnnSetTensor4dDescriptor(l.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.batch, l.out_c, l.out_h, l.out_w); 
        cudnnSetTensor4dDescriptor(l.normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l.out_c, 1, 1); 
#endif
    }
#endif
    l.activation = activation;
    fprintf(stderr, "connected                            %4d  ->  %4d\n", inputs, outputs);
    return l;
}

void update_connected_layer(layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;
    axpy_cpu(l.outputs, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.outputs, momentum, l.bias_updates, 1);

    if(l.batch_normalize){
        axpy_cpu(l.outputs, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.outputs, momentum, l.scale_updates, 1);
    }

    axpy_cpu(l.inputs*l.outputs, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(l.inputs*l.outputs, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(l.inputs*l.outputs, momentum, l.weight_updates, 1);
}

/*
**@brief 全连接层的前向传播函数
**@param 输入: l 当前的全连接层
**             net 整个网络
**@note 全连接层的前向传播相对简单，首先初始化输出l.output全为0，在进行相关参数赋值之后，直接调用gemm_nt()完成Wx操作
**      而后根据判断是否需要BN，如果需要则进行BN操作，完了之后为每一个输出元素添加偏置得到Wx+b，最后使用激活函数处理
        每一个输出元素，得到f(Wx+b)
**
*/
void forward_connected_layer(layer l, network net)
{
    /*初始化全连接层的所有输出(包含所有batch)为0*/
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);

    /*m:全连接层接收的一个batch的图片张数*/
    int m = l.batch;

    /*k:全连接层单张输入图片元素个数*/
    int k = l.inputs;

    /*n:全连接层对应单张输入图片的输出元素个数*/
    int n = l.outputs;

    /*a:全连接层的输入元素，维度为l.batch*l.inputs(包含真个batch的输入),可视为l.batch行,l.inputs列,每行就是一张输入图片*/
    float *a = net.input;

    /*b:全连接层的所有权重，维度为1.batch*l.outputs(包含整个batch的输出)*/
    float *b = l.weights;

    /*c:全连接层的所有输出(包含所有batch)，维度为l.batch*l.outputs(包含整个batch的输出)*/
    float *c = l.output;
    gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);
    if(l.batch_normalize){
        /*forward_batchnorm_layer()定义在batchnorm_layer.c中*/
        /*
                *
        void forward_batchnorm_layer(layer l, network net)
        {
            if(l.type == BATCHNORM) copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
            copy_cpu(l.outputs*l.batch, l.output, 1, l.x, 1);
            if(net.train){
                mean_cpu(l.output, l.batch, l.out_c, l.out_h*l.out_w, l.mean);
                variance_cpu(l.output, l.mean, l.batch, l.out_c, l.out_h*l.out_w, l.variance);

                scal_cpu(l.out_c, .99, l.rolling_mean, 1);
                axpy_cpu(l.out_c, .01, l.mean, 1, l.rolling_mean, 1);
                scal_cpu(l.out_c, .99, l.rolling_variance, 1);
                axpy_cpu(l.out_c, .01, l.variance, 1, l.rolling_variance, 1);

                normalize_cpu(l.output, l.mean, l.variance, l.batch, l.out_c, l.out_h*l.out_w);   
                copy_cpu(l.outputs*l.batch, l.output, 1, l.x_norm, 1);
            } else {
                normalize_cpu(l.output, l.rolling_mean, l.rolling_variance, l.batch, l.out_c, l.out_h*l.out_w);
            }
            scale_bias(l.output, l.scales, l.batch, l.out_c, l.out_h*l.out_w);
            add_bias(l.output, l.biases, l.batch, l.out_c, l.out_h*l.out_w);
        }
        */
        forward_batchnorm_layer(l, net);
    } else {
        add_bias(l.output, l.biases, l.batch, l.outputs, 1);
    }

    /*前向传播最后一步：前面得到每一个输入元素的加权输入Wx+b，这一步利用激活函数处理l.output中的每一个输出元素
    **最终得到全连接层的输出f(Wx+b)
    */
    activate_array(l.output, l.outputs*l.batch, l.activation);
}

/*
**
**@brief 全连接层反向传播函数
**@param 输入:l 当前全连接层
**@param      net 整个网络
**@note 计算当前层的敏感度图l.delta(注意是反向传播)，
**      全连接层的偏置更新值(基于完全计算完的l.delta)
**      ，判断是否进行BN，如果进行，则完成BN操作，再接着计算当前层权重更新值，最后计算上一层
**      的敏感度图(完成大部分计算)。相比于卷积神经网络，全连接层很多的计算变得更为直接，不需要
**      im2col_cpu()或者col2im_cpu()函数对数据重排来重拍去，直接矩阵相乘就可以搞定
*/
void backward_connected_layer(layer l, network net)
{
    /*完成当前层敏感度图的计算：当前全连接层下一层不管是什么类型的网络，都会完成当前敏感度的绝大部分计算
    **（上一层敏感度乘以上一层与当前层之间的权重）,此处只需要将l.delta中每一个元素乘以激活函数对加权输入的导数即可
    **gradient_array()函数完成激活函数对加权输入的导数，并乘以之前得到的l.delta，得到当前层的最终的l.delta(误差函数对加权输入的导数)
    **
    */
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);

    if(l.batch_normalize){
        backward_batchnorm_layer(l, net);
    } else {
        backward_bias(l.bias_updates, l.delta, l.batch, l.outputs, 1);
    }

    /*计算当前全连接层的权重更新值*/
    int m = l.outputs;

    
    int k = l.batch;

    int n = l.inputs;

    float *a = l.delta;

    float *b = net.input;

    float *c = l.weight_updates;

    /* 由行列匹配规则可知，需要将a转置，故而调用gemm()函数，转置a实际上是想把batch中所有图片的影响叠加。*/
    gemm(1,0,m,n,k,1,a,m,b,n,1,c,n);

    // 最终得到的c维度为l.outputs*l.inputs，对应所有权重的更新值

    /*m：a'的行，值为l.outputs，含义为每张图片输出的元素个数*/
    m = l.batch;

    /*k：a’的列数，值为l.batch，含义为一个batch中含有的图片张数*/
    k = l.outputs;

     /*n：b的列数，值为l.inputs，含义为每张输入图片的元素个数*/
    n = l.inputs;

    /*a: 当前全连接层敏感度图，维度为l.batch*l.outputs*/
    a = l.delta;

    /*b: 当前全连接层所有输入，维度为l.batch*l.inputs*/
    b = l.weights;

     /*当前全连接层权重更新值，维度为l.outputs*l.inputs(权重个数)*/
    c = net.delta;

    if(c) gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
}


void denormalize_connected_layer(layer l)
{
    int i, j;
    for(i = 0; i < l.outputs; ++i){
        float scale = l.scales[i]/sqrt(l.rolling_variance[i] + .000001);
        for(j = 0; j < l.inputs; ++j){
            l.weights[i*l.inputs + j] *= scale;
        }
        l.biases[i] -= l.rolling_mean[i] * scale;
        l.scales[i] = 1;
        l.rolling_mean[i] = 0;
        l.rolling_variance[i] = 1;
    }
}


void statistics_connected_layer(layer l)
{
    if(l.batch_normalize){
        printf("Scales ");
        print_statistics(l.scales, l.outputs);
        /*
           printf("Rolling Mean ");
           print_statistics(l.rolling_mean, l.outputs);
           printf("Rolling Variance ");
           print_statistics(l.rolling_variance, l.outputs);
         */
    }
    printf("Biases ");
    print_statistics(l.biases, l.outputs);
    printf("Weights ");
    print_statistics(l.weights, l.outputs);
}

#ifdef GPU

void pull_connected_layer(layer l)
{
    cuda_pull_array(l.weights_gpu, l.weights, l.inputs*l.outputs);
    cuda_pull_array(l.biases_gpu, l.biases, l.outputs);
    cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.inputs*l.outputs);
    cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
    if (l.batch_normalize){
        cuda_pull_array(l.scales_gpu, l.scales, l.outputs);
        cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.outputs);
        cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.outputs);
    }
}

void push_connected_layer(layer l)
{
    cuda_push_array(l.weights_gpu, l.weights, l.inputs*l.outputs);
    cuda_push_array(l.biases_gpu, l.biases, l.outputs);
    cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.inputs*l.outputs);
    cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
    if (l.batch_normalize){
        cuda_push_array(l.scales_gpu, l.scales, l.outputs);
        cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.outputs);
        cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.outputs);
    }
}

void update_connected_layer_gpu(layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;
    if(a.adam){
        adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.inputs*l.outputs, batch, a.t);
        adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.outputs, batch, a.t);
        if(l.scales_gpu){
            adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.outputs, batch, a.t);
        }
    }else{
        axpy_gpu(l.outputs, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
        scal_gpu(l.outputs, momentum, l.bias_updates_gpu, 1);

        if(l.batch_normalize){
            axpy_gpu(l.outputs, learning_rate/batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
            scal_gpu(l.outputs, momentum, l.scale_updates_gpu, 1);
        }

        axpy_gpu(l.inputs*l.outputs, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
        axpy_gpu(l.inputs*l.outputs, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
        scal_gpu(l.inputs*l.outputs, momentum, l.weight_updates_gpu, 1);
    }
}

void forward_connected_layer_gpu(layer l, network net)
{
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);

    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    float * a = net.input_gpu;
    float * b = l.weights_gpu;
    float * c = l.output_gpu;
    gemm_gpu(0,1,m,n,k,1,a,k,b,k,1,c,n);

    if (l.batch_normalize) {
        forward_batchnorm_layer_gpu(l, net);
    } else {
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.outputs, 1);
    }
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
}

void backward_connected_layer_gpu(layer l, network net)
{
    constrain_gpu(l.outputs*l.batch, 1, l.delta_gpu, 1);
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    if(l.batch_normalize){
        backward_batchnorm_layer_gpu(l, net);
    } else {
        backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.outputs, 1);
    }

    int m = l.outputs;
    int k = l.batch;
    int n = l.inputs;
    float * a = l.delta_gpu;
    float * b = net.input_gpu;
    float * c = l.weight_updates_gpu;
    gemm_gpu(1,0,m,n,k,1,a,m,b,n,1,c,n);

    m = l.batch;
    k = l.outputs;
    n = l.inputs;

    a = l.delta_gpu;
    b = l.weights_gpu;
    c = net.delta_gpu;

    if(c) gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
}
#endif
