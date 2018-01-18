#include "blas.h"

#include <math.h>
#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void reorg_cpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out)
{
    // 输入参数代表的意义：
    // w: wide 宽度
    // h: high 高度
    // c: channel 通道
    // batch: batch
    // stride:步长
    // forward: forward==1代表前向传播，否则为后向传播
    // x: 输入数组
    // out: 输出数组
    //该函数的目的是对x数组进行打乱重组
    int b,i,j,k;
    int out_c = c/(stride*stride);
    for(b = 0; b < batch; ++b){
        for(k = 0; k < c; ++k){
            for(j = 0; j < h; ++j){
                for(i = 0; i < w; ++i){
                    int in_index  = i + w*(j + h*(k + c*b));
                    int c2 = k % out_c;
                    int offset = k / out_c;
                    int w2 = i*stride + offset % stride;
                    int h2 = j*stride + offset / stride;
                    int out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b));
                    if(forward) out[out_index] = x[in_index];
                    else out[in_index] = x[out_index];
                }
            }
        }
    }
}

//flatten用来将输入“压平”，即把多维的输入一维化
// 如果  x = [1.000000 2.000000 3.000000]
//           [4.000000 5.000000 6.000000]
//那么
// 1.前向传播flatten处理的结果为: x = [1.000000 3.000000 5.000000 2.000000 4.000000 6.000000]
// 2.后向传播flatten处理的结果为: x = [1.000000 4.000000 2.000000 5.000000 3.000000 6.000000]
void flatten(float *x, int size, int layers, int batch, int forward)
{
    float *swap = calloc(size*layers*batch, sizeof(float));
    int i,c,b;
    for(b = 0; b < batch; ++b){
        for(c = 0; c < layers; ++c){
            for(i = 0; i < size; ++i){
                int i1 = b*layers*size + c*size + i;
                int i2 = b*layers*size + i*layers + c;
                if (forward) swap[i2] = x[i1];
                else swap[i1] = x[i2];
            }
        }
    }
    memcpy(x, swap, size*layers*batch*sizeof(float));
    free(swap);
}

void weighted_sum_cpu(float *a, float *b, float *s, int n, float *c)
{
    int i;
    for(i = 0; i < n; ++i){
        c[i] = s[i]*a[i] + (1-s[i])*(b ? b[i] : 0);
    }
}

void weighted_delta_cpu(float *a, float *b, float *s, float *da, float *db, float *ds, int n, float *dc)
{
    int i;
    for(i = 0; i < n; ++i){
        if(da) da[i] += dc[i] * s[i];
        if(db) db[i] += dc[i] * (1-s[i]);
        ds[i] += dc[i] * (a[i] - b[i]);
    }
}

void shortcut_cpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float *out)
{
    int stride = w1/w2;
    int sample = w2/w1;
    assert(stride == h1/h2);
    assert(sample == h2/h1);
    if(stride < 1) stride = 1;
    if(sample < 1) sample = 1;
    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;

    int i,j,k,b;
    for(b = 0; b < batch; ++b){
        for(k = 0; k < minc; ++k){
            for(j = 0; j < minh; ++j){
                for(i = 0; i < minw; ++i){
                    int out_index = i*sample + w2*(j*sample + h2*(k + c2*b));
                    int add_index = i*stride + w1*(j*stride + h1*(k + c1*b));
                    out[out_index] += add[add_index];
                }
            }
        }
    }
}


/*
**有组织的计算输入数据x的平均值，输出的mean是一个矢量，比如如果x是多张3通道图片，那么mean的维度就为通道数3
**也就是每张输入图片会得到3张特征图，称为1,2,3通道
**由于每次训练的输入都是一个batch的图片，因此最终会输出batch张3通道图片，mean中的第一个元素就是第一个通道上
**全部batch张输出特征图所有元素的额平均值，2,3通道依次类推
**mean_cpu的主要用处之一就应该是实现batch normalization的第一步了
**输入:
**      x 包含所有数据，比如l.output,其包含的元素个数为l.batch*l.outputs
**      batch 一个batch中包含的图片张数，即l.batch
**      filters 该层神经网络的filter个数，也是该层网络输出图片的通道数
**      spatial 该层神经网络每张输出特征图的尺寸，也即等于l.out_w*l.out_h
**      mean 求得平均值，维度为filters，也即每个滤波器对应有一个均值(每个滤波器会处理所有图片)
**Note1: mean_cpu()函数是一个纯粹的数学计算函数，有组织的计算x中某些数据的均值，x的具体存储结构视情况而定
**Note2: 该函数的调用具体可以以参考batchnorm_layer.c中forward_batchnorm_layer()函数
**Note3：x中包含了众多数据，mean中的每个元素究竟对应x中的哪些数据的平均值呢?
**       x为l.output,(可以这么理解),有l.batch个块,每一块有l.c个切片，每个切片中可以理解为一个二维数组(l.h行，l.w列)
**       若filter为3，每次输入batch=64张图片，那么将输出64张3通道的图片，而mean中每个元素就是某个单通道上所有64张图片所有元素的平均值
**       比如第1个通道上，所有64张图片像素平均值
**Note4: 在全连接层的前向传播函数中：sptial=1，因为全连接层的输出可以看做1*1的特征图
**
*/
void mean_cpu(float *x, int batch, int filters, int spatial, float *mean)
{
    /*scale即求均值中的分母项*/
    float scale = 1./(batch * spatial);
    int i,j,k;
    /*外层循环:为filters,也即mean的维度，每次循环将得到一个平均值*/
    for(i = 0; i < filters; ++i){
        mean[i] = 0;
        /*中层循环:的次数为batch,也即叠加每张输入图片对应的某一通道上的输出*/
        for(j = 0; j < batch; ++j){
            /*内层循环:叠加一张输入特征图中的所有像素值*/
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                mean[i] += x[index];
            }
        }
        mean[i] *= scale;
    }
}

/*
**
**计算输入x中每个元素的方差
**该函数的主要用处之一应该就是batch normalization的第二步了
**输入：
**      x:包含所有数据，比如l.output,包含的元素个数为l.batch*l.outputs
**      batch:一个batch中包含的图片张数，即l.batch
**      filters:该层神经网络的滤波器个数,也即该层网络输出图片的通道数
**      spatial:该层神经网络每张输出特征图的尺寸，也即等于l.out_w*l.out_h
**      mean:求得平均值，维度为filters，也即每个录波器都对应一个均值(每个滤波器会处理所有图片)
*/
void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{   
    /*
    *为什么计算方差分母要减1
    *事实上，在统计学中，往往采用的方差计算公式都会让分母减1，这时因为所有数据的方差是基于均值
    *这个固定点来计算的，对于n个数据的样本，在均值固定的情况下，其采样自由度为n-1(只要n-1个数据
    *固定，第n个可以由均值推导出)
    */
    float scale = 1./(batch * spatial - 1);
    int i,j,k;
    for(i = 0; i < filters; ++i){
        variance[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                /*每个元素减去均值求平方*/
                variance[i] += pow((x[index] - mean[i]), 2);
            }
        }
        variance[i] *= scale;
    }
}

void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
    int b, f, i;
    for(b = 0; b < batch; ++b){
        for(f = 0; f < filters; ++f){
            for(i = 0; i < spatial; ++i){
                int index = b*filters*spatial + f*spatial + i;
                x[index] = (x[index] - mean[f])/(sqrt(variance[f]) + .000001f);
            }
        }
    }
}

void const_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] = ALPHA;
}

void mul_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] *= X[i*INCX];
}

void pow_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] = pow(X[i*INCX], ALPHA);
}



/*
**axpy是线性代数中的一种基本操作，完成y= alpha*x + y，其中x,y为矢量，alpha为实数系数
**输入:N:X中包含的有效元素个数
**     ALPHA:实数系数
**     X:参与运算的矢量X
**     INCX:步长(倍数步长)，即X中凡是INCX的倍数编号参与运算
**     Y:输出
**
*/

void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] += ALPHA*X[i*INCX];
}

void scal_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] *= ALPHA;
}


/*
**
**初始化X数组所有元素的值为ALPHA
**输入:N: X中包含的有效元素个数
**     ALPHA:初始化的值
**     X:待初始化的float数组
**     INCX:步长(倍数步长),即X中凡是INCX的倍数的编号进行初始化赋值操作
**
**
*/
void fill_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] = ALPHA;
}

void deinter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT)
{
    int i, j;
    int index = 0;
    for(j = 0; j < B; ++j) {
        for(i = 0; i < NX; ++i){
            if(X) X[j*NX + i] += OUT[index];
            ++index;
        }
        for(i = 0; i < NY; ++i){
            if(Y) Y[j*NY + i] += OUT[index];
            ++index;
        }
    }
}

void inter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT)
{
    int i, j;
    int index = 0;
    for(j = 0; j < B; ++j) {
        for(i = 0; i < NX; ++i){
            OUT[index++] = X[j*NX + i];
        }
        for(i = 0; i < NY; ++i){
            OUT[index++] = Y[j*NY + i];
        }
    }
}

/*
**  将输入X中的数据复制到Y中（值复制，并不是指针复制，即之后X与Y之间再无关联）
**  输入： N       X中包含的有效元素个数
**        X       待初始化的float数组指针
**        INCX    步长（倍数步长），即X中凡是INCX的倍数编号进行初始化赋值操作
*/

void copy_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] = X[i*INCX];
}

void mult_add_into_cpu(int N, float *X, float *Y, float *Z)
{
    int i;
    for(i = 0; i < N; ++i) Z[i] += X[i]*Y[i];
}

void smooth_l1_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float diff = truth[i] - pred[i];
        float abs_val = fabs(diff);
        if(abs_val < 1) {
            error[i] = diff * diff;
            delta[i] = diff;
        }
        else {
            error[i] = 2*abs_val - 1;
            delta[i] = (diff < 0) ? 1 : -1;
        }
    }
}

void l1_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float diff = truth[i] - pred[i];
        error[i] = fabs(diff);
        delta[i] = diff > 0 ? 1 : -1;
    }
}


/*
**l2范数
**计算预测数组与真实标签数组中每对元素的l2范数值，或者说是计算squared error，
**注意此函数，并没有求和，没有将所有误差加起来，而是对网络输出的每个元素计算
**误差的平方值
**输入：n:输入元素的个数，也即pred中元素的个数，也是truth中的元素的个数
**      pred:网络最终的输出值，或者说网络的预测值，其中输出元素个数为n(也即最后一层网络神经元的个数为n)
**      truth:真实标签值，其中元素个数为n
**      delta:相当于本函数的输出,为网络的敏感度图(一般为cost_layer.c的敏感度图)
**      error:相当于本函数的输出，包含每个输出元素的squared error
**说明:这个函数一般被cost_layer.c调用，用于计算cost_layer每个输出的均方误差，除此之外，还有一个重要的用途
**     就是计算网络最后一层的敏感度图，在darknet中，最后一层比较多的情况是cost_l
**
*/
void l2_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float diff = truth[i] - pred[i];
        error[i] = diff * diff;
        delta[i] = diff;
    }
}

//求内积
float dot_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    float dot = 0;
    for(i = 0; i < N; ++i) dot += X[i*INCX] * Y[i*INCY];
    return dot;
}

/*
** 输入： input   一组输入图片数据（含义见下面softmax_cpu()注释，下同）
**       n       一组输入数据中含有的元素个数n=l.inputs/l.groups
**       temp    温度参数，关于softmax的温度参数，可以搜索一下softmax with temperature，应该会有很多的
**       stride  跨度
**       output  这一组输入图片数据对应的输出（也即l.output中与这一组输入对应的某一部分）
** 说明：本函数实现的就是标准的softmax函数处理，唯一有点变化的就是在做指数运算之前，将每个输入元素减去了该组输入元素中的最大值，以增加数值稳定性，
**      关于此，可以参考博客：http://freemind.pluskid.org/machine-learning/softmax-vs-softmax-loss-numerical-stability/，
**      这篇博客写的不错，博客中还提到了softmax-loss，此处没有实现（此处实现的也即博客中提到的softmax函数，将softmax-loss分开实现了）。
*/

void softmax(float *input, int n, float temp, int stride, float *output)
{
    int i;
    float sum = 0;
    float largest = -FLT_MAX;
    for(i = 0; i < n; ++i){
        if(input[i*stride] > largest) largest = input[i*stride];
    }
    for(i = 0; i < n; ++i){
        float e = exp(input[i*stride]/temp - largest/temp);
        sum += e;
        output[i*stride] = e;
    }
    for(i = 0; i < n; ++i){
        output[i*stride] /= sum;
    }
}


/*
**
**@brief 对输入input进行softmax处理得到输出output
**@param input softmax层所有输入数据(包含整个batch的),即net.input(上一层的输出)
**@param n 一组输入数据中含有元素的个数 n = l.inputs/l.groups
**@param batch 一个batch中含有图片的张数(等于net.batch)
**@param batch_offset 一张输入图片含有的元素的个数，即值等于l.inputs(所以叫做batch_offset,目的就是要借助该参数在input中整张整张照片移位)
**@param groups 一张输入图片的元素被分成了几组，值为l.groups(这个参数由配置文件指定，如果未指定，则默认为1)
**@param group_offset 值等于1，组偏移(在每张图片元素中整租整租偏移)
**@param stride 跨度，这个参数类似于axpy_cpu()函数中的INCX参数，一定注意不同于卷积层中的l.stride，这个参数是指按照stride间隔从每组输入
**              数据中抽取元素，即会出去所有索引为stride倍数的输入元素，而其他的输入元素，实际没有用到，stride=1时，显然，相当于没有这个参数
**              所有输入数据都用到了(这个参在softmax_layer层中，相当于没用，因为在forward_softmax_layer()中，调用该函数时，
**              stride已经被写死为1，并不能更改)
**@param temp softmax的温度参数l.temperature，softmax的温度参数T，当T很大时，即趋正无穷时，
**            所有的激活值对应的激活概率趋近于相同(激活概率差异性小)，当T很低时，即趋于0时，
**            不同的激活值对应的激活概率差异也越大。
**@param output 经softmax处理之后得到的输出l.output(即概率)，与input具有相同的元素个数(见make_softmax_layer())，由此可知
**              stride的值必然为1，不然output的元素个数肯定少于input的元素个数(所以对softmax来说，感觉设置stride是没有必要的)
**@Note 以上注释针对的是softmax_layer，另有不同地方调用本函数的在调用出进行详细注释:上面的注释出现了新的量词单位,这里理清下关系：
**      输入input中包括batch中所有图片的输入数据，其中一张图片具有inputs个元素，一张图片的元素又分成了groups组，每组元素个数为n=l.inputs/l.groups
**
*/
void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
{
    int g, b;
    /*遍历batch中的每张图片*/
    for(b = 0; b < batch; ++b){
        /*每张图片又按组遍历*/
        for(g = 0; g < groups; ++g){
            softmax(input + b*batch_offset + g*group_offset, n, temp, stride, output + b*batch_offset + g*group_offset);
        }
    }
}

