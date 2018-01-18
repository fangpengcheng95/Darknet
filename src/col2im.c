#include <stdio.h>
#include <math.h>

/*
**@brief 将输入图像im的channel通道上的第row行，col列像素灰度值加上val(直接修改im的值，因此im相当于是返回值)
**@param im 输入图像
**@param channels 输入图像的im通道数
**@param height 输入图像im的高度(行)
**@param width 输入图像im的宽度(列)
**@param row 需要加上val的像素所在的行数(补0之后的行数，因此需要先减去pad才能得到真正im中的行数)
**@param col 需要加上val的像素所在的列数(补0之后的列数，因此需要先减去pad才能得到真正的im中的列数)
**@param channel 需要加上val的像素坐在的通道数
**@param pad 四周补0长度
**@param val 像素灰度添加值
*/

void col2im_add_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad, float val)
{
    row -= pad;
    col -= pad;

    /*边界检查，超过边界则直接返回，什么都不做*/
    if (row < 0 || col < 0 ||
        row >= height || col >= width) return;
    im[col + width*(row + height*channel)] += val;
}

//This one might be too, can't remember.
/*
**@brief 此函数与im2col_cpu()函数的流程相反，目地是将im2col_cpu()函数重排得到的图片data_col恢复至正常的图像矩阵排列
**并与data_im相加，最终data_im相当于是输出值
**要注意的是,data_im的尺寸是在函数外确定的，且没有显示地将data_col转为一个与data_im尺寸相同的矩阵，而是将其中元素直接加在data_im
**对应元素上(data_im初始所有元素值都为0)，得到的data_im尺寸为l.c * l.h * l.w，即为当前层的输入图像尺寸，上一层的输出图像尺寸，按行存储
**可视为l.c行，l.h * l.w列，即其中每行对应一张输出特征图的敏感度图(实际上这还不是最终的敏感度)
**还差一个环节：乘以激活函数对加权输入的导数，这将在下一次调用backward_covolutional_layer时完成
**
**@example 第L-1层每张输入图片(本例子只分析单张输入图片)的输出5*5*3(3 维输出通道数)，第L层共有2个卷积核，每个卷积核的尺寸为3*3，stride = 2
**第L - 1层的输出是第L层的输入，第L层的2个卷积核同时对上一层的3个通道的输出做卷积运算，为了做到这一点，需要调用im2col_cpu()函数将
**上一层的输出，也就是本层的输入重排为27行4列的图，也就是5*5*3变换至27*4，你会发现总的元素个数变多了(75增多到了98)
**这是因为卷积核stride=2，小于卷积核的尺寸3，因此卷积在两个连续位置做卷积，会有重叠部分，而im2col_cpu()函数为了便于卷积运算，完全将其
**铺排开来，并没有在空间上避免重复元素，因此像素元素会增多。此外，之所以是27行，是因为卷积核尺寸是3*3，而上一层的输出即本层输入有3个通道，
**为了同时给3个通道做卷积运算，需要将3个通道上的输入一起考虑，即得到3*3*3行，4列是因为对于5*5的图像，使用3*3的卷积核，stride=2的卷积跨度，
**最终会得到2*2的特征图，也就是4个元素。除了调用im2col_cpu()对输入图像做重排，相应的，也要将所有卷积核重排成一个2*27的矩阵，为什么是2呢？
**因为有2个卷积核，为了做到同时将两个卷积核作用到输入图像上，需要将两个核合到一个矩阵中，每个核对应一行，因此有2行，那么为什么是27呢？因为每个核
**3*3*3(3个通道)=27，所以实际上一个卷积核有9*3 = 27个元素，综述，得到2*27的卷积核矩阵与27*4的输入图像矩阵，两个矩阵相乘，即可完成将2个卷积核同时
**作用于3通道的输入图像上,最终得到2*4的矩阵
**@note 可以看出im2col_cpu()这个函数的重要性，而此处的col2im_cpu()是一个逆过程，主要用于反向传播中，由L层的敏感图(sensitivity map)，反向求得第L-1层
**的敏感度图。顺承上面的例子，第L-1层的输出是一个是一个5*5*3(l.w = l.h = 5,l.c = 3)的矩阵，也就是敏感度图的维度为5*5*3(每个输出元素，对应一个敏感度值)
**第L层的输出是一个2*4的矩阵，敏感度图的维度为2*4，假设已经计算得到了第L层2*4的敏感度图，那么现在的问题是，如何由第L层的2*4敏感度图以及2个卷积核(2*27)
**反向获取第L-1层的敏感度图呢？此处参考博客：https://www.zybuluo.com/hanbingtao/note/485480，给出了一种很好的求解方式，但是darknet并不是这样做的，
**为什么？因为前面有im2col_cpu()，im2col_cpu()函数中的重排方式，使得我们不再需要博客中提到的将sensitivity map还原为步长为1的sensitivity map，
**只需要col2im_cpu()就可以了，过程是怎么样的呢？看backward_covolutional_layer()函数中if(net.delta)中的语句就知道了，此处仅讨论讨论col2im_cpu()的过程
**在backward_convolutional_layer()已经得到了data_col，这个矩阵含有了所有的第L-1层敏感度的信息，但遗憾的是，不能直接用，需要整理，因为此时data_col还是一个
**27*4的矩阵，而我们知道第L-1层的敏感图是一个5*5*3的矩阵，如何将一个27*4变换至一个5*5*3的矩阵是本函数要完成的工作，前面说到的27*4元素个数多于5*5*3个，
** 输入： data_col    backward_convolutional_layer()中计算得到的包含上一层所有敏感度信息的矩阵，行数为l.n*l.size*l.size（l代表本层/当前层），
**                    列数为l.out_h*l.out_w（对于本例子，行数为27,列数为4,上一层为第L-1层，本层是第L层） 
**       channels    当前层输入图像的通道数（对于本例子，为3）
**       height      当前层输入图像的行数（对于本例子，为5）
**       width       当前层输入图像的列数（对于本例子，为5）
**       ksize       当前层卷积核尺寸（对于本例子，为3）
**       stride      当前层卷积跨度（对于本例子，为2）
**       pad         当前层对输入图像做卷积时四周补0的长度
**       data_im     经col2im_cpu()重排恢复之后得到的输出矩阵，也即上一层的敏感度图，尺寸为l.c * l.h * l.w（刚好就是上一层的输出当前层输入的尺寸，
                     对于本例子，5行5列3通道），注意data_im的尺寸，是在本函数之外就已经确定的，不是在本函数内部计算出来的，这与im2col_cpu()不同，
                     im2col_cpu()计算得到的data_col的尺寸都是在函数内部计算得到的，
**                   并不是事先指定的。也就是说，col2im_cpu()函数完成的是指定尺寸的输入矩阵往指定尺寸的输出矩阵的转换。
** 原理：原理比较复杂，很难用文字叙述，博客：https://www.zybuluo.com/hanbingtao/note/485480中基本原理说得很详细了，但是此处的实现与博客中并不一样，
**      所以具体实现的原理此处简要叙述一下，具体见个人博客。
**      第L-1层得到l.h*l.w*l.c输出，也是第L层的输入，经L层卷积及激活函数处理之后，得到l.out_h*l.out_w*l.out_c的输出，
**      也就是由l.h*l.w*l.c-->l.out_h*l.out_w*l.out_c，
**      由于第L层有多个卷积核，所以第L-1层中的一个输出元素会流入到第L层多个输出中，除此之外，由于卷积核之间的重叠，
**      也导致部分元素流入到第L层的多个输出中，这两种情况，都导致第L-1层中的某个敏感度会与第L层多个输出有关，
**      为清晰，还是用上面的例子来解释，第L-1层得到5*5*3(3*25)的输出，第L层得到2*2*2（2*4）的输出，
**      在backward_convolutional_layer()已经计算得到的data_col实际是27*2矩阵与2*4矩阵相乘的结果，
**      为方便，我们记27*2的矩阵为a，记2*4矩阵为b，那么a中一行（2个元素）与b中一列（2个元素）相乘对应这什么呢？对应第一情况，
**      因为有两个卷积核，使得L-1中一个输出至少与L层中两个输出有关系，经此矩阵相乘，得到27*4的矩阵，
**      已经考虑了第一种情况（27*4这个矩阵中的每一个元素都是两个卷积核影响结果的求和），那么接下来的就是要考虑第二种情况：
**      卷积核重叠导致的一对多关系，具体做法就是将data_col中对应相同像素的值相加，这是由
**      im2col_cpu()函数决定的（可以配合im2col_cpu()来理解），因为im2col_cpu()将这些重叠元素也铺陈保存在data_col中，所以接下来，
**      只要按照im2col_cpu()逆向将这些重叠元素的影响叠加就可以了，
**      大致就是这个思路，具体的实现细节可能得见个人博客了（这段写的有点罗嗦～）。
**很显然要从27*4变成5*5*3，肯定会将某些元素相加合并(下面的col2im_add_pixel()函数就是干这个的)，具体怎样
*/
void col2im_cpu(float* data_col,
         int channels,  int height,  int width,
         int ksize,  int stride, int pad, float* data_im) 
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                double val = data_col[col_index];
                col2im_add_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad, val);
            }
        }
    }
}

