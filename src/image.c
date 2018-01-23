#include "image.h"
#include "utils.h"
#include "blas.h"
#include "cuda.h"
#include <stdio.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int windows = 0;

float colors[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };

float get_color(int c, int x, int max)
{
    float ratio = ((float)x/max)*5;
    int i = floor(ratio);
    int j = ceil(ratio);
    ratio -= i;
    float r = (1-ratio) * colors[i][c] + ratio*colors[j][c];
    //printf("%f\n", r);
    return r;
}

image mask_to_rgb(image mask)
{
    int n = mask.c;
    image im = make_image(mask.w, mask.h, 3);
    int i, j;
    for(j = 0; j < n; ++j){
        int offset = j*123457 % n;
        float red = get_color(2,offset,n);
        float green = get_color(1,offset,n);
        float blue = get_color(0,offset,n);
        for(i = 0; i < im.w*im.h; ++i){
            im.data[i + 0*im.w*im.h] += mask.data[j*im.h*im.w + i]*red;
            im.data[i + 1*im.w*im.h] += mask.data[j*im.h*im.w + i]*green;
            im.data[i + 2*im.w*im.h] += mask.data[j*im.h*im.w + i]*blue;
        }
    }
    return im;
}

static float get_pixel(image m, int x, int y, int c)
{
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c*m.h*m.w + y*m.w + x];
}
static float get_pixel_extend(image m, int x, int y, int c)
{
    if(x < 0 || x >= m.w || y < 0 || y >= m.h) return 0;
    /*
    if(x < 0) x = 0;
    if(x >= m.w) x = m.w-1;
    if(y < 0) y = 0;
    if(y >= m.h) y = m.h-1;
    */
    if(c < 0 || c >= m.c) return 0;
    return get_pixel(m, x, y, c);
}
static void set_pixel(image m, int x, int y, int c, float val)
{
    if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c) return;
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] = val;
}
static void add_pixel(image m, int x, int y, int c, float val)
{
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] += val;
}


/*
** 双线性插值得到(x,y)处的像素值
** 输入： im    输入图像
**       x     像素坐标（列），亚像素
**       y     像素坐标（行），亚像素
**       c     所在通道数
** 返回：亚像素坐标(x,y)处的像素值
*/
static float bilinear_interpolate(image im, float x, float y, int c)
{
     //往下取整得到整数坐标
    int ix = (int) floorf(x);
    int iy = (int) floorf(y);

    float dx = x - ix;
    float dy = y - iy;

    float val = (1-dy) * (1-dx) * get_pixel_extend(im, ix, iy, c) + 
        dy     * (1-dx) * get_pixel_extend(im, ix, iy+1, c) + 
        (1-dy) *   dx   * get_pixel_extend(im, ix+1, iy, c) +
        dy     *   dx   * get_pixel_extend(im, ix+1, iy+1, c);
    return val;
}


void composite_image(image source, image dest, int dx, int dy)
{
    int x,y,k;
    for(k = 0; k < source.c; ++k){
        for(y = 0; y < source.h; ++y){
            for(x = 0; x < source.w; ++x){
                float val = get_pixel(source, x, y, k);
                float val2 = get_pixel_extend(dest, dx+x, dy+y, k);
                set_pixel(dest, dx+x, dy+y, k, val * val2);
            }
        }
    }
}

image border_image(image a, int border)
{
    image b = make_image(a.w + 2*border, a.h + 2*border, a.c);
    int x,y,k;
    for(k = 0; k < b.c; ++k){
        for(y = 0; y < b.h; ++y){
            for(x = 0; x < b.w; ++x){
                float val = get_pixel_extend(a, x - border, y - border, k);
                if(x - border < 0 || x - border >= a.w || y - border < 0 || y - border >= a.h) val = 1;
                set_pixel(b, x, y, k, val);
            }
        }
    }
    return b;
}

image tile_images(image a, image b, int dx)
{
    if(a.w == 0) return copy_image(b);
    image c = make_image(a.w + b.w + dx, (a.h > b.h) ? a.h : b.h, (a.c > b.c) ? a.c : b.c);
    fill_cpu(c.w*c.h*c.c, 1, c.data, 1);
    embed_image(a, c, 0, 0); 
    composite_image(b, c, a.w + dx, 0);
    return c;
}

image get_label(image **characters, char *string, int size)
{
    if(size > 7) size = 7;
    image label = make_empty_image(0,0,0);
    while(*string){
        image l = characters[size][(int)*string];
        image n = tile_images(label, l, -size - 1 + (size+1)/2);
        free_image(label);
        label = n;
        ++string;
    }
    image b = border_image(label, label.h*.25);
    free_image(label);
    return b;
}

void draw_label(image a, int r, int c, image label, const float *rgb)
{
    int w = label.w;
    int h = label.h;
    if (r - h >= 0) r = r - h;

    int i, j, k;
    for(j = 0; j < h && j + r < a.h; ++j){
        for(i = 0; i < w && i + c < a.w; ++i){
            for(k = 0; k < label.c; ++k){
                float val = get_pixel(label, i, j, k);
                set_pixel(a, i+c, j+r, k, rgb[k] * val);
            }
        }
    }
}

void draw_box(image a, int x1, int y1, int x2, int y2, float r, float g, float b)
{
    //normalize_image(a);
    int i;
    if(x1 < 0) x1 = 0;
    if(x1 >= a.w) x1 = a.w-1;
    if(x2 < 0) x2 = 0;
    if(x2 >= a.w) x2 = a.w-1;

    if(y1 < 0) y1 = 0;
    if(y1 >= a.h) y1 = a.h-1;
    if(y2 < 0) y2 = 0;
    if(y2 >= a.h) y2 = a.h-1;

    for(i = x1; i <= x2; ++i){
        a.data[i + y1*a.w + 0*a.w*a.h] = r;
        a.data[i + y2*a.w + 0*a.w*a.h] = r;

        a.data[i + y1*a.w + 1*a.w*a.h] = g;
        a.data[i + y2*a.w + 1*a.w*a.h] = g;

        a.data[i + y1*a.w + 2*a.w*a.h] = b;
        a.data[i + y2*a.w + 2*a.w*a.h] = b;
    }
    for(i = y1; i <= y2; ++i){
        a.data[x1 + i*a.w + 0*a.w*a.h] = r;
        a.data[x2 + i*a.w + 0*a.w*a.h] = r;

        a.data[x1 + i*a.w + 1*a.w*a.h] = g;
        a.data[x2 + i*a.w + 1*a.w*a.h] = g;

        a.data[x1 + i*a.w + 2*a.w*a.h] = b;
        a.data[x2 + i*a.w + 2*a.w*a.h] = b;
    }
}

void draw_box_width(image a, int x1, int y1, int x2, int y2, int w, float r, float g, float b)
{
    int i;
    for(i = 0; i < w; ++i){
        draw_box(a, x1+i, y1+i, x2-i, y2-i, r, g, b);
    }
}

void draw_bbox(image a, box bbox, int w, float r, float g, float b)
{
    int left  = (bbox.x-bbox.w/2)*a.w;
    int right = (bbox.x+bbox.w/2)*a.w;
    int top   = (bbox.y-bbox.h/2)*a.h;
    int bot   = (bbox.y+bbox.h/2)*a.h;

    int i;
    for(i = 0; i < w; ++i){
        draw_box(a, left+i, top+i, right-i, bot-i, r, g, b);
    }
}

/*
**
**@brief 加载data/labels/文件夹中所有的标签图片。所谓标签图片，就是仅含有单个字符的小图片，各标签图片组合在一起
**就可以得到完成的类标签，data/labels含有8套美国标准ASCII码32~127号字符，每套间仅大小(尺寸)不同，以应对不同大小的图片
*@return *返回:image**,二维数组，8行128列，实际有效为8行96列，因为前32列为0；每行包括一套32号~127号的ASCII标准码字符标签图片
**@note image**实际有效值为后面的96列，前32列为0指针，之所以还要保留前32列，是保持秩序统一，便于之后的访问，访问时，直接将ASCII码
**转为整型值即可得到在image**中的索引号，利于查找定位
**
*/

/*
**image的定义
**typedef struct {
    int w;
    int h;
    int c;
    float *data;
} image;
**
*/

image **load_alphabet()
{
    int i, j;

    /*共有8套，每套尺寸不同*/
    const int nsize = 8;

    /*alphabets是一个二维image指针，每行代表一套，包含32~127个字符图片指针*/
    /*为每行第一个字符标签图片动态分配内存。共有8行，每行的首个元素都是存储image的指针*/
    /*因此calloc(8，sizeof(image))(为指针分配内存不是为指针变量本身分配内存，而是为其所指的内存块分配内存)*/
    /*calloc()这个动态分配函数会初始化元素值为0指针(malloc不会，之前内存中遗留了什么就是什么，不会重新初始化为0)*/
    /*而这里需要重新初始化为0，因此这里使用的是calloc*/
    image **alphabets = calloc(nsize, sizeof(image));

    /*外循环，循环8次，读入8套字符标签图片*/
    for(j = 0; j < nsize; ++j){

        /*为每列动态分配内存，这里其实没有128个元素，只有96个元素，但是依然分配了128个元素
        **是为了便于之后的访问：直接将ASCII码转为整型值就可以得到在image中的索引，
        **定位到相应的字符标签图片
        */
        alphabets[j] = calloc(128, sizeof(image));

        /*内循环，从32开始，读入32号~127号字符标签图片*/
        for(i = 32; i < 127; ++i){
            char buff[256];

            /*格式化并打印各种数据到buffer
            **按照指定格式将字符串输出至字符串数组buffer中，得到每个字符标签图片的完整名称
            */
            sprintf(buff, "data/labels/%d_%d.png", i, j);

            /*
            image load_image_color(char *filename, int w, int h)
            {
                return load_image(filename, w, h, 3);
            }
            */
            /*读入彩色图(3通道)*/
            alphabets[j][i] = load_image_color(buff, 0, 0);
        }
    }
    return alphabets;
}


/*
**@brief 检测最后一步：遍历得到所有的检测框，找出其中最大所属概率，判断是否大于指定阈值，
**如果大于则在输入图片上绘制检测信息，同时在终端输入检测结果，如果小于则什么都不做
**@param im 输入图片，将在im上绘制检测结果，绘制内容包括定位用矩形框，由26个单字母拼接而成的类别标签
**@param num 整张图片所拥有的box(物体检测框)个数,输入图片经过检测之后，会划分成l.w*l.h的网络，每个网络中会提取l.n个box
**           每个box对应一个包含所有类别所属概率的数组，数组中元素最大并且大于某一个指定概率阈值对应的类别将作为这个box
             最终所检测出的物体类别
**@param thresh 概率阈值，每个box最大的所属类别概率必须大于这个阈值才接受这个检测结果，否则将该box作废(即不认为其包含物体)
**@param boxes  包含所有的box信息，是一个一维数组(大小即为num)，每个元素都是一个box数据类型，包含一个box需的信息，矩形框的中心
                坐标x，y以及矩形框的宽w，高h
**@param probs 这是一个二维数组，行为num，列为物体类别总数(就是该模型所能检测的物体类别的总数)，或者说训练数据集中所含有的所有物体类别总数
               训练集中不包含的物体肯定无法检测，即每行对应一个box，每行包含该框所属所有物体类别的概率。每行最大的元素所对应的物体类别为
               该检测框最有可能所属的物体类别，当然，最终判定是否属于该类别还需要看其是否大于一定概率阈值
**@param names 可以理解成一个一维字符串数组(C中没有字符串的概念，用C字符数组实现字符串)，每行存储一个物体的名称，共有classes行
**@param alphabet
**@param classes 物体类别总数
**@details 最终图片检测出的box总数并不是由网络配置文件指定的，而是由网络设计决定的，也就是根据前面网络的参数逐层推理计算决定的。darknet中，
**         检测模型最后一层往往是region_layer层，这一层输入的l.w，l.h参数就是最终图片所划分的网络数，而每个网络中所检测的矩形框是有配置文件
**         指定的，就是配置文件中的num参数，比如yolo.cfg中最后一层region层中的num参数，这样最后检测得到的box共有l.w*l.h*num个，每个box都
**         对应有一个概率数组，数组的大小等于该模型能够检测的物体的类别总数，这些物体都是排好序的，与数组的索引一致，比如/data文件夹下有一个
**         coco.names文件，包含coco数据集中的所有数据类别，该文件中按顺序列出了这些类别的名称，这些顺序也是数组索引的顺序
**
*/
void draw_detections(image im, int num, float thresh, box *boxes, float **probs, float **masks, char **names, image **alphabet, int classes)
{
    int i,j;

    for(i = 0; i < num; ++i){

        /*寻找出概率最大的类别索引值，返回的class即为所属物体的索引值，每个prob[i]都是一个大小为classes的一维数组
        **(包含该检测框属于各个物体的概率)
        */
        char labelstr[4096] = {0};
        int class = -1;
        for(j = 0; j < classes; ++j){
            if (probs[i][j] > thresh){
                if (class < 0) {
                    strcat(labelstr, names[j]);
                    class = j;
                } else {
                    strcat(labelstr, ", ");
                    strcat(labelstr, names[j]);
                }
                /*在终端输出检测结果：类别：概率*/
                printf("%s: %.0f%%\n", names[j], probs[i][j]*100);
            }
        }
        if(class >= 0){
            int width = im.h * .006;

            /*
               if(0){
               width = pow(prob, 1./2.)*10+1;
               alphabet = 0;
               }
             */

            //printf("%d %s: %.0f%%\n", i, names[class], prob*100);
            int offset = class*123457 % classes;
            float red = get_color(2,offset,classes);
            float green = get_color(1,offset,classes);
            float blue = get_color(0,offset,classes);
            float rgb[3];

            //width = prob*20+2;

            rgb[0] = red;
            rgb[1] = green;
            rgb[2] = blue;
            box b = boxes[i];

            /*
            **计算矩形框在图片中真实像素坐标(box中所含的都是比例坐标)，left为矩形框左横坐标，right为矩形框右横坐标
            **top为矩形框上方纵坐标，bot为矩形框下方纵坐标(这里也是按照OpenCV的习惯，y轴往下，x轴往右，原点为左上角)
            */
            int left  = (b.x-b.w/2.)*im.w;
            int right = (b.x+b.w/2.)*im.w;
            int top   = (b.y-b.h/2.)*im.h;
            int bot   = (b.y+b.h/2.)*im.h;

            /*越界检测*/
            if(left < 0) left = 0;
            if(right > im.w-1) right = im.w-1;
            if(top < 0) top = 0;
            if(bot > im.h-1) bot = im.h-1;

            draw_box_width(im, left, top, right, bot, width, red, green, blue);
            if (alphabet) {
                image label = get_label(alphabet, labelstr, (im.h*.03)/10);
                draw_label(im, top + width, left, label, rgb);
                free_image(label);
            }
            if (masks){
                image mask = float_to_image(14, 14, 1, masks[i]);
                image resized_mask = resize_image(mask, b.w*im.w, b.h*im.h);
                image tmask = threshold_image(resized_mask, .5);
                embed_image(tmask, im, left, top);
                free_image(mask);
                free_image(resized_mask);
                free_image(tmask);
            }
        }
    }
}

void transpose_image(image im)
{
    assert(im.w == im.h);
    int n, m;
    int c;
    for(c = 0; c < im.c; ++c){
        for(n = 0; n < im.w-1; ++n){
            for(m = n + 1; m < im.w; ++m){
                float swap = im.data[m + im.w*(n + im.h*c)];
                im.data[m + im.w*(n + im.h*c)] = im.data[n + im.w*(m + im.h*c)];
                im.data[n + im.w*(m + im.h*c)] = swap;
            }
        }
    }
}

void rotate_image_cw(image im, int times)
{
    assert(im.w == im.h);
    times = (times + 400) % 4;
    int i, x, y, c;
    int n = im.w;
    for(i = 0; i < times; ++i){
        for(c = 0; c < im.c; ++c){
            for(x = 0; x < n/2; ++x){
                for(y = 0; y < (n-1)/2 + 1; ++y){
                    float temp = im.data[y + im.w*(x + im.h*c)];
                    im.data[y + im.w*(x + im.h*c)] = im.data[n-1-x + im.w*(y + im.h*c)];
                    im.data[n-1-x + im.w*(y + im.h*c)] = im.data[n-1-y + im.w*(n-1-x + im.h*c)];
                    im.data[n-1-y + im.w*(n-1-x + im.h*c)] = im.data[x + im.w*(n-1-y + im.h*c)];
                    im.data[x + im.w*(n-1-y + im.h*c)] = temp;
                }
            }
        }
    }
}

/*左右翻转图片a(本地翻转，即a也是输出)*/
void flip_image(image a)
{
    int i,j,k;
    for(k = 0; k < a.c; ++k){
        for(i = 0; i < a.h; ++i){
            for(j = 0; j < a.w/2; ++j){
                /*index是从左到右的顺序编号*/
                /*int index = j + [a.w * a.h * k + a.w * i]*/
                int index = j + a.w*(i + a.h*(k));

                //flip是从右到左的顺序编号
                /*int flip = ( a.w - j - 1) + [a.w * a.h * k + a.w * i]*/
                int flip = (a.w - j - 1) + a.w*(i + a.h*(k));

                /*左右交换像素值*/
                float swap = a.data[flip];
                a.data[flip] = a.data[index];
                a.data[index] = swap;
            }
        }
    }
}

image image_distance(image a, image b)
{
    int i,j;
    image dist = make_image(a.w, a.h, 1);
    for(i = 0; i < a.c; ++i){
        for(j = 0; j < a.h*a.w; ++j){
            dist.data[j] += pow(a.data[i*a.h*a.w+j]-b.data[i*a.h*a.w+j],2);
        }
    }
    for(j = 0; j < a.h*a.w; ++j){
        dist.data[j] = sqrt(dist.data[j]);
    }
    return dist;
}

void ghost_image(image source, image dest, int dx, int dy)
{
    int x,y,k;
    float max_dist = sqrt((-source.w/2. + .5)*(-source.w/2. + .5));
    for(k = 0; k < source.c; ++k){
        for(y = 0; y < source.h; ++y){
            for(x = 0; x < source.w; ++x){
                float dist = sqrt((x - source.w/2. + .5)*(x - source.w/2. + .5) + (y - source.h/2. + .5)*(y - source.h/2. + .5));
                float alpha = (1 - dist/max_dist);
                if(alpha < 0) alpha = 0;
                float v1 = get_pixel(source, x,y,k);
                float v2 = get_pixel(dest, dx+x,dy+y,k);
                float val = alpha*v1 + (1-alpha)*v2;
                set_pixel(dest, dx+x, dy+y, k, val);
            }
        }
    }
}

/*
**
**将输入source图片的像素值嵌入到目标图片dest中
**@param source 源图片
**@param dest 目标图片(相当于该图片的输出)
**@param dx 列偏移(dx=(source.w-dest.w)/2),因为源尺寸一般小于目标图片尺寸，所以要将源图嵌入目标图中心
**          源图需要在目标图上偏移dx开始插入，如下图：
**@param dy 行偏移(dy=(source.h-dest.h)/2)
**@details 下图所示，外层为目标图(dest)，内层为源图(source)，源图尺寸不超过目标图，源图嵌入到目标图中心
**
*  ###############
*  #             #
*  #<-->######   #
*  # dx #    #   #
*  #    ######   #
*  #      dy     #
*  ###############
**
**@details 此函数是将源图片嵌入到目标图片中心，意味着源图片的尺寸(宽高)不大于目标图片的尺寸，
**         目标图片在输入函数之前已经有初始化值了，因此如果源图片的尺寸小于目标图片尺寸，那么
**         源图片会覆盖掉目标图片中新区域的像素值，而目标图片四周多出的像素值则保持初始化值
*/
void embed_image(image source, image dest, int dx, int dy)
{
    int x,y,k;
    for(k = 0; k < source.c; ++k){
        for(y = 0; y < source.h; ++y){
            for(x = 0; x < source.w; ++x){

                /*获取k通道y行x列处的像素值*/
                float val = get_pixel(source, x,y,k);

                /*设置目标图中k通道dy+y行，dx+x列处的像素值为val*/
                set_pixel(dest, dx+x, dy+y, k, val);
            }
        }
    }
}

image collapse_image_layers(image source, int border)
{
    int h = source.h;
    h = (h+border)*source.c - border;
    image dest = make_image(source.w, h, 1);
    int i;
    for(i = 0; i < source.c; ++i){
        image layer = get_image_layer(source, i);
        int h_offset = i*(source.h+border);
        embed_image(layer, dest, 0, h_offset);
        free_image(layer);
    }
    return dest;
}

/*
**将图像的im所有通道的像素值严格的限制在0.0~1.0之内(像素值已经归一化),(小于0的设为0，大于1的设为1，其他保持不变)
*/
void constrain_image(image im)
{
    int i;
    for(i = 0; i < im.w*im.h*im.c; ++i){
        if(im.data[i] < 0) im.data[i] = 0;
        if(im.data[i] > 1) im.data[i] = 1;
    }
}

void normalize_image(image p)
{
    int i;
    float min = 9999999;
    float max = -999999;

    for(i = 0; i < p.h*p.w*p.c; ++i){
        float v = p.data[i];
        if(v < min) min = v;
        if(v > max) max = v;
    }
    if(max - min < .000000001){
        min = 0;
        max = 1;
    }
    for(i = 0; i < p.c*p.w*p.h; ++i){
        p.data[i] = (p.data[i] - min)/(max-min);
    }
}

void normalize_image2(image p)
{
    float *min = calloc(p.c, sizeof(float));
    float *max = calloc(p.c, sizeof(float));
    int i,j;
    for(i = 0; i < p.c; ++i) min[i] = max[i] = p.data[i*p.h*p.w];

    for(j = 0; j < p.c; ++j){
        for(i = 0; i < p.h*p.w; ++i){
            float v = p.data[i+j*p.h*p.w];
            if(v < min[j]) min[j] = v;
            if(v > max[j]) max[j] = v;
        }
    }
    for(i = 0; i < p.c; ++i){
        if(max[i] - min[i] < .000000001){
            min[i] = 0;
            max[i] = 1;
        }
    }
    for(j = 0; j < p.c; ++j){
        for(i = 0; i < p.w*p.h; ++i){
            p.data[i+j*p.h*p.w] = (p.data[i+j*p.h*p.w] - min[j])/(max[j]-min[j]);
        }
    }
    free(min);
    free(max);
}

void copy_image_into(image src, image dest)
{
    memcpy(dest.data, src.data, src.h*src.w*src.c*sizeof(float));
}

image copy_image(image p)
{
    image copy = p;
    copy.data = calloc(p.h*p.w*p.c, sizeof(float));
    memcpy(copy.data, p.data, p.h*p.w*p.c*sizeof(float));
    return copy;
}

void rgbgr_image(image im)
{
    int i;
    for(i = 0; i < im.w*im.h; ++i){
        float swap = im.data[i];
        im.data[i] = im.data[i+im.w*im.h*2];
        im.data[i+im.w*im.h*2] = swap;
    }
}

#ifdef OPENCV
void show_image_cv(image p, const char *name, IplImage *disp)
{
    int x,y,k;
    if(p.c == 3) rgbgr_image(p);
    //normalize_image(copy);

    char buff[256];
    //sprintf(buff, "%s (%d)", name, windows);
    sprintf(buff, "%s", name);

    int step = disp->widthStep;
    cvNamedWindow(buff, CV_WINDOW_NORMAL); 
    //cvMoveWindow(buff, 100*(windows%10) + 200*(windows/10), 100*(windows%10));
    ++windows;
    for(y = 0; y < p.h; ++y){
        for(x = 0; x < p.w; ++x){
            for(k= 0; k < p.c; ++k){
                disp->imageData[y*step + x*p.c + k] = (unsigned char)(get_pixel(p,x,y,k)*255);
            }
        }
    }
    if(0){
        int w = 448;
        int h = w*p.h/p.w;
        if(h > 1000){
            h = 1000;
            w = h*p.w/p.h;
        }
        IplImage *buffer = disp;
        disp = cvCreateImage(cvSize(w, h), buffer->depth, buffer->nChannels);
        cvResize(buffer, disp, CV_INTER_LINEAR);
        cvReleaseImage(&buffer);
    }
    cvShowImage(buff, disp);
}
#endif

void show_image(image p, const char *name)
{
#ifdef OPENCV
    IplImage *disp = cvCreateImage(cvSize(p.w,p.h), IPL_DEPTH_8U, p.c);
    image copy = copy_image(p);
    constrain_image(copy);
    show_image_cv(copy, name, disp);
    free_image(copy);
    cvReleaseImage(&disp);
#else
    fprintf(stderr, "Not compiled with OpenCV, saving to %s.png instead\n", name);
    save_image(p, name);
#endif
}

#ifdef OPENCV

void ipl_into_image(IplImage* src, image im)
{
    unsigned char *data = (unsigned char *)src->imageData;
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    int step = src->widthStep;
    int i, j, k;

    for(i = 0; i < h; ++i){
        for(k= 0; k < c; ++k){
            for(j = 0; j < w; ++j){
                im.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
            }
        }
    }
}

image ipl_to_image(IplImage* src)
{
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    image out = make_image(w, h, c);
    ipl_into_image(src, out);
    return out;
}

image load_image_cv(char *filename, int channels)
{
    IplImage* src = 0;
    int flag = -1;
    if (channels == 0) flag = -1;
    else if (channels == 1) flag = 0;
    else if (channels == 3) flag = 1;
    else {
        fprintf(stderr, "OpenCV can't force load with %d channels\n", channels);
    }

    if( (src = cvLoadImage(filename, flag)) == 0 )
    {
        fprintf(stderr, "Cannot load image \"%s\"\n", filename);
        char buff[256];
        sprintf(buff, "echo %s >> bad.list", filename);
        system(buff);
        return make_image(10,10,3);
        //exit(0);
    }
    image out = ipl_to_image(src);
    cvReleaseImage(&src);
    rgbgr_image(out);
    return out;
}

void flush_stream_buffer(CvCapture *cap, int n)
{
    int i;
    for(i = 0; i < n; ++i) {
        cvQueryFrame(cap);
    }
}

image get_image_from_stream(CvCapture *cap)
{
    IplImage* src = cvQueryFrame(cap);
    if (!src) return make_empty_image(0,0,0);
    image im = ipl_to_image(src);
    rgbgr_image(im);
    return im;
}

int fill_image_from_stream(CvCapture *cap, image im)
{
    IplImage* src = cvQueryFrame(cap);
    if (!src) return 0;
    ipl_into_image(src, im);
    rgbgr_image(im);
    return 1;
}

void save_image_jpg(image p, const char *name)
{
    image copy = copy_image(p);
    if(p.c == 3) rgbgr_image(copy);
    int x,y,k;

    char buff[256];
    sprintf(buff, "%s.jpg", name);

    IplImage *disp = cvCreateImage(cvSize(p.w,p.h), IPL_DEPTH_8U, p.c);
    int step = disp->widthStep;
    for(y = 0; y < p.h; ++y){
        for(x = 0; x < p.w; ++x){
            for(k= 0; k < p.c; ++k){
                disp->imageData[y*step + x*p.c + k] = (unsigned char)(get_pixel(copy,x,y,k)*255);
            }
        }
    }
    cvSaveImage(buff, disp,0);
    cvReleaseImage(&disp);
    free_image(copy);
}
#endif

void save_image_png(image im, const char *name)
{
    char buff[256];
    //sprintf(buff, "%s (%d)", name, windows);
    sprintf(buff, "%s.png", name);
    unsigned char *data = calloc(im.w*im.h*im.c, sizeof(char));
    int i,k;
    for(k = 0; k < im.c; ++k){
        for(i = 0; i < im.w*im.h; ++i){
            data[i*im.c+k] = (unsigned char) (255*im.data[i + k*im.w*im.h]);
        }
    }
    int success = stbi_write_png(buff, im.w, im.h, im.c, data, im.w*im.c);
    free(data);
    if(!success) fprintf(stderr, "Failed to write image %s\n", buff);
}

void save_image(image im, const char *name)
{
#ifdef OPENCV
    save_image_jpg(im, name);
#else
    save_image_png(im, name);
#endif
}


void show_image_layers(image p, char *name)
{
    int i;
    char buff[256];
    for(i = 0; i < p.c; ++i){
        sprintf(buff, "%s - Layer %d", name, i);
        image layer = get_image_layer(p, i);
        show_image(layer, buff);
        free_image(layer);
    }
}

void show_image_collapsed(image p, char *name)
{
    image c = collapse_image_layers(p, 1);
    show_image(c, name);
    free_image(c);
}

/*
**创建一张空image，指定图片的宽，高，通道三个属性
**同时初始化数据为0，(此函数还未为图片数据分配内存)
**@param w 图片的宽度
**@param h 图片的高度
**@param c 图片的通道
**@return image
*/
image make_empty_image(int w, int h, int c)
{
    image out;
    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}


/*
**创建一张image，并未image的图像数据动态分配内存
**@param w 图片宽
**@param h 图片高
**@param c 图片通道
**@return image，已经指定宽，高，通道数属性，并已经动态分配好内存的image，可以往里面存入数据了
**
*/
image make_image(int w, int h, int c)
{
    /*创建一张空图片(仅指定了图片的三个属性，为分配图像数据的内存)*/
    image out = make_empty_image(w,h,c);

    /*为图像数据动态分配内存：总共有h*w*c个元素，每个元素的字节数为sizeof(float)*/
    out.data = calloc(h*w*c, sizeof(float));
    return out;
}

image make_random_image(int w, int h, int c)
{
    image out = make_empty_image(w,h,c);
    out.data = calloc(h*w*c, sizeof(float));
    int i;
    for(i = 0; i < w*h*c; ++i){
        out.data[i] = (rand_normal() * .25) + .5;
    }
    return out;
}

image float_to_image(int w, int h, int c, float *data)
{
    image out = make_empty_image(w,h,c);
    out.data = data;
    return out;
}


/*
** 先用双线性插值对输入图像im进行重排得到一个虚拟中间图（之所以称为虚拟，是因为这个中间图并不是一个真实存在的变量），
** 而后将中间图嵌入到canvas中（中间图尺寸比canvas小）或者将canvas当作一个mask在im上抠图（canvas尺寸小于中间图尺寸）（canvas是帆布/画布的意思）
** 输入： im    源图
**       w     中间图的宽度
**       h     中间图的高度
**       dx    中间图插入到canvas的x方向上的偏移
**       dy    中间图插入到canvas的y方向上的偏移
**       canvas 目标图（在传入到本函数之前，所有像素值已经初始化为某个值，比如0.5）
** 说明： 此函数是实现图像数据增强手段中的一种：平移（不含旋转等其他变换）。先用双线性插值将源图im重排至一个中间图，im与中间图的长宽比不一定一样，
**      而后将中间图放入到输出图canvas中（这两者的长宽比也不一定一样），分两种情况，如果中间图的尺寸小于输出图canvas，
**      显然，中间图是填不满canvas的，那么就将中间图随机的嵌入到canvas的某个位置（dx,dy就是起始位置，此时二者大于0,相对canvas的起始位置，这两个数是在函数外随机生成的），
**      因为canvas已经在函数外初始化了（比如所有像素值初始化为0.5），剩下没填满的就保持为初始化值；
**      如果canvas的尺寸小于中间图尺寸，那么就将canvas当作一个mask，在中间图上随机位置（dx,dy就是起始位置，此时二者小于0,相对中间图的起始位置）抠图。
**      因为canvas与中间图的长宽比不一样，因此，有可能在一个方向是嵌入情况，而在另一个方向上是mask情况，总之理解就可以了。
**      可以参考一下：
**      https://medium.com/@vivek.yadav/dealing-with-unbalanced-data-generating-additional-data-by-jittering-the-original-image-7497fe2119c3
**      上面网址给出了100张对原始图像进行增强之后的图片，可以看到很多图片有填满的，也有未填满的（无黑区），且位置随机。
**      (当然，网址中给出的图片包含了多种用于图片数据增强的变换，此函数仅仅完成最简单的一种：平移)
**
*/
void place_image(image im, int w, int h, int dx, int dy, image canvas)
{
    int x, y, c;

    /*外层循环，通道数量*/
    for(c = 0; c < im.c; ++c){

        /*中层循环和内层循环次数分别为中间图的行数和中间图的列数，这两个循环下来，实际可得中间图的所有像素值(当然，此处并没有形成一个真实的中间图变量)*/
        for(y = 0; y < h; ++y){
            for(x = 0; x < w; ++x){

                /*x为中间图的列坐标，((x / w) * im.w)得到的中间图对应在原图上的列坐标(按比例得到，亚像素坐标) */
                int rx = ((float)x / w) * im.w;

                /*y为中间图的行坐标，( (y / h) * im.h)得到的中间图对应在原图上的行坐标*/
                /*Tips:这里的代码显然存在不足，应该放入中层循环，以减少不必要的计算*/
                int ry = ((float)y / h) * im.h;

                /*利用原图进行双线性插值得到中间图在c通道x行y列处的像素值*/
                float val = bilinear_interpolate(im, rx, ry, c);

                /*设置canvas中c通道，y+dy行 x+dx列的像素值为val*/
                /*dx，dy可大于0，也可以小于0，大于0很好理解，对于小于0的情况，x+dx，y+dy会有一段小于0的，这时
                **set_pixel()函数无作为,直到x+dx , y+dy大于0时，才有作为，这样canvas相当于其一个mask的作用
                */
                set_pixel(canvas, x + dx, y + dy, c, val);
            }
        }
    }
}

image center_crop_image(image im, int w, int h)
{
    int m = (im.w < im.h) ? im.w : im.h;   
    image c = crop_image(im, (im.w - m) / 2, (im.h - m)/2, m, m);
    image r = resize_image(c, w, h);
    free_image(c);
    return r;
}

image rotate_crop_image(image im, float rad, float s, int w, int h, float dx, float dy, float aspect)
{
    int x, y, c;
    float cx = im.w/2.;
    float cy = im.h/2.;
    image rot = make_image(w, h, im.c);
    for(c = 0; c < im.c; ++c){
        for(y = 0; y < h; ++y){
            for(x = 0; x < w; ++x){
                float rx = cos(rad)*((x - w/2.)/s*aspect + dx/s*aspect) - sin(rad)*((y - h/2.)/s + dy/s) + cx;
                float ry = sin(rad)*((x - w/2.)/s*aspect + dx/s*aspect) + cos(rad)*((y - h/2.)/s + dy/s) + cy;
                float val = bilinear_interpolate(im, rx, ry, c);
                set_pixel(rot, x, y, c, val);
            }
        }
    }
    return rot;
}

image rotate_image(image im, float rad)
{
    int x, y, c;
    float cx = im.w/2.;
    float cy = im.h/2.;
    image rot = make_image(im.w, im.h, im.c);
    for(c = 0; c < im.c; ++c){
        for(y = 0; y < im.h; ++y){
            for(x = 0; x < im.w; ++x){
                float rx = cos(rad)*(x-cx) - sin(rad)*(y-cy) + cx;
                float ry = sin(rad)*(x-cx) + cos(rad)*(y-cy) + cy;
                float val = bilinear_interpolate(im, rx, ry, c);
                set_pixel(rot, x, y, c, val);
            }
        }
    }
    return rot;
}

/*将图片m的所有像素赋值为s，常用于初始化图片*/
void fill_image(image m, float s)
{
    int i;
    for(i = 0; i < m.h*m.w*m.c; ++i) m.data[i] = s;
}

void translate_image(image m, float s)
{
    int i;
    for(i = 0; i < m.h*m.w*m.c; ++i) m.data[i] += s;
}

void scale_image(image m, float s)
{
    int i;
    for(i = 0; i < m.h*m.w*m.c; ++i) m.data[i] *= s;
}

image crop_image(image im, int dx, int dy, int w, int h)
{
    image cropped = make_image(w, h, im.c);
    int i, j, k;
    for(k = 0; k < im.c; ++k){
        for(j = 0; j < h; ++j){
            for(i = 0; i < w; ++i){
                int r = j + dy;
                int c = i + dx;
                float val = 0;
                r = constrain_int(r, 0, im.h-1);
                c = constrain_int(c, 0, im.w-1);
                val = get_pixel(im, c, r, k);
                set_pixel(cropped, i, j, k, val);
            }
        }
    }
    return cropped;
}

int best_3d_shift_r(image a, image b, int min, int max)
{
    if(min == max) return min;
    int mid = floor((min + max) / 2.);
    image c1 = crop_image(b, 0, mid, b.w, b.h);
    image c2 = crop_image(b, 0, mid+1, b.w, b.h);
    float d1 = dist_array(c1.data, a.data, a.w*a.h*a.c, 10);
    float d2 = dist_array(c2.data, a.data, a.w*a.h*a.c, 10);
    free_image(c1);
    free_image(c2);
    if(d1 < d2) return best_3d_shift_r(a, b, min, mid);
    else return best_3d_shift_r(a, b, mid+1, max);
}

int best_3d_shift(image a, image b, int min, int max)
{
    int i;
    int best = 0;
    float best_distance = FLT_MAX;
    for(i = min; i <= max; i += 2){
        image c = crop_image(b, 0, i, b.w, b.h);
        float d = dist_array(c.data, a.data, a.w*a.h*a.c, 100);
        if(d < best_distance){
            best_distance = d;
            best = i;
        }
        printf("%d %f\n", i, d);
        free_image(c);
    }
    return best;
}

void composite_3d(char *f1, char *f2, char *out, int delta)
{
    if(!out) out = "out";
    image a = load_image(f1, 0,0,0);
    image b = load_image(f2, 0,0,0);
    int shift = best_3d_shift_r(a, b, -a.h/100, a.h/100);

    image c1 = crop_image(b, 10, shift, b.w, b.h);
    float d1 = dist_array(c1.data, a.data, a.w*a.h*a.c, 100);
    image c2 = crop_image(b, -10, shift, b.w, b.h);
    float d2 = dist_array(c2.data, a.data, a.w*a.h*a.c, 100);

    if(d2 < d1 && 0){
        image swap = a;
        a = b;
        b = swap;
        shift = -shift;
        printf("swapped, %d\n", shift);
    }
    else{
        printf("%d\n", shift);
    }

    image c = crop_image(b, delta, shift, a.w, a.h);
    int i;
    for(i = 0; i < c.w*c.h; ++i){
        c.data[i] = a.data[i];
    }
#ifdef OPENCV
    save_image_jpg(c, out);
#else
    save_image(c, out);
#endif
}



void letterbox_image_into(image im, int w, int h, image boxed)
{
    int new_w = im.w;
    int new_h = im.h;

    
    if (((float)w/im.w) < ((float)h/im.h)) {
        new_w = w;
        new_h = (im.h * w)/im.w;
    } else {
        new_h = h;
        new_w = (im.w * h)/im.h;
    }

    image resized = resize_image(im, new_w, new_h);

    embed_image(resized, boxed, (w-new_w)/2, (h-new_h)/2); 
    
    free_image(resized);
}


/** 按照神经网络能够接受处理的图片尺寸对输入图片进行尺寸调整（主要包括插值缩放与嵌入两个步骤）.
* @param im   输入图片（读入的原始图片）
* @param w    网络处理的标准图片宽度（列）
* @param h    网络处理的标准图片高度（行）
* @return boxed，image类型，其尺寸为神经网络能够处理的标准图片尺寸
* @details 此函数常用于将图片输入卷积神经网络之前，因为构建神经网络时，一般会指定神经网络第一层接受的输入图片尺寸，
*          比如yolo.cfg神经网络配置文件中，就指定了height=416,width=416，也就是说图片输入进神经网络之前，
*          需要标准化图片的尺寸为416,416（此时w=416,h=416）.流程主要包括两步：
*          1）利用插值等比例缩放图片尺寸，缩放后图片resized的尺寸与原图尺寸的比例为w/im.w与h/im.h中的较小值；
*          2）等比例缩放后的图片resized，还不是神经网络能够处理的标准尺寸（但是resized宽、高二者有一个等于标准尺寸），
*             第二步进一步将缩放后的图片resized嵌入到标准尺寸图片boxed中并返回
*/ 
image letterbox_image(image im, int w, int h)
{
     /*确认缩放后图片(resized)的尺寸大小*/
    int new_w = im.w;
    int new_h = im.h;

    /*缩放后的图片的尺寸与原图成等比例关系，比例值为w/im.w h/im.h的较小者*/
    /*
    **总之最后的结果有两种：1）new_w = w， new_h = im.h * ( w / im.w )
    **                      2) new_w = im.w * ( h / im.h) , new_h = h
    */
    /*也即resized的宽高有一个跟标准尺寸w或h相同，另一个按照比例确定*/

    if (((float)w/im.w) < ((float)h/im.h)) {
        new_w = w;
        /*new_h = im.h * ( w / im.w)*/
        new_h = (im.h * w)/im.w;
    } else {
        new_h = h;
        new_w = (im.w * h)/im.h;
    }

     /*第一步：缩放图片，使缩放后的图片尺寸为new_w，new_h*/
    image resized = resize_image(im, new_w, new_h);

    /*创建标准尺寸图片，并初始化所有像素值为0.5*/
    image boxed = make_image(w, h, im.c);
    fill_image(boxed, .5);

    //int i;
    //for(i = 0; i < boxed.w*boxed.h*boxed.c; ++i) boxed.data[i] = 0;
     /*第二步：将缩放后的图片嵌入到标准尺寸图片中*/
    embed_image(resized, boxed, (w-new_w)/2, (h-new_h)/2);

    /*切记释放没用的resized的内存，然后返回最终的标准尺寸图片*/ 
    free_image(resized);
    return boxed;
}

image resize_max(image im, int max)
{
    int w = im.w;
    int h = im.h;
    if(w > h){
        h = (h * max) / w;
        w = max;
    } else {
        w = (w * max) / h;
        h = max;
    }
    if(w == im.w && h == im.h) return im;
    image resized = resize_image(im, w, h);
    return resized;
}

image resize_min(image im, int min)
{
    int w = im.w;
    int h = im.h;
    if(w < h){
        h = (h * min) / w;
        w = min;
    } else {
        w = (w * min) / h;
        h = min;
    }
    if(w == im.w && h == im.h) return im;
    image resized = resize_image(im, w, h);
    return resized;
}

image random_crop_image(image im, int w, int h)
{
    int dx = rand_int(0, im.w - w);
    int dy = rand_int(0, im.h - h);
    image crop = crop_image(im, dx, dy, w, h);
    return crop;
}

augment_args random_augment_args(image im, float angle, float aspect, int low, int high, int w, int h)
{
    augment_args a = {0};
    aspect = rand_scale(aspect);
    int r = rand_int(low, high);
    int min = (im.h < im.w*aspect) ? im.h : im.w*aspect;
    float scale = (float)r / min;

    float rad = rand_uniform(-angle, angle) * TWO_PI / 360.;

    float dx = (im.w*scale/aspect - w) / 2.;
    float dy = (im.h*scale - w) / 2.;
    //if(dx < 0) dx = 0;
    //if(dy < 0) dy = 0;
    dx = rand_uniform(-dx, dx);
    dy = rand_uniform(-dy, dy);

    a.rad = rad;
    a.scale = scale;
    a.w = w;
    a.h = h;
    a.dx = dx;
    a.dy = dy;
    a.aspect = aspect;
    return a;
}

image random_augment_image(image im, float angle, float aspect, int low, int high, int w, int h)
{
    augment_args a = random_augment_args(im, angle, aspect, low, high, w, h);
    image crop = rotate_crop_image(im, a.rad, a.scale, a.w, a.h, a.dx, a.dy, a.aspect);
    return crop;
}

float three_way_max(float a, float b, float c)
{
    return (a > b) ? ( (a > c) ? a : c) : ( (b > c) ? b : c) ;
}

float three_way_min(float a, float b, float c)
{
    return (a < b) ? ( (a < c) ? a : c) : ( (b < c) ? b : c) ;
}

void yuv_to_rgb(image im)
{
    assert(im.c == 3);
    int i, j;
    float r, g, b;
    float y, u, v;
    for(j = 0; j < im.h; ++j){
        for(i = 0; i < im.w; ++i){
            y = get_pixel(im, i , j, 0);
            u = get_pixel(im, i , j, 1);
            v = get_pixel(im, i , j, 2);

            r = y + 1.13983*v;
            g = y + -.39465*u + -.58060*v;
            b = y + 2.03211*u;

            set_pixel(im, i, j, 0, r);
            set_pixel(im, i, j, 1, g);
            set_pixel(im, i, j, 2, b);
        }
    }
}

void rgb_to_yuv(image im)
{
    assert(im.c == 3);
    int i, j;
    float r, g, b;
    float y, u, v;
    for(j = 0; j < im.h; ++j){
        for(i = 0; i < im.w; ++i){
            r = get_pixel(im, i , j, 0);
            g = get_pixel(im, i , j, 1);
            b = get_pixel(im, i , j, 2);

            y = .299*r + .587*g + .114*b;
            u = -.14713*r + -.28886*g + .436*b;
            v = .615*r + -.51499*g + -.10001*b;

            set_pixel(im, i, j, 0, y);
            set_pixel(im, i, j, 1, u);
            set_pixel(im, i, j, 2, v);
        }
    }
}

// http://www.cs.rit.edu/~ncs/color/t_convert.html
void rgb_to_hsv(image im)
{
    assert(im.c == 3);
    int i, j;
    float r, g, b;
    float h, s, v;
    for(j = 0; j < im.h; ++j){
        for(i = 0; i < im.w; ++i){
            r = get_pixel(im, i , j, 0);
            g = get_pixel(im, i , j, 1);
            b = get_pixel(im, i , j, 2);
            float max = three_way_max(r,g,b);
            float min = three_way_min(r,g,b);
            float delta = max - min;
            v = max;
            if(max == 0){
                s = 0;
                h = 0;
            }else{
                s = delta/max;
                if(r == max){
                    h = (g - b) / delta;
                } else if (g == max) {
                    h = 2 + (b - r) / delta;
                } else {
                    h = 4 + (r - g) / delta;
                }
                if (h < 0) h += 6;
                h = h/6.;
            }
            set_pixel(im, i, j, 0, h);
            set_pixel(im, i, j, 1, s);
            set_pixel(im, i, j, 2, v);
        }
    }
}

void hsv_to_rgb(image im)
{
    assert(im.c == 3);
    int i, j;
    float r, g, b;
    float h, s, v;
    float f, p, q, t;
    for(j = 0; j < im.h; ++j){
        for(i = 0; i < im.w; ++i){
            h = 6 * get_pixel(im, i , j, 0);
            s = get_pixel(im, i , j, 1);
            v = get_pixel(im, i , j, 2);
            if (s == 0) {
                r = g = b = v;
            } else {
                int index = floor(h);
                f = h - index;
                p = v*(1-s);
                q = v*(1-s*f);
                t = v*(1-s*(1-f));
                if(index == 0){
                    r = v; g = t; b = p;
                } else if(index == 1){
                    r = q; g = v; b = p;
                } else if(index == 2){
                    r = p; g = v; b = t;
                } else if(index == 3){
                    r = p; g = q; b = v;
                } else if(index == 4){
                    r = t; g = p; b = v;
                } else {
                    r = v; g = p; b = q;
                }
            }
            set_pixel(im, i, j, 0, r);
            set_pixel(im, i, j, 1, g);
            set_pixel(im, i, j, 2, b);
        }
    }
}

void grayscale_image_3c(image im)
{
    assert(im.c == 3);
    int i, j, k;
    float scale[] = {0.299, 0.587, 0.114};
    for(j = 0; j < im.h; ++j){
        for(i = 0; i < im.w; ++i){
            float val = 0;
            for(k = 0; k < 3; ++k){
                val += scale[k]*get_pixel(im, i, j, k);
            }
            im.data[0*im.h*im.w + im.w*j + i] = val;
            im.data[1*im.h*im.w + im.w*j + i] = val;
            im.data[2*im.h*im.w + im.w*j + i] = val;
        }
    }
}

image grayscale_image(image im)
{
    assert(im.c == 3);
    int i, j, k;
    image gray = make_image(im.w, im.h, 1);
    float scale[] = {0.299, 0.587, 0.114};
    for(k = 0; k < im.c; ++k){
        for(j = 0; j < im.h; ++j){
            for(i = 0; i < im.w; ++i){
                gray.data[i+im.w*j] += scale[k]*get_pixel(im, i, j, k);
            }
        }
    }
    return gray;
}

image threshold_image(image im, float thresh)
{
    int i;
    image t = make_image(im.w, im.h, im.c);
    for(i = 0; i < im.w*im.h*im.c; ++i){
        t.data[i] = im.data[i]>thresh ? 1 : 0;
    }
    return t;
}

image blend_image(image fore, image back, float alpha)
{
    assert(fore.w == back.w && fore.h == back.h && fore.c == back.c);
    image blend = make_image(fore.w, fore.h, fore.c);
    int i, j, k;
    for(k = 0; k < fore.c; ++k){
        for(j = 0; j < fore.h; ++j){
            for(i = 0; i < fore.w; ++i){
                float val = alpha * get_pixel(fore, i, j, k) + 
                    (1 - alpha)* get_pixel(back, i, j, k);
                set_pixel(blend, i, j, k, val);
            }
        }
    }
    return blend;
}


/*
** 将图像c通道的所有元素值乘以一个因子v（放大或者缩小）
** 输入： im    输入图片 
**       c     要进行元素值缩放的通道编号
**       v     因子
** 该函数用处：比如用于数据增强，对图像的饱和度以及明度进行缩放
*/
void scale_image_channel(image im, int c, float v)
{
    int i, j;
    for(j = 0; j < im.h; ++j){
        for(i = 0; i < im.w; ++i){
            // 获取im图像c通道j行i列的元素值，而后乘以因子v进行缩放，然后更新当前像素值
            float pix = get_pixel(im, i, j, c);
            pix = pix*v;
            set_pixel(im, i, j, c, pix);
        }
    }
}

void translate_image_channel(image im, int c, float v)
{
    int i, j;
    for(j = 0; j < im.h; ++j){
        for(i = 0; i < im.w; ++i){
            float pix = get_pixel(im, i, j, c);
            pix = pix+v;
            set_pixel(im, i, j, c, pix);
        }
    }
}

image binarize_image(image im)
{
    image c = copy_image(im);
    int i;
    for(i = 0; i < im.w * im.h * im.c; ++i){
        if(c.data[i] > .5) c.data[i] = 1;
        else c.data[i] = 0;
    }
    return c;
}

void saturate_image(image im, float sat)
{
    rgb_to_hsv(im);
    scale_image_channel(im, 1, sat);
    hsv_to_rgb(im);
    constrain_image(im);
}

void hue_image(image im, float hue)
{
    rgb_to_hsv(im);
    int i;
    for(i = 0; i < im.w*im.h; ++i){
        im.data[i] = im.data[i] + hue;
        if (im.data[i] > 1) im.data[i] -= 1;
        if (im.data[i] < 0) im.data[i] += 1;
    }
    hsv_to_rgb(im);
    constrain_image(im);
}

void exposure_image(image im, float sat)
{
    rgb_to_hsv(im);
    scale_image_channel(im, 2, sat);
    hsv_to_rgb(im);
    constrain_image(im);
}


/*
** 先将输入的彩色图片im由RGB颜色空间转换至hsv空间，而后在hsv空间的h,s,v三通道上在添加噪声，以实现数据增强
** 输入： im     读入的彩色图片
**       hue    色调偏差值
**       saturation    色彩饱和度（取值范围0~1）缩放因子
**       exposure    明度（色彩明亮程度，0~1）缩放因子
*/
void distort_image(image im, float hue, float sat, float val)
{   
    /*将图像由rgb空间转至hsv空间*/
    rgb_to_hsv(im);

    /*缩放第二通道即s通道的值以进行图像jitter(绕动，或者说添加噪声)，进而实现数据增强*/
    scale_image_channel(im, 1, sat);
    scale_image_channel(im, 2, val);
    int i;

    /*对于第0通道即h通道，直接添加指定偏差值来实现图像jitter*/
    for(i = 0; i < im.w*im.h; ++i){
        im.data[i] = im.data[i] + hue;
        if (im.data[i] > 1) im.data[i] -= 1;
        if (im.data[i] < 0) im.data[i] += 1;
    }

    /*将图像变回rgb空间*/
    hsv_to_rgb(im);

    /*可能之前在hsv空间添加的绕动使得变换回RGB空间之后，其像素值不再合理范围之内
    **通过constrain_image()函数将严格限制在0~1范围内(小于0设置为0，大于1的设置为1，其他保持不变)
    */
    constrain_image(im);
}


/*
** 此函数先将输入的颜色图片由RGB颜色空间转换至hsv空间，而后在hsv空间的h,s,v三通道上在添加噪声，以实现数据增强
** 输入： im     读入的彩色图片
**       hue    色调（取值0度到360度）偏差最大值，实际色调偏差为-hue~hue之间的随机值
**       saturation    色彩饱和度（取值范围0~1）缩放最大值，实际为范围内的随机值
**       exposure    明度（色彩明亮程度，0~1）缩放最大值，实际为范围内的随机值
** 说明：色调正常都是0度到360度的，但是这里并没有乘以60，所以范围正常为0~6，此外，rgb_to_hsv()函数最后还除以了6进行了类似归一化的操作，
**      不知道是何意，总之不管怎样，在hsv_to_rgb()函数中将hsv转回至rgb配套上就可以了
*/
void random_distort_image(image im, float hue, float saturation, float exposure)
{
    /*下面依次产生具体的色调，饱和度，明度的偏差值*/
    float dhue = rand_uniform(-hue, hue);
    float dsat = rand_scale(saturation);
    float dexp = rand_scale(exposure);

    /*在hsv空间h,s,v三个通道上，为图像增加噪声（对于h是添加偏差，对于s,v是缩放），以进行图像数据增强*/
    distort_image(im, dhue, dsat, dexp);
}

void saturate_exposure_image(image im, float sat, float exposure)
{
    rgb_to_hsv(im);
    scale_image_channel(im, 1, sat);
    scale_image_channel(im, 2, exposure);
    hsv_to_rgb(im);
    constrain_image(im);
}


/*
**  重排图片的尺寸
**  输入： im  待重排的图片
**        w   新的图片宽度
**        h   新的图片高度
**  返回：目标图片（缩放后的图片）resized
**  思路：1）此处重排并不是保持像素个数意义上的重排，像素个数是可以变的，即图像缩放，这就涉及到像素插值操作;
**       2）重排操作分两步完成，第一步图像行数不变，仅缩放图像的列数，而后在水平方向上（x方向）进行像素线性插值，
**       3）第二步，在第一步基础上，保持列数不变（列在第一步已经缩放完成），缩放行数，在竖直方向（y方向）进行像素线性插值，
**         两步叠加在一起，其实就相当于是双线性插值
*/
image resize_image(image im, int w, int h)
{
    image resized = make_image(w, h, im.c);   
    image part = make_image(w, im.h, im.c);
    int r, c, k;
    float w_scale = (float)(im.w - 1) / (w - 1);
    float h_scale = (float)(im.h - 1) / (h - 1);
    for(k = 0; k < im.c; ++k){
        for(r = 0; r < im.h; ++r){
            for(c = 0; c < w; ++c){
                float val = 0;
                if(c == w-1 || im.w == 1){
                    val = get_pixel(im, im.w-1, r, k);
                } else {
                    float sx = c*w_scale;
                    int ix = (int) sx;
                    float dx = sx - ix;
                    val = (1 - dx) * get_pixel(im, ix, r, k) + dx * get_pixel(im, ix+1, r, k);
                }
                set_pixel(part, c, r, k, val);
            }
        }
    }
    for(k = 0; k < im.c; ++k){
        for(r = 0; r < h; ++r){
            float sy = r*h_scale;
            int iy = (int) sy;
            float dy = sy - iy;
            for(c = 0; c < w; ++c){
                float val = (1-dy) * get_pixel(part, c, iy, k);
                set_pixel(resized, c, r, k, val);
            }
            if(r == h-1 || im.h == 1) continue;
            for(c = 0; c < w; ++c){
                float val = dy * get_pixel(part, c, iy+1, k);
                add_pixel(resized, c, r, k, val);
            }
        }
    }

    free_image(part);
    return resized;
}


void test_resize(char *filename)
{
    image im = load_image(filename, 0,0, 3);
    float mag = mag_array(im.data, im.w*im.h*im.c);
    printf("L2 Norm: %f\n", mag);
    image gray = grayscale_image(im);

    image c1 = copy_image(im);
    image c2 = copy_image(im);
    image c3 = copy_image(im);
    image c4 = copy_image(im);
    distort_image(c1, .1, 1.5, 1.5);
    distort_image(c2, -.1, .66666, .66666);
    distort_image(c3, .1, 1.5, .66666);
    distort_image(c4, .1, .66666, 1.5);


    show_image(im,   "Original");
    show_image(gray, "Gray");
    show_image(c1, "C1");
    show_image(c2, "C2");
    show_image(c3, "C3");
    show_image(c4, "C4");
#ifdef OPENCV
    while(1){
        image aug = random_augment_image(im, 0, .75, 320, 448, 320, 320);
        show_image(aug, "aug");
        free_image(aug);


        float exposure = 1.15;
        float saturation = 1.15;
        float hue = .05;

        image c = copy_image(im);

        float dexp = rand_scale(exposure);
        float dsat = rand_scale(saturation);
        float dhue = rand_uniform(-hue, hue);

        distort_image(c, dhue, dsat, dexp);
        show_image(c, "rand");
        printf("%f %f %f\n", dhue, dsat, dexp);
        free_image(c);
        cvWaitKey(0);
    }
#endif
}


image load_image_stb(char *filename, int channels)
{
    int w, h, c;
    unsigned char *data = stbi_load(filename, &w, &h, &c, channels);
    if (!data) {
        fprintf(stderr, "Cannot load image \"%s\"\nSTB Reason: %s\n", filename, stbi_failure_reason());
        exit(0);
    }
    if(channels) c = channels;
    int i,j,k;
    image im = make_image(w, h, c);
    for(k = 0; k < c; ++k){
        for(j = 0; j < h; ++j){
            for(i = 0; i < w; ++i){
                int dst_index = i + w*j + w*h*k;
                int src_index = k + c*i + c*w*j;
                im.data[dst_index] = (float)data[src_index]/255.;
            }
        }
    }
    free(data);
    return im;
}

image load_image(char *filename, int w, int h, int c)
{
#ifdef OPENCV
    image out = load_image_cv(filename, c);
#else
    image out = load_image_stb(filename, c);
#endif

    if((h && w) && (h != out.h || w != out.w)){
        image resized = resize_image(out, w, h);
        free_image(out);
        out = resized;
    }
    return out;
}

image load_image_color(char *filename, int w, int h)
{
    return load_image(filename, w, h, 3);
}

image get_image_layer(image m, int l)
{
    image out = make_image(m.w, m.h, 1);
    int i;
    for(i = 0; i < m.h*m.w; ++i){
        out.data[i] = m.data[i+l*m.h*m.w];
    }
    return out;
}
void print_image(image m)
{
    int i, j, k;
    for(i =0 ; i < m.c; ++i){
        for(j =0 ; j < m.h; ++j){
            for(k = 0; k < m.w; ++k){
                printf("%.2lf, ", m.data[i*m.h*m.w + j*m.w + k]);
                if(k > 30) break;
            }
            printf("\n");
            if(j > 30) break;
        }
        printf("\n");
    }
    printf("\n");
}

image collapse_images_vert(image *ims, int n)
{
    int color = 1;
    int border = 1;
    int h,w,c;
    w = ims[0].w;
    h = (ims[0].h + border) * n - border;
    c = ims[0].c;
    if(c != 3 || !color){
        w = (w+border)*c - border;
        c = 1;
    }

    image filters = make_image(w, h, c);
    int i,j;
    for(i = 0; i < n; ++i){
        int h_offset = i*(ims[0].h+border);
        image copy = copy_image(ims[i]);
        //normalize_image(copy);
        if(c == 3 && color){
            embed_image(copy, filters, 0, h_offset);
        }
        else{
            for(j = 0; j < copy.c; ++j){
                int w_offset = j*(ims[0].w+border);
                image layer = get_image_layer(copy, j);
                embed_image(layer, filters, w_offset, h_offset);
                free_image(layer);
            }
        }
        free_image(copy);
    }
    return filters;
} 

image collapse_images_horz(image *ims, int n)
{
    int color = 1;
    int border = 1;
    int h,w,c;
    int size = ims[0].h;
    h = size;
    w = (ims[0].w + border) * n - border;
    c = ims[0].c;
    if(c != 3 || !color){
        h = (h+border)*c - border;
        c = 1;
    }

    image filters = make_image(w, h, c);
    int i,j;
    for(i = 0; i < n; ++i){
        int w_offset = i*(size+border);
        image copy = copy_image(ims[i]);
        //normalize_image(copy);
        if(c == 3 && color){
            embed_image(copy, filters, w_offset, 0);
        }
        else{
            for(j = 0; j < copy.c; ++j){
                int h_offset = j*(size+border);
                image layer = get_image_layer(copy, j);
                embed_image(layer, filters, w_offset, h_offset);
                free_image(layer);
            }
        }
        free_image(copy);
    }
    return filters;
} 

void show_image_normalized(image im, const char *name)
{
    image c = copy_image(im);
    normalize_image(c);
    show_image(c, name);
    free_image(c);
}

void show_images(image *ims, int n, char *window)
{
    image m = collapse_images_vert(ims, n);
    /*
       int w = 448;
       int h = ((float)m.h/m.w) * 448;
       if(h > 896){
       h = 896;
       w = ((float)m.w/m.h) * 896;
       }
       image sized = resize_image(m, w, h);
     */
    normalize_image(m);
    save_image(m, window);
    show_image(m, window);
    free_image(m);
}

void free_image(image m)
{
    if(m.data){
        free(m.data);
    }
}
