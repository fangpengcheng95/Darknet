#ifndef BOX_H
#define BOX_H
#include "darknet.h"

/**
 **box的定义在darknet.h中
 ** typedef struct{
 **   float x, y, w, h;
 ** } box;
 **物体检测定位矩形框，即用于定位物体的矩形框，
 **矩形框中心坐标dx（横坐标）,dy（纵坐标）
 **矩形框宽dw，高dh总共4个参数值，
 **不包含矩形框中所包含的物体的类别编号值。4个值都是比例坐标，也即占整个图片宽高的比例（dx为占图宽的比例，dy为占图高的比例）。
 */

typedef struct{
    float dx, dy, dw, dh;
} dbox;

float box_rmse(box a, box b);
dbox diou(box a, box b);
box decode_box(box b, box anchor);
box encode_box(box b, box anchor);

#endif
