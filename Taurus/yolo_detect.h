#ifndef YOLOV5_DETECT_H
#define YOLOV5_DETECT_H


#include "hi_comm_video.h"
#include "hi_type.h"

#if __cplusplus
extern "C" {
#endif


/**
 * 加载Yolov5模型
*/
HI_S32 Yolov5DetectModelLoad(uintptr_t *model);
/**
 * 卸载Yolov5模型
*/
HI_S32 Yolov5DetectModelUnLoad(uintptr_t model);
/**
 * 检测模型推理
*/
// HI_S32 Yolov5DetectModelCal(uintptr_t model, VIDEO_FRAME_INFO_S *srcFrm,VIDEO_FRAME_INFO_S *dstFrm);
HI_S32 Yolov5DetectModelCal(uintptr_t model, VIDEO_FRAME_INFO_S *srcFrm,VIDEO_FRAME_INFO_S *dstFrm,int num_1,char *uartRead_1);


#ifdef __cplusplus
}
#endif
#endif