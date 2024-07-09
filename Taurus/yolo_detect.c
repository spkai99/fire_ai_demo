#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>

#include "sample_comm_nnie.h"
#include "ai_infer_process.h"
#include "sample_media_ai.h"
#include "vgs_img.h"
#include "ive_img.h"
#include "misc_util.h"
#include "hisignalling.h"
#include <termios.h>

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* End of #ifdef __cplusplus */



#define DETECT_OBJ_MAX 32
#define RET_NUM_MAX 4
    // Draw the width of the line
#define WIDTH_LIMIT 32
#define HEIGHT_LIMIT 32

static IVE_IMAGE_S img;
#define DRAW_RETC_THICK 2    // Draw the width of the line
// #define MODEL_FILE_DETECT "/userdata/models/yolo_detect/yolov5_drone.wk" // darknet framework wk model
#define MODEL_FILE_DETECT "/userdata/models/yolo_detect/yolov5_fire.wk" // darknet framework wk model
#define YOLO_MIN(a, b) ((a) > (b) ? (b) : (a))
#define YOLO_MAX(a, b) ((a) < (b) ? (b) : (a))
static IVE_IMAGE_S imgIn;
static IVE_IMAGE_S imgDst;
static VIDEO_FRAME_INFO_S frmIn;
static VIDEO_FRAME_INFO_S frmDst;
static int uartFd = 0;
extern int num_distance;
extern int threshold_distance;


void yolo_result_sort(yolo_result *output_result)
    { // 目前用这个做排序

        yolo_result *comparable_node = NULL; // 右节点，挨个指向右边所有节点

        yolo_result *comparable_next_node = NULL;

        yolo_result *current_node = output_result; // 左节点，其与右边每个节点做比较

        yolo_result *current_next_node = NULL;

        yolo_result temp_node = {0};

        while (current_node != NULL)
        {

            comparable_node = current_node->next;

            current_next_node = current_node->next; // 记录后续节点，方便调换数据后维持链表完整

            while (comparable_node != NULL)
            {

                comparable_next_node = comparable_node->next; // 记录后续节点，方便调换数据后维持链表完整

                if (current_node->score >= comparable_node->score)
                { // 如果大于它，说明后面的比它小，比较下一个

                    comparable_node = comparable_node->next;
                }
                else
                {
                    // 当大于 current_confidence 时，数据做调换，内存不变，小的放后面去
                    memcpy(&temp_node, current_node, sizeof(yolo_result));

                    memcpy(current_node, comparable_node, sizeof(yolo_result));

                    memcpy(comparable_node, &temp_node, sizeof(yolo_result));

                    current_node->next = current_next_node; // 链表接好

                    comparable_node->next = comparable_next_node;

                    comparable_node = comparable_node->next; // 更新位置，因为当前节点已经小于current_node ，不必再做比较
                }
            }

            current_node = current_node->next;
        }
    }

    void yolo_nms(yolo_result *output_result, float iou_threshold)
    {

        yolo_result *comparable_node = NULL; // 右节点，挨个指向右边所有节点

        yolo_result *comparable_former_node = NULL;

        yolo_result *current_node = output_result; // 左节点，其与右边每个节点做比较

        yolo_result *temp_node = NULL;

        float overlap_left_x = 0.0f;

        float overlap_left_y = 0.0f;

        float overlap_right_x = 0.0f;

        float overlap_right_y = 0.0f;

        float current_area = 0.0f, comparable_area = 0.0f, overlap_area = 0.0f;

        float nms_ratio = 0.0f;

        float overlap_w = 0.0f, overlap_h = 0.0f;

        //
        while (current_node != NULL)
        {

            comparable_node = current_node->next;

            comparable_former_node = current_node;
            // printf("current_node->score = %f\n", current_node->score);
            current_area = (current_node->right_down_x - current_node->left_up_x) * (current_node->right_down_y - current_node->left_up_y);

            while (comparable_node != NULL)
            {

                if (current_node->class_index != comparable_node->class_index)
                { // 如果类别不一致，没必要做 nms

                    comparable_former_node = comparable_node;

                    comparable_node = comparable_node->next;
                    continue;
                }
                // printf("comparable_node->score = %f\n", comparable_node->score);
                comparable_area = (comparable_node->right_down_x - comparable_node->left_up_x) * (comparable_node->right_down_y - comparable_node->left_up_y);

                overlap_left_x = YOLO_MAX(current_node->left_up_x, comparable_node->left_up_x);
                overlap_left_y = YOLO_MAX(current_node->left_up_y, comparable_node->left_up_y);

                overlap_right_x = YOLO_MIN(current_node->right_down_x, comparable_node->right_down_x);
                overlap_right_y = YOLO_MIN(current_node->right_down_y, comparable_node->right_down_y);

                overlap_w = YOLO_MAX((overlap_right_x - overlap_left_x), 0.0F);
                overlap_h = YOLO_MAX((overlap_right_y - overlap_left_y), 0.0F);
                overlap_area = YOLO_MAX((overlap_w * overlap_h), 0.0f); // 重叠区域面积

                nms_ratio = overlap_area / (current_area + comparable_area - overlap_area);

                if (nms_ratio > iou_threshold)
                { // 重叠过大，去掉

                    temp_node = comparable_node;

                    comparable_node = comparable_node->next;

                    comparable_former_node->next = comparable_node; // 链表接好

                    free(temp_node);
                }
                else
                {

                    comparable_former_node = comparable_node;

                    comparable_node = comparable_node->next;
                }
            }
            // printf("loop end \n");
            current_node = current_node->next;
        }
    }

    void printf_result(yolo_result *temp)
    {
        // printf("--------------------\n");

        while (temp != NULL)
        {

            printf("output_result->left_up_x = %f\t", temp->left_up_x);
            printf("output_result->left_up_y = %f\n", temp->left_up_y);

            printf("output_result->right_down_x = %f\t", temp->right_down_x);
            printf("output_result->right_down_y = %f\n", temp->right_down_y);

            printf("output_result->class_index = %d\t", temp->class_index);
            printf("output_result->score = %f\n\n", temp->score);

            temp = temp->next;
        }
        // printf("--------------------\n");
    }

    yolo_result* CopyYoloResult(yolo_result* source) {
        if (source == NULL) {
            return NULL; // 如果源为空，返回空指针
        }

        yolo_result* dest = (yolo_result*)malloc(sizeof(yolo_result));  // 创建新节点

        // 复制数据成员的值
        dest->left_up_x = source->left_up_x;
        dest->left_up_y = source->left_up_y;
        dest->right_down_x = source->right_down_x;
        dest->right_down_y = source->right_down_y;
        dest->class_index = source->class_index;
        dest->score = source->score;

        // 递归复制下一个节点
        dest->next = CopyYoloResult(source->next);

        return dest; // 返回新节点的指针
    }

 void release_result(yolo_result *output_result)
    {

        yolo_result *temp = NULL;

        while (output_result != NULL)
        {

            temp = output_result;

            output_result = output_result->next;

            free(temp);
        }
    }
/**
 * 加载Yolov5模型
*/
HI_S32 Yolov5DetectModelLoad(uintptr_t *model)
{
    SAMPLE_SVP_NNIE_CFG_S *self = NULL;
    HI_S32 ret;
    ret = CnnCreate(&self, MODEL_FILE_DETECT);

    SAMPLE_PRT("RET=%u\n", ret);
    *model = ret < 0 ? 0 : (uintptr_t)self; // 首先判断ret < 0是否成立，若成立则返回":"前边的项0
    SAMPLE_PRT("Load YOLO detect model success\n");

    uartFd = UartOpenInit();
    if (uartFd < 0) {
        printf("uart1 open failed\r\n");
    } else {
        printf("uart1 open successed\r\n");
    }
    return ret;
}
/**
 * 卸载Yolov5模型
*/
HI_S32 Yolov5DetectModelUnLoad(uintptr_t model)
{
    CnnDestroy((SAMPLE_SVP_NNIE_CFG_S*)model);
    close(uartFd);
    SAMPLE_PRT("UnLoad YOLO detect model success\n");
    return 0;
}

static void YOLOTran(yolo_result* result, int srcWidth, int srcHeight, int dstWidth, int dstHeight)
{
    if (!srcWidth || !srcHeight) {
        HI_ASSERT(srcWidth && srcHeight);
    } else {
        if (srcWidth != 0 && srcHeight != 0) {
            result->left_up_x = result->left_up_x * dstWidth / srcWidth * HI_OVEN_BASE / HI_OVEN_BASE;
            result->right_down_x = result->right_down_x * dstWidth / srcWidth * HI_OVEN_BASE / HI_OVEN_BASE;
            result->left_up_y = result->left_up_y * dstHeight / srcHeight * HI_OVEN_BASE / HI_OVEN_BASE;
            result->right_down_y = result->right_down_y * dstHeight / srcHeight * HI_OVEN_BASE / HI_OVEN_BASE;
        }
    }
}
/**
 * 检测模型推理
*/
// 串口使用(在interconnection_server/hisignalling.c中)
//    int UartRead(int uartFd, char *buf, int len, int timeoutMs);
//    int UartSend(int fd, char *buf, int len);
HI_S32 Yolov5DetectModelCal(uintptr_t model, VIDEO_FRAME_INFO_S *srcFrm,VIDEO_FRAME_INFO_S *dstFrm,int num_1,char *uartRead_1)
{
    SAMPLE_SVP_NNIE_CFG_S *self = (SAMPLE_SVP_NNIE_CFG_S *)model;
    HI_S32 ret;
    yolo_result *output_result =NULL;
    yolo_result *origin_data = NULL;
    float left_x = 130.0f;  //预设中心区域左上角x坐标
    float left_y = 55.0f;   //预设中心区域左上角y坐标
    float right_x = 250.0f; //预设中心区域右下角x坐标
    float right_y = 155.0f; //预设中心区域右下角y坐标
    const float area_thres = 4900;   //面积阈值作为触发喷水条件
    float areas = 0.0f;     //用来存储面积
    float center_x = 0.0;   //通过输出目标检测结果的坐标计算center_X和center_y  与预设区域比较发送相应指令
    float center_y = 0.0;

    int flag = 1;   //每次外设部分执行完功能 向3516发OK  初始化默认为1
    //串口用到的变量
    // unsigned char uartReadBuff[3] = {0};
    unsigned char uartsend[8] = {0};
    // int readLen = 0;
    // unsigned int RecvLen = strlen("OK");

    ret = FrmToOrigImg((VIDEO_FRAME_INFO_S*)srcFrm, &img);    //改变原始图像数据格式
    SAMPLE_CHECK_EXPR_RET(ret !=HI_SUCCESS, ret,"Yolov5 detect for YUV frm to Img FAIL,ret =%#X\n",ret);
    ret = YOLOV5CalImg(self,&img,&output_result);
    if (output_result != NULL)
    {
        yolo_result_sort(output_result);//输出结果排序，为下nms做准备
        yolo_nms(output_result,0.001f); //nms
        // printf_result(output_result);
        origin_data = CopyYoloResult(output_result);
        YOLOTran(output_result, 384, 216,dstFrm->stVFrame.u32Width, dstFrm->stVFrame.u32Height);
        MppFrmDrawYOLORects(dstFrm, output_result, 1, RGB888_RED, DRAW_RETC_THICK);
    }
    // readLen = UartRead(uartFd,uartReadBuff,RecvLen,0);  /* 1000 :time out */
    printf("num_1:--------------%d",num_1);
    // if(readLen>0){
    //     printf("uartReadBuff read data:%s\n",uartReadBuff);
    //     if(strcmp(uartReadBuff,"OK")==0)
    //     {
    //        strcpy(uartRead_1,uartReadBuff); //传回来OK表示可以进行下一步操作
    //        flag =1;
    //     }
    // }
    uartsend[0]='@';
    uartsend[2]='\r';
    uartsend[3]='\n';
    while(origin_data!=NULL){
        areas = (int)(origin_data->right_down_x-origin_data->left_up_x)*(origin_data->right_down_y-origin_data->left_up_y);
        center_x = (origin_data->right_down_x + origin_data->left_up_x) / 2;
        center_y = (origin_data->right_down_y + origin_data->left_up_y) / 2;
        if (num_1%2==0){
            //控制小车移动以及舵机俯仰
             if(center_x<left_x)
            {uartsend[1]='L';   //控制小车向左转动
             UartSend(uartFd,(unsigned char *)uartsend,strlen(uartsend));
             } else if(center_x>right_x)
            {uartsend[1]='R';   //控制小车向右转动
             UartSend(uartFd,(unsigned char *)uartsend,strlen(uartsend));
             } else{
            if (center_y<left_y)
            {uartsend[1]='U';
            UartSend(uartFd,(unsigned char *)uartsend,strlen(uartsend));}  //舵机向上移动
            else if(center_y>right_y)
            {uartsend[1]='D';   //舵机向下移动
            UartSend(uartFd,(unsigned char *)uartsend,strlen(uartsend));
            }else{
                uartsend[1]='W';  // w 向前移动   保证目标物在视野中央
                if(areas>=area_thres){
                    uartsend[1]='O';     //可以喷水  
                }
                UartSend(uartFd,(unsigned char *)uartsend,strlen(uartsend));
                // else{
                //     uartsend[1]='N';     //NONONO
                // }
            // }
            }}
            //灭火
            // if(!(center_x<left_x||center_x>right_x||center_y<left_y||center_y>right_y))
            //   {  

            //   }}
        //发送串口指令
        // if(flag)
        // {
        // UartSend(uartFd,(unsigned char *)uartsend,strlen(uartsend));
        printf("uartReadBuff read data:%s\n",uartsend);
            // flag = 0;
        // }
        origin_data = origin_data->next;
        break;
    }
    return ret;
}


#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* End of #ifdef __cplusplus */
