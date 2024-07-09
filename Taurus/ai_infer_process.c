/*
 * Copyright (c) 2022 HiSilicon (Shanghai) Technologies CO., LIMITED.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <signal.h>
#include <pthread.h>
#include <sys/prctl.h>
#include <math.h>
#include <assert.h>

#include "hi_common.h"
#include "hi_comm_sys.h"
#include "hi_comm_svp.h"
#include "sample_comm_svp.h"
#include "hi_comm_ive.h"
#include "sample_svp_nnie_software.h"
#include "sample_media_ai.h"
#include "ai_infer_process.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* End of #ifdef __cplusplus */


#define yolo_layer_num 3 // yolo layer 层数

#ifndef YOLO_MIN
#define YOLO_MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef YOLO_MAX
#define YOLO_MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif

#define IMAGE_W 384.0f // 输入图片大小

#define IMAGE_H 384.0f

typedef struct anchor_w_h
{
	float anchor_w;
	
	float anchor_h;
}anchor_w_h;




anchor_w_h  anchor_grids[3][3] = {{{10.0f, 13.0f}, {16.0f, 30.0f}, {33.0f, 23.0f}}, // small yolo layer 层 anchor

									{{30.0f, 61.0f}, {62.0f, 45.0f}, {59.0f, 119.0f}}, // middle yolo layer 层 anchor

									{{116.0f, 90.0f}, {156.0f, 198.0f}, {373.0f, 326.0f}}}; // large yolo layer 层 anchor
	
float strides[3] = {8.0f, 16.0f, 32.0f}; // 每个 yolo 层，grid 大小，与上面顺序对应

int map_size[3] = {48, 24, 12}; // 每个 yolo 层，feature map size 大小，与上面顺序对应 输入图片大小为384

#define USLEEP_TIME   100 // 100: usleep time, in microseconds

#define ARRAY_SUBSCRIPT_0     0
#define ARRAY_SUBSCRIPT_1     1
#define ARRAY_SUBSCRIPT_2     2
#define ARRAY_SUBSCRIPT_3     3
#define ARRAY_SUBSCRIPT_4     4
#define ARRAY_SUBSCRIPT_5     5
#define ARRAY_SUBSCRIPT_6     6
#define ARRAY_SUBSCRIPT_7     7
#define ARRAY_SUBSCRIPT_8     8
#define ARRAY_SUBSCRIPT_9     9

#define ARRAY_SUBSCRIPT_OFFSET_1    1
#define ARRAY_SUBSCRIPT_OFFSET_2    2
#define ARRAY_SUBSCRIPT_OFFSET_3    3

#define THRESH_MIN         0.25

/*
 * CNN 参数
 * CNN parameter
 */
static SAMPLE_SVP_NNIE_MODEL_S g_stCnnModel = {0};
static SAMPLE_SVP_NNIE_PARAM_S g_stCnnNnieParam = {0};
static SAMPLE_SVP_NNIE_CNN_SOFTWARE_PARAM_S g_stCnnSoftwareParam = {0};

/*
 * Yolov2 参数
 * Yolov2 parameter
 */
static SAMPLE_SVP_NNIE_MODEL_S g_stYolov2Model = {0};
static SAMPLE_SVP_NNIE_PARAM_S g_stYolov2NnieParam = {0};
static SAMPLE_SVP_NNIE_YOLOV2_SOFTWARE_PARAM_S g_stYolov2SoftwareParam = {0};

/*
 * 函数：Cnn software参数初始化
 * function : Cnn software parameter init
 */
static HI_S32 SampleSvpNnieCnnSoftwareParaInit(SAMPLE_SVP_NNIE_CFG_S* pstNnieCfg,
    SAMPLE_SVP_NNIE_PARAM_S *pstCnnPara, SAMPLE_SVP_NNIE_CNN_SOFTWARE_PARAM_S* pstCnnSoftWarePara)
{
    HI_U32 u32GetTopNMemSize;
    HI_U32 u32GetTopBufSize;
    HI_U32 u32GetTopFrameSize;
    HI_U32 u32TotalSize;
    HI_U32 u32ClassNum = pstCnnPara->pstModel->astSeg[0].astDstNode[0].unShape.stWhc.u32Width;
    HI_U64 u64PhyAddr = 0;
    HI_U8* pu8VirAddr = NULL;
    HI_S32 s32Ret;

    /*
     * 获取内存大小
     * Get mem size
     */
    u32GetTopFrameSize = pstCnnSoftWarePara->u32TopN*sizeof(SAMPLE_SVP_NNIE_CNN_GETTOPN_UNIT_S);
    u32GetTopNMemSize = SAMPLE_SVP_NNIE_ALIGN16(u32GetTopFrameSize)*pstNnieCfg->u32MaxInputNum;
    u32GetTopBufSize = u32ClassNum*sizeof(SAMPLE_SVP_NNIE_CNN_GETTOPN_UNIT_S);
    u32TotalSize = u32GetTopNMemSize + u32GetTopBufSize;

    /*
     * 在用户态分配MMZ内存
     * Malloc memory in user mode
     */
    s32Ret = SAMPLE_COMM_SVP_MallocMem("SAMPLE_CNN_INIT", NULL, (HI_U64*)&u64PhyAddr,
        (void**)&pu8VirAddr, u32TotalSize);
    SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,Malloc memory failed!\n");
    memset_s(pu8VirAddr, u32TotalSize, 0, u32TotalSize);

    /*
     * 初始化GetTopN参数
     * Init GetTopN param
     */
    pstCnnSoftWarePara->stGetTopN.u32Num = pstNnieCfg->u32MaxInputNum;
    pstCnnSoftWarePara->stGetTopN.unShape.stWhc.u32Chn = 1;
    pstCnnSoftWarePara->stGetTopN.unShape.stWhc.u32Height = 1;
    pstCnnSoftWarePara->stGetTopN.unShape.stWhc.u32Width = u32GetTopFrameSize / sizeof(HI_U32);
    pstCnnSoftWarePara->stGetTopN.u32Stride = SAMPLE_SVP_NNIE_ALIGN16(u32GetTopFrameSize);
    pstCnnSoftWarePara->stGetTopN.u64PhyAddr = u64PhyAddr;
    pstCnnSoftWarePara->stGetTopN.u64VirAddr = (HI_U64)(HI_UL)pu8VirAddr;

    /*
     * 初始化AssistBuf参数
     * Init AssistBuf
     */
    pstCnnSoftWarePara->stAssistBuf.u32Size = u32GetTopBufSize;
    pstCnnSoftWarePara->stAssistBuf.u64PhyAddr = u64PhyAddr + u32GetTopNMemSize;
    pstCnnSoftWarePara->stAssistBuf.u64VirAddr = (HI_U64)(HI_UL)pu8VirAddr + u32GetTopNMemSize;

    return s32Ret;
}

/*
 * 函数：Cnn software参数去初始化
 * function : Cnn software deinit
 */
static HI_S32 SampleSvpNnieCnnSoftwareDeinit(SAMPLE_SVP_NNIE_CNN_SOFTWARE_PARAM_S* pstCnnSoftWarePara)
{
    HI_S32 s32Ret = HI_SUCCESS;
    SAMPLE_SVP_CHECK_EXPR_RET(pstCnnSoftWarePara == NULL, HI_INVALID_VALUE, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error, pstCnnSoftWarePara can't be NULL!\n");
    if (pstCnnSoftWarePara->stGetTopN.u64PhyAddr != 0 && pstCnnSoftWarePara->stGetTopN.u64VirAddr != 0) {
        SAMPLE_SVP_MMZ_FREE(pstCnnSoftWarePara->stGetTopN.u64PhyAddr,
            pstCnnSoftWarePara->stGetTopN.u64VirAddr);
        pstCnnSoftWarePara->stGetTopN.u64PhyAddr = 0;
        pstCnnSoftWarePara->stGetTopN.u64VirAddr = 0;
    }
    return s32Ret;
}

/*
 * 函数：Cnn去初始化
 * function : Cnn Deinit
 */
static HI_S32 SampleSvpNnieCnnDeinit(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_CNN_SOFTWARE_PARAM_S* pstSoftWareParam, SAMPLE_SVP_NNIE_MODEL_S* pstNnieModel)
{
    HI_S32 s32Ret = HI_SUCCESS;
    /*
     * SAMPLE_SVP_NNIE_PARAM_S参数去初始化
     * Hardware param deinit
     */
    if (pstNnieParam != NULL) {
        s32Ret = SAMPLE_COMM_SVP_NNIE_ParamDeinit(pstNnieParam);
        SAMPLE_SVP_CHECK_EXPR_TRACE(HI_SUCCESS != s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error,SAMPLE_COMM_SVP_NNIE_ParamDeinit failed!\n");
    }
    /*
     * SAMPLE_SVP_NNIE_CNN_SOFTWARE_PARAM_S参数去初始化
     * Software param deinit
     */
    if (pstSoftWareParam != NULL) {
        s32Ret = SampleSvpNnieCnnSoftwareDeinit(pstSoftWareParam);
        SAMPLE_SVP_CHECK_EXPR_TRACE(HI_SUCCESS != s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error,SampleSvpNnieCnnSoftwareDeinit failed!\n");
    }
    /*
     * SAMPLE_SVP_NNIE_MODEL_S参数去初始化
     * Model deinit
     */
    if (pstNnieModel != NULL) {
        s32Ret = SAMPLE_COMM_SVP_NNIE_UnloadModel(pstNnieModel);
        SAMPLE_SVP_CHECK_EXPR_TRACE(HI_SUCCESS != s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error,SAMPLE_COMM_SVP_NNIE_UnloadModel failed!\n");
    }
    return s32Ret;
}

/*
 * 函数：Cnn参数初始化
 * function : Cnn Param init
 */
static HI_S32 SampleSvpNnieCnnParamInit(SAMPLE_SVP_NNIE_CFG_S* pstNnieCfg,
    SAMPLE_SVP_NNIE_PARAM_S *pstCnnPara, SAMPLE_SVP_NNIE_CNN_SOFTWARE_PARAM_S* pstCnnSoftWarePara)
{
    HI_S32 s32Ret;
    /*
     * 初始化hardware参数
     * Init hardware param
     */
    s32Ret = SAMPLE_COMM_SVP_NNIE_ParamInit(pstNnieCfg, pstCnnPara);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, INIT_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x),SAMPLE_COMM_SVP_NNIE_ParamInit failed!\n", s32Ret);

    /*
     * 初始化software参数
     * Init software param
     */
    if (pstCnnSoftWarePara != NULL) {
        s32Ret = SampleSvpNnieCnnSoftwareParaInit(pstNnieCfg, pstCnnPara, pstCnnSoftWarePara);
        SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, INIT_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error(%#x),SampleSvpNnieCnnSoftwareParaInit failed!\n", s32Ret);
    }

    return s32Ret;
INIT_FAIL_0:
    s32Ret = SampleSvpNnieCnnDeinit(pstCnnPara, pstCnnSoftWarePara, NULL);
    SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x),SampleSvpNnieCnnDeinit failed!\n", s32Ret);
    return HI_FAILURE;
}

/*
 * 基于模型文件创建CNN模型
 * Create CNN model based mode file
 */
int CnnCreate(SAMPLE_SVP_NNIE_CFG_S **model, const char* modelFile)
{
    SAMPLE_SVP_NNIE_CFG_S *self;
    HI_U32 u32PicNum = 1;
    HI_S32 s32Ret;

    self = (SAMPLE_SVP_NNIE_CFG_S*)malloc(sizeof(*self));
    HI_ASSERT(self);
    if (memset_s(self, sizeof(*self), 0x00, sizeof(*self)) != EOK) {
        HI_ASSERT(0);
    }
    /*
     * 设置配置参数
     * Set configuration parameter
     */
    self->pszPic = NULL;
    self->u32MaxInputNum = u32PicNum; // max input image num in each batch
    self->u32MaxRoiNum = 0;
    self->aenNnieCoreId[0] = SVP_NNIE_ID_0; // set NNIE core
    g_stCnnSoftwareParam.u32TopN = 5; // 5: value of the u32TopN

    /*
     * 加载CNN模型
     * Load cnn model
     */
    SAMPLE_SVP_TRACE_INFO("Cnn Load model!\n");
    s32Ret = SAMPLE_COMM_SVP_NNIE_LoadModel((char*)modelFile, &g_stCnnModel);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CNN_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_COMM_SVP_NNIE_LoadModel failed!\n");

    /*
     * CNN参数初始化
     * Cnn软件参数设置在SampleSvpNnieCnnSoftwareParaInit,
     * 如果用户更改了网络结构，请确保
     * SampleSvpNnieCnnSoftwareParaInit函数中的参数设置是正确的
     *
     * CNN parameter initialization
     * Cnn software parameters are set in SampleSvpNnieCnnSoftwareParaInit,
     * if user has changed net struct, please make sure the parameter settings in
     * SampleSvpNnieCnnSoftwareParaInit function are correct
     */
    SAMPLE_SVP_TRACE_INFO("Cnn parameter initialization!\n");
    g_stCnnNnieParam.pstModel = &g_stCnnModel.stModel;
    s32Ret = SampleSvpNnieCnnParamInit(self, &g_stCnnNnieParam, &g_stCnnSoftwareParam);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CNN_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SampleSvpNnieCnnParamInit failed!\n");

    /*
     * 模型关键数据
     * Model important information
     */
    SAMPLE_PRT("model={ type=%x, frmNum=%u, chnNum=%u, w=%u, h=%u, stride=%u }\n",
        g_stCnnNnieParam.astSegData[0].astSrc[0].enType,
        g_stCnnNnieParam.astSegData[0].astSrc[0].u32Num,
        g_stCnnNnieParam.astSegData[0].astSrc[0].unShape.stWhc.u32Chn,
        g_stCnnNnieParam.astSegData[0].astSrc[0].unShape.stWhc.u32Width,
        g_stCnnNnieParam.astSegData[0].astSrc[0].unShape.stWhc.u32Height,
        g_stCnnNnieParam.astSegData[0].astSrc[0].u32Stride);

    /*
     * 记录TskBuf地址信息
     * Record TskBuf address information
     */
    s32Ret = HI_MPI_SVP_NNIE_AddTskBuf(&(g_stCnnNnieParam.astForwardCtrl[0].stTskBuf));
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CNN_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,HI_MPI_SVP_NNIE_AddTskBuf failed!\n");
    *model = self;
    return 0;

    CNN_FAIL_0:
        SampleSvpNnieCnnDeinit(&g_stCnnNnieParam, &g_stCnnSoftwareParam, &g_stCnnModel);
        *model = NULL;
        return -1;
}

/*
 * 销毁CNN模型
 * Destroy CNN model
 */
void CnnDestroy(SAMPLE_SVP_NNIE_CFG_S *self)
{
    HI_S32 s32Ret;

    /*
     * 移除TskBuf地址信息
     * Remove TskBuf address information
     */
    s32Ret = HI_MPI_SVP_NNIE_RemoveTskBuf(&(g_stCnnNnieParam.astForwardCtrl[0].stTskBuf));
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CNN_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,HI_MPI_SVP_NNIE_RemoveTskBuf failed!\n");

    CNN_FAIL_0:
        SampleSvpNnieCnnDeinit(&g_stCnnNnieParam, &g_stCnnSoftwareParam, &g_stCnnModel);
        free(self);
}

static HI_S32 FillNnieByImg(SAMPLE_SVP_NNIE_CFG_S* pstNnieCfg,
    SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam, int segId, int nodeId, const IVE_IMAGE_S *img)
{
    HI_U32 i;
    HI_U32 j;
    HI_U32 n;
    HI_U32 u32Height = 0;
    HI_U32 u32Width = 0;
    HI_U32 u32Chn = 0;
    HI_U32 u32Stride = 0;
    HI_U32 u32VarSize;
    HI_U8 *pu8PicAddr = NULL;

    /*
     * 获取数据大小
     * Get data size
     */
    if (SVP_BLOB_TYPE_U8 <= pstNnieParam->astSegData[segId].astSrc[nodeId].enType &&
        SVP_BLOB_TYPE_YVU422SP >= pstNnieParam->astSegData[segId].astSrc[nodeId].enType) {
        u32VarSize = sizeof(HI_U8);
    } else {
        u32VarSize = sizeof(HI_U32);
    }

    /*
     * 填充源数据
     * Fill src data
     */
    if (SVP_BLOB_TYPE_SEQ_S32 == pstNnieParam->astSegData[segId].astSrc[nodeId].enType) {
        HI_ASSERT(0);
    } else {
        u32Height = pstNnieParam->astSegData[segId].astSrc[nodeId].unShape.stWhc.u32Height;
        u32Width = pstNnieParam->astSegData[segId].astSrc[nodeId].unShape.stWhc.u32Width;
        u32Chn = pstNnieParam->astSegData[segId].astSrc[nodeId].unShape.stWhc.u32Chn;
        u32Stride = pstNnieParam->astSegData[segId].astSrc[nodeId].u32Stride;
        pu8PicAddr = SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_U8,
            pstNnieParam->astSegData[segId].astSrc[nodeId].u64VirAddr);

        if (SVP_BLOB_TYPE_YVU420SP == pstNnieParam->astSegData[segId].astSrc[nodeId].enType) {
            HI_ASSERT(pstNnieParam->astSegData[segId].astSrc[nodeId].u32Num == 1);
            for (n = 0; n < pstNnieParam->astSegData[segId].astSrc[nodeId].u32Num; n++) {
                // Y
                const uint8_t *srcData = (const uint8_t*)(uintptr_t)img->au64VirAddr[0];
                HI_ASSERT(srcData);
                for (j = 0; j < u32Height; j++) {
                    if (memcpy_s(pu8PicAddr, u32Width * u32VarSize, srcData, u32Width * u32VarSize) != EOK) {
                        HI_ASSERT(0);
                    }
                    pu8PicAddr += u32Stride;
                    srcData += img->au32Stride[0];
                }
                // UV
                srcData = (const uint8_t*)(uintptr_t)img->au64VirAddr[1];
                HI_ASSERT(srcData);
                for (j = 0; j < u32Height / 2; j++) { // 2: 1/2Height
                    if (memcpy_s(pu8PicAddr, u32Width * u32VarSize, srcData, u32Width * u32VarSize) != EOK) {
                        HI_ASSERT(0);
                    }
                    pu8PicAddr += u32Stride;
                    srcData += img->au32Stride[1];
                }
            }
        } else if (SVP_BLOB_TYPE_YVU422SP == pstNnieParam->astSegData[segId].astSrc[nodeId].enType) {
            HI_ASSERT(0);
        } else {
            for (n = 0; n < pstNnieParam->astSegData[segId].astSrc[nodeId].u32Num; n++) {
                for (i = 0; i < u32Chn; i++) {
                    const uint8_t *srcData = (const uint8_t*)(uintptr_t)img->au64VirAddr[i];
                    HI_ASSERT(srcData);
                    for (j = 0; j < u32Height; j++) {
                        if (memcpy_s(pu8PicAddr, u32Width * u32VarSize, srcData, u32Width * u32VarSize) != EOK) {
                            HI_ASSERT(0);
                        }
                        pu8PicAddr += u32Stride;
                        srcData += img->au32Stride[i];
                    }
                }
            }
        }

        SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[segId].astSrc[nodeId].u64PhyAddr,
            SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID, pstNnieParam->astSegData[segId].astSrc[nodeId].u64VirAddr),
            pstNnieParam->astSegData[segId].astSrc[nodeId].u32Num*u32Chn*u32Height*u32Stride);
    }

    return HI_SUCCESS;
}

void CnnFetchRes(SVP_BLOB_S *pstGetTopN, HI_U32 u32TopN, RecogNumInfo resBuf[], int resSize, int* resLen)
{
    HI_ASSERT(pstGetTopN);
    HI_U32 i;
    HI_U32 j = 0;
    HI_U32 *pu32Tmp = NULL;
    HI_U32 u32Stride = pstGetTopN->u32Stride;
    if (memset_s(resBuf, resSize * sizeof(resBuf[0]), 0x00, resSize * sizeof(resBuf[0])) != EOK) {
        HI_ASSERT(0);
    }

    int resId = 0;
    pu32Tmp = (HI_U32*)((HI_UL)pstGetTopN->u64VirAddr + j * u32Stride);
    for (i = 0; i < u32TopN * 2 && resId < resSize; i += 2, resId++) { // 2: u32TopN*2
        resBuf[resId].num = pu32Tmp[i];
        resBuf[resId].score = pu32Tmp[i + 1];
    }
    *resLen = resId;
}

/*
 * 函数：NNIE网络预测
 * function : NNIE Forward
 */
static HI_S32 SAMPLE_SVP_NNIE_Forward(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S* pstInputDataIdx,
    SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S* pstProcSegIdx, HI_BOOL bInstant)
{
    HI_S32 s32Ret = HI_SUCCESS;
    HI_U32 i;
    HI_U32 j;
    HI_BOOL bFinish = HI_FALSE;
    SVP_NNIE_HANDLE hSvpNnieHandle = 0;
    HI_U32 u32TotalStepNum = 0;

    SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].stTskBuf.u64PhyAddr,
        SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID,
        pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].stTskBuf.u64VirAddr),
        pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].stTskBuf.u32Size);

    for (i = 0; i < pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].u32DstNum; i++) {
        if (pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].enType == SVP_BLOB_TYPE_SEQ_S32) {
            for (j = 0; j < pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Num; j++) {
                u32TotalStepNum += *(SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_U32,
                    pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stSeq.u64VirAddrStep) + j);
            }
            SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64PhyAddr,
                SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID,
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr),
                u32TotalStepNum*pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Stride);
        } else {
            SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64PhyAddr,
                SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID,
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr),
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Num*
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stWhc.u32Chn*
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stWhc.u32Height*
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Stride);
        }
    }

    /*
     * 根据节点名称设置输入blob
     * Set input blob according to node name
     */
    if (pstInputDataIdx->u32SegIdx != pstProcSegIdx->u32SegIdx) {
        for (i = 0; i < pstNnieParam->pstModel->astSeg[pstProcSegIdx->u32SegIdx].u16SrcNum; i++) {
            for (j = 0; j < pstNnieParam->pstModel->astSeg[pstInputDataIdx->u32SegIdx].u16DstNum; j++) {
                if (strncmp(pstNnieParam->pstModel->astSeg[pstInputDataIdx->u32SegIdx].astDstNode[j].szName,
                    pstNnieParam->pstModel->astSeg[pstProcSegIdx->u32SegIdx].astSrcNode[i].szName,
                    SVP_NNIE_NODE_NAME_LEN) == 0) {
                    pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astSrc[i] =
                        pstNnieParam->astSegData[pstInputDataIdx->u32SegIdx].astDst[j];
                    break;
                }
            }
            SAMPLE_SVP_CHECK_EXPR_RET((j == pstNnieParam->pstModel->astSeg[pstInputDataIdx->u32SegIdx].u16DstNum),
                HI_FAILURE, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error,can't find %d-th seg's %d-th src blob!\n",
                pstProcSegIdx->u32SegIdx, i);
        }
    }

    /*
     * 多节点输入输出的CNN类型网络预测
     * CNN-type network prediction with multi-node input and output
     */
    s32Ret = HI_MPI_SVP_NNIE_Forward(&hSvpNnieHandle,
        pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astSrc,
        pstNnieParam->pstModel, pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst,
        &pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx], bInstant);
    SAMPLE_SVP_CHECK_EXPR_RET(s32Ret != HI_SUCCESS, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,HI_MPI_SVP_NNIE_Forward failed!\n");

    if (bInstant) {
        /*
         * 查询任务是否完成
         * Query whether the task is completed
         */
        while (HI_ERR_SVP_NNIE_QUERY_TIMEOUT == (s32Ret =
            HI_MPI_SVP_NNIE_Query(pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].enNnieId,
            hSvpNnieHandle, &bFinish, HI_TRUE))) {
            usleep(USLEEP_TIME);
            SAMPLE_SVP_TRACE(SAMPLE_SVP_ERR_LEVEL_INFO,
                "HI_MPI_SVP_NNIE_Query Query timeout!\n");
        }
    }
    u32TotalStepNum = 0;

    for (i = 0; i < pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].u32DstNum; i++) {
        if (SVP_BLOB_TYPE_SEQ_S32 == pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].enType) {
            for (j = 0; j < pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Num; j++) {
                u32TotalStepNum += *(SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_U32,
                    pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stSeq.u64VirAddrStep) + j);
            }
            SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64PhyAddr,
                SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID,
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr),
                u32TotalStepNum*pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Stride);
        } else {
            SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64PhyAddr,
                SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID,
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr),
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Num*
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stWhc.u32Chn*
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stWhc.u32Height*
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Stride);
        }
    }

    return s32Ret;
}

/*
 * 对一帧图像进行推理
 * Calculate a frame of image
 */
int CnnCalImg(SAMPLE_SVP_NNIE_CFG_S* self,
    const IVE_IMAGE_S *img, RecogNumInfo resBuf[], int resSize, int* resLen)
{
    HI_S32 s32Ret;
    SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S stInputDataIdx = {0};
    SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S stProcSegIdx = {0};

    /*
     * 填充源数据
     * Fill src data
     */
    self->pszPic = NULL;
    stInputDataIdx.u32SegIdx = 0;
    stInputDataIdx.u32NodeIdx = 0;
    s32Ret = FillNnieByImg(self, &g_stCnnNnieParam, 0, 0, img);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CNN_FAIL_1, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_FillSrcData failed!\n");

    /*
     * NNIE推理(process the 0-th segment)
     * NNIE process(process the 0-th segment)
     */
    stProcSegIdx.u32SegIdx = 0;
    s32Ret = SAMPLE_SVP_NNIE_Forward(&g_stCnnNnieParam, &stInputDataIdx, &stProcSegIdx, HI_TRUE);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CNN_FAIL_1, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_Forward failed!\n");

    /*
     * CPU处理
     * 如果用户更改了网络结构，请确保SAMPLE_SVP_NNIE_Cnn_GetTopN函数的输入数据是正确的
     *
     * Software process
     * if user has changed net struct, please make sure SAMPLE_SVP_NNIE_Cnn_GetTopN
     * function's input datas are correct
     */
    s32Ret = SAMPLE_SVP_NNIE_Cnn_GetTopN(&g_stCnnNnieParam, &g_stCnnSoftwareParam);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CNN_FAIL_1, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_CnnGetTopN failed!\n");

    /*
     * 打印结果
     * Print result
     */
    CnnFetchRes(&g_stCnnSoftwareParam.stGetTopN, g_stCnnSoftwareParam.u32TopN, resBuf, resSize, resLen);
    return 0;

    CNN_FAIL_1:
        return -1;
}

float sigmoid(float x){
	return (1.0f / ((float)exp((float)(-x)) + 1.0f));
}

/**
 * YOLOv5结果处理
*/
static unsigned int yolo_result_process(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam, float *strides, anchor_w_h (*anchor_grids)[3], int *map_size, yolo_result **output_result, float confidence_threshold)

{
    HI_S32 output_num = 0;
	
	HI_S32 anchor_num = 0;
	
	HI_S32 feature_length = 0;

    HI_S32 u32Stride = 0;

	float anchor_w = 0.0f;

	float anchor_h = 0.0f;

	int x = 0;
	
	int y = 0;

	HI_S32* output_addr = NULL;

	float confidence = 0.0f;

	float class_confidence = 0.0f;

	//float confidence_threshold = 0.4f;

	float pred_x = 0.0f;

	float pred_y = 0.0f;

	float pred_w = 0.0f;

	float pred_h = 0.0f;

	yolo_result *current = NULL;
	
	yolo_result *former = NULL;
	
	*output_result = NULL;

	unsigned int resltu_num = 0;
    for (int yolo_layer_index = 0; yolo_layer_index < yolo_layer_num; yolo_layer_index ++) { // 3 yolo layer 
    
        // if(SVP_BLOB_TYPE_S32 ==pstNnieParam->astSegData[0].astDst[yolo_layer_index].enType){
        //      SAMPLE_PRT("------------------------enType-------------------------------------\n");
        // }

		feature_length = pstNnieParam->astSegData[0].astDst[yolo_layer_index].unShape.stWhc.u32Width; // 1600 / 400 / 100
		
		output_num = pstNnieParam->astSegData[0].astDst[yolo_layer_index].unShape.stWhc.u32Height; // 53
		
		anchor_num = pstNnieParam->astSegData[0].astDst[yolo_layer_index].unShape.stWhc.u32Chn; // 3

		u32Stride = pstNnieParam->astSegData[0].astDst[yolo_layer_index].u32Stride; // 3

        //SAMPLE_PRT("u32Stride=%u,feature_length=%u,output_num=%u,anchor_num=%u\n",u32Stride,feature_length,output_num,anchor_num);

		output_addr = (HI_S32* )((HI_U8* )pstNnieParam->astSegData[0].astDst[yolo_layer_index].u64VirAddr); // yolo 输出的首地址
		
		for (int anchor_index = 0; anchor_index < anchor_num; anchor_index ++){ // 每个 grid 上有三个 anchor
				
				anchor_w = anchor_grids[yolo_layer_index][anchor_index].anchor_w;
				anchor_h = anchor_grids[yolo_layer_index][anchor_index].anchor_h;
				
				for (int coord_x_y = 0; coord_x_y < feature_length; coord_x_y ++){ // feature size 拉直后的长度，如 1600 400 100.
					y = coord_x_y / map_size[yolo_layer_index];
					x = coord_x_y % map_size[yolo_layer_index];
					
					confidence = *(output_addr + anchor_index * feature_length * output_num + 4 * feature_length + coord_x_y) /4096.0f;;
            
					confidence = sigmoid(confidence);
					// if (confidence > 0.15f){
                    //     SAMPLE_PRT("confidence=%f\n",confidence);
                    // }
					if (confidence > 0.25f){
						// SAMPLE_PRT("confidence=%f\n",confidence);
						for (int output_index = 5; output_index < output_num; output_index ++){
							class_confidence = *(output_addr + anchor_index * feature_length * output_num + output_index * feature_length + coord_x_y) / 4096.0f;  // class confidence
							class_confidence = sigmoid(class_confidence) * confidence;
                            //SAMPLE_PRT("class_confidence=%f\n",class_confidence);
							// 注意，yolo v5 的类别置信度并不需要选出个最大值，它的 label 是多标签，所以并不是 softmax
							if (class_confidence > 0.25f){
								
								pred_x = *(output_addr + anchor_index * feature_length * output_num + 0 * feature_length + coord_x_y) / 4096.0f; // x
								pred_y = *(output_addr + anchor_index * feature_length * output_num + 1 * feature_length + coord_x_y) / 4096.0f; // y
								pred_w = *(output_addr + anchor_index * feature_length * output_num + 2 * feature_length + coord_x_y) / 4096.0f; // w
								pred_h = *(output_addr + anchor_index * feature_length * output_num + 3 * feature_length + coord_x_y) / 4096.0f; // h 

								pred_x = sigmoid(pred_x);
								pred_y = sigmoid(pred_y);
								pred_w = sigmoid(pred_w);
								pred_h = sigmoid(pred_h);
								// bbox 输出结果处理
								pred_x = (pred_x * 2.0f - 0.5f + (float)x) * strides[yolo_layer_index];
								pred_y = (pred_y * 2.0f - 0.5f + (float)y) * strides[yolo_layer_index];
								pred_w = (pred_w * 2.0f) * (pred_w * 2.0f) * anchor_w;
								pred_h = (pred_h * 2.0f) * (pred_h * 2.0f) * anchor_h;
								
								current = (yolo_result *) malloc(sizeof(yolo_result));
								// 坐标转换 (x y w h) -> (x y x y)
								current->left_up_x = YOLO_MAX((pred_x - 0.5f * (pred_w - 1.0f)), 0.0f);
								current->left_up_y = YOLO_MAX((pred_y - 0.5f * (pred_h - 1.0f)), 0.0f);
								current->right_down_x = YOLO_MIN((pred_x + 0.5f * (pred_w - 1.0f)), IMAGE_W);
								current->right_down_y = YOLO_MIN((pred_y + 0.5f * (pred_h - 1.0f)), IMAGE_H);
								
								current->class_index = output_index - 5; // 类别索引，减 5 是因为前五个数据是 bbox + confidence 输出.
								current->score = class_confidence; // 置信度
								current->next = NULL;
								resltu_num ++;
								if (current->right_down_x - current->left_up_x >1 
                                &&current->right_down_y - current->left_up_y >1){

                                     if (*output_result == NULL){ // 存储结果
                                        *output_result = current;
                                        former = current;
                                    
                                    }else{
                                        former->next = current;
                                        former = former->next;
                                    }
                                }
								current = NULL;
							}
						}
					}
				}
		}
    }
	return resltu_num;
}

/*
*函数：YOLOv5对一帧图像推理
*/
int YOLOV5CalImg(SAMPLE_SVP_NNIE_CFG_S* self,const IVE_IMAGE_S *img, yolo_result **output_result)
{
    HI_S32 s32Ret;
    SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S stInputDataIdx = {0};
    SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S stProcSegIdx = {0};

    /*
    * 填充源数据 fill src data
    */  
    self->pszPic = NULL;
    stInputDataIdx.u32SegIdx = 0;
    stInputDataIdx.u32NodeIdx = 0;

    s32Ret = FillNnieByImg(self,&g_stCnnNnieParam, 0, 0, img);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CNN_FAIL_1, SAMPLE_SVP_ERR_LEVEL_ERROR,
    "Error,SAMPLE_SVP_NNIE_FillSrcData failed!\n");
    /**
     * NNIE推理 
    */
    stProcSegIdx.u32SegIdx = 0;
    s32Ret = SAMPLE_SVP_NNIE_Forward(&g_stCnnNnieParam, &stInputDataIdx,&stProcSegIdx,HI_TRUE);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CNN_FAIL_1, SAMPLE_SVP_ERR_LEVEL_ERROR,
    "Error,SAMPLE_SVP_NNIE_Forward failed!\n");
    yolo_result * output_result1 = NULL;
    unsigned int result_num = yolo_result_process(&g_stCnnNnieParam,strides,anchor_grids,map_size,&output_result1,0.15f);
    *output_result=output_result1;
    return 0;

CNN_FAIL_1:
    return -1;

}

/*
 * 函数：Yolov2 software参数初始化
 * function : Yolov2 software param init
 */
static HI_S32 SampleSvpNnieYolov2SoftwareInit(SAMPLE_SVP_NNIE_CFG_S* pstCfg,
    SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam, SAMPLE_SVP_NNIE_YOLOV2_SOFTWARE_PARAM_S* pstSoftWareParam)
{
    HI_S32 s32Ret;
    HI_U32 u32ClassNum = 0;
    HI_U32 u32BboxNum;
    HI_U32 u32TotalSize = 0;
    HI_U32 u32DstRoiSize;
    HI_U32 u32DstScoreSize;
    HI_U32 u32ClassRoiNumSize;
    HI_U32 u32TmpBufTotalSize;
    HI_U64 u64PhyAddr = 0;
    HI_U8* pu8VirAddr = NULL;

    pstSoftWareParam->u32OriImHeight = pstNnieParam->astSegData[0].astSrc[0].unShape.stWhc.u32Height;
    pstSoftWareParam->u32OriImWidth = pstNnieParam->astSegData[0].astSrc[0].unShape.stWhc.u32Width;
    pstSoftWareParam->u32BboxNumEachGrid = 5; // 5: 2BboxNumEachGrid
    pstSoftWareParam->u32ClassNum = 1; // 5: class number
    pstSoftWareParam->u32GridNumHeight = 12; // 12: GridNumHeight
    pstSoftWareParam->u32GridNumWidth = 20; // 20: GridNumWidth
    pstSoftWareParam->u32NmsThresh = (HI_U32)(0.3f*SAMPLE_SVP_NNIE_QUANT_BASE);
    pstSoftWareParam->u32ConfThresh = (HI_U32)(0.25f*SAMPLE_SVP_NNIE_QUANT_BASE);
    pstSoftWareParam->u32MaxRoiNum = 10;  // 10: MaxRoiNum
    pstSoftWareParam->af32Bias[0] = 0.52; // 0.52: af32Bias[0] value
    pstSoftWareParam->af32Bias[1] = 0.61; // 0.61: af32Bias[1] value
    pstSoftWareParam->af32Bias[ARRAY_SUBSCRIPT_2] = 1.05; // 1.05: af32Bias[ARRAY_SUBSCRIPT_2] value
    pstSoftWareParam->af32Bias[ARRAY_SUBSCRIPT_3] = 1.12; // 1.12: af32Bias[ARRAY_SUBSCRIPT_3] value
    pstSoftWareParam->af32Bias[ARRAY_SUBSCRIPT_4] = 1.85; // 1.85: af32Bias[ARRAY_SUBSCRIPT_4] value
    pstSoftWareParam->af32Bias[ARRAY_SUBSCRIPT_5] = 2.05; // 2.05: af32Bias[ARRAY_SUBSCRIPT_5] value
    pstSoftWareParam->af32Bias[ARRAY_SUBSCRIPT_6] = 4.63; // 4.63: af32Bias[ARRAY_SUBSCRIPT_6] value
    pstSoftWareParam->af32Bias[ARRAY_SUBSCRIPT_7] = 4.49; // 4.49: af32Bias[ARRAY_SUBSCRIPT_7] value
    pstSoftWareParam->af32Bias[ARRAY_SUBSCRIPT_8] = 7.15; // 7.15: af32Bias[ARRAY_SUBSCRIPT_8] value
    pstSoftWareParam->af32Bias[ARRAY_SUBSCRIPT_9] = 7.56; // 7.56: af32Bias[ARRAY_SUBSCRIPT_9] value

    /*
     * 申请辅助内存空间
     * Malloc assist buffer memory
     */
    u32ClassNum = pstSoftWareParam->u32ClassNum + 1;
    u32BboxNum = pstSoftWareParam->u32BboxNumEachGrid*pstSoftWareParam->u32GridNumHeight*
        pstSoftWareParam->u32GridNumWidth;
    u32TmpBufTotalSize = SAMPLE_SVP_NNIE_Yolov2_GetResultTmpBuf(pstSoftWareParam);
    u32DstRoiSize = SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum * u32BboxNum * sizeof(HI_U32) * SAMPLE_SVP_NNIE_COORDI_NUM);
    u32DstScoreSize = SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum * u32BboxNum * sizeof(HI_U32));
    u32ClassRoiNumSize = SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum * sizeof(HI_U32));
    u32TotalSize = u32TotalSize + u32DstRoiSize + u32DstScoreSize + u32ClassRoiNumSize + u32TmpBufTotalSize;
    s32Ret = SAMPLE_COMM_SVP_MallocCached("SAMPLE_YOLOV2_INIT", NULL, (HI_U64*)&u64PhyAddr,
        (void**)&pu8VirAddr, u32TotalSize);
    SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,Malloc memory failed!\n");
    memset_s(pu8VirAddr, u32TotalSize, 0, u32TotalSize);
    SAMPLE_COMM_SVP_FlushCache(u64PhyAddr, (void*)pu8VirAddr, u32TotalSize);

    /*
     * 设置每个tmp buffer地址
     * Set each tmp buffer addr
     */
    pstSoftWareParam->stGetResultTmpBuf.u64PhyAddr = u64PhyAddr;
    pstSoftWareParam->stGetResultTmpBuf.u64VirAddr = (HI_U64)((HI_UL)pu8VirAddr);

    /*
     * 设置结果blob
     * Set result blob
     */
    pstSoftWareParam->stDstRoi.enType = SVP_BLOB_TYPE_S32;
    pstSoftWareParam->stDstRoi.u64PhyAddr = u64PhyAddr + u32TmpBufTotalSize;
    pstSoftWareParam->stDstRoi.u64VirAddr = (HI_U64)((HI_UL)pu8VirAddr + u32TmpBufTotalSize);
    pstSoftWareParam->stDstRoi.u32Stride = SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum *
        u32BboxNum * sizeof(HI_U32) * SAMPLE_SVP_NNIE_COORDI_NUM);
    pstSoftWareParam->stDstRoi.u32Num = 1;
    pstSoftWareParam->stDstRoi.unShape.stWhc.u32Chn = 1;
    pstSoftWareParam->stDstRoi.unShape.stWhc.u32Height = 1;
    pstSoftWareParam->stDstRoi.unShape.stWhc.u32Width = u32ClassNum *
        u32BboxNum*SAMPLE_SVP_NNIE_COORDI_NUM;

    pstSoftWareParam->stDstScore.enType = SVP_BLOB_TYPE_S32;
    pstSoftWareParam->stDstScore.u64PhyAddr = u64PhyAddr + u32TmpBufTotalSize + u32DstRoiSize;
    pstSoftWareParam->stDstScore.u64VirAddr = (HI_U64)((HI_UL)pu8VirAddr + u32TmpBufTotalSize + u32DstRoiSize);
    pstSoftWareParam->stDstScore.u32Stride = SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum * u32BboxNum * sizeof(HI_U32));
    pstSoftWareParam->stDstScore.u32Num = 1;
    pstSoftWareParam->stDstScore.unShape.stWhc.u32Chn = 1;
    pstSoftWareParam->stDstScore.unShape.stWhc.u32Height = 1;
    pstSoftWareParam->stDstScore.unShape.stWhc.u32Width = u32ClassNum*u32BboxNum;

    pstSoftWareParam->stClassRoiNum.enType = SVP_BLOB_TYPE_S32;
    pstSoftWareParam->stClassRoiNum.u64PhyAddr = u64PhyAddr + u32TmpBufTotalSize +
        u32DstRoiSize + u32DstScoreSize;
    pstSoftWareParam->stClassRoiNum.u64VirAddr = (HI_U64)((HI_UL)pu8VirAddr + u32TmpBufTotalSize +
        u32DstRoiSize + u32DstScoreSize);
    pstSoftWareParam->stClassRoiNum.u32Stride = SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum*sizeof(HI_U32));
    pstSoftWareParam->stClassRoiNum.u32Num = 1;
    pstSoftWareParam->stClassRoiNum.unShape.stWhc.u32Chn = 1;
    pstSoftWareParam->stClassRoiNum.unShape.stWhc.u32Height = 1;
    pstSoftWareParam->stClassRoiNum.unShape.stWhc.u32Width = u32ClassNum;

    return s32Ret;
}

/*
 * 函数：Yolov2 software参数去初始化
 * function : Yolov2 software param Deinit
 */
static HI_S32 SampleSvpNnieYolov2SoftwareDeinit(SAMPLE_SVP_NNIE_YOLOV2_SOFTWARE_PARAM_S* pstSoftWareParam)
{
    HI_S32 s32Ret = HI_SUCCESS;
    SAMPLE_SVP_CHECK_EXPR_RET(pstSoftWareParam == NULL, HI_INVALID_VALUE, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error, pstSoftWareParam can't be NULL!\n");
    if (pstSoftWareParam->stGetResultTmpBuf.u64PhyAddr != 0 && pstSoftWareParam->stGetResultTmpBuf.u64VirAddr != 0) {
        SAMPLE_SVP_MMZ_FREE(pstSoftWareParam->stGetResultTmpBuf.u64PhyAddr,
            pstSoftWareParam->stGetResultTmpBuf.u64VirAddr);
        pstSoftWareParam->stGetResultTmpBuf.u64PhyAddr = 0;
        pstSoftWareParam->stGetResultTmpBuf.u64VirAddr = 0;
        pstSoftWareParam->stDstRoi.u64PhyAddr = 0;
        pstSoftWareParam->stDstRoi.u64VirAddr = 0;
        pstSoftWareParam->stDstScore.u64PhyAddr = 0;
        pstSoftWareParam->stDstScore.u64VirAddr = 0;
        pstSoftWareParam->stClassRoiNum.u64PhyAddr = 0;
        pstSoftWareParam->stClassRoiNum.u64VirAddr = 0;
    }
    return s32Ret;
}

/*
 * 函数：Yolov2去初始化
 * function : Yolov2 Deinit
 */
static HI_S32 SampleSvpNnieYolov2Deinit(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_YOLOV2_SOFTWARE_PARAM_S* pstSoftWareParam, SAMPLE_SVP_NNIE_MODEL_S *pstNnieModel)
{
    HI_S32 s32Ret = HI_SUCCESS;
    /*
     * SAMPLE_SVP_NNIE_PARAM_S参数去初始化
     * Hardware param deinit
     */
    if (pstNnieParam != NULL) {
        s32Ret = SAMPLE_COMM_SVP_NNIE_ParamDeinit(pstNnieParam);
        SAMPLE_SVP_CHECK_EXPR_TRACE(HI_SUCCESS != s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error,SAMPLE_COMM_SVP_NNIE_ParamDeinit failed!\n");
    }
    /*
     * SAMPLE_SVP_NNIE_YOLOV2_SOFTWARE_PARAM_S参数去初始化
     * Software deinit
     */
    if (pstSoftWareParam != NULL) {
        s32Ret = SampleSvpNnieYolov2SoftwareDeinit(pstSoftWareParam);
        SAMPLE_SVP_CHECK_EXPR_TRACE(HI_SUCCESS != s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error,SampleSvpNnieYolov2SoftwareDeinit failed!\n");
    }

    /*
     * SAMPLE_SVP_NNIE_MODEL_S参数去初始化
     * Model deinit
     */
    if (pstNnieModel != NULL) {
        s32Ret = SAMPLE_COMM_SVP_NNIE_UnloadModel(pstNnieModel);
        SAMPLE_SVP_CHECK_EXPR_TRACE(HI_SUCCESS != s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error,SAMPLE_COMM_SVP_NNIE_UnloadModel failed!\n");
    }
    return s32Ret;
}

/*
 * 函数：Yolov2参数初始化
 * function : Yolov2 Param init
 */
static HI_S32 SampleSvpNnieYolov2ParamInit(SAMPLE_SVP_NNIE_CFG_S* pstCfg,
    SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam, SAMPLE_SVP_NNIE_YOLOV2_SOFTWARE_PARAM_S* pstSoftWareParam)
{
    HI_S32 s32Ret;
    /*
     * 初始化hardware参数
     * Init hardware param
     */
    s32Ret = SAMPLE_COMM_SVP_NNIE_ParamInit(pstCfg, pstNnieParam);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, INIT_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x),SAMPLE_COMM_SVP_NNIE_ParamInit failed!\n", s32Ret);

    /*
     * 初始化software参数
     * Init software param
     */
    s32Ret = SampleSvpNnieYolov2SoftwareInit(pstCfg, pstNnieParam,
        pstSoftWareParam);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, INIT_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x),SAMPLE_SVP_NNIE_Yolov1_SoftwareInit failed!\n", s32Ret);
    return s32Ret;
INIT_FAIL_0:
    s32Ret = SampleSvpNnieYolov2Deinit(pstNnieParam, pstSoftWareParam, NULL);
    SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x),SAMPLE_SVP_NNIE_Yolov1_Deinit failed!\n", s32Ret);
    return HI_FAILURE;
}

/*
 * 函数：基于模型文件创建Yolov2模型
 * function : Creat Yolov2 model basad mode file
 */
int Yolo2Create(SAMPLE_SVP_NNIE_CFG_S **model, const char* modelFile)
{
    SAMPLE_SVP_NNIE_CFG_S *self;
    HI_U32 u32PicNum = 1;
    HI_S32 s32Ret;

    self = (SAMPLE_SVP_NNIE_CFG_S*)malloc(sizeof(*self));
    HI_ASSERT(self);
    memset_s(self, sizeof(*self), 0x00, sizeof(*self));

    /*
     * 设置配置参数
     * Set configuration parameter
     */
    self->pszPic = NULL;
    self->u32MaxInputNum = u32PicNum; // max input image num in each batch
    self->u32MaxRoiNum = 0;
    self->aenNnieCoreId[0] = SVP_NNIE_ID_0; // set NNIE core

    /*
     * 加载Yolov2模型
     * Load Yolov2 model
     */
    SAMPLE_SVP_TRACE_INFO("Yolov2 Load model!\n");
    s32Ret = SAMPLE_COMM_SVP_NNIE_LoadModel((char*)modelFile, &g_stYolov2Model);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, YOLOV2_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error, SAMPLE_COMM_SVP_NNIE_LoadModel failed!\n");

    /*
     * Yolov2参数初始化
     * Yolov2软件参数设置在SampleSvpNnieYolov2SoftwareInit,
     * 如果用户更改了网络结构，请确保
     * SampleSvpNnieYolov2SoftwareInit函数中的参数设置是正确的
     *
     * Yolov2 parameter initialization
     * Yolov2 software parameters are set in SampleSvpNnieYolov2SoftwareInit,
     * if user has changed net struct, please make sure the parameter settings in
     * SampleSvpNnieYolov2SoftwareInit function are correct
     */
    SAMPLE_SVP_TRACE_INFO("Yolov2 parameter initialization!\n");
    g_stYolov2NnieParam.pstModel = &g_stYolov2Model.stModel;
    s32Ret = SampleSvpNnieYolov2ParamInit(self, &g_stYolov2NnieParam, &g_stYolov2SoftwareParam);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, YOLOV2_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SampleSvpNnieYolov2ParamInit failed!\n");

    /*
     * 模型关键数据
     * Model important information
     */
    SAMPLE_PRT("model.base={ type=%x, frmNum=%u, chnNum=%u, w=%u, h=%u, stride=%u }\n",
        g_stYolov2NnieParam.astSegData[0].astSrc[0].enType,
        g_stYolov2NnieParam.astSegData[0].astSrc[0].u32Num,
        g_stYolov2NnieParam.astSegData[0].astSrc[0].unShape.stWhc.u32Chn,
        g_stYolov2NnieParam.astSegData[0].astSrc[0].unShape.stWhc.u32Width,
        g_stYolov2NnieParam.astSegData[0].astSrc[0].unShape.stWhc.u32Height,
        g_stYolov2NnieParam.astSegData[0].astSrc[0].u32Stride);
    SAMPLE_PRT("model.soft={ class=%u, ori.w=%u, ori.h=%u, bnum=%u, \
        grid.w=%u, grid.h=%u, nmsThresh=%u, confThresh=%u, u32MaxRoiNum=%u }\n",
        g_stYolov2SoftwareParam.u32ClassNum,
        g_stYolov2SoftwareParam.u32OriImWidth,
        g_stYolov2SoftwareParam.u32OriImHeight,
        g_stYolov2SoftwareParam.u32BboxNumEachGrid,
        g_stYolov2SoftwareParam.u32GridNumWidth,
        g_stYolov2SoftwareParam.u32GridNumHeight,
        g_stYolov2SoftwareParam.u32NmsThresh,
        g_stYolov2SoftwareParam.u32ConfThresh,
        g_stYolov2SoftwareParam.u32MaxRoiNum);

    *model = self;
    return 0;

    YOLOV2_FAIL_0:
        SAMPLE_PRT("Yolo2Create SampleSvpNnieYolov2Deinit\n");
        SampleSvpNnieYolov2Deinit(&g_stYolov2NnieParam, &g_stYolov2SoftwareParam, &g_stYolov2Model);
        *model = NULL;
        return -1;
}

/*
 * 销毁Yolov2模型
 * Destroy Yolov2 model
 */
void Yolo2Destory(SAMPLE_SVP_NNIE_CFG_S *self)
{
    SampleSvpNnieYolov2Deinit(&g_stYolov2NnieParam, &g_stYolov2SoftwareParam, &g_stYolov2Model);
    SAMPLE_COMM_SVP_CheckSysExit();
    free(self);
}

/*
 * 获取结果
 * Fetch result
 */
static void Yolo2FetchRes(SVP_BLOB_S *pstDstScore, SVP_BLOB_S *pstDstRoi, SVP_BLOB_S *pstClassRoiNum,
    DetectObjInfo resBuf[], int resSize, int* resLen)
{
    HI_U32 i;
    HI_U32 j;
    HI_U32 u32RoiNumBias = 0;
    HI_U32 u32ScoreBias;
    HI_U32 u32BboxBias;
    HI_FLOAT f32Score;
    HI_S32* ps32Score = SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_S32, pstDstScore->u64VirAddr);
    HI_S32* ps32Roi = SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_S32, pstDstRoi->u64VirAddr);
    HI_S32* ps32ClassRoiNum = SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_S32, pstClassRoiNum->u64VirAddr);
    HI_U32 u32ClassNum = pstClassRoiNum->unShape.stWhc.u32Width;

    HI_ASSERT(u32ClassNum == 2); // 2: the number of class
    HI_ASSERT(resSize > 0);
    int resId = 0;
    *resLen = 0;
    memset_s(resBuf, resSize * sizeof(resBuf[0]), 0x00, resSize * sizeof(resBuf[0]));

    u32RoiNumBias += ps32ClassRoiNum[0];
    for (i = 1; i < u32ClassNum; i++) {
        u32ScoreBias = u32RoiNumBias;
        u32BboxBias = u32RoiNumBias * SAMPLE_SVP_NNIE_COORDI_NUM;
        /*
         * 如果置信度分数大于结果阈值，则打印结果
         * If the confidence score greater than result threshold, the result will be printed
         */
        if ((HI_FLOAT)ps32Score[u32ScoreBias] / SAMPLE_SVP_NNIE_QUANT_BASE >=
            THRESH_MIN && ps32ClassRoiNum[i] != 0) {
        }
        for (j = 0; j < (HI_U32)ps32ClassRoiNum[i]; j++) {
            f32Score = (HI_FLOAT)ps32Score[u32ScoreBias + j] / SAMPLE_SVP_NNIE_QUANT_BASE;
            if (f32Score < THRESH_MIN) {
                SAMPLE_PRT("f32Score:%.2f\n", f32Score);
                break;
            }
            if (resId >= resSize) {
                SAMPLE_PRT("yolo2 resBuf full\n");
                break;
            }
            resBuf[resId].cls = 1; // class 1
            resBuf[resId].score = f32Score;

            RectBox *box = &resBuf[resId].box;
            box->xmin = ps32Roi[u32BboxBias + j * SAMPLE_SVP_NNIE_COORDI_NUM];
            box->ymin = ps32Roi[u32BboxBias + j * SAMPLE_SVP_NNIE_COORDI_NUM + ARRAY_SUBSCRIPT_OFFSET_1];
            box->xmax = ps32Roi[u32BboxBias + j * SAMPLE_SVP_NNIE_COORDI_NUM + ARRAY_SUBSCRIPT_OFFSET_2];
            box->ymax = ps32Roi[u32BboxBias + j * SAMPLE_SVP_NNIE_COORDI_NUM + ARRAY_SUBSCRIPT_OFFSET_3];
            if (box->xmin >= box->xmax || box->ymin >= box->ymax) {
                SAMPLE_PRT("yolo1_orig: {%d, %d, %d, %d}, %f, discard for coord ERR\n",
                    box->xmin, box->ymin, box->xmax, box->ymax, f32Score);
            } else {
                resId++;
            }
        }
        u32RoiNumBias += ps32ClassRoiNum[i];
    }

    *resLen = resId;
}

/*
 * 对一帧yuv图片进行推理
 * Calculation yuv image
 */
int Yolo2CalImg(SAMPLE_SVP_NNIE_CFG_S* self,
    const IVE_IMAGE_S *img, DetectObjInfo resBuf[], int resSize, int* resLen)
{
    SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S stInputDataIdx = {0};
    SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S stProcSegIdx = {0};
    HI_S32 s32Ret;

    /*
     * 填充源数据
     * Fill src data
     */
    self->pszPic = NULL;
    stInputDataIdx.u32SegIdx = 0;
    stInputDataIdx.u32NodeIdx = 0;

    s32Ret = FillNnieByImg(self, &g_stYolov2NnieParam, 0, 0, img);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, YOLOV2_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_FillSrcData failed!\n");

    /*
     * NNIE推理(process the 0-th segment)
     * NNIE process(process the 0-th segment)
     */
    stProcSegIdx.u32SegIdx = 0;
    s32Ret = SAMPLE_SVP_NNIE_Forward(&g_stYolov2NnieParam, &stInputDataIdx, &stProcSegIdx, HI_TRUE);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, YOLOV2_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_Forward failed!\n");

    /*
     * CPU处理
     * 如果用户更改了网络结构，请确保SAMPLE_SVP_NNIE_Yolov2_GetResult函数的输入数据是正确的
     *
     * Software process
     * if user has changed net struct, please make sure SAMPLE_SVP_NNIE_Yolov2_GetResult
     * function's input datas are correct
     */
    s32Ret = SAMPLE_SVP_NNIE_Yolov2_GetResult(&g_stYolov2NnieParam, &g_stYolov2SoftwareParam);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, YOLOV2_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_Yolov2_GetResult failed!\n");

    Yolo2FetchRes(&g_stYolov2SoftwareParam.stDstScore,
        &g_stYolov2SoftwareParam.stDstRoi, &g_stYolov2SoftwareParam.stClassRoiNum, resBuf, resSize, resLen);
    return 0;

    YOLOV2_FAIL_0:
        return -1;
}

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* End of #ifdef __cplusplus */
