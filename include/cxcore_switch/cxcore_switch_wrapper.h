//CVG_LicenseBegin==============================================================
//
//	Copyright@ Institut TELECOM 2005
//		http://www.institut-telecom.fr/en_accueil.html
//	
//	This software is a GPU accelerated library for computer-vision. It 
//	supports an OPENCV-like extensible interface for easily porting OPENCV 
//	applications.
//	
//	Contacts :
//		patrick.horain@it-sudparis.eu
//		gpucv-developers@picoforge.int-evry.fr
//	
//	Project's Home Page :
//		https://picoforge.int-evry.fr/cgi-bin/twiki/view/Gpucv/Web/WebHome
//	
//	This software is governed by the CeCILL-B license under French law and
//	abiding by the rules of distribution of free software.  You can  use, 
//	modify and/ or redistribute the software under the terms of the CeCILL-B
//	license as circulated by CEA, CNRS and INRIA at the following URL
//	"http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html". 
//	
//================================================================CVG_LicenseEnd
#ifdef __cplusplus
	#include <GPUCVSwitch/macro.h>
	#include <GPUCVCore/GpuTextureManager.h>
	#include <GPUCVSwitch/Cl_Dll.h>
	#include <GPUCVSwitch/switch.h>
	using namespace std;
	using namespace GCV;
#endif

#define _GPUCV_FORCE_OPENCV_NP 1
#include <includecv.h>
#define CVAPI(MSG) MSG
#ifndef __CXCORE_SWIT_H
#define __CXCORE_SWIT_H


#include <cxcore_switch/cxcore_switch.h>

#define cvAbsDiff	 cvgswAbsDiff
#define cvAbsDiffS	 cvgswAbsDiffS
#define cvAdd	 cvgswAdd
#define cvAddS	 cvgswAddS
#define cvAddWeighted	 cvgswAddWeighted
#define cvAlloc	 cvgswAlloc
#define cvAnd	 cvgswAnd
#define cvAndS	 cvgswAndS
#define cvAttrValue	 cvgswAttrValue
#define cvAvg	 cvgswAvg
#define cvAvgSdv	 cvgswAvgSdv
#define cvBackProjectPCA	 cvgswBackProjectPCA
#define cvCalcCovarMatrix	 cvgswCalcCovarMatrix
#define cvCalcPCA	 cvgswCalcPCA
#define cvCartToPolar	 cvgswCartToPolar
#define cvCbrt	 cvgswCbrt
#define cvChangeSeqBlock	 cvgswChangeSeqBlock
#define cvCheckArr	 cvgswCheckArr
#define cvCheckHardwareSupport	 cvgswCheckHardwareSupport
#define cvCheckTermCriteria	 cvgswCheckTermCriteria
#define cvCircle	 cvgswCircle
#define cvClearGraph	 cvgswClearGraph
#define cvClearMemStorage	 cvgswClearMemStorage
#define cvClearND	 cvgswClearND
#define cvClearSeq	 cvgswClearSeq
#define cvClearSet	 cvgswClearSet
#define cvClipLine	 cvgswClipLine
#define cvClone	 cvgswClone
#define cvCloneGraph	 cvgswCloneGraph
#define cvCloneImage	 cvgswCloneImage
#define cvCloneMat	 cvgswCloneMat
#define cvCloneMatND	 cvgswCloneMatND
#define cvCloneSeq	 cvgswCloneSeq
#define cvCloneSparseMat	 cvgswCloneSparseMat
#define cvCmp	 cvgswCmp
#define cvCmpS	 cvgswCmpS
#define cvColorToScalar	 cvgswColorToScalar
#define cvCompleteSymm	 cvgswCompleteSymm
#define cvConvertScale	 cvgswConvertScale
#define cvConvertScaleAbs	 cvgswConvertScaleAbs
#define cvCopy	 cvgswCopy
#define cvCountNonZero	 cvgswCountNonZero
#define cvCreateChildMemStorage	 cvgswCreateChildMemStorage
#define cvCreateData	 cvgswCreateData
#define cvCreateGraph	 cvgswCreateGraph
#define cvCreateGraphScanner	 cvgswCreateGraphScanner
#define cvCreateImage	 cvgswCreateImage
#define cvCreateImageHeader	 cvgswCreateImageHeader
#define cvCreateMat	 cvgswCreateMat
#define cvCreateMatHeader	 cvgswCreateMatHeader
#define cvCreateMatND	 cvgswCreateMatND
#define cvCreateMatNDHeader	 cvgswCreateMatNDHeader
#define cvCreateMemStorage	 cvgswCreateMemStorage
#define cvCreateSeq	 cvgswCreateSeq
#define cvCreateSeqBlock	 cvgswCreateSeqBlock
#define cvCreateSet	 cvgswCreateSet
#define cvCreateSparseMat	 cvgswCreateSparseMat
#define cvCrossProduct	 cvgswCrossProduct
#define cvCvtSeqToArray	 cvgswCvtSeqToArray
#define cvDCT	 cvgswDCT
#define cvDFT	 cvgswDFT
#define cvDecRefData	 cvgswDecRefData
#define cvDet	 cvgswDet
#define cvDiv	 cvgswDiv
#define cvDotProduct	 cvgswDotProduct
#define cvDrawContours	 cvgswDrawContours
#define cvEigenVV	 cvgswEigenVV
#define cvEllipse	 cvgswEllipse
#define cvEllipse2Poly	 cvgswEllipse2Poly
#define cvEllipseBox	 cvgswEllipseBox
#define cvEndWriteSeq	 cvgswEndWriteSeq
#define cvEndWriteStruct	 cvgswEndWriteStruct
#define cvError	 cvgswError
#define cvErrorFromIppStatus	 cvgswErrorFromIppStatus
#define cvErrorStr	 cvgswErrorStr
#define cvExp	 cvgswExp
#define cvFastArctan	 cvgswFastArctan
#define cvFillConvexPoly	 cvgswFillConvexPoly
#define cvFillPoly	 cvgswFillPoly
#define cvFindGraphEdge	 cvgswFindGraphEdge
#define cvFindGraphEdgeByPtr	 cvgswFindGraphEdgeByPtr
#define cvFindType	 cvgswFindType
#define cvFirstType	 cvgswFirstType
#define cvFlip	 cvgswFlip
#define cvFlushSeqWriter	 cvgswFlushSeqWriter
#define cvFont	 cvgswFont
#define cvFree_	 cvgswFree_
#define cvGEMM	 cvgswGEMM
#define cvGet1D	 cvgswGet1D
#define cvGet2D	 cvgswGet2D
#define cvGet3D	 cvgswGet3D
#define cvGetCol	 cvgswGetCol
#define cvGetCols	 cvgswGetCols
#define cvGetDiag	 cvgswGetDiag
#define cvGetDimSize	 cvgswGetDimSize
#define cvGetDims	 cvgswGetDims
#define cvGetElemType	 cvgswGetElemType
#define cvGetErrInfo	 cvgswGetErrInfo
#define cvGetErrMode	 cvgswGetErrMode
#define cvGetErrStatus	 cvgswGetErrStatus
#define cvGetFileNode	 cvgswGetFileNode
#define cvGetFileNodeByName	 cvgswGetFileNodeByName
#define cvGetFileNodeName	 cvgswGetFileNodeName
#define cvGetHashedKey	 cvgswGetHashedKey
#define cvGetImage	 cvgswGetImage
#define cvGetImageCOI	 cvgswGetImageCOI
#define cvGetImageROI	 cvgswGetImageROI
#define cvGetMat	 cvgswGetMat
#define cvGetModuleInfo	 cvgswGetModuleInfo
#define cvGetND	 cvgswGetND
#define cvGetNextSparseNode	 cvgswGetNextSparseNode
#define cvGetNumThreads	 cvgswGetNumThreads
#define cvGetOptimalDFTSize	 cvgswGetOptimalDFTSize
#define cvGetRawData	 cvgswGetRawData
#define cvGetReal1D	 cvgswGetReal1D
#define cvGetReal2D	 cvgswGetReal2D
#define cvGetReal3D	 cvgswGetReal3D
#define cvGetRealND	 cvgswGetRealND
#define cvGetRootFileNode	 cvgswGetRootFileNode
#define cvGetRow	 cvgswGetRow
#define cvGetRows	 cvgswGetRows
#define cvGetSeqElem	 cvgswGetSeqElem
#define cvGetSeqReaderPos	 cvgswGetSeqReaderPos
#define cvGetSetElem	 cvgswGetSetElem
#define cvGetSize	 cvgswGetSize
#define cvGetSubRect	 cvgswGetSubRect
#define cvGetTextSize	 cvgswGetTextSize
#define cvGetThreadNum	 cvgswGetThreadNum
#define cvGetTickCount	 cvgswGetTickCount
#define cvGetTickFrequency	 cvgswGetTickFrequency
#define cvGraphAddEdge	 cvgswGraphAddEdge
#define cvGraphAddEdgeByPtr	 cvgswGraphAddEdgeByPtr
#define cvGraphAddVtx	 cvgswGraphAddVtx
#define cvGraphRemoveEdge	 cvgswGraphRemoveEdge
#define cvGraphRemoveEdgeByPtr	 cvgswGraphRemoveEdgeByPtr
#define cvGraphRemoveVtx	 cvgswGraphRemoveVtx
#define cvGraphRemoveVtxByPtr	 cvgswGraphRemoveVtxByPtr
#define cvGraphVtxDegree	 cvgswGraphVtxDegree
#define cvGraphVtxDegreeByPtr	 cvgswGraphVtxDegreeByPtr
#define cvGuiBoxReport	 cvgswGuiBoxReport
#define cvInRange	 cvgswInRange
#define cvInRangeS	 cvgswInRangeS
#define cvIncRefData	 cvgswIncRefData
#define cvInitFont	 cvgswInitFont
#define cvInitImageHeader	 cvgswInitImageHeader
#define cvInitLineIterator	 cvgswInitLineIterator
#define cvInitMatHeader	 cvgswInitMatHeader
#define cvInitMatNDHeader	 cvgswInitMatNDHeader
#define cvInitNArrayIterator	 cvgswInitNArrayIterator
#define cvInitSparseMatIterator	 cvgswInitSparseMatIterator
#define cvInitTreeNodeIterator	 cvgswInitTreeNodeIterator
#define cvInsertNodeIntoTree	 cvgswInsertNodeIntoTree
#define cvInvert	 cvgswInvert
#define cvKMeans2	 cvgswKMeans2
#define cvLUT	 cvgswLUT
#define cvLine	 cvgswLine
#define cvLoad	 cvgswLoad
#define cvLog	 cvgswLog
#define cvMahalanobis	 cvgswMahalanobis
#define cvMakeSeqHeaderForArray	 cvgswMakeSeqHeaderForArray
#define cvMax	 cvgswMax
#define cvMaxS	 cvgswMaxS
#define cvMemStorageAlloc	 cvgswMemStorageAlloc
#define cvMemStorageAllocString	 cvgswMemStorageAllocString
#define cvMerge	 cvgswMerge
#define cvMin	 cvgswMin
#define cvMinMaxLoc	 cvgswMinMaxLoc
#define cvMinS	 cvgswMinS
#define cvMixChannels	 cvgswMixChannels
#define cvMul	 cvgswMul
#define cvMulSpectrums	 cvgswMulSpectrums
#define cvMulTransposed	 cvgswMulTransposed
#define cvNextGraphItem	 cvgswNextGraphItem
#define cvNextNArraySlice	 cvgswNextNArraySlice
#define cvNextTreeNode	 cvgswNextTreeNode
#define cvNorm	 cvgswNorm
#define cvNormalize	 cvgswNormalize
#define cvNot	 cvgswNot
#define cvNulDevReport	 cvgswNulDevReport
#define cvOpenFileStorage	 cvgswOpenFileStorage
#define cvOr	 cvgswOr
#define cvOrS	 cvgswOrS
#define cvPerspectiveTransform	 cvgswPerspectiveTransform
#define cvPolarToCart	 cvgswPolarToCart
#define cvPolyLine	 cvgswPolyLine
#define cvPow	 cvgswPow
#define cvPrevTreeNode	 cvgswPrevTreeNode
#define cvProjectPCA	 cvgswProjectPCA
#define cvPtr1D	 cvgswPtr1D
#define cvPtr2D	 cvgswPtr2D
#define cvPtr3D	 cvgswPtr3D
#define cvPtrND	 cvgswPtrND
#define cvPutText	 cvgswPutText
#define cvRandArr	 cvgswRandArr
#define cvRandShuffle	 cvgswRandShuffle
#define cvRange	 cvgswRange
#define cvRawDataToScalar	 cvgswRawDataToScalar
#define cvRead	 cvgswRead
#define cvReadByName	 cvgswReadByName
#define cvReadInt	 cvgswReadInt
#define cvReadIntByName	 cvgswReadIntByName
#define cvReadRawData	 cvgswReadRawData
#define cvReadRawDataSlice	 cvgswReadRawDataSlice
#define cvReadReal	 cvgswReadReal
#define cvReadRealByName	 cvgswReadRealByName
#define cvReadString	 cvgswReadString
#define cvReadStringByName	 cvgswReadStringByName
#define cvRectangle	 cvgswRectangle
#define cvRectangleR	 cvgswRectangleR
#define cvRedirectError	 cvgswRedirectError
#define cvReduce	 cvgswReduce
#define cvRegisterModule	 cvgswRegisterModule
#define cvRegisterType	 cvgswRegisterType
#define cvRelease	 cvgswRelease
#define cvReleaseData	 cvgswReleaseData
#define cvReleaseFileStorage	 cvgswReleaseFileStorage
#define cvReleaseGraphScanner	 cvgswReleaseGraphScanner
#define cvReleaseImage	 cvgswReleaseImage
#define cvReleaseImageHeader	 cvgswReleaseImageHeader
#define cvReleaseMat	 cvgswReleaseMat
#define cvReleaseMatND	 cvgswReleaseMatND
#define cvReleaseMemStorage	 cvgswReleaseMemStorage
#define cvReleaseSparseMat	 cvgswReleaseSparseMat
#define cvRemoveNodeFromTree	 cvgswRemoveNodeFromTree
#define cvRepeat	 cvgswRepeat
#define cvResetImageROI	 cvgswResetImageROI
#define cvReshape	 cvgswReshape
#define cvReshapeMatND	 cvgswReshapeMatND
#define cvRestoreMemStoragePos	 cvgswRestoreMemStoragePos
#define cvSVBkSb	 cvgswSVBkSb
#define cvSVD	 cvgswSVD
#define cvSave	 cvgswSave
#define cvSaveMemStoragePos	 cvgswSaveMemStoragePos
#define cvScalarToRawData	 cvgswScalarToRawData
#define cvScaleAdd	 cvgswScaleAdd
#define cvSeqElemIdx	 cvgswSeqElemIdx
#define cvSeqInsert	 cvgswSeqInsert
#define cvSeqInsertSlice	 cvgswSeqInsertSlice
#define cvSeqInvert	 cvgswSeqInvert
#define cvSeqPartition	 cvgswSeqPartition
#define cvSeqPop	 cvgswSeqPop
#define cvSeqPopFront	 cvgswSeqPopFront
#define cvSeqPopMulti	 cvgswSeqPopMulti
#define cvSeqPush	 cvgswSeqPush
#define cvSeqPushFront	 cvgswSeqPushFront
#define cvSeqPushMulti	 cvgswSeqPushMulti
#define cvSeqRemove	 cvgswSeqRemove
#define cvSeqRemoveSlice	 cvgswSeqRemoveSlice
#define cvSeqSearch	 cvgswSeqSearch
#define cvSeqSlice	 cvgswSeqSlice
#define cvSeqSort	 cvgswSeqSort
#define cvSet	 cvgswSet
#define cvSet1D	 cvgswSet1D
#define cvSet2D	 cvgswSet2D
#define cvSet3D	 cvgswSet3D
#define cvSetAdd	 cvgswSetAdd
#define cvSetData	 cvgswSetData
#define cvSetErrMode	 cvgswSetErrMode
#define cvSetErrStatus	 cvgswSetErrStatus
#define cvSetIPLAllocators	 cvgswSetIPLAllocators
#define cvSetIdentity	 cvgswSetIdentity
#define cvSetImageCOI	 cvgswSetImageCOI
#define cvSetImageROI	 cvgswSetImageROI
#define cvSetMemoryManager	 cvgswSetMemoryManager
#define cvSetND	 cvgswSetND
#define cvSetNew	 cvgswSetNew
#define cvSetNumThreads	 cvgswSetNumThreads
#define cvSetReal1D	 cvgswSetReal1D
#define cvSetReal2D	 cvgswSetReal2D
#define cvSetReal3D	 cvgswSetReal3D
#define cvSetRealND	 cvgswSetRealND
#define cvSetRemove	 cvgswSetRemove
#define cvSetRemoveByPtr	 cvgswSetRemoveByPtr
#define cvSetSeqBlockSize	 cvgswSetSeqBlockSize
#define cvSetSeqReaderPos	 cvgswSetSeqReaderPos
#define cvSetZero	 cvgswSetZero
#define cvSliceLength	 cvgswSliceLength
#define cvSolve	 cvgswSolve
#define cvSolveCubic	 cvgswSolveCubic
#define cvSolvePoly	 cvgswSolvePoly
#define cvSort	 cvgswSort
#define cvSplit	 cvgswSplit
#define cvStartAppendToSeq	 cvgswStartAppendToSeq
#define cvStartNextStream	 cvgswStartNextStream
#define cvStartReadRawData	 cvgswStartReadRawData
#define cvStartReadSeq	 cvgswStartReadSeq
#define cvStartWriteSeq	 cvgswStartWriteSeq
#define cvStartWriteStruct	 cvgswStartWriteStruct
#define cvStdErrReport	 cvgswStdErrReport
#define cvSub	 cvgswSub
#define cvSubRS	 cvgswSubRS
#define cvSubS	 cvgswSubS
#define cvSum	 cvgswSum
#define cvTrace	 cvgswTrace
#define cvTransform	 cvgswTransform
#define cvTranspose	 cvgswTranspose
#define cvTreeToNodeSeq	 cvgswTreeToNodeSeq
#define cvTypeOf	 cvgswTypeOf
#define cvUnregisterType	 cvgswUnregisterType
#define cvUseOptimized	 cvgswUseOptimized
#define cvWrite	 cvgswWrite
#define cvWriteComment	 cvgswWriteComment
#define cvWriteFileNode	 cvgswWriteFileNode
#define cvWriteInt	 cvgswWriteInt
#define cvWriteRawData	 cvgswWriteRawData
#define cvWriteReal	 cvgswWriteReal
#define cvWriteString	 cvgswWriteString
#define cvXor	 cvgswXor
#define cvXorS	 cvgswXorS
/*........End Declaration.............*/


#endif //__CXCORE_SWIT_H