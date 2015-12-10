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
#ifndef __CV_SWIT_H
#define __CV_SWIT_H


#include <cv_switch/cv_switch.h>

#define cv2DRotationMatrix	 cvgsw2DRotationMatrix
#define cvAcc	 cvgswAcc
#define cvAdaptiveThreshold	 cvgswAdaptiveThreshold
#define cvApproxChains	 cvgswApproxChains
#define cvApproxPoly	 cvgswApproxPoly
#define cvArcLength	 cvgswArcLength
#define cvBoundingRect	 cvgswBoundingRect
#define cvBoxPoints	 cvgswBoxPoints
#define cvCalcAffineFlowPyrLK	 cvgswCalcAffineFlowPyrLK
#define cvCalcArrBackProject	 cvgswCalcArrBackProject
#define cvCalcArrBackProjectPatch	 cvgswCalcArrBackProjectPatch
#define cvCalcArrHist	 cvgswCalcArrHist
#define cvCalcBayesianProb	 cvgswCalcBayesianProb
#define cvCalcEMD2	 cvgswCalcEMD2
#define cvCalcGlobalOrientation	 cvgswCalcGlobalOrientation
#define cvCalcHist	 cvgswCalcHist
#define cvCalcMatMulDeriv	 cvgswCalcMatMulDeriv
#define cvCalcMotionGradient	 cvgswCalcMotionGradient
#define cvCalcOpticalFlowBM	 cvgswCalcOpticalFlowBM
#define cvCalcOpticalFlowFarneback	 cvgswCalcOpticalFlowFarneback
#define cvCalcOpticalFlowHS	 cvgswCalcOpticalFlowHS
#define cvCalcOpticalFlowLK	 cvgswCalcOpticalFlowLK
#define cvCalcOpticalFlowPyrLK	 cvgswCalcOpticalFlowPyrLK
#define cvCalcProbDensity	 cvgswCalcProbDensity
#define cvCalcSubdivVoronoi2D	 cvgswCalcSubdivVoronoi2D
#define cvCalibrateCamera2	 cvgswCalibrateCamera2
#define cvCalibrationMatrixValues	 cvgswCalibrationMatrixValues
#define cvCamShift	 cvgswCamShift
#define cvCanny	 cvgswCanny
#define cvCheckChessboard	 cvgswCheckChessboard
#define cvCheckContourConvexity	 cvgswCheckContourConvexity
#define cvClearHist	 cvgswClearHist
#define cvClearSubdivVoronoi2D	 cvgswClearSubdivVoronoi2D
#define cvCompareHist	 cvgswCompareHist
#define cvComposeRT	 cvgswComposeRT
#define cvComputeCorrespondEpilines	 cvgswComputeCorrespondEpilines
#define cvContourArea	 cvgswContourArea
#define cvContourFromContourTree	 cvgswContourFromContourTree
#define cvConvertMaps	 cvgswConvertMaps
#define cvConvertPointsHomogeneous	 cvgswConvertPointsHomogeneous
#define cvConvexHull2	 cvgswConvexHull2
#define cvConvexityDefects	 cvgswConvexityDefects
#define cvCopyHist	 cvgswCopyHist
#define cvCopyMakeBorder	 cvgswCopyMakeBorder
#define cvCornerEigenValsAndVecs	 cvgswCornerEigenValsAndVecs
#define cvCornerHarris	 cvgswCornerHarris
#define cvCornerMinEigenVal	 cvgswCornerMinEigenVal
#define cvCorrectMatches	 cvgswCorrectMatches
#define cvCreateContourTree	 cvgswCreateContourTree
#define cvCreateHist	 cvgswCreateHist
#define cvCreateKalman	 cvgswCreateKalman
#define cvCreatePOSITObject	 cvgswCreatePOSITObject
#define cvCreatePyramid	 cvgswCreatePyramid
#define cvCreateStereoBMState	 cvgswCreateStereoBMState
#define cvCreateStereoGCState	 cvgswCreateStereoGCState
#define cvCreateStructuringElementEx	 cvgswCreateStructuringElementEx
#define cvCreateSubdiv2D	 cvgswCreateSubdiv2D
#define cvCreateSubdivDelaunay2D	 cvgswCreateSubdivDelaunay2D
#define cvCvtColor	 cvgswCvtColor
#define cvDecomposeProjectionMatrix	 cvgswDecomposeProjectionMatrix
#define cvDilate	 cvgswDilate
#define cvDistTransform	 cvgswDistTransform
#define cvDrawChessboardCorners	 cvgswDrawChessboardCorners
#define cvEndFindContours	 cvgswEndFindContours
#define cvEqualizeHist	 cvgswEqualizeHist
#define cvErode	 cvgswErode
#define cvEstimateRigidTransform	 cvgswEstimateRigidTransform
#define cvExtractMSER	 cvgswExtractMSER
#define cvExtractSURF	 cvgswExtractSURF
#define cvFilter2D	 cvgswFilter2D
#define cvFindChessboardCorners	 cvgswFindChessboardCorners
#define cvFindContours	 cvgswFindContours
#define cvFindCornerSubPix	 cvgswFindCornerSubPix
#define cvFindExtrinsicCameraParams2	 cvgswFindExtrinsicCameraParams2
#define cvFindFeatures	 cvgswFindFeatures
#define cvFindFeaturesBoxed	 cvgswFindFeaturesBoxed
#define cvFindFundamentalMat	 cvgswFindFundamentalMat
#define cvFindHomography	 cvgswFindHomography
#define cvFindNearestPoint2D	 cvgswFindNearestPoint2D
#define cvFindNextContour	 cvgswFindNextContour
#define cvFindStereoCorrespondenceBM	 cvgswFindStereoCorrespondenceBM
#define cvFindStereoCorrespondenceGC	 cvgswFindStereoCorrespondenceGC
#define cvFitEllipse2	 cvgswFitEllipse2
#define cvFitLine	 cvgswFitLine
#define cvFloodFill	 cvgswFloodFill
#define cvGetAffineTransform	 cvgswGetAffineTransform
#define cvGetCentralMoment	 cvgswGetCentralMoment
#define cvGetHuMoments	 cvgswGetHuMoments
#define cvGetMinMaxHistValue	 cvgswGetMinMaxHistValue
#define cvGetNormalizedCentralMoment	 cvgswGetNormalizedCentralMoment
#define cvGetOptimalNewCameraMatrix	 cvgswGetOptimalNewCameraMatrix
#define cvGetPerspectiveTransform	 cvgswGetPerspectiveTransform
#define cvGetQuadrangleSubPix	 cvgswGetQuadrangleSubPix
#define cvGetRectSubPix	 cvgswGetRectSubPix
#define cvGetSpatialMoment	 cvgswGetSpatialMoment
#define cvGetStarKeypoints	 cvgswGetStarKeypoints
#define cvGetValidDisparityROI	 cvgswGetValidDisparityROI
#define cvGoodFeaturesToTrack	 cvgswGoodFeaturesToTrack
#define cvHaarDetectObjects	 cvgswHaarDetectObjects
#define cvHoughCircles	 cvgswHoughCircles
#define cvHoughLines2	 cvgswHoughLines2
#define cvInitIntrinsicParams2D	 cvgswInitIntrinsicParams2D
#define cvInitSubdivDelaunay2D	 cvgswInitSubdivDelaunay2D
#define cvInitUndistortMap	 cvgswInitUndistortMap
#define cvInitUndistortRectifyMap	 cvgswInitUndistortRectifyMap
#define cvInpaint	 cvgswInpaint
#define cvIntegral	 cvgswIntegral
#define cvKalmanCorrect	 cvgswKalmanCorrect
#define cvKalmanPredict	 cvgswKalmanPredict
#define cvLSHAdd	 cvgswLSHAdd
#define cvLSHQuery	 cvgswLSHQuery
#define cvLSHRemove	 cvgswLSHRemove
#define cvLaplace	 cvgswLaplace
#define cvLinearPolar	 cvgswLinearPolar
#define cvLoadHaarClassifierCascade	 cvgswLoadHaarClassifierCascade
#define cvLogPolar	 cvgswLogPolar
#define cvMSERParams	 cvgswMSERParams
#define cvMakeHistHeaderForArray	 cvgswMakeHistHeaderForArray
#define cvMatchContourTrees	 cvgswMatchContourTrees
#define cvMatchShapes	 cvgswMatchShapes
#define cvMatchTemplate	 cvgswMatchTemplate
#define cvMaxRect	 cvgswMaxRect
#define cvMeanShift	 cvgswMeanShift
#define cvMinAreaRect2	 cvgswMinAreaRect2
#define cvMinEnclosingCircle	 cvgswMinEnclosingCircle
#define cvMoments	 cvgswMoments
#define cvMorphologyEx	 cvgswMorphologyEx
#define cvMultiplyAcc	 cvgswMultiplyAcc
#define cvNormalizeHist	 cvgswNormalizeHist
#define cvPOSIT	 cvgswPOSIT
#define cvPointPolygonTest	 cvgswPointPolygonTest
#define cvPointSeqFromMat	 cvgswPointSeqFromMat
#define cvPreCornerDetect	 cvgswPreCornerDetect
#define cvProjectPoints2	 cvgswProjectPoints2
#define cvPyrDown	 cvgswPyrDown
#define cvPyrMeanShiftFiltering	 cvgswPyrMeanShiftFiltering
#define cvPyrSegmentation	 cvgswPyrSegmentation
#define cvPyrUp	 cvgswPyrUp
#define cvRANSACUpdateNumIters	 cvgswRANSACUpdateNumIters
#define cvRQDecomp3x3	 cvgswRQDecomp3x3
#define cvReadChainPoint	 cvgswReadChainPoint
#define cvReleaseFeatureTree	 cvgswReleaseFeatureTree
#define cvReleaseHaarClassifierCascade	 cvgswReleaseHaarClassifierCascade
#define cvReleaseHist	 cvgswReleaseHist
#define cvReleaseKalman	 cvgswReleaseKalman
#define cvReleaseLSH	 cvgswReleaseLSH
#define cvReleasePOSITObject	 cvgswReleasePOSITObject
#define cvReleasePyramid	 cvgswReleasePyramid
#define cvReleaseStereoBMState	 cvgswReleaseStereoBMState
#define cvReleaseStereoGCState	 cvgswReleaseStereoGCState
#define cvReleaseStructuringElement	 cvgswReleaseStructuringElement
#define cvRemap	 cvgswRemap
#define cvReprojectImageTo3D	 cvgswReprojectImageTo3D
#define cvResize	 cvgswResize
#define cvRodrigues2	 cvgswRodrigues2
#define cvRunHaarClassifierCascade	 cvgswRunHaarClassifierCascade
#define cvRunningAvg	 cvgswRunningAvg
#define cvSURFParams	 cvgswSURFParams
#define cvSURFPoint	 cvgswSURFPoint
#define cvSampleLine	 cvgswSampleLine
#define cvSegmentMotion	 cvgswSegmentMotion
#define cvSetHistBinRanges	 cvgswSetHistBinRanges
#define cvSetImagesForHaarClassifierCascade	 cvgswSetImagesForHaarClassifierCascade
#define cvSmooth	 cvgswSmooth
#define cvSnakeImage	 cvgswSnakeImage
#define cvSobel	 cvgswSobel
#define cvSquareAcc	 cvgswSquareAcc
#define cvStarDetectorParams	 cvgswStarDetectorParams
#define cvStarKeypoint	 cvgswStarKeypoint
#define cvStartFindContours	 cvgswStartFindContours
#define cvStartReadChainPoints	 cvgswStartReadChainPoints
#define cvStereoCalibrate	 cvgswStereoCalibrate
#define cvStereoRectify	 cvgswStereoRectify
#define cvStereoRectifyUncalibrated	 cvgswStereoRectifyUncalibrated
#define cvSubdiv2DEdgeDst	 cvgswSubdiv2DEdgeDst
#define cvSubdiv2DEdgeOrg	 cvgswSubdiv2DEdgeOrg
#define cvSubdiv2DGetEdge	 cvgswSubdiv2DGetEdge
#define cvSubdiv2DLocate	 cvgswSubdiv2DLocate
#define cvSubdiv2DNextEdge	 cvgswSubdiv2DNextEdge
#define cvSubdiv2DRotateEdge	 cvgswSubdiv2DRotateEdge
#define cvSubdiv2DSymEdge	 cvgswSubdiv2DSymEdge
#define cvSubdivDelaunay2DInsert	 cvgswSubdivDelaunay2DInsert
#define cvSubstituteContour	 cvgswSubstituteContour
#define cvThreshHist	 cvgswThreshHist
#define cvThreshold	 cvgswThreshold
#define cvTriangleArea	 cvgswTriangleArea
#define cvTriangulatePoints	 cvgswTriangulatePoints
#define cvUndistort2	 cvgswUndistort2
#define cvUndistortPoints	 cvgswUndistortPoints
#define cvUpdateMotionHistory	 cvgswUpdateMotionHistory
#define cvValidateDisparity	 cvgswValidateDisparity
#define cvWarpAffine	 cvgswWarpAffine
#define cvWarpPerspective	 cvgswWarpPerspective
#define cvWatershed	 cvgswWatershed
/*........End Declaration.............*/


#endif //__CV_SWIT_H