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
#ifndef __CV_SWITCH_H
#define __CV_SWITCH_H

#include <cv_switch/config.h>
#ifdef __cplusplus
_CV_SWITCH_EXPORT  void cvg_cv_switch_RegisterTracerSingletons(SG_TRC::TTCL_APPLI_TRACER<SG_TRC::SG_TRC_Default_Trc_Type> * _pAppliTracer, SG_TRC::CL_TRACING_EVENT_LIST *_pEventList);
#endif
_CV_SWITCH_EXPORT_C CvMat* cvgsw2DRotationMatrix(CvPoint2D32f center, double angle, double scale, CvMat* map_matrix);
_CV_SWITCH_EXPORT_C void cvgswAcc( CvArr* image, CvArr* sum,  CvArr* mask CV_DEFAULT(NULL));
_CV_SWITCH_EXPORT_C void cvgswAdaptiveThreshold( CvArr* src, CvArr* dst, double max_value, int adaptive_method CV_DEFAULT(CV_ADAPTIVE_THRESH_MEAN_C), int threshold_type CV_DEFAULT(CV_THRESH_BINARY), int block_size CV_DEFAULT(3), double param1 CV_DEFAULT(5));
_CV_SWITCH_EXPORT_C CvSeq* cvgswApproxChains(CvSeq* src_seq, CvMemStorage* storage, int method CV_DEFAULT(CV_CHAIN_APPROX_SIMPLE), double parameter CV_DEFAULT(0), int minimal_perimeter CV_DEFAULT(0), int recursive CV_DEFAULT(0));
_CV_SWITCH_EXPORT_C CvSeq* cvgswApproxPoly(const  void* src_seq, int header_size, CvMemStorage* storage, int method, double parameter, int parameter2 CV_DEFAULT(0));
_CV_SWITCH_EXPORT_C double cvgswArcLength(const  void* curve, CvSlice slice CV_DEFAULT(CV_WHOLE_SEQ), int is_closed CV_DEFAULT(-1));
_CV_SWITCH_EXPORT_C CvRect cvgswBoundingRect(CvArr* points, int update CV_DEFAULT(0));
_CV_SWITCH_EXPORT_C void cvgswBoxPoints(CvBox2D box, CvPoint2D32f *  pt);
_CV_SWITCH_EXPORT_C void cvgswCalcAffineFlowPyrLK( CvArr* prev,  CvArr* curr, CvArr* prev_pyr, CvArr* curr_pyr, const  CvPoint2D32f* prev_features, CvPoint2D32f* curr_features, float* matrices, int count, CvSize win_size, int level, char* status, float* track_error, CvTermCriteria criteria, int flags);
_CV_SWITCH_EXPORT_C void cvgswCalcArrBackProject(CvArr** image, CvArr* dst, const  CvHistogram* hist);
_CV_SWITCH_EXPORT_C void cvgswCalcArrBackProjectPatch(CvArr** image, CvArr* dst, CvSize range, CvHistogram* hist, int method, double factor);
_CV_SWITCH_EXPORT_C void cvgswCalcArrHist(CvArr** arr, CvHistogram* hist, int accumulate CV_DEFAULT(0),  CvArr* mask CV_DEFAULT(NULL));
_CV_SWITCH_EXPORT_C void cvgswCalcBayesianProb(CvHistogram** src, int number, CvHistogram** dst);
_CV_SWITCH_EXPORT_C float cvgswCalcEMD2( CvArr* signature1,  CvArr* signature2, int distance_type, CvDistanceFunction distance_func CV_DEFAULT(NULL),  CvArr* cost_matrix CV_DEFAULT(NULL), CvArr* flow CV_DEFAULT(NULL), float* lower_bound CV_DEFAULT(NULL), void* userdata CV_DEFAULT(NULL));
_CV_SWITCH_EXPORT_C double cvgswCalcGlobalOrientation( CvArr* orientation,  CvArr* mask,  CvArr* mhi, double timestamp, double duration);
_CV_SWITCH_EXPORT_C void cvgswCalcHist(IplImage** image, CvHistogram* hist, int accumulate CV_DEFAULT(0),  CvArr* mask CV_DEFAULT(NULL));
_CV_SWITCH_EXPORT_C void cvgswCalcMatMulDeriv( CvMat* A,  CvMat* B, CvMat* dABdA, CvMat* dABdB);
_CV_SWITCH_EXPORT_C void cvgswCalcMotionGradient( CvArr* mhi, CvArr* mask, CvArr* orientation, double delta1, double delta2, int aperture_size CV_DEFAULT(3));
_CV_SWITCH_EXPORT_C void cvgswCalcOpticalFlowBM( CvArr* prev,  CvArr* curr, CvSize block_size, CvSize shift_size, CvSize max_range, int use_previous, CvArr* velx, CvArr* vely);
_CV_SWITCH_EXPORT_C void cvgswCalcOpticalFlowFarneback( CvArr* prev,  CvArr* next, CvArr* flow, double pyr_scale, int levels, int winsize, int iterations, int poly_n, double poly_sigma, int flags);
_CV_SWITCH_EXPORT_C void cvgswCalcOpticalFlowHS( CvArr* prev,  CvArr* curr, int use_previous, CvArr* velx, CvArr* vely, double lambda, CvTermCriteria criteria);
_CV_SWITCH_EXPORT_C void cvgswCalcOpticalFlowLK( CvArr* prev,  CvArr* curr, CvSize win_size, CvArr* velx, CvArr* vely);
_CV_SWITCH_EXPORT_C void cvgswCalcOpticalFlowPyrLK( CvArr* prev,  CvArr* curr, CvArr* prev_pyr, CvArr* curr_pyr, const  CvPoint2D32f* prev_features, CvPoint2D32f* curr_features, int count, CvSize win_size, int level, char* status, float* track_error, CvTermCriteria criteria, int flags);
_CV_SWITCH_EXPORT_C void cvgswCalcProbDensity(const  CvHistogram* hist1, const  CvHistogram* hist2, CvHistogram* dst_hist, double scale CV_DEFAULT(255));
_CV_SWITCH_EXPORT_C void cvgswCalcSubdivVoronoi2D(CvSubdiv2D* subdiv);
_CV_SWITCH_EXPORT_C double cvgswCalibrateCamera2( CvMat* object_points,  CvMat* image_points,  CvMat* point_counts, CvSize image_size, CvMat* camera_matrix, CvMat* distortion_coeffs, CvMat* rotation_vectors CV_DEFAULT(NULL), CvMat* translation_vectors CV_DEFAULT(NULL), int flags CV_DEFAULT(0));
_CV_SWITCH_EXPORT_C void cvgswCalibrationMatrixValues(const  CvMat * camera_matrix, CvSize image_size, double aperture_width CV_DEFAULT(0), double aperture_height CV_DEFAULT(0), double * fovx CV_DEFAULT(NULL), double * fovy CV_DEFAULT(NULL), double * focal_length CV_DEFAULT(NULL), CvPoint2D64f * principal_point CV_DEFAULT(NULL), double * pixel_aspect_ratio CV_DEFAULT(NULL));
_CV_SWITCH_EXPORT_C int cvgswCamShift( CvArr* prob_image, CvRect window, CvTermCriteria criteria, CvConnectedComp* comp, CvBox2D* box CV_DEFAULT(NULL));
_CV_SWITCH_EXPORT_C void cvgswCanny( CvArr* image, CvArr* edges, double threshold1, double threshold2, int aperture_size CV_DEFAULT(3));
_CV_SWITCH_EXPORT_C int cvgswCheckChessboard(IplImage* src, CvSize size);
_CV_SWITCH_EXPORT_C int cvgswCheckContourConvexity( CvArr* contour);
_CV_SWITCH_EXPORT_C void cvgswClearHist(CvHistogram* hist);
_CV_SWITCH_EXPORT_C void cvgswClearSubdivVoronoi2D(CvSubdiv2D* subdiv);
_CV_SWITCH_EXPORT_C double cvgswCompareHist(const  CvHistogram* hist1, const  CvHistogram* hist2, int method);
_CV_SWITCH_EXPORT_C void cvgswComposeRT( CvMat* _rvec1,  CvMat* _tvec1,  CvMat* _rvec2,  CvMat* _tvec2, CvMat* _rvec3, CvMat* _tvec3, CvMat* dr3dr1 CV_DEFAULT(0), CvMat* dr3dt1 CV_DEFAULT(0), CvMat* dr3dr2 CV_DEFAULT(0), CvMat* dr3dt2 CV_DEFAULT(0), CvMat* dt3dr1 CV_DEFAULT(0), CvMat* dt3dt1 CV_DEFAULT(0), CvMat* dt3dr2 CV_DEFAULT(0), CvMat* dt3dt2 CV_DEFAULT(0));
_CV_SWITCH_EXPORT_C void cvgswComputeCorrespondEpilines( CvMat* points, int which_image,  CvMat* fundamental_matrix, CvMat* correspondent_lines);
_CV_SWITCH_EXPORT_C double cvgswContourArea( CvArr* contour, CvSlice slice CV_DEFAULT(CV_WHOLE_SEQ), int oriented CV_DEFAULT(0));
_CV_SWITCH_EXPORT_C CvSeq* cvgswContourFromContourTree(const  CvContourTree* tree, CvMemStorage* storage, CvTermCriteria criteria);
_CV_SWITCH_EXPORT_C void cvgswConvertMaps( CvArr* mapx,  CvArr* mapy, CvArr* mapxy, CvArr* mapalpha);
_CV_SWITCH_EXPORT_C void cvgswConvertPointsHomogeneous( CvMat* src, CvMat* dst);
_CV_SWITCH_EXPORT_C CvSeq* cvgswConvexHull2( CvArr* input, void* hull_storage CV_DEFAULT(NULL), int orientation CV_DEFAULT(CV_CLOCKWISE), int return_points CV_DEFAULT(0));
_CV_SWITCH_EXPORT_C CvSeq* cvgswConvexityDefects( CvArr* contour,  CvArr* convexhull, CvMemStorage* storage CV_DEFAULT(NULL));
_CV_SWITCH_EXPORT_C void cvgswCopyHist(const  CvHistogram* src, CvHistogram** dst);
_CV_SWITCH_EXPORT_C void cvgswCopyMakeBorder( CvArr* src, CvArr* dst, CvPoint offset, int bordertype, CvScalar value CV_DEFAULT(cvScalarAll(0)));
_CV_SWITCH_EXPORT_C void cvgswCornerEigenValsAndVecs( CvArr* image, CvArr* eigenvv, int block_size, int aperture_size CV_DEFAULT(3));
_CV_SWITCH_EXPORT_C void cvgswCornerHarris( CvArr* image, CvArr* harris_responce, int block_size, int aperture_size CV_DEFAULT(3), double k CV_DEFAULT(0.04));
_CV_SWITCH_EXPORT_C void cvgswCornerMinEigenVal( CvArr* image, CvArr* eigenval, int block_size, int aperture_size CV_DEFAULT(3));
_CV_SWITCH_EXPORT_C void cvgswCorrectMatches(CvMat* F, CvMat* points1, CvMat* points2, CvMat* new_points1, CvMat* new_points2);
_CV_SWITCH_EXPORT_C CvContourTree* cvgswCreateContourTree(const  CvSeq* contour, CvMemStorage* storage, double threshold);
_CV_SWITCH_EXPORT_C CvHistogram* cvgswCreateHist(int dims, int* sizes, int type, float** ranges CV_DEFAULT(NULL), int uniform CV_DEFAULT(1));
_CV_SWITCH_EXPORT_C CvKalman* cvgswCreateKalman(int dynam_params, int measure_params, int control_params CV_DEFAULT(0));
_CV_SWITCH_EXPORT_C CvPOSITObject* cvgswCreatePOSITObject(CvPoint3D32f* points, int point_count);
_CV_SWITCH_EXPORT_C CVAPI(CvMat**) cvgswCreatePyramid( CvArr* img, int extra_layers, double rate, const  CvSize* layer_sizes CV_DEFAULT(0), CvArr* bufarr CV_DEFAULT(0), int calc CV_DEFAULT(1), int filter CV_DEFAULT(CV_GAUSSIAN_5x5));
#ifdef _WINDOWS
_CV_SWITCH_EXPORT_C CvStereoBMState* cvgswCreateStereoBMState(int preset CV_DEFAULT(CV_STEREO_BM_BASIC), int numberOfDisparities CV_DEFAULT(0));
_CV_SWITCH_EXPORT_C CvStereoGCState* cvgswCreateStereoGCState(int numberOfDisparities, int maxIters);
#endif
_CV_SWITCH_EXPORT_C IplConvKernel* cvgswCreateStructuringElementEx(int cols, int rows, int anchor_x, int anchor_y, int shape, int* values CV_DEFAULT(NULL));
_CV_SWITCH_EXPORT_C CvSubdiv2D* cvgswCreateSubdiv2D(int subdiv_type, int header_size, int vtx_size, int quadedge_size, CvMemStorage* storage);
_CV_SWITCH_EXPORT_C CvSubdiv2D* cvgswCreateSubdivDelaunay2D(CvRect rect, CvMemStorage* storage);
_CV_SWITCH_EXPORT_C void cvgswCvtColor( CvArr* src, CvArr* dst, int code);
_CV_SWITCH_EXPORT_C void cvgswDecomposeProjectionMatrix(const  CvMat * projMatr, CvMat * calibMatr, CvMat * rotMatr, CvMat * posVect, CvMat * rotMatrX CV_DEFAULT(NULL), CvMat * rotMatrY CV_DEFAULT(NULL), CvMat * rotMatrZ CV_DEFAULT(NULL), CvPoint3D64f * eulerAngles CV_DEFAULT(NULL));
_CV_SWITCH_EXPORT_C void cvgswDilate( CvArr* src, CvArr* dst, IplConvKernel* element CV_DEFAULT(NULL), int iterations CV_DEFAULT(1));
_CV_SWITCH_EXPORT_C void cvgswDistTransform( CvArr* src, CvArr* dst, int distance_type CV_DEFAULT(CV_DIST_L2), int mask_size CV_DEFAULT(3), const  float* mask CV_DEFAULT(NULL), CvArr* labels CV_DEFAULT(NULL));
_CV_SWITCH_EXPORT_C void cvgswDrawChessboardCorners(CvArr* image, CvSize pattern_size, CvPoint2D32f* corners, int count, int pattern_was_found);
_CV_SWITCH_EXPORT_C CvSeq* cvgswEndFindContours(CvContourScanner* scanner);
_CV_SWITCH_EXPORT_C void cvgswEqualizeHist( CvArr* src, CvArr* dst);
_CV_SWITCH_EXPORT_C void cvgswErode( CvArr* src, CvArr* dst, IplConvKernel* element CV_DEFAULT(NULL), int iterations CV_DEFAULT(1));
_CV_SWITCH_EXPORT_C int cvgswEstimateRigidTransform( CvArr* A,  CvArr* B, CvMat* M, int full_affine);
#ifdef _WINDOWS
_CV_SWITCH_EXPORT_C void cvgswExtractMSER(CvArr* _img, CvArr* _mask, CvSeq** contours, CvMemStorage* storage, CvMSERParams params);
_CV_SWITCH_EXPORT_C void cvgswExtractSURF( CvArr* img,  CvArr* mask, CvSeq** keypoints, CvSeq** descriptors, CvMemStorage* storage, CvSURFParams params, int useProvidedKeyPts CV_DEFAULT(0));
#endif
_CV_SWITCH_EXPORT_C void cvgswFilter2D( CvArr* src, CvArr* dst,  CvMat* kernel, CvPoint anchor CV_DEFAULT(cvPoint(-1,-1)));
_CV_SWITCH_EXPORT_C int cvgswFindChessboardCorners(const  void* image, CvSize pattern_size, CvPoint2D32f* corners, int* corner_count CV_DEFAULT(NULL), int flags CV_DEFAULT(CV_CALIB_CB_ADAPTIVE_THRESH+CV_CALIB_CB_NORMALIZE_IMAGE));
_CV_SWITCH_EXPORT_C int cvgswFindContours(CvArr* image, CvMemStorage* storage, CvSeq** first_contour, int header_size CV_DEFAULT(sizeof(CvContour)), int mode CV_DEFAULT(CV_RETR_LIST), int method CV_DEFAULT(CV_CHAIN_APPROX_SIMPLE), CvPoint offset CV_DEFAULT(cvPoint(0,0)));
_CV_SWITCH_EXPORT_C void cvgswFindCornerSubPix( CvArr* image, CvPoint2D32f* corners, int count, CvSize win, CvSize zero_zone, CvTermCriteria criteria);
_CV_SWITCH_EXPORT_C void cvgswFindExtrinsicCameraParams2( CvMat* object_points,  CvMat* image_points,  CvMat* camera_matrix,  CvMat* distortion_coeffs, CvMat* rotation_vector, CvMat* translation_vector, int use_extrinsic_guess CV_DEFAULT(0));
_CV_SWITCH_EXPORT_C void cvgswFindFeatures(struct CvFeatureTree* tr,  CvMat* query_points, CvMat* indices, CvMat* dist, int k, int emax CV_DEFAULT(20));
_CV_SWITCH_EXPORT_C int cvgswFindFeaturesBoxed(struct CvFeatureTree* tr, CvMat* bounds_min, CvMat* bounds_max, CvMat* out_indices);
_CV_SWITCH_EXPORT_C int cvgswFindFundamentalMat( CvMat* points1,  CvMat* points2, CvMat* fundamental_matrix, int method CV_DEFAULT(CV_FM_RANSAC), double param1 CV_DEFAULT(3.), double param2 CV_DEFAULT(0.99), CvMat* status CV_DEFAULT(NULL));
_CV_SWITCH_EXPORT_C int cvgswFindHomography( CvMat* src_points,  CvMat* dst_points, CvMat* homography, int method CV_DEFAULT(0), double ransacReprojThreshold CV_DEFAULT(0), CvMat* mask CV_DEFAULT(0));
_CV_SWITCH_EXPORT_C CvSubdiv2DPoint* cvgswFindNearestPoint2D(CvSubdiv2D* subdiv, CvPoint2D32f pt);
_CV_SWITCH_EXPORT_C CvSeq* cvgswFindNextContour(CvContourScanner scanner);
#ifdef _WINDOWS
_CV_SWITCH_EXPORT_C void cvgswFindStereoCorrespondenceBM( CvArr* left,  CvArr* right, CvArr* disparity, CvStereoBMState* state);
_CV_SWITCH_EXPORT_C void cvgswFindStereoCorrespondenceGC( CvArr* left,  CvArr* right, CvArr* disparityLeft, CvArr* disparityRight, CvStereoGCState* state, int useDisparityGuess CV_DEFAULT(0));
#endif
_CV_SWITCH_EXPORT_C CvBox2D cvgswFitEllipse2( CvArr* points);
_CV_SWITCH_EXPORT_C void cvgswFitLine( CvArr* points, int dist_type, double param, double reps, double aeps, float* line);
_CV_SWITCH_EXPORT_C void cvgswFloodFill(CvArr* image, CvPoint seed_point, CvScalar new_val, CvScalar lo_diff CV_DEFAULT(cvScalarAll(0)), CvScalar up_diff CV_DEFAULT(cvScalarAll(0)), CvConnectedComp* comp CV_DEFAULT(NULL), int flags CV_DEFAULT(4), CvArr* mask CV_DEFAULT(NULL));
_CV_SWITCH_EXPORT_C CvMat* cvgswGetAffineTransform(const  CvPoint2D32f * src, const  CvPoint2D32f * dst, CvMat * map_matrix);
_CV_SWITCH_EXPORT_C double cvgswGetCentralMoment(CvMoments* moments, int x_order, int y_order);
_CV_SWITCH_EXPORT_C void cvgswGetHuMoments(CvMoments* moments, CvHuMoments* hu_moments);
_CV_SWITCH_EXPORT_C void cvgswGetMinMaxHistValue(const  CvHistogram* hist, float* min_value, float* max_value, int* min_idx CV_DEFAULT(NULL), int* max_idx CV_DEFAULT(NULL));
_CV_SWITCH_EXPORT_C double cvgswGetNormalizedCentralMoment(CvMoments* moments, int x_order, int y_order);
_CV_SWITCH_EXPORT_C void cvgswGetOptimalNewCameraMatrix( CvMat* camera_matrix,  CvMat* dist_coeffs, CvSize image_size, double alpha, CvMat* new_camera_matrix, CvSize new_imag_size CV_DEFAULT(cvSize(0,0)), CvRect* valid_pixel_ROI CV_DEFAULT(0));
_CV_SWITCH_EXPORT_C CvMat* cvgswGetPerspectiveTransform(const  CvPoint2D32f* src, const  CvPoint2D32f* dst, CvMat* map_matrix);
_CV_SWITCH_EXPORT_C void cvgswGetQuadrangleSubPix( CvArr* src, CvArr* dst,  CvMat* map_matrix);
_CV_SWITCH_EXPORT_C void cvgswGetRectSubPix( CvArr* src, CvArr* dst, CvPoint2D32f center);
_CV_SWITCH_EXPORT_C double cvgswGetSpatialMoment(CvMoments* moments, int x_order, int y_order);
_CV_SWITCH_EXPORT_C CvSeq* cvgswGetStarKeypoints( CvArr* img, CvMemStorage* storage, CvStarDetectorParams params CV_DEFAULT(cvStarDetectorParams()));
_CV_SWITCH_EXPORT_C CvRect cvgswGetValidDisparityROI(CvRect roi1, CvRect roi2, int minDisparity, int numberOfDisparities, int SADWindowSize);
_CV_SWITCH_EXPORT_C void cvgswGoodFeaturesToTrack( CvArr* image, CvArr* eig_image, CvArr* temp_image, CvPoint2D32f* corners, int* corner_count, double quality_level, double min_distance,  CvArr* mask CV_DEFAULT(NULL), int block_size CV_DEFAULT(3), int use_harris CV_DEFAULT(0), double k CV_DEFAULT(0.04));
_CV_SWITCH_EXPORT_C CvSeq* cvgswHaarDetectObjects( CvArr* image, CvHaarClassifierCascade* cascade, CvMemStorage* storage, double scale_factor CV_DEFAULT(1.1), int min_neighbors CV_DEFAULT(3), int flags CV_DEFAULT(0), CvSize min_size CV_DEFAULT(cvSize(0,0)));
_CV_SWITCH_EXPORT_C CvSeq* cvgswHoughCircles(CvArr* image, void* circle_storage, int method, double dp, double min_dist, double param1 CV_DEFAULT(100), double param2 CV_DEFAULT(100), int min_radius CV_DEFAULT(0), int max_radius CV_DEFAULT(0));
_CV_SWITCH_EXPORT_C CvSeq* cvgswHoughLines2(CvArr* image, void* line_storage, int method, double rho, double theta, int threshold, double param1 CV_DEFAULT(0), double param2 CV_DEFAULT(0));
_CV_SWITCH_EXPORT_C void cvgswInitIntrinsicParams2D( CvMat* object_points,  CvMat* image_points,  CvMat* npoints, CvSize image_size, CvMat* camera_matrix, double aspect_ratio CV_DEFAULT(1.));
_CV_SWITCH_EXPORT_C void cvgswInitSubdivDelaunay2D(CvSubdiv2D* subdiv, CvRect rect);
_CV_SWITCH_EXPORT_C void cvgswInitUndistortMap( CvMat* camera_matrix,  CvMat* distortion_coeffs, CvArr* mapx, CvArr* mapy);
_CV_SWITCH_EXPORT_C void cvgswInitUndistortRectifyMap( CvMat* camera_matrix,  CvMat* dist_coeffs, const  CvMat * R,  CvMat* new_camera_matrix, CvArr* mapx, CvArr* mapy);
_CV_SWITCH_EXPORT_C void cvgswInpaint( CvArr* src,  CvArr* inpaint_mask, CvArr* dst, double inpaintRange, int flags);
_CV_SWITCH_EXPORT_C void cvgswIntegral( CvArr* image, CvArr* sum, CvArr* sqsum CV_DEFAULT(NULL), CvArr* tilted_sum CV_DEFAULT(NULL));
_CV_SWITCH_EXPORT_C const CvMat* cvgswKalmanCorrect(CvKalman* kalman,  CvMat* measurement);
_CV_SWITCH_EXPORT_C const CvMat* cvgswKalmanPredict(CvKalman* kalman,  CvMat* control CV_DEFAULT(NULL));
_CV_SWITCH_EXPORT_C void cvgswLSHAdd(struct CvLSH* lsh,  CvMat* data, CvMat* indices CV_DEFAULT(0));
_CV_SWITCH_EXPORT_C void cvgswLSHQuery(struct CvLSH* lsh,  CvMat* query_points, CvMat* indices, CvMat* dist, int k, int emax);
_CV_SWITCH_EXPORT_C void cvgswLSHRemove(struct CvLSH* lsh,  CvMat* indices);
_CV_SWITCH_EXPORT_C void cvgswLaplace( CvArr* src, CvArr* dst, int aperture_size CV_DEFAULT(3));
_CV_SWITCH_EXPORT_C void cvgswLinearPolar( CvArr* src, CvArr* dst, CvPoint2D32f center, double maxRadius, int flags CV_DEFAULT(CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS));
_CV_SWITCH_EXPORT_C CvHaarClassifierCascade* cvgswLoadHaarClassifierCascade(const  char* directory, CvSize orig_window_size);
_CV_SWITCH_EXPORT_C void cvgswLogPolar( CvArr* src, CvArr* dst, CvPoint2D32f center, double M, int flags CV_DEFAULT(CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS));
_CV_SWITCH_EXPORT_C CVAPI(CvMSERParams) cvgswMSERParams(int delta CV_DEFAULT(5), int min_area CV_DEFAULT(60), int max_area CV_DEFAULT(14400), float max_variation CV_DEFAULT(.25f), float min_diversity CV_DEFAULT(.2f), int max_evolution CV_DEFAULT(200), double area_threshold CV_DEFAULT(1.01), double min_margin CV_DEFAULT(.003), int edge_blur_size CV_DEFAULT(5));
_CV_SWITCH_EXPORT_C CvHistogram* cvgswMakeHistHeaderForArray(int dims, int* sizes, CvHistogram* hist, float* data, float** ranges CV_DEFAULT(NULL), int uniform CV_DEFAULT(1));
_CV_SWITCH_EXPORT_C double cvgswMatchContourTrees(const  CvContourTree* tree1, const  CvContourTree* tree2, int method, double threshold);
_CV_SWITCH_EXPORT_C double cvgswMatchShapes(const  void* object1, const  void* object2, int method, double parameter CV_DEFAULT(0));
_CV_SWITCH_EXPORT_C void cvgswMatchTemplate( CvArr* image,  CvArr* templ, CvArr* result, int method);
_CV_SWITCH_EXPORT_C CvRect cvgswMaxRect(const  CvRect* rect1, const  CvRect* rect2);
_CV_SWITCH_EXPORT_C int cvgswMeanShift( CvArr* prob_image, CvRect window, CvTermCriteria criteria, CvConnectedComp* comp);
_CV_SWITCH_EXPORT_C CvBox2D cvgswMinAreaRect2( CvArr* points, CvMemStorage* storage CV_DEFAULT(NULL));
_CV_SWITCH_EXPORT_C int cvgswMinEnclosingCircle( CvArr* points, CvPoint2D32f* center, float* radius);
_CV_SWITCH_EXPORT_C void cvgswMoments( CvArr* arr, CvMoments* moments, int binary CV_DEFAULT(0));
_CV_SWITCH_EXPORT_C void cvgswMorphologyEx( CvArr* src, CvArr* dst, CvArr* temp, IplConvKernel* element, int operation, int iterations CV_DEFAULT(1));
_CV_SWITCH_EXPORT_C void cvgswMultiplyAcc( CvArr* image1,  CvArr* image2, CvArr* acc,  CvArr* mask CV_DEFAULT(NULL));
_CV_SWITCH_EXPORT_C void cvgswNormalizeHist(CvHistogram* hist, double factor);
_CV_SWITCH_EXPORT_C void cvgswPOSIT(CvPOSITObject* posit_object, CvPoint2D32f* image_points, double focal_length, CvTermCriteria criteria, CvMatr32f rotation_matrix, CvVect32f translation_vector);
_CV_SWITCH_EXPORT_C double cvgswPointPolygonTest( CvArr* contour, CvPoint2D32f pt, int measure_dist);
_CV_SWITCH_EXPORT_C CvSeq* cvgswPointSeqFromMat(int seq_kind,  CvArr* mat, CvContour* contour_header, CvSeqBlock* block);
_CV_SWITCH_EXPORT_C void cvgswPreCornerDetect( CvArr* image, CvArr* corners, int aperture_size CV_DEFAULT(3));
_CV_SWITCH_EXPORT_C void cvgswProjectPoints2( CvMat* object_points,  CvMat* rotation_vector,  CvMat* translation_vector,  CvMat* camera_matrix,  CvMat* distortion_coeffs, CvMat* image_points, CvMat* dpdrot CV_DEFAULT(NULL), CvMat* dpdt CV_DEFAULT(NULL), CvMat* dpdf CV_DEFAULT(NULL), CvMat* dpdc CV_DEFAULT(NULL), CvMat* dpddist CV_DEFAULT(NULL), double aspect_ratio CV_DEFAULT(0));
_CV_SWITCH_EXPORT_C void cvgswPyrDown( CvArr* src, CvArr* dst, int filter CV_DEFAULT(CV_GAUSSIAN_5x5));
_CV_SWITCH_EXPORT_C void cvgswPyrMeanShiftFiltering( CvArr* src, CvArr* dst, double sp, double sr, int max_level CV_DEFAULT(1), CvTermCriteria termcrit CV_DEFAULT(cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,5,1)));
_CV_SWITCH_EXPORT_C void cvgswPyrSegmentation(IplImage* src, IplImage* dst, CvMemStorage* storage, CvSeq** comp, int level, double threshold1, double threshold2);
_CV_SWITCH_EXPORT_C void cvgswPyrUp( CvArr* src, CvArr* dst, int filter CV_DEFAULT(CV_GAUSSIAN_5x5));
_CV_SWITCH_EXPORT_C int cvgswRANSACUpdateNumIters(double p, double err_prob, int model_points, int max_iters);
_CV_SWITCH_EXPORT_C void cvgswRQDecomp3x3(const  CvMat * matrixM, CvMat * matrixR, CvMat * matrixQ, CvMat * matrixQx CV_DEFAULT(NULL), CvMat * matrixQy CV_DEFAULT(NULL), CvMat * matrixQz CV_DEFAULT(NULL), CvPoint3D64f * eulerAngles CV_DEFAULT(NULL));
_CV_SWITCH_EXPORT_C CvPoint cvgswReadChainPoint(CvChainPtReader* reader);
_CV_SWITCH_EXPORT_C void cvgswReleaseFeatureTree(struct CvFeatureTree* tr);
_CV_SWITCH_EXPORT_C void cvgswReleaseHaarClassifierCascade(CvHaarClassifierCascade** cascade);
_CV_SWITCH_EXPORT_C void cvgswReleaseHist(CvHistogram** hist);
_CV_SWITCH_EXPORT_C void cvgswReleaseKalman(CvKalman** kalman);
_CV_SWITCH_EXPORT_C void cvgswReleaseLSH(struct CvLSH** lsh);
_CV_SWITCH_EXPORT_C void cvgswReleasePOSITObject(CvPOSITObject** posit_object);
_CV_SWITCH_EXPORT_C void cvgswReleasePyramid(CvMat*** pyramid, int extra_layers);
_CV_SWITCH_EXPORT_C void cvgswReleaseStereoBMState(CvStereoBMState** state);
_CV_SWITCH_EXPORT_C void cvgswReleaseStereoGCState(CvStereoGCState** state);
_CV_SWITCH_EXPORT_C void cvgswReleaseStructuringElement(IplConvKernel** element);
_CV_SWITCH_EXPORT_C void cvgswRemap( CvArr* src, CvArr* dst,  CvArr* mapx,  CvArr* mapy, int flags CV_DEFAULT(CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS), CvScalar fillval CV_DEFAULT(cvScalarAll(0)));
_CV_SWITCH_EXPORT_C void cvgswReprojectImageTo3D( CvArr* disparityImage, CvArr* _3dImage,  CvMat* Q, int handleMissingValues CV_DEFAULT(0));
_CV_SWITCH_EXPORT_C void cvgswResize( CvArr* src, CvArr* dst, int interpolation CV_DEFAULT( CV_INTER_LINEAR ));
_CV_SWITCH_EXPORT_C int cvgswRodrigues2( CvMat* src, CvMat* dst, CvMat* jacobian CV_DEFAULT(0));
_CV_SWITCH_EXPORT_C int cvgswRunHaarClassifierCascade(const  CvHaarClassifierCascade* cascade, CvPoint pt, int start_stage CV_DEFAULT(0));
_CV_SWITCH_EXPORT_C void cvgswRunningAvg( CvArr* image, CvArr* acc, double alpha,  CvArr* mask CV_DEFAULT(NULL));
_CV_SWITCH_EXPORT_C CVAPI(CvSURFParams) cvgswSURFParams(double hessianThreshold, int extended CV_DEFAULT(0));
_CV_SWITCH_EXPORT_C CvSURFPoint cvgswSURFPoint(CvPoint2D32f pt, int laplacian, int size, float dir CV_DEFAULT(0), float hessian CV_DEFAULT(0));
_CV_SWITCH_EXPORT_C int cvgswSampleLine( CvArr* image, CvPoint pt1, CvPoint pt2, void* buffer, int connectivity CV_DEFAULT(8));
_CV_SWITCH_EXPORT_C CvSeq* cvgswSegmentMotion( CvArr* mhi, CvArr* seg_mask, CvMemStorage* storage, double timestamp, double seg_thresh);
_CV_SWITCH_EXPORT_C void cvgswSetHistBinRanges(CvHistogram* hist, float** ranges, int uniform CV_DEFAULT(1));
_CV_SWITCH_EXPORT_C void cvgswSetImagesForHaarClassifierCascade(CvHaarClassifierCascade* cascade,  CvArr* sum,  CvArr* sqsum,  CvArr* tilted_sum, double scale);
_CV_SWITCH_EXPORT_C void cvgswSmooth( CvArr* src, CvArr* dst, int smoothtype CV_DEFAULT(CV_GAUSSIAN), int size1 CV_DEFAULT(3), int size2 CV_DEFAULT(0), double sigma1 CV_DEFAULT(0), double sigma2 CV_DEFAULT(0));
_CV_SWITCH_EXPORT_C void cvgswSnakeImage( IplImage* image, CvPoint* points, int length, float* alpha, float* beta, float* gamma, int coeff_usage, CvSize win, CvTermCriteria criteria, int calc_gradient CV_DEFAULT(1));
_CV_SWITCH_EXPORT_C void cvgswSobel( CvArr* src, CvArr* dst, int xorder, int yorder, int aperture_size CV_DEFAULT(3));
_CV_SWITCH_EXPORT_C void cvgswSquareAcc( CvArr* image, CvArr* sqsum,  CvArr* mask CV_DEFAULT(NULL));
_CV_SWITCH_EXPORT_C CvStarDetectorParams cvgswStarDetectorParams(int maxSize CV_DEFAULT(45), int responseThreshold CV_DEFAULT(30), int lineThresholdProjected CV_DEFAULT(10), int lineThresholdBinarized CV_DEFAULT(8), int suppressNonmaxSize CV_DEFAULT(5));
_CV_SWITCH_EXPORT_C CvStarKeypoint cvgswStarKeypoint(CvPoint pt, int size, float response);
_CV_SWITCH_EXPORT_C CvContourScanner cvgswStartFindContours(CvArr* image, CvMemStorage* storage, int header_size CV_DEFAULT(sizeof(CvContour)), int mode CV_DEFAULT(CV_RETR_LIST), int method CV_DEFAULT(CV_CHAIN_APPROX_SIMPLE), CvPoint offset CV_DEFAULT(cvPoint(0,0)));
_CV_SWITCH_EXPORT_C void cvgswStartReadChainPoints(CvChain* chain, CvChainPtReader* reader);
_CV_SWITCH_EXPORT_C double cvgswStereoCalibrate( CvMat* object_points,  CvMat* image_points1,  CvMat* image_points2,  CvMat* npoints, CvMat* camera_matrix1, CvMat* dist_coeffs1, CvMat* camera_matrix2, CvMat* dist_coeffs2, CvSize image_size, CvMat* R, CvMat* T, CvMat* E CV_DEFAULT(0), CvMat* F CV_DEFAULT(0), CvTermCriteria term_crit CV_DEFAULT(cvTermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,30,1e-6)), int flags CV_DEFAULT(CV_CALIB_FIX_INTRINSIC));
_CV_SWITCH_EXPORT_C void cvgswStereoRectify( CvMat* camera_matrix1,  CvMat* camera_matrix2,  CvMat* dist_coeffs1,  CvMat* dist_coeffs2, CvSize image_size,  CvMat* R,  CvMat* T, CvMat* R1, CvMat* R2, CvMat* P1, CvMat* P2, CvMat* Q CV_DEFAULT(0), int flags CV_DEFAULT(CV_CALIB_ZERO_DISPARITY), double alpha CV_DEFAULT(-1), CvSize new_image_size CV_DEFAULT(cvSize(0,0)), CvRect* valid_pix_ROI1 CV_DEFAULT(0), CvRect* valid_pix_ROI2 CV_DEFAULT(0));
_CV_SWITCH_EXPORT_C int cvgswStereoRectifyUncalibrated( CvMat* points1,  CvMat* points2,  CvMat* F, CvSize img_size, CvMat* H1, CvMat* H2, double threshold CV_DEFAULT(5));
_CV_SWITCH_EXPORT_C CvSubdiv2DPoint* cvgswSubdiv2DEdgeDst(CvSubdiv2DEdge edge);
_CV_SWITCH_EXPORT_C CvSubdiv2DPoint* cvgswSubdiv2DEdgeOrg(CvSubdiv2DEdge edge);
_CV_SWITCH_EXPORT_C CvSubdiv2DEdge cvgswSubdiv2DGetEdge(CvSubdiv2DEdge edge, CvNextEdgeType type);
_CV_SWITCH_EXPORT_C CvSubdiv2DPointLocation cvgswSubdiv2DLocate(CvSubdiv2D* subdiv, CvPoint2D32f pt, CvSubdiv2DEdge* edge, CvSubdiv2DPoint** vertex CV_DEFAULT(NULL));
_CV_SWITCH_EXPORT_C CvSubdiv2DEdge cvgswSubdiv2DNextEdge(CvSubdiv2DEdge edge);
_CV_SWITCH_EXPORT_C CvSubdiv2DEdge cvgswSubdiv2DRotateEdge(CvSubdiv2DEdge edge, int rotate);
_CV_SWITCH_EXPORT_C CvSubdiv2DEdge cvgswSubdiv2DSymEdge(CvSubdiv2DEdge edge);
_CV_SWITCH_EXPORT_C CvSubdiv2DPoint* cvgswSubdivDelaunay2DInsert(CvSubdiv2D* subdiv, CvPoint2D32f pt);
_CV_SWITCH_EXPORT_C void cvgswSubstituteContour(CvContourScanner scanner, CvSeq* new_contour);
_CV_SWITCH_EXPORT_C void cvgswThreshHist(CvHistogram* hist, double threshold);
_CV_SWITCH_EXPORT_C double cvgswThreshold( CvArr* src, CvArr* dst, double threshold, double max_value, int threshold_type);
_CV_SWITCH_EXPORT_C double cvgswTriangleArea(CvPoint2D32f a, CvPoint2D32f b, CvPoint2D32f c);
_CV_SWITCH_EXPORT_C void cvgswTriangulatePoints(CvMat* projMatr1, CvMat* projMatr2, CvMat* projPoints1, CvMat* projPoints2, CvMat* points4D);
_CV_SWITCH_EXPORT_C void cvgswUndistort2( CvArr* src, CvArr* dst,  CvMat* camera_matrix,  CvMat* distortion_coeffs,  CvMat* new_camera_matrix CV_DEFAULT(0));
_CV_SWITCH_EXPORT_C void cvgswUndistortPoints( CvMat* src, CvMat* dst,  CvMat* camera_matrix,  CvMat* dist_coeffs,  CvMat* R CV_DEFAULT(0),  CvMat* P CV_DEFAULT(0));
_CV_SWITCH_EXPORT_C void cvgswUpdateMotionHistory( CvArr* silhouette, CvArr* mhi, double timestamp, double duration);
_CV_SWITCH_EXPORT_C void cvgswValidateDisparity(CvArr* disparity,  CvArr* cost, int minDisparity, int numberOfDisparities, int disp12MaxDiff CV_DEFAULT(1));
_CV_SWITCH_EXPORT_C void cvgswWarpAffine( CvArr* src, CvArr* dst,  CvMat* map_matrix, int flags CV_DEFAULT(CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS), CvScalar fillval CV_DEFAULT(cvScalarAll(0)));
_CV_SWITCH_EXPORT_C void cvgswWarpPerspective( CvArr* src, CvArr* dst,  CvMat* map_matrix, int flags CV_DEFAULT(CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS), CvScalar fillval CV_DEFAULT(cvScalarAll(0)));
_CV_SWITCH_EXPORT_C void cvgswWatershed( CvArr* image, CvArr* markers);
/*........End Declaration.............*/


#endif //__CV_SWITCH_H
