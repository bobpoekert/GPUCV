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
#ifndef __CXCORE_SWITCH_H
#define __CXCORE_SWITCH_H

#include <cxcore_switch/config.h>
#ifdef __cplusplus
_CXCORE_SWITCH_EXPORT  void cvg_cxcore_switch_RegisterTracerSingletons(SG_TRC::TTCL_APPLI_TRACER<SG_TRC::SG_TRC_Default_Trc_Type> * _pAppliTracer, SG_TRC::CL_TRACING_EVENT_LIST *_pEventList);
#endif
_CXCORE_SWITCH_EXPORT_C void cvgswAbsDiff( CvArr* src1,  CvArr* src2, CvArr* dst);
_CXCORE_SWITCH_EXPORT_C void cvgswAbsDiffS( CvArr* src, CvArr* dst, CvScalar value);
_CXCORE_SWITCH_EXPORT_C void cvgswAdd( CvArr* src1,  CvArr* src2, CvArr* dst,  CvArr* mask CV_DEFAULT(NULL));
_CXCORE_SWITCH_EXPORT_C void cvgswAddS( CvArr* src, CvScalar value, CvArr* dst,  CvArr* mask CV_DEFAULT(NULL));
_CXCORE_SWITCH_EXPORT_C void cvgswAddWeighted( CvArr* src1, double alpha,  CvArr* src2, double beta, double gamma, CvArr* dst);
_CXCORE_SWITCH_EXPORT_C void* cvgswAlloc(size_t size);
_CXCORE_SWITCH_EXPORT_C void cvgswAnd( CvArr* src1,  CvArr* src2, CvArr* dst,  CvArr* mask CV_DEFAULT(NULL));
_CXCORE_SWITCH_EXPORT_C void cvgswAndS( CvArr* src, CvScalar value, CvArr* dst,  CvArr* mask CV_DEFAULT(NULL));
_CXCORE_SWITCH_EXPORT_C const char* cvgswAttrValue(const  CvAttrList* attr, const  char* attr_name);
_CXCORE_SWITCH_EXPORT_C CvScalar cvgswAvg( CvArr* arr,  CvArr* mask CV_DEFAULT(NULL));
_CXCORE_SWITCH_EXPORT_C void cvgswAvgSdv( CvArr* arr, CvScalar* mean, CvScalar* std_dev,  CvArr* mask CV_DEFAULT(NULL));
_CXCORE_SWITCH_EXPORT_C void cvgswBackProjectPCA( CvArr* proj,  CvArr* mean,  CvArr* eigenvects, CvArr* result);
_CXCORE_SWITCH_EXPORT_C void cvgswCalcCovarMatrix( CvArr** vects, int count, CvArr* cov_mat, CvArr* avg, int flags);
_CXCORE_SWITCH_EXPORT_C void cvgswCalcPCA( CvArr* data, CvArr* mean, CvArr* eigenvals, CvArr* eigenvects, int flags);
_CXCORE_SWITCH_EXPORT_C void cvgswCartToPolar( CvArr* x,  CvArr* y, CvArr* magnitude, CvArr* angle CV_DEFAULT(NULL), int angle_in_degrees CV_DEFAULT(0));
_CXCORE_SWITCH_EXPORT_C float cvgswCbrt(float value);
_CXCORE_SWITCH_EXPORT_C void cvgswChangeSeqBlock(void* reader, int direction);
_CXCORE_SWITCH_EXPORT_C int cvgswCheckArr( CvArr* arr, int flags CV_DEFAULT(0), double min_val CV_DEFAULT(0), double max_val CV_DEFAULT(0));
_CXCORE_SWITCH_EXPORT_C int cvgswCheckHardwareSupport(int feature);
_CXCORE_SWITCH_EXPORT_C CvTermCriteria cvgswCheckTermCriteria(CvTermCriteria criteria, double default_eps, int default_max_iters);
_CXCORE_SWITCH_EXPORT_C void cvgswCircle(CvArr* img, CvPoint center, int radius, CvScalar color, int thickness CV_DEFAULT(1), int line_type CV_DEFAULT(8), int shift CV_DEFAULT(0));
_CXCORE_SWITCH_EXPORT_C void cvgswClearGraph(CvGraph* graph);
_CXCORE_SWITCH_EXPORT_C void cvgswClearMemStorage(CvMemStorage* storage);
_CXCORE_SWITCH_EXPORT_C void cvgswClearND(CvArr* arr, const  int* idx);
_CXCORE_SWITCH_EXPORT_C void cvgswClearSeq(CvSeq* seq);
_CXCORE_SWITCH_EXPORT_C void cvgswClearSet(CvSet* set_header);
_CXCORE_SWITCH_EXPORT_C int cvgswClipLine(CvSize img_size, CvPoint* pt1, CvPoint* pt2);
_CXCORE_SWITCH_EXPORT_C void* cvgswClone(const  void* struct_ptr);
_CXCORE_SWITCH_EXPORT_C CvGraph* cvgswCloneGraph(const  CvGraph* graph, CvMemStorage* storage);
_CXCORE_SWITCH_EXPORT_C IplImage* cvgswCloneImage( IplImage* image);
_CXCORE_SWITCH_EXPORT_C CvMat* cvgswCloneMat( CvMat* mat);
_CXCORE_SWITCH_EXPORT_C CvMatND* cvgswCloneMatND(const  CvMatND* mat);
_CXCORE_SWITCH_EXPORT_C CvSeq* cvgswCloneSeq(const  CvSeq* seq, CvMemStorage* storage CV_DEFAULT(NULL));
_CXCORE_SWITCH_EXPORT_C CvSparseMat* cvgswCloneSparseMat(const  CvSparseMat* mat);
_CXCORE_SWITCH_EXPORT_C void cvgswCmp( CvArr* src1,  CvArr* src2, CvArr* dst, int cmp_op);
_CXCORE_SWITCH_EXPORT_C void cvgswCmpS( CvArr* src, double value, CvArr* dst, int cmp_op);
_CXCORE_SWITCH_EXPORT_C CvScalar cvgswColorToScalar(double packed_color, int arrtype);
_CXCORE_SWITCH_EXPORT_C void cvgswCompleteSymm(CvMat* matrix, int LtoR CV_DEFAULT(0));
_CXCORE_SWITCH_EXPORT_C void cvgswConvertScale( CvArr* src, CvArr* dst, double scale CV_DEFAULT(1), double shift CV_DEFAULT(0));
_CXCORE_SWITCH_EXPORT_C void cvgswConvertScaleAbs( CvArr* src, CvArr* dst, double scale CV_DEFAULT(1), double shift CV_DEFAULT(0));
_CXCORE_SWITCH_EXPORT_C void cvgswCopy( CvArr* src, CvArr* dst,  CvArr* mask CV_DEFAULT(NULL));
_CXCORE_SWITCH_EXPORT_C int cvgswCountNonZero( CvArr* arr);
_CXCORE_SWITCH_EXPORT_C CvMemStorage* cvgswCreateChildMemStorage(CvMemStorage* parent);
_CXCORE_SWITCH_EXPORT_C void cvgswCreateData(CvArr* arr);
_CXCORE_SWITCH_EXPORT_C CvGraph* cvgswCreateGraph(int graph_flags, int header_size, int vtx_size, int edge_size, CvMemStorage* storage);
_CXCORE_SWITCH_EXPORT_C CvGraphScanner* cvgswCreateGraphScanner(CvGraph* graph, CvGraphVtx* vtx CV_DEFAULT(NULL), int mask CV_DEFAULT(CV_GRAPH_ALL_ITEMS));
_CXCORE_SWITCH_EXPORT_C IplImage* cvgswCreateImage(CvSize size, int depth, int channels);
_CXCORE_SWITCH_EXPORT_C IplImage* cvgswCreateImageHeader(CvSize size, int depth, int channels);
_CXCORE_SWITCH_EXPORT_C CvMat* cvgswCreateMat(int rows, int cols, int type);
_CXCORE_SWITCH_EXPORT_C CvMat* cvgswCreateMatHeader(int rows, int cols, int type);
_CXCORE_SWITCH_EXPORT_C CvMatND* cvgswCreateMatND(int dims, const  int* sizes, int type);
_CXCORE_SWITCH_EXPORT_C CvMatND* cvgswCreateMatNDHeader(int dims, const  int* sizes, int type);
_CXCORE_SWITCH_EXPORT_C CvMemStorage* cvgswCreateMemStorage(int block_size CV_DEFAULT(0));
_CXCORE_SWITCH_EXPORT_C CvSeq* cvgswCreateSeq(int seq_flags, int header_size, int elem_size, CvMemStorage* storage);
_CXCORE_SWITCH_EXPORT_C void cvgswCreateSeqBlock(CvSeqWriter* writer);
_CXCORE_SWITCH_EXPORT_C CvSet* cvgswCreateSet(int set_flags, int header_size, int elem_size, CvMemStorage* storage);
_CXCORE_SWITCH_EXPORT_C CvSparseMat* cvgswCreateSparseMat(int dims, const  int* sizes, int type);
_CXCORE_SWITCH_EXPORT_C void cvgswCrossProduct( CvArr* src1,  CvArr* src2, CvArr* dst);
_CXCORE_SWITCH_EXPORT_C void* cvgswCvtSeqToArray(const  CvSeq* seq, void* elements, CvSlice slice CV_DEFAULT(CV_WHOLE_SEQ));
_CXCORE_SWITCH_EXPORT_C void cvgswDCT( CvArr* src, CvArr* dst, int flags);
_CXCORE_SWITCH_EXPORT_C void cvgswDFT( CvArr* src, CvArr* dst, int flags, int nonzero_rows CV_DEFAULT(0));
_CXCORE_SWITCH_EXPORT_C void cvgswDecRefData(CvArr* arr);
_CXCORE_SWITCH_EXPORT_C double cvgswDet( CvArr* mat);
_CXCORE_SWITCH_EXPORT_C void cvgswDiv( CvArr* src1,  CvArr* src2, CvArr* dst, double scale CV_DEFAULT(1));
_CXCORE_SWITCH_EXPORT_C double cvgswDotProduct( CvArr* src1,  CvArr* src2);
_CXCORE_SWITCH_EXPORT_C void cvgswDrawContours(CvArr * img, CvSeq* contour, CvScalar external_color, CvScalar hole_color, int max_level, int thickness CV_DEFAULT(1), int line_type CV_DEFAULT(8), CvPoint offset CV_DEFAULT(cvPoint(0,0)));
_CXCORE_SWITCH_EXPORT_C void cvgswEigenVV(CvArr* mat, CvArr* evects, CvArr* evals, double eps CV_DEFAULT(0), int lowindex CV_DEFAULT(-1), int highindex CV_DEFAULT(-1));
_CXCORE_SWITCH_EXPORT_C void cvgswEllipse(CvArr* img, CvPoint center, CvSize axes, double angle, double start_angle, double end_angle, CvScalar color, int thickness CV_DEFAULT(1), int line_type CV_DEFAULT(8), int shift CV_DEFAULT(0));
_CXCORE_SWITCH_EXPORT_C int cvgswEllipse2Poly(CvPoint center, CvSize axes, int angle, int arc_start, int arc_end, CvPoint * pts, int delta);
_CXCORE_SWITCH_EXPORT_C void cvgswEllipseBox(CvArr* img, CvBox2D box, CvScalar color, int thickness CV_DEFAULT(1), int line_type CV_DEFAULT(8), int shift CV_DEFAULT(0));
_CXCORE_SWITCH_EXPORT_C CvSeq* cvgswEndWriteSeq(CvSeqWriter* writer);
_CXCORE_SWITCH_EXPORT_C void cvgswEndWriteStruct(CvFileStorage* fs);
_CXCORE_SWITCH_EXPORT_C void cvgswError(int status, const  char* func_name, const  char* err_msg, const  char* file_name, int line);
_CXCORE_SWITCH_EXPORT_C int cvgswErrorFromIppStatus(int ipp_status);
_CXCORE_SWITCH_EXPORT_C const char* cvgswErrorStr(int status);
_CXCORE_SWITCH_EXPORT_C void cvgswExp( CvArr* src, CvArr* dst);
_CXCORE_SWITCH_EXPORT_C float cvgswFastArctan(float y, float x);
_CXCORE_SWITCH_EXPORT_C void cvgswFillConvexPoly(CvArr* img, const  CvPoint* pts, int npts, CvScalar color, int line_type CV_DEFAULT(8), int shift CV_DEFAULT(0));
_CXCORE_SWITCH_EXPORT_C void cvgswFillPoly(CvArr* img, CvPoint** pts, const  int* npts, int contours, CvScalar color, int line_type CV_DEFAULT(8), int shift CV_DEFAULT(0));
_CXCORE_SWITCH_EXPORT_C CvGraphEdge* cvgswFindGraphEdge(const  CvGraph* graph, int start_idx, int end_idx);
_CXCORE_SWITCH_EXPORT_C CvGraphEdge* cvgswFindGraphEdgeByPtr(const  CvGraph* graph, const  CvGraphVtx* start_vtx, const  CvGraphVtx* end_vtx);
_CXCORE_SWITCH_EXPORT_C CvTypeInfo* cvgswFindType(const  char* type_name);
_CXCORE_SWITCH_EXPORT_C CvTypeInfo* cvgswFirstType();
_CXCORE_SWITCH_EXPORT_C void cvgswFlip( CvArr* src, CvArr* dst CV_DEFAULT(NULL), int flip_mode CV_DEFAULT(0));
_CXCORE_SWITCH_EXPORT_C void cvgswFlushSeqWriter(CvSeqWriter* writer);
_CXCORE_SWITCH_EXPORT_C CvFont cvgswFont(double scale, int thickness CV_DEFAULT(1));
_CXCORE_SWITCH_EXPORT_C void cvgswFree_(void* ptr);
_CXCORE_SWITCH_EXPORT_C void cvgswGEMM( CvArr* src1,  CvArr* src2, double alpha,  CvArr* src3, double beta, CvArr* dst, int tABC CV_DEFAULT(0));
_CXCORE_SWITCH_EXPORT_C CvScalar cvgswGet1D( CvArr* arr, int idx0);
_CXCORE_SWITCH_EXPORT_C CvScalar cvgswGet2D( CvArr* arr, int idx0, int idx1);
_CXCORE_SWITCH_EXPORT_C CvScalar cvgswGet3D( CvArr* arr, int idx0, int idx1, int idx2);
_CXCORE_SWITCH_EXPORT_C CvMat* cvgswGetCol( CvArr* arr, CvMat* submat, int col);
_CXCORE_SWITCH_EXPORT_C CvMat* cvgswGetCols( CvArr* arr, CvMat* submat, int start_col, int end_col);
_CXCORE_SWITCH_EXPORT_C CvMat* cvgswGetDiag( CvArr* arr, CvMat* submat, int diag CV_DEFAULT(0));
_CXCORE_SWITCH_EXPORT_C int cvgswGetDimSize( CvArr* arr, int index);
_CXCORE_SWITCH_EXPORT_C int cvgswGetDims( CvArr* arr, int* sizes CV_DEFAULT(NULL));
_CXCORE_SWITCH_EXPORT_C int cvgswGetElemType( CvArr* arr);
_CXCORE_SWITCH_EXPORT_C int cvgswGetErrInfo(const  char** errcode_desc, const  char** description, const  char** filename, int* line);
_CXCORE_SWITCH_EXPORT_C int cvgswGetErrMode();
_CXCORE_SWITCH_EXPORT_C int cvgswGetErrStatus();
_CXCORE_SWITCH_EXPORT_C CvFileNode* cvgswGetFileNode(CvFileStorage* fs, CvFileNode* map, const  CvStringHashNode* key, int create_missing CV_DEFAULT(0));
_CXCORE_SWITCH_EXPORT_C CvFileNode* cvgswGetFileNodeByName(const  CvFileStorage* fs, const  CvFileNode* map, const  char* name);
_CXCORE_SWITCH_EXPORT_C const char* cvgswGetFileNodeName(const  CvFileNode* node);
_CXCORE_SWITCH_EXPORT_C CvStringHashNode* cvgswGetHashedKey(CvFileStorage* fs, const  char* name, int len CV_DEFAULT(-1), int create_missing CV_DEFAULT(0));
_CXCORE_SWITCH_EXPORT_C IplImage* cvgswGetImage( CvArr* arr, IplImage* image_header);
_CXCORE_SWITCH_EXPORT_C int cvgswGetImageCOI( IplImage* image);
_CXCORE_SWITCH_EXPORT_C CvRect cvgswGetImageROI( IplImage* image);
_CXCORE_SWITCH_EXPORT_C CvMat* cvgswGetMat( CvArr* arr, CvMat* header, int* coi CV_DEFAULT(NULL), int allowND CV_DEFAULT(0));
_CXCORE_SWITCH_EXPORT_C void cvgswGetModuleInfo(const  char* module_name, const  char** version, const  char** loaded_addon_plugins);
_CXCORE_SWITCH_EXPORT_C CvScalar cvgswGetND( CvArr* arr, const  int* idx);
_CXCORE_SWITCH_EXPORT_C CvSparseNode* cvgswGetNextSparseNode(CvSparseMatIterator* mat_iterator);
_CXCORE_SWITCH_EXPORT_C int cvgswGetNumThreads();
_CXCORE_SWITCH_EXPORT_C int cvgswGetOptimalDFTSize(int size0);
_CXCORE_SWITCH_EXPORT_C void cvgswGetRawData( CvArr* arr, uchar** data, int* step CV_DEFAULT(NULL), CvSize* roi_size CV_DEFAULT(NULL));
_CXCORE_SWITCH_EXPORT_C double cvgswGetReal1D( CvArr* arr, int idx0);
_CXCORE_SWITCH_EXPORT_C double cvgswGetReal2D( CvArr* arr, int idx0, int idx1);
_CXCORE_SWITCH_EXPORT_C double cvgswGetReal3D( CvArr* arr, int idx0, int idx1, int idx2);
_CXCORE_SWITCH_EXPORT_C double cvgswGetRealND( CvArr* arr, const  int* idx);
_CXCORE_SWITCH_EXPORT_C CvFileNode* cvgswGetRootFileNode(const  CvFileStorage* fs, int stream_index CV_DEFAULT(0));
_CXCORE_SWITCH_EXPORT_C CvMat* cvgswGetRow( CvArr* arr, CvMat* submat, int row);
_CXCORE_SWITCH_EXPORT_C CvMat* cvgswGetRows( CvArr* arr, CvMat* submat, int start_row, int end_row, int delta_row CV_DEFAULT(1));
_CXCORE_SWITCH_EXPORT_C CVAPI(schar*) cvgswGetSeqElem(const  CvSeq* seq, int index);
_CXCORE_SWITCH_EXPORT_C int cvgswGetSeqReaderPos(CvSeqReader* reader);
_CXCORE_SWITCH_EXPORT_C CvSetElem* cvgswGetSetElem(const  CvSet* set_header, int index);
_CXCORE_SWITCH_EXPORT_C CvSize cvgswGetSize( CvArr* arr);
_CXCORE_SWITCH_EXPORT_C CvMat* cvgswGetSubRect( CvArr* arr, CvMat* submat, CvRect rect);
_CXCORE_SWITCH_EXPORT_C void cvgswGetTextSize(const  char* text_string, const  CvFont* font, CvSize* text_size, int* baseline);
_CXCORE_SWITCH_EXPORT_C int cvgswGetThreadNum();
_CXCORE_SWITCH_EXPORT_C int64 cvgswGetTickCount();
_CXCORE_SWITCH_EXPORT_C double cvgswGetTickFrequency();
_CXCORE_SWITCH_EXPORT_C int cvgswGraphAddEdge(CvGraph* graph, int start_idx, int end_idx, const  CvGraphEdge* edge CV_DEFAULT(NULL), CvGraphEdge** inserted_edge CV_DEFAULT(NULL));
_CXCORE_SWITCH_EXPORT_C int cvgswGraphAddEdgeByPtr(CvGraph* graph, CvGraphVtx* start_vtx, CvGraphVtx* end_vtx, const  CvGraphEdge* edge CV_DEFAULT(NULL), CvGraphEdge** inserted_edge CV_DEFAULT(NULL));
_CXCORE_SWITCH_EXPORT_C int cvgswGraphAddVtx(CvGraph* graph, const  CvGraphVtx* vtx CV_DEFAULT(NULL), CvGraphVtx** inserted_vtx CV_DEFAULT(NULL));
_CXCORE_SWITCH_EXPORT_C void cvgswGraphRemoveEdge(CvGraph* graph, int start_idx, int end_idx);
_CXCORE_SWITCH_EXPORT_C void cvgswGraphRemoveEdgeByPtr(CvGraph* graph, CvGraphVtx* start_vtx, CvGraphVtx* end_vtx);
_CXCORE_SWITCH_EXPORT_C int cvgswGraphRemoveVtx(CvGraph* graph, int index);
_CXCORE_SWITCH_EXPORT_C int cvgswGraphRemoveVtxByPtr(CvGraph* graph, CvGraphVtx* vtx);
_CXCORE_SWITCH_EXPORT_C int cvgswGraphVtxDegree(const  CvGraph* graph, int vtx_idx);
_CXCORE_SWITCH_EXPORT_C int cvgswGraphVtxDegreeByPtr(const  CvGraph* graph, const  CvGraphVtx* vtx);
_CXCORE_SWITCH_EXPORT_C int cvgswGuiBoxReport(int status, const  char* func_name, const  char* err_msg, const  char* file_name, int line, void* userdata);
_CXCORE_SWITCH_EXPORT_C void cvgswInRange( CvArr* src,  CvArr* lower,  CvArr* upper, CvArr* dst);
_CXCORE_SWITCH_EXPORT_C void cvgswInRangeS( CvArr* src, CvScalar lower, CvScalar upper, CvArr* dst);
_CXCORE_SWITCH_EXPORT_C int cvgswIncRefData(CvArr* arr);
_CXCORE_SWITCH_EXPORT_C void cvgswInitFont(CvFont* font, int font_face, double hscale, double vscale, double shear CV_DEFAULT(0), int thickness CV_DEFAULT(1), int line_type CV_DEFAULT(8));
_CXCORE_SWITCH_EXPORT_C IplImage* cvgswInitImageHeader(IplImage* image, CvSize size, int depth, int channels, int origin CV_DEFAULT(0), int align CV_DEFAULT(4));
_CXCORE_SWITCH_EXPORT_C int cvgswInitLineIterator( CvArr* image, CvPoint pt1, CvPoint pt2, CvLineIterator* line_iterator, int connectivity CV_DEFAULT(8), int left_to_right CV_DEFAULT(0));
_CXCORE_SWITCH_EXPORT_C CvMat* cvgswInitMatHeader(CvMat* mat, int rows, int cols, int type, void* data CV_DEFAULT(NULL), int step CV_DEFAULT(CV_AUTOSTEP));
_CXCORE_SWITCH_EXPORT_C CvMatND* cvgswInitMatNDHeader(CvMatND* mat, int dims, const  int* sizes, int type, void* data CV_DEFAULT(NULL));
_CXCORE_SWITCH_EXPORT_C int cvgswInitNArrayIterator(int count, CvArr** arrs,  CvArr* mask, CvMatND* stubs, CvNArrayIterator* array_iterator, int flags CV_DEFAULT(0));
_CXCORE_SWITCH_EXPORT_C CvSparseNode* cvgswInitSparseMatIterator(const  CvSparseMat* mat, CvSparseMatIterator* mat_iterator);
_CXCORE_SWITCH_EXPORT_C void cvgswInitTreeNodeIterator(CvTreeNodeIterator* tree_iterator, const  void* first, int max_level);
_CXCORE_SWITCH_EXPORT_C void cvgswInsertNodeIntoTree(void* node, void* parent, void* frame);
_CXCORE_SWITCH_EXPORT_C double cvgswInvert( CvArr* src, CvArr* dst, int method CV_DEFAULT(CV_LU));
_CXCORE_SWITCH_EXPORT_C int cvgswKMeans2( CvArr* samples, int cluster_count, CvArr* labels, CvTermCriteria termcrit, int attempts CV_DEFAULT(1), CvRNG* rng CV_DEFAULT(0), int flags CV_DEFAULT(0), CvArr* _centers CV_DEFAULT(0), double* compactness CV_DEFAULT(0));
_CXCORE_SWITCH_EXPORT_C void cvgswLUT( CvArr* src, CvArr* dst,  CvArr* lut);
_CXCORE_SWITCH_EXPORT_C void cvgswLine(CvArr* img, CvPoint pt1, CvPoint pt2, CvScalar color, int thickness CV_DEFAULT(1), int line_type CV_DEFAULT(8), int shift CV_DEFAULT(0));
_CXCORE_SWITCH_EXPORT_C void* cvgswLoad(const  char* filename, CvMemStorage* memstorage CV_DEFAULT(NULL), const  char* name CV_DEFAULT(NULL), const  char** real_name CV_DEFAULT(NULL));
_CXCORE_SWITCH_EXPORT_C void cvgswLog( CvArr* src, CvArr* dst);
_CXCORE_SWITCH_EXPORT_C double cvgswMahalanobis( CvArr* vec1,  CvArr* vec2,  CvArr* mat);
_CXCORE_SWITCH_EXPORT_C CvSeq* cvgswMakeSeqHeaderForArray(int seq_type, int header_size, int elem_size, void* elements, int total, CvSeq* seq, CvSeqBlock* block);
_CXCORE_SWITCH_EXPORT_C void cvgswMax( CvArr* src1,  CvArr* src2, CvArr* dst);
_CXCORE_SWITCH_EXPORT_C void cvgswMaxS( CvArr* src, double value, CvArr* dst);
_CXCORE_SWITCH_EXPORT_C void* cvgswMemStorageAlloc(CvMemStorage* storage, size_t size);
_CXCORE_SWITCH_EXPORT_C CvString cvgswMemStorageAllocString(CvMemStorage* storage, const  char* ptr, int len CV_DEFAULT(-1));
_CXCORE_SWITCH_EXPORT_C void cvgswMerge( CvArr* src0,  CvArr* src1,  CvArr* src2,  CvArr* src3, CvArr* dst);
_CXCORE_SWITCH_EXPORT_C void cvgswMin( CvArr* src1,  CvArr* src2, CvArr* dst);
_CXCORE_SWITCH_EXPORT_C void cvgswMinMaxLoc( CvArr* arr, double* min_val, double* max_val, CvPoint* min_loc CV_DEFAULT(NULL), CvPoint* max_loc CV_DEFAULT(NULL),  CvArr* mask CV_DEFAULT(NULL));
_CXCORE_SWITCH_EXPORT_C void cvgswMinS( CvArr* src, double value, CvArr* dst);
_CXCORE_SWITCH_EXPORT_C void cvgswMixChannels( CvArr** src, int src_count, CvArr** dst, int dst_count, const  int* from_to, int pair_count);
_CXCORE_SWITCH_EXPORT_C void cvgswMul( CvArr* src1,  CvArr* src2, CvArr* dst, double scale CV_DEFAULT(1));
_CXCORE_SWITCH_EXPORT_C void cvgswMulSpectrums( CvArr* src1,  CvArr* src2, CvArr* dst, int flags);
_CXCORE_SWITCH_EXPORT_C void cvgswMulTransposed( CvArr* src, CvArr* dst, int order,  CvArr* delta CV_DEFAULT(NULL), double scale CV_DEFAULT(1.));
_CXCORE_SWITCH_EXPORT_C int cvgswNextGraphItem(CvGraphScanner* scanner);
_CXCORE_SWITCH_EXPORT_C int cvgswNextNArraySlice(CvNArrayIterator* array_iterator);
_CXCORE_SWITCH_EXPORT_C void* cvgswNextTreeNode(CvTreeNodeIterator* tree_iterator);
_CXCORE_SWITCH_EXPORT_C double cvgswNorm( CvArr* arr1,  CvArr* arr2 CV_DEFAULT(NULL), int norm_type CV_DEFAULT(CV_L2),  CvArr* mask CV_DEFAULT(NULL));
_CXCORE_SWITCH_EXPORT_C void cvgswNormalize( CvArr* src, CvArr* dst, double a CV_DEFAULT(1.), double b CV_DEFAULT(0.), int norm_type CV_DEFAULT(CV_L2),  CvArr* mask CV_DEFAULT(NULL));
_CXCORE_SWITCH_EXPORT_C void cvgswNot( CvArr* src, CvArr* dst);
_CXCORE_SWITCH_EXPORT_C int cvgswNulDevReport(int status, const  char* func_name, const  char* err_msg, const  char* file_name, int line, void* userdata);
_CXCORE_SWITCH_EXPORT_C CvFileStorage* cvgswOpenFileStorage(const  char* filename, CvMemStorage* memstorage, int flags);
_CXCORE_SWITCH_EXPORT_C void cvgswOr( CvArr* src1,  CvArr* src2, CvArr* dst,  CvArr* mask CV_DEFAULT(NULL));
_CXCORE_SWITCH_EXPORT_C void cvgswOrS( CvArr* src, CvScalar value, CvArr* dst,  CvArr* mask CV_DEFAULT(NULL));
_CXCORE_SWITCH_EXPORT_C void cvgswPerspectiveTransform( CvArr* src, CvArr* dst,  CvMat* mat);
_CXCORE_SWITCH_EXPORT_C void cvgswPolarToCart( CvArr* magnitude,  CvArr* angle, CvArr* x, CvArr* y, int angle_in_degrees CV_DEFAULT(0));
_CXCORE_SWITCH_EXPORT_C void cvgswPolyLine(CvArr* img, CvPoint** pts, const  int* npts, int contours, int is_closed, CvScalar color, int thickness CV_DEFAULT(1), int line_type CV_DEFAULT(8), int shift CV_DEFAULT(0));
_CXCORE_SWITCH_EXPORT_C void cvgswPow( CvArr* src, CvArr* dst, double power);
_CXCORE_SWITCH_EXPORT_C void* cvgswPrevTreeNode(CvTreeNodeIterator* tree_iterator);
_CXCORE_SWITCH_EXPORT_C void cvgswProjectPCA( CvArr* data,  CvArr* mean,  CvArr* eigenvects, CvArr* result);
_CXCORE_SWITCH_EXPORT_C uchar* cvgswPtr1D( CvArr* arr, int idx0, int* type CV_DEFAULT(NULL));
_CXCORE_SWITCH_EXPORT_C uchar* cvgswPtr2D( CvArr* arr, int idx0, int idx1, int* type CV_DEFAULT(NULL));
_CXCORE_SWITCH_EXPORT_C uchar* cvgswPtr3D( CvArr* arr, int idx0, int idx1, int idx2, int* type CV_DEFAULT(NULL));
_CXCORE_SWITCH_EXPORT_C uchar* cvgswPtrND( CvArr* arr, const  int* idx, int* type CV_DEFAULT(NULL), int create_node CV_DEFAULT(1), unsigned* precalc_hashval CV_DEFAULT(NULL));
_CXCORE_SWITCH_EXPORT_C void cvgswPutText(CvArr* img, const  char* text, CvPoint org, const  CvFont* font, CvScalar color);
_CXCORE_SWITCH_EXPORT_C void cvgswRandArr(CvRNG* rng, CvArr* arr, int dist_type, CvScalar param1, CvScalar param2);
_CXCORE_SWITCH_EXPORT_C void cvgswRandShuffle(CvArr* mat, CvRNG* rng, double iter_factor CV_DEFAULT(1.));
_CXCORE_SWITCH_EXPORT_C CvArr* cvgswRange(CvArr* mat, double start, double end);
_CXCORE_SWITCH_EXPORT_C void cvgswRawDataToScalar(const  void* data, int type, CvScalar* scalar);
_CXCORE_SWITCH_EXPORT_C void* cvgswRead(CvFileStorage* fs, CvFileNode* node, CvAttrList* attributes CV_DEFAULT(NULL));
_CXCORE_SWITCH_EXPORT_C void* cvgswReadByName(CvFileStorage* fs, const  CvFileNode* map, const  char* name, CvAttrList* attributes CV_DEFAULT(NULL));
_CXCORE_SWITCH_EXPORT_C int cvgswReadInt(const  CvFileNode* node, int default_value CV_DEFAULT(0));
_CXCORE_SWITCH_EXPORT_C int cvgswReadIntByName(const  CvFileStorage* fs, const  CvFileNode* map, const  char* name, int default_value CV_DEFAULT(0));
_CXCORE_SWITCH_EXPORT_C void cvgswReadRawData(const  CvFileStorage* fs, const  CvFileNode* src, void* dst, const  char* dt);
_CXCORE_SWITCH_EXPORT_C void cvgswReadRawDataSlice(const  CvFileStorage* fs, CvSeqReader* reader, int count, void* dst, const  char* dt);
_CXCORE_SWITCH_EXPORT_C double cvgswReadReal(const  CvFileNode* node, double default_value CV_DEFAULT(0.));
_CXCORE_SWITCH_EXPORT_C double cvgswReadRealByName(const  CvFileStorage* fs, const  CvFileNode* map, const  char* name, double default_value CV_DEFAULT(0.));
_CXCORE_SWITCH_EXPORT_C const char* cvgswReadString(const  CvFileNode* node, const  char* default_value CV_DEFAULT(NULL));
_CXCORE_SWITCH_EXPORT_C const char* cvgswReadStringByName(const  CvFileStorage* fs, const  CvFileNode* map, const  char* name, const  char* default_value CV_DEFAULT(NULL));
_CXCORE_SWITCH_EXPORT_C void cvgswRectangle(CvArr* img, CvPoint pt1, CvPoint pt2, CvScalar color, int thickness CV_DEFAULT(1), int line_type CV_DEFAULT(8), int shift CV_DEFAULT(0));
_CXCORE_SWITCH_EXPORT_C void cvgswRectangleR(CvArr* img, CvRect r, CvScalar color, int thickness CV_DEFAULT(1), int line_type CV_DEFAULT(8), int shift CV_DEFAULT(0));
_CXCORE_SWITCH_EXPORT_C CvErrorCallback cvgswRedirectError(CvErrorCallback error_handler, void* userdata CV_DEFAULT(NULL), void** prev_userdata CV_DEFAULT(NULL));
_CXCORE_SWITCH_EXPORT_C void cvgswReduce( CvArr* src, CvArr* dst, int dim CV_DEFAULT(-1), int op CV_DEFAULT(CV_REDUCE_SUM));
_CXCORE_SWITCH_EXPORT_C int cvgswRegisterModule(const  CvModuleInfo* module_info);
_CXCORE_SWITCH_EXPORT_C void cvgswRegisterType(const  CvTypeInfo* info);
_CXCORE_SWITCH_EXPORT_C void cvgswRelease(void** struct_ptr);
_CXCORE_SWITCH_EXPORT_C void cvgswReleaseData(CvArr* arr);
_CXCORE_SWITCH_EXPORT_C void cvgswReleaseFileStorage(CvFileStorage** fs);
_CXCORE_SWITCH_EXPORT_C void cvgswReleaseGraphScanner(CvGraphScanner** scanner);
_CXCORE_SWITCH_EXPORT_C void cvgswReleaseImage(IplImage** image);
_CXCORE_SWITCH_EXPORT_C void cvgswReleaseImageHeader(IplImage** image);
_CXCORE_SWITCH_EXPORT_C void cvgswReleaseMat(CvMat** mat);
_CXCORE_SWITCH_EXPORT_C void cvgswReleaseMatND(CvMatND** mat);
_CXCORE_SWITCH_EXPORT_C void cvgswReleaseMemStorage(CvMemStorage** storage);
_CXCORE_SWITCH_EXPORT_C void cvgswReleaseSparseMat(CvSparseMat** mat);
_CXCORE_SWITCH_EXPORT_C void cvgswRemoveNodeFromTree(void* node, void* frame);
_CXCORE_SWITCH_EXPORT_C void cvgswRepeat( CvArr* src, CvArr* dst);
_CXCORE_SWITCH_EXPORT_C void cvgswResetImageROI(IplImage* image);
_CXCORE_SWITCH_EXPORT_C CvMat* cvgswReshape( CvArr* arr, CvMat* header, int new_cn, int new_rows CV_DEFAULT(0));
_CXCORE_SWITCH_EXPORT_C CvArr* cvgswReshapeMatND( CvArr* arr, int sizeof_header, CvArr* header, int new_cn, int new_dims, int* new_sizes);
_CXCORE_SWITCH_EXPORT_C void cvgswRestoreMemStoragePos(CvMemStorage* storage, CvMemStoragePos* pos);
_CXCORE_SWITCH_EXPORT_C void cvgswSVBkSb( CvArr* W,  CvArr* U,  CvArr* V,  CvArr* B, CvArr* X, int flags);
_CXCORE_SWITCH_EXPORT_C void cvgswSVD(CvArr* A, CvArr* W, CvArr* U CV_DEFAULT(NULL), CvArr* V CV_DEFAULT(NULL), int flags CV_DEFAULT(0));
_CXCORE_SWITCH_EXPORT_C void cvgswSave(const  char* filename, const  void* struct_ptr, const  char* name CV_DEFAULT(NULL), const  char* comment CV_DEFAULT(NULL), CvAttrList attributes CV_DEFAULT(cvAttrList()));
_CXCORE_SWITCH_EXPORT_C void cvgswSaveMemStoragePos(const  CvMemStorage* storage, CvMemStoragePos* pos);
_CXCORE_SWITCH_EXPORT_C void cvgswScalarToRawData(const  CvScalar* scalar, void* data, int type, int extend_to_12 CV_DEFAULT(0));
_CXCORE_SWITCH_EXPORT_C void cvgswScaleAdd( CvArr* src1, CvScalar scale,  CvArr* src2, CvArr* dst);
_CXCORE_SWITCH_EXPORT_C int cvgswSeqElemIdx(const  CvSeq* seq, const  void* element, CvSeqBlock** block CV_DEFAULT(NULL));
_CXCORE_SWITCH_EXPORT_C CVAPI(schar*) cvgswSeqInsert(CvSeq* seq, int before_index, const  void* element CV_DEFAULT(NULL));
_CXCORE_SWITCH_EXPORT_C void cvgswSeqInsertSlice(CvSeq* seq, int before_index,  CvArr* from_arr);
_CXCORE_SWITCH_EXPORT_C void cvgswSeqInvert(CvSeq* seq);
_CXCORE_SWITCH_EXPORT_C int cvgswSeqPartition(const  CvSeq* seq, CvMemStorage* storage, CvSeq** labels, CvCmpFunc is_equal, void* userdata);
_CXCORE_SWITCH_EXPORT_C void cvgswSeqPop(CvSeq* seq, void* element CV_DEFAULT(NULL));
_CXCORE_SWITCH_EXPORT_C void cvgswSeqPopFront(CvSeq* seq, void* element CV_DEFAULT(NULL));
_CXCORE_SWITCH_EXPORT_C void cvgswSeqPopMulti(CvSeq* seq, void* elements, int count, int in_front CV_DEFAULT(0));
_CXCORE_SWITCH_EXPORT_C CVAPI(schar*) cvgswSeqPush(CvSeq* seq, const  void* element CV_DEFAULT(NULL));
_CXCORE_SWITCH_EXPORT_C CVAPI(schar*) cvgswSeqPushFront(CvSeq* seq, const  void* element CV_DEFAULT(NULL));
_CXCORE_SWITCH_EXPORT_C void cvgswSeqPushMulti(CvSeq* seq, const  void* elements, int count, int in_front CV_DEFAULT(0));
_CXCORE_SWITCH_EXPORT_C void cvgswSeqRemove(CvSeq* seq, int index);
_CXCORE_SWITCH_EXPORT_C void cvgswSeqRemoveSlice(CvSeq* seq, CvSlice slice);
_CXCORE_SWITCH_EXPORT_C CVAPI(schar*) cvgswSeqSearch(CvSeq* seq, const  void* elem, CvCmpFunc func, int is_sorted, int* elem_idx, void* userdata CV_DEFAULT(NULL));
_CXCORE_SWITCH_EXPORT_C CvSeq* cvgswSeqSlice(const  CvSeq* seq, CvSlice slice, CvMemStorage* storage CV_DEFAULT(NULL), int copy_data CV_DEFAULT(0));
_CXCORE_SWITCH_EXPORT_C void cvgswSeqSort(CvSeq* seq, CvCmpFunc func, void* userdata CV_DEFAULT(NULL));
_CXCORE_SWITCH_EXPORT_C void cvgswSet(CvArr* arr, CvScalar value,  CvArr* mask CV_DEFAULT(NULL));
_CXCORE_SWITCH_EXPORT_C void cvgswSet1D(CvArr* arr, int idx0, CvScalar value);
_CXCORE_SWITCH_EXPORT_C void cvgswSet2D(CvArr* arr, int idx0, int idx1, CvScalar value);
_CXCORE_SWITCH_EXPORT_C void cvgswSet3D(CvArr* arr, int idx0, int idx1, int idx2, CvScalar value);
_CXCORE_SWITCH_EXPORT_C int cvgswSetAdd(CvSet* set_header, CvSetElem* elem CV_DEFAULT(NULL), CvSetElem** inserted_elem CV_DEFAULT(NULL));
_CXCORE_SWITCH_EXPORT_C void cvgswSetData(CvArr* arr, void* data, int step);
_CXCORE_SWITCH_EXPORT_C int cvgswSetErrMode(int mode);
_CXCORE_SWITCH_EXPORT_C void cvgswSetErrStatus(int status);
_CXCORE_SWITCH_EXPORT_C void cvgswSetIPLAllocators(Cv_iplCreateImageHeader create_header, Cv_iplAllocateImageData allocate_data, Cv_iplDeallocate deallocate, Cv_iplCreateROI create_roi, Cv_iplCloneImage clone_image);
_CXCORE_SWITCH_EXPORT_C void cvgswSetIdentity(CvArr* mat, CvScalar value CV_DEFAULT(cvRealScalar(1)));
_CXCORE_SWITCH_EXPORT_C void cvgswSetImageCOI(IplImage* image, int coi);
_CXCORE_SWITCH_EXPORT_C void cvgswSetImageROI(IplImage* image, CvRect rect);
_CXCORE_SWITCH_EXPORT_C void cvgswSetMemoryManager(CvAllocFunc alloc_func CV_DEFAULT(NULL), CvFreeFunc free_func CV_DEFAULT(NULL), void* userdata CV_DEFAULT(NULL));
_CXCORE_SWITCH_EXPORT_C void cvgswSetND(CvArr* arr, const  int* idx, CvScalar value);
_CXCORE_SWITCH_EXPORT_C CvSetElem* cvgswSetNew(CvSet* set_header);
_CXCORE_SWITCH_EXPORT_C void cvgswSetNumThreads(int threads CV_DEFAULT(0));
_CXCORE_SWITCH_EXPORT_C void cvgswSetReal1D(CvArr* arr, int idx0, double value);
_CXCORE_SWITCH_EXPORT_C void cvgswSetReal2D(CvArr* arr, int idx0, int idx1, double value);
_CXCORE_SWITCH_EXPORT_C void cvgswSetReal3D(CvArr* arr, int idx0, int idx1, int idx2, double value);
_CXCORE_SWITCH_EXPORT_C void cvgswSetRealND(CvArr* arr, const  int* idx, double value);
_CXCORE_SWITCH_EXPORT_C void cvgswSetRemove(CvSet* set_header, int index);
_CXCORE_SWITCH_EXPORT_C void cvgswSetRemoveByPtr(CvSet* set_header, void* elem);
_CXCORE_SWITCH_EXPORT_C void cvgswSetSeqBlockSize(CvSeq* seq, int delta_elems);
_CXCORE_SWITCH_EXPORT_C void cvgswSetSeqReaderPos(CvSeqReader* reader, int index, int is_relative CV_DEFAULT(0));
_CXCORE_SWITCH_EXPORT_C void cvgswSetZero(CvArr* arr);
_CXCORE_SWITCH_EXPORT_C int cvgswSliceLength(CvSlice slice, const  CvSeq* seq);
_CXCORE_SWITCH_EXPORT_C int cvgswSolve( CvArr* src1,  CvArr* src2, CvArr* dst, int method CV_DEFAULT(CV_LU));
_CXCORE_SWITCH_EXPORT_C int cvgswSolveCubic( CvMat* coeffs, CvMat* roots);
_CXCORE_SWITCH_EXPORT_C void cvgswSolvePoly( CvMat* coeffs, CvMat * roots2, 			int maxiter CV_DEFAULT(20), int fig CV_DEFAULT(100));
_CXCORE_SWITCH_EXPORT_C void cvgswSort( CvArr* src, CvArr* dst CV_DEFAULT(NULL), CvArr* idxmat CV_DEFAULT(NULL), int flags CV_DEFAULT(0));
_CXCORE_SWITCH_EXPORT_C void cvgswSplit( CvArr* src, CvArr* dst0, CvArr* dst1, CvArr* dst2, CvArr* dst3);
_CXCORE_SWITCH_EXPORT_C void cvgswStartAppendToSeq(CvSeq* seq, CvSeqWriter* writer);
_CXCORE_SWITCH_EXPORT_C void cvgswStartNextStream(CvFileStorage* fs);
_CXCORE_SWITCH_EXPORT_C void cvgswStartReadRawData(const  CvFileStorage* fs, const  CvFileNode* src, CvSeqReader* reader);
_CXCORE_SWITCH_EXPORT_C void cvgswStartReadSeq(const  CvSeq* seq, CvSeqReader* reader, int reverse CV_DEFAULT(0));
_CXCORE_SWITCH_EXPORT_C void cvgswStartWriteSeq(int seq_flags, int header_size, int elem_size, CvMemStorage* storage, CvSeqWriter* writer);
_CXCORE_SWITCH_EXPORT_C void cvgswStartWriteStruct(CvFileStorage* fs, const  char* name, int struct_flags, const  char* type_name CV_DEFAULT(NULL), CvAttrList attributes CV_DEFAULT(cvAttrList()));
_CXCORE_SWITCH_EXPORT_C int cvgswStdErrReport(int status, const  char* func_name, const  char* err_msg, const  char* file_name, int line, void* userdata);
_CXCORE_SWITCH_EXPORT_C void cvgswSub( CvArr* src1,  CvArr* src2, CvArr* dst,  CvArr* mask CV_DEFAULT(NULL));
_CXCORE_SWITCH_EXPORT_C void cvgswSubRS( CvArr* src, CvScalar value, CvArr* dst,  CvArr* mask CV_DEFAULT(NULL));
_CXCORE_SWITCH_EXPORT_C void cvgswSubS( CvArr* src, CvScalar value, CvArr* dst,  CvArr* mask CV_DEFAULT(NULL));
_CXCORE_SWITCH_EXPORT_C CvScalar cvgswSum( CvArr* arr);
_CXCORE_SWITCH_EXPORT_C CvScalar cvgswTrace( CvArr* mat);
_CXCORE_SWITCH_EXPORT_C void cvgswTransform( CvArr* src, CvArr* dst,  CvMat* transmat,  CvMat* shiftvec CV_DEFAULT(NULL));
_CXCORE_SWITCH_EXPORT_C void cvgswTranspose( CvArr* src, CvArr* dst);
_CXCORE_SWITCH_EXPORT_C CvSeq* cvgswTreeToNodeSeq(const  void* first, int header_size, CvMemStorage* storage);
_CXCORE_SWITCH_EXPORT_C CvTypeInfo* cvgswTypeOf(const  void* struct_ptr);
_CXCORE_SWITCH_EXPORT_C void cvgswUnregisterType(const  char* type_name);
_CXCORE_SWITCH_EXPORT_C int cvgswUseOptimized(int on_off);
_CXCORE_SWITCH_EXPORT_C void cvgswWrite(CvFileStorage* fs, const  char* name, const  void* ptr, CvAttrList attributes CV_DEFAULT(cvAttrList()));
_CXCORE_SWITCH_EXPORT_C void cvgswWriteComment(CvFileStorage* fs, const  char* comment, int eol_comment);
_CXCORE_SWITCH_EXPORT_C void cvgswWriteFileNode(CvFileStorage* fs, const  char* new_node_name, const  CvFileNode* node, int embed);
_CXCORE_SWITCH_EXPORT_C void cvgswWriteInt(CvFileStorage* fs, const  char* name, int value);
_CXCORE_SWITCH_EXPORT_C void cvgswWriteRawData(CvFileStorage* fs, const  void* src, int len, const  char* dt);
_CXCORE_SWITCH_EXPORT_C void cvgswWriteReal(CvFileStorage* fs, const  char* name, double value);
_CXCORE_SWITCH_EXPORT_C void cvgswWriteString(CvFileStorage* fs, const  char* name, const  char* str, int quote CV_DEFAULT(0));
_CXCORE_SWITCH_EXPORT_C void cvgswXor( CvArr* src1,  CvArr* src2, CvArr* dst,  CvArr* mask CV_DEFAULT(NULL));
_CXCORE_SWITCH_EXPORT_C void cvgswXorS( CvArr* src, CvScalar value, CvArr* dst,  CvArr* mask CV_DEFAULT(NULL));
/*........End Declaration.............*/


#endif //__CXCORE_SWITCH_H