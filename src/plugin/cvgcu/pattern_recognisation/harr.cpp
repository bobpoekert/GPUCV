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

#if 0
/* Haar features calculation */


#include "StdAfx.h"
#include <cvgcu/config.h>
#include <stdio.h>
#include <cvgcu/harr.h>
//#include <cxcoregcu/cxcoregcu_ankit.h>

using namespace GCV;
/* these settings affect the quality of detection: change with care */
#define CV_ADJUST_FEATURES 1
#define CV_ADJUST_WEIGHTS  0
#define GPUCV_HAAR	0
#define GPUCV_HAAR_SCALE_FACTOR 1
#define GPUCV_HAAR_DEBUG 0

typedef int sumtype;
typedef double sqsumtype;

typedef struct CvHidHaarFeature
{
	struct
	{
		sumtype *p0, *p1, *p2, *p3;
		float weight;
	}
	rect[CV_HAAR_FEATURE_MAX];
}
CvHidHaarFeature;


typedef struct CvHidHaarTreeNode
{
	CvHidHaarFeature feature;
	float threshold;
	int left;
	int right;
}
CvHidHaarTreeNode;


typedef struct CvHidHaarClassifier
{
	int count;
	//CvHaarFeature* orig_feature;
	CvHidHaarTreeNode* node;
	float* alpha;
}
CvHidHaarClassifier;


typedef struct CvHidHaarStageClassifier
{
	int  count;
	float threshold;
	CvHidHaarClassifier* classifier;
	int two_rects;

	struct CvHidHaarStageClassifier* next;
	struct CvHidHaarStageClassifier* child;
	struct CvHidHaarStageClassifier* parent;
}
CvHidHaarStageClassifier;


struct CvHidHaarClassifierCascade
{
	int  count;
	int  is_stump_based;
	int  has_tilted_features;
	int  is_tree;
	double inv_window_area;
	CvMat sum, sqsum, tilted;
	CvHidHaarStageClassifier* stage_classifier;
	sqsumtype *pq0, *pq1, *pq2, *pq3;
	sumtype *p0, *p1, *p2, *p3;

	void** ipp_stages;
};
#if 0
icvHaarClassifierInitAlloc_32f_t icvHaarClassifierInitAlloc_32f_p = 0;
icvHaarClassifierFree_32f_t icvHaarClassifierFree_32f_p = 0;
icvApplyHaarClassifier_32s32f_C1R_t icvApplyHaarClassifier_32s32f_C1R_p = 0;
icvRectStdDev_32s32f_C1R_t icvRectStdDev_32s32f_C1R_p = 0;

const int icv_object_win_border = 1;
const float icv_stage_threshold_bias = 0.0001f;


static int is_equal( const void* _r1, const void* _r2, void* )
{
	const CvRect* r1 = (const CvRect*)_r1;
	const CvRect* r2 = (const CvRect*)_r2;
	int distance = cvRound(r1->width*0.2);

	return r2->x <= r1->x + distance &&
		r2->x >= r1->x - distance &&
		r2->y <= r1->y + distance &&
		r2->y >= r1->y - distance &&
		r2->width <= cvRound( r1->width * 1.2 ) &&
		cvRound( r2->width * 1.2 ) >= r1->width;
}

static void
icvReleaseHidHaarClassifierCascade( CvHidHaarClassifierCascade** _cascade )
{
	if( _cascade && *_cascade )
	{
		CvHidHaarClassifierCascade* cascade = *_cascade;
		if( cascade->ipp_stages && icvHaarClassifierFree_32f_p )
		{
			int i;
			for( i = 0; i < cascade->count; i++ )
			{
				if( cascade->ipp_stages[i] )
					icvHaarClassifierFree_32f_p( cascade->ipp_stages[i] );
			}
		}
		cvFree( &cascade->ipp_stages );
		cvFree( _cascade );
	}
}



/* create more efficient internal representation of haar classifier cascade */


static CvHidHaarClassifierCascade*
icvCreateHidHaarClassifierCascade( CvHaarClassifierCascade* cascade )
{


	//intialisation
	CvRect* ipp_features = 0;
	float *ipp_weights = 0, *ipp_thresholds = 0, *ipp_val1 = 0, *ipp_val2 = 0;
	int* ipp_counts = 0;

	CvHidHaarClassifierCascade* out = 0;

	CV_FUNCNAME( "icvCreateHidHaarClassifierCascade" );

	__BEGIN__;

	int i, j, k, l;
	int datasize;
	int total_classifiers = 0;
	int total_nodes = 0;
	char errorstr[100];
	CvHidHaarClassifier* haar_classifier_ptr;
	CvHidHaarTreeNode* haar_node_ptr;
	CvSize orig_window_size;
	int has_tilted_features = 0;
	int max_count = 0;


	/////checking the inputs

	if( !CV_IS_HAAR_CLASSIFIER(cascade) )
		CV_ERROR( !cascade ? CV_StsNullPtr : CV_StsBadArg, "Invalid classifier pointer" );

	if( cascade->hid_cascade )
		CV_ERROR( CV_StsError, "hid_cascade has been already created" );

	if( !cascade->stage_classifier )
		CV_ERROR( CV_StsNullPtr, "" );

	if( cascade->count <= 0 )
		CV_ERROR( CV_StsOutOfRange, "Negative number of cascade stages" );

	orig_window_size = cascade->orig_window_size;

	/* check input structure correctness and calculate total memory size needed for
	internal representation of the classifier cascade */
	for( i = 0; i < cascade->count; i++ )
	{
		CvHaarStageClassifier* stage_classifier = cascade->stage_classifier + i;

		if( !stage_classifier->classifier ||
			stage_classifier->count <= 0 )
		{
			sprintf( errorstr, "header of the stage classifier #%d is invalid "
				"(has null pointers or non-positive classfier count)", i );
			CV_ERROR( CV_StsError, errorstr );
		}

		max_count = MAX( max_count, stage_classifier->count );
		total_classifiers += stage_classifier->count;

		for( j = 0; j < stage_classifier->count; j++ )
		{
			CvHaarClassifier* classifier = stage_classifier->classifier + j;

			total_nodes += classifier->count;
			for( l = 0; l < classifier->count; l++ )
			{
				for( k = 0; k < CV_HAAR_FEATURE_MAX; k++ )
				{
					if( classifier->haar_feature[l].rect[k].r.width )
					{
						CvRect r = classifier->haar_feature[l].rect[k].r;
						int tilted = classifier->haar_feature[l].tilted;
						has_tilted_features |= tilted != 0;
						if( r.width < 0 || r.height < 0 || r.y < 0 ||
							r.x + r.width > orig_window_size.width
							||
							(!tilted &&
							(r.x < 0 || r.y + r.height > orig_window_size.height))
							||
							(tilted && (r.x - r.height < 0 ||
							r.y + r.width + r.height > orig_window_size.height)))
						{
							sprintf( errorstr, "rectangle #%d of the classifier #%d of "
								"the stage classifier #%d is not inside "
								"the reference (original) cascade window", k, j, i );
							CV_ERROR( CV_StsNullPtr, errorstr );
						}
					}
				}
			}
		}
	}

	// this is an upper boundary for the whole hidden cascade size
	datasize = sizeof(CvHidHaarClassifierCascade) +
		sizeof(CvHidHaarStageClassifier)*cascade->count +
		sizeof(CvHidHaarClassifier) * total_classifiers +
		sizeof(CvHidHaarTreeNode) * total_nodes +
		sizeof(void*)*(total_nodes + total_classifiers);

	CV_CALL( out = (CvHidHaarClassifierCascade*)cvAlloc( datasize ));
	memset( out, 0, sizeof(*out) );

	/* init header */
	out->count = cascade->count;
	out->stage_classifier = (CvHidHaarStageClassifier*)(out + 1);
	haar_classifier_ptr = (CvHidHaarClassifier*)(out->stage_classifier + cascade->count);
	haar_node_ptr = (CvHidHaarTreeNode*)(haar_classifier_ptr + total_classifiers);

	out->is_stump_based = 0;
	out->has_tilted_features = has_tilted_features;
	out->is_tree = 0;

	/* initialize internal representation */
	for( i = 0; i < cascade->count; i++ )
	{
		CvHaarStageClassifier* stage_classifier = cascade->stage_classifier + i;
		CvHidHaarStageClassifier* hid_stage_classifier = out->stage_classifier + i;

		hid_stage_classifier->count = stage_classifier->count;
		hid_stage_classifier->threshold = stage_classifier->threshold - icv_stage_threshold_bias;
		hid_stage_classifier->classifier = haar_classifier_ptr;
		hid_stage_classifier->two_rects = 1;
		haar_classifier_ptr += stage_classifier->count;

		hid_stage_classifier->parent = (stage_classifier->parent == -1)
			? NULL : out->stage_classifier + stage_classifier->parent;
		hid_stage_classifier->next = (stage_classifier->next == -1)
			? NULL : out->stage_classifier + stage_classifier->next;
		hid_stage_classifier->child = (stage_classifier->child == -1)
			? NULL : out->stage_classifier + stage_classifier->child;

		out->is_tree |= hid_stage_classifier->next != NULL;

		for( j = 0; j < stage_classifier->count; j++ )
		{
			CvHaarClassifier* classifier = stage_classifier->classifier + j;
			CvHidHaarClassifier* hid_classifier = hid_stage_classifier->classifier + j;
			int node_count = classifier->count;
			float* alpha_ptr = (float*)(haar_node_ptr + node_count);

			hid_classifier->count = node_count;
			hid_classifier->node = haar_node_ptr;
			hid_classifier->alpha = alpha_ptr;

			for( l = 0; l < node_count; l++ )
			{
				CvHidHaarTreeNode* node = hid_classifier->node + l;
				CvHaarFeature* feature = classifier->haar_feature + l;
				memset( node, -1, sizeof(*node) );
				node->threshold = classifier->threshold[l];
				node->left = classifier->left[l];
				node->right = classifier->right[l];

				if( fabs(feature->rect[2].weight) < DBL_EPSILON ||
					feature->rect[2].r.width == 0 ||
					feature->rect[2].r.height == 0 )
					memset( &(node->feature.rect[2]), 0, sizeof(node->feature.rect[2]) );
				else
					hid_stage_classifier->two_rects = 0;
			}

			memcpy( alpha_ptr, classifier->alpha, (node_count+1)*sizeof(alpha_ptr[0]));
			haar_node_ptr =
				(CvHidHaarTreeNode*)cvAlignPtr(alpha_ptr+node_count+1, sizeof(void*));

			out->is_stump_based &= node_count == 1;
		}
	}

	//
	// NOTE: Currently, OpenMP is implemented and IPP modes are incompatible.
	//
#ifndef _OPENMP
	{
		int can_use_ipp = icvHaarClassifierInitAlloc_32f_p != 0 &&
			icvHaarClassifierFree_32f_p != 0 &&
			icvApplyHaarClassifier_32s32f_C1R_p != 0 &&
			icvRectStdDev_32s32f_C1R_p != 0 &&
			!out->has_tilted_features && !out->is_tree && out->is_stump_based;

		if( can_use_ipp )
		{
			int ipp_datasize = cascade->count*sizeof(out->ipp_stages[0]);
			float ipp_weight_scale=(float)(1./((orig_window_size.width-icv_object_win_border*2)*
				(orig_window_size.height-icv_object_win_border*2)));

			CV_CALL( out->ipp_stages = (void**)cvAlloc( ipp_datasize ));
			memset( out->ipp_stages, 0, ipp_datasize );

			CV_CALL( ipp_features = (CvRect*)cvAlloc( max_count*3*sizeof(ipp_features[0]) ));
			CV_CALL( ipp_weights = (float*)cvAlloc( max_count*3*sizeof(ipp_weights[0]) ));
			CV_CALL( ipp_thresholds = (float*)cvAlloc( max_count*sizeof(ipp_thresholds[0]) ));
			CV_CALL( ipp_val1 = (float*)cvAlloc( max_count*sizeof(ipp_val1[0]) ));
			CV_CALL( ipp_val2 = (float*)cvAlloc( max_count*sizeof(ipp_val2[0]) ));
			CV_CALL( ipp_counts = (int*)cvAlloc( max_count*sizeof(ipp_counts[0]) ));

			for( i = 0; i < cascade->count; i++ )
			{
				CvHaarStageClassifier* stage_classifier = cascade->stage_classifier + i;
				for( j = 0, k = 0; j < stage_classifier->count; j++ )
				{
					CvHaarClassifier* classifier = stage_classifier->classifier + j;
					int rect_count = 2 + (classifier->haar_feature->rect[2].r.width != 0);

					ipp_thresholds[j] = classifier->threshold[0];
					ipp_val1[j] = classifier->alpha[0];
					ipp_val2[j] = classifier->alpha[1];
					ipp_counts[j] = rect_count;

					for( l = 0; l < rect_count; l++, k++ )
					{
						ipp_features[k] = classifier->haar_feature->rect[l].r;
						//ipp_features[k].y = orig_window_size.height - ipp_features[k].y - ipp_features[k].height;
						ipp_weights[k] = classifier->haar_feature->rect[l].weight*ipp_weight_scale;
					}
				}

				if( icvHaarClassifierInitAlloc_32f_p( &out->ipp_stages[i],
					ipp_features, ipp_weights, ipp_thresholds,
					ipp_val1, ipp_val2, ipp_counts, stage_classifier->count ) < 0 )
					break;
			}

			if( i < cascade->count )
			{
				for( j = 0; j < i; j++ )
					if( icvHaarClassifierFree_32f_p && out->ipp_stages[i] )
						icvHaarClassifierFree_32f_p( out->ipp_stages[i] );
				cvFree( &out->ipp_stages );
			}
		}
	}
#endif

	cascade->hid_cascade = out;
	assert( (char*)haar_node_ptr - (char*)out <= datasize );

	__END__;

	if( cvGetErrStatus() < 0 )
		icvReleaseHidHaarClassifierCascade( &out );

	cvFree( &ipp_features );
	cvFree( &ipp_weights );
	cvFree( &ipp_thresholds );
	cvFree( &ipp_val1 );
	cvFree( &ipp_val2 );
	cvFree( &ipp_counts );

	return out;
}


CvSeq* passfunction(CvHaarClassifierCascade* cascade, int stop_width, int stop_height,CvSize win_size,CvMat * temp,CvSeq * seq)

{
	int pass, stage_offset = 0;

	int npass = 1;
	int ystep =2;

	for( pass = 0; pass < npass; pass++ )
	{
		GPUCV_PROFILE_CURRENT_FCT("sumPass",temp, 1, GpuCVSettings::GPUCV_SETTINGS_PROFILING_OPER);

		for( int _iy = 0; _iy < stop_height; _iy++)
		{
			int iy = cvRound(_iy*ystep);
			int _ix, _xstep = 1;

			uchar* mask_row = temp->data.ptr + temp->step * iy;

			for( _ix = 0; _ix < stop_width; _ix += _xstep )
			{
				int ix = cvRound(_ix*ystep); // it really should be ystep

				if( pass == 0 )
				{
					int result=1;
					_xstep = 2;

					{
						//GPUCV_PROFILE_CURRENT_FCT("runhaar", img, 1, GpuCVSettings::GPUCV_SETTINGS_PROFILING_OPER);
						result = cvRunHaarClassifierCascade(cascade, cvPoint(ix,iy), 0 );
					}
					if( result > 0 )
					{
						if( pass < npass - 1 )
							mask_row[ix] = 1;
						else
						{
							CvRect rect = cvRect(ix,iy,win_size.width,win_size.height);
							cvSeqPush( seq, &rect );
						}
					}
					if( result < 0 )
						_xstep = 1;
				}
				else if( mask_row[ix] )
				{
					int result = cvRunHaarClassifierCascade(cascade, cvPoint(ix,iy),stage_offset );

					if( result > 0 )
					{
						if( pass == npass - 1 )
						{
							CvRect rect = cvRect(ix,iy,win_size.width,win_size.height);
							cvSeqPush( seq, &rect );
						}
					}
					else
						mask_row[ix] = 0;
				}
			}
		}
		stage_offset = cascade->hid_cascade->count;
		cascade->hid_cascade->count = cascade->count;
	}
	return seq;

}



/*CV_IMPL*/ CvSeq* cvgHaarDetectObjects(const CvArr* _img, CvHaarClassifierCascade* cascade, CvMemStorage* storage, double scale_factor, int min_neighbors, int flags, CvSize min_size )
{
	int split_stage = 2;

	CvMat stub, *img = (CvMat*)_img;
	CvMat *temp = 0, *sum = 0, *tempsum = 0,*tilted = 0, *sqsum = 0, *sqsum32f = 0,*sum32f = 0,*norm_img = 0, *sumcanny = 0,*tempsumcanny = 0, *img_small = 0;
	CvSeq* seq = 0;
	CvSeq* tempseq = 0;
	CvSeq* seq2 = 0;
	CvSeq* idx_seq = 0;
	CvSeq* result_seq = 0;
	CvMemStorage* temp_storage = 0;
	CvAvgComp* comps = 0;
	int i;

#ifdef _OPENMP
	CvSeq* seq_thread[CV_MAX_THREADS] = {0};
	int max_threads = 0;
#endif

	CV_FUNCNAME( "cvHaarDetectObjects" );

	__BEGIN__;

	double factor;
	int npass = 2, coi;
	int do_canny_pruning = flags & CV_HAAR_DO_CANNY_PRUNING;

	if( !CV_IS_HAAR_CLASSIFIER(cascade) )
		CV_ERROR( !cascade ? CV_StsNullPtr : CV_StsBadArg, "Invalid classifier cascade" );

	if( !storage )
		CV_ERROR( CV_StsNullPtr, "Null storage pointer" );

	CV_CALL( img = cvGetMat( img, &stub, &coi ));
	if( coi )
		CV_ERROR( CV_BadCOI, "COI is not supported" );

	if( CV_MAT_DEPTH(img->type) != CV_8U )
		CV_ERROR( CV_StsUnsupportedFormat, "Only 8-bit images are supported" );

	CV_CALL( temp = cvCreateMat( img->rows, img->cols, CV_8UC1 ));
	CV_CALL( sum = cvCreateMat( img->rows + 1, img->cols + 1, CV_32SC1 ));
	CV_CALL( sum32f = cvCreateMat( img->rows + 1, img->cols + 1, CV_32FC1 ));
	CV_CALL( tempsum = cvCreateMat( img->rows + 1, img->cols + 1, CV_32SC1 ));
	CV_CALL( sqsum32f = cvCreateMat( img->rows + 1, img->cols + 1, CV_32FC1 ));
	CV_CALL( temp_storage = cvCreateChildMemStorage( storage ));
	CV_CALL( sqsum = cvCreateMat( img->rows + 1, img->cols + 1, CV_64FC1 ));



#ifdef _OPENMP
	max_threads = cvGetNumThreads();
	for( i = 0; i < max_threads; i++ )
	{
		CvMemStorage* temp_storage_thread;
		CV_CALL( temp_storage_thread = cvCreateMemStorage(0));
		CV_CALL( seq_thread[i] = cvCreateSeq( 0, sizeof(CvSeq),
			sizeof(CvRect), temp_storage_thread ));
	}
#endif

	if( !cascade->hid_cascade )
		CV_CALL( icvCreateHidHaarClassifierCascade(cascade) );

	if( cascade->hid_cascade->has_tilted_features )
		tilted = cvCreateMat( img->rows + 1, img->cols + 1, CV_32SC1 );

	seq = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvRect), temp_storage );
	tempseq = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvRect), temp_storage );
	seq2 = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvAvgComp), temp_storage );
	result_seq = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvAvgComp), storage );

	if( min_neighbors == 0 )
		seq = result_seq;

	if( CV_MAT_CN(img->type) > 1 )
	{

#if GPUCV_HAAR
		//GPU???
		cvCvtColor( img, temp, CV_BGR2GRAY );
#else
		cvCvtColor(img,temp,CV_BGR2GRAY);
#endif
		img = temp;
	}

	if( flags & CV_HAAR_SCALE_IMAGE )
	{
		CvSize win_size0 = cascade->orig_window_size;
		int use_ipp = cascade->hid_cascade->ipp_stages != 0 &&
			icvApplyHaarClassifier_32s32f_C1R_p != 0;

		for( factor = 1; ; factor *= scale_factor)
		{
			int positive = 0;
			int x, y;
			CvSize win_size = { cvRound(win_size0.width*factor),
				cvRound(win_size0.height*factor) };
			CvSize sz = { cvRound( img->cols/factor ), cvRound( img->rows/factor ) };
			CvSize sz1 = { sz.width - win_size0.width, sz.height - win_size0.height };
			CvRect rect1 = { icv_object_win_border, icv_object_win_border,
				win_size0.width - icv_object_win_border*2,
				win_size0.height - icv_object_win_border*2 };
			CvMat img1, sum1,tempsum1, sqsum1, norm1, tilted1, mask1;
			CvMat* _tilted = 0;

			if( sz1.width <= 0 || sz1.height <= 0 )
				break;
			if( win_size.width < min_size.width || win_size.height < min_size.height )
				continue;

			img1 = cvMat( sz.height, sz.width, CV_8UC1, img_small->data.ptr );
			sum1 = cvMat( sz.height+1, sz.width+1, CV_32SC1, sum->data.ptr );
			tempsum1 = cvMat( sz.height+1, sz.width+1, CV_32SC1, tempsum->data.ptr );
			sqsum1 = cvMat( sz.height+1, sz.width+1, CV_32SC1, sqsum->data.ptr );

			if( tilted )
			{
				tilted1 = cvMat( sz.height+1, sz.width+1, CV_32SC1, tilted->data.ptr );
				_tilted = &tilted1;
			}
			norm1 = cvMat( sz1.height, sz1.width, CV_32FC1, norm_img ? norm_img->data.ptr : 0 );
			mask1 = cvMat( sz1.height, sz1.width, CV_8UC1, temp->data.ptr );


			cvResize( img, &img1, CV_INTER_LINEAR );
#if GPUCV_HAAR
			//cvIntegral( &img1, &sum1, &sqsum1, _tilted );
			cvgCudaIntegral( &img1, &sum1, &sqsum1, _tilted );


#else
			{_PROFILE_FCT_CREATE_START("cvIntegral( &img1, &sum1, &sqsum1, _tilted );");
			cvIntegral( &img1, &sum1, &sqsum1, _tilted );
			}

			//cvgShowMatrix(&sum1,1,30,30);
#endif



			if( !use_ipp )
			{
				cvSetImagesForHaarClassifierCascade( cascade, &sum1, &sqsum1, 0, 1. );
				for( y = 0, positive = 0; y < sz1.height; y++ )
					for( x = 0; x < sz1.width; x++ )
					{
						mask1.data.ptr[mask1.step*y + x] =
							cvRunHaarClassifierCascade( cascade, cvPoint(x,y), 0 ) > 0;
						positive += mask1.data.ptr[mask1.step*y + x];
					}
			}

			if( positive > 0 )
			{
				for( y = 0; y < sz1.height; y++ )
					for( x = 0; x < sz1.width; x++ )
						if( mask1.data.ptr[mask1.step*y + x] != 0 )
						{
							CvRect obj_rect = { cvRound(y*factor), cvRound(x*factor),
								win_size.width, win_size.height };
							cvSeqPush( seq, &obj_rect );
						}
			}
		}
	}
	else
	{
#if GPUCV_HAAR
		//cvIntegral (img,sum ,);
		cvgCudaIntegral( img, sum32f ,sqsum32f,tilted);
		cvgSynchronize(sqsum32f);
		std::cout << "\n" << (SGE::ToCharStr(sqsum32f->type));
		std::cout << "\n" << (SGE::ToCharStr(sqsum->type));
		cvConvertScale(sum32f,sqsum);

		/*cvgShowMatrix("32f",sqsum32f);
		cvIntegral( img, sum ,sqsum,tilted);
		cvgShowMatrix("64f",sqsum);

		/*cvConvertScale(sqsum32f,sqsum);
		cvgShowMatrix("square sum",sqsum);*/

#else
		{
			_PROFILE_FCT_CREATE_START("cvIntegral( img, sum, sqsum, tilted );");
			cvIntegral(img,sum,sqsum,tilted);
			cvgShowMatrix("square sum",sqsum);
		}
#endif

		//_PROFILE_FCT_CREATE_START("sum");
		{
			GPUCV_PROFILE_CURRENT_FCT("sum", img, 1, GpuCVSettings::GPUCV_SETTINGS_PROFILING_OPER);


			if( (unsigned)split_stage >= (unsigned)cascade->count ||
				cascade->hid_cascade->is_tree )
			{
				split_stage = cascade->count;
				npass = 1;
			}

			for( factor = 1; factor*cascade->orig_window_size.width < img->cols - 10 &&
				factor*cascade->orig_window_size.height < img->rows - 10;
				factor *= scale_factor )
			{
				const double ystep = MAX( 2, factor );
				CvSize win_size = { cvRound( cascade->orig_window_size.width * factor ),
					cvRound( cascade->orig_window_size.height * factor )};
				CvRect equ_rect = { 0, 0, 0, 0 };
				int *p0 = 0, *p1 = 0, *p2 = 0, *p3 = 0;
				int *pq0 = 0, *pq1 = 0, *pq2 = 0, *pq3 = 0;

				int stop_height = cvRound((img->rows - win_size.height) / ystep);

				if( win_size.width < min_size.width || win_size.height < min_size.height )
					continue;
				//cvgSynchronize(sqsum);
				cvSetImagesForHaarClassifierCascade( cascade, sum, sqsum, tilted, factor );
				cvZero( temp );


				cascade->hid_cascade->count = split_stage;
				int stop_width = cvRound((img->cols - win_size.width) / ystep);

				tempseq = passfunction(cascade,stop_width,stop_height,win_size,temp,seq);
				seq = tempseq;
			}
		}
	}// benchmarking sum


	{
		GPUCV_PROFILE_CURRENT_FCT("runhaar", img, 1, GpuCVSettings::GPUCV_SETTINGS_PROFILING_OPER);
		int f = cvRunHaarClassifierCascade(cascade, cvPoint(50,50), 0 );
	}

	if( min_neighbors != 0 )
	{
		// group retrieved rectangles in order to filter out noise
		int ncomp = cvSeqPartition( seq, 0, &idx_seq, is_equal, 0 );
		CV_CALL( comps = (CvAvgComp*)cvAlloc( (ncomp+1)*sizeof(comps[0])));
		memset( comps, 0, (ncomp+1)*sizeof(comps[0]));

		// count number of neighbors
		for( i = 0; i < seq->total; i++ )
		{
			CvRect r1 = *(CvRect*)cvGetSeqElem( seq, i );
			int idx = *(int*)cvGetSeqElem( idx_seq, i );
			assert( (unsigned)idx < (unsigned)ncomp );

			comps[idx].neighbors++;

			comps[idx].rect.x += r1.x;
			comps[idx].rect.y += r1.y;
			comps[idx].rect.width += r1.width;
			comps[idx].rect.height += r1.height;
		}

		// calculate average bounding box
		for( i = 0; i < ncomp; i++ )
		{
			int n = comps[i].neighbors;
			if( n >= min_neighbors )
			{
				CvAvgComp comp;
				comp.rect.x = (comps[i].rect.x*2 + n)/(2*n);
				comp.rect.y = (comps[i].rect.y*2 + n)/(2*n);
				comp.rect.width = (comps[i].rect.width*2 + n)/(2*n);
				comp.rect.height = (comps[i].rect.height*2 + n)/(2*n);
				comp.neighbors = comps[i].neighbors;

				cvSeqPush( seq2, &comp );
			}
		}

		// filter out small face rectangles inside large face rectangles
		for( i = 0; i < seq2->total; i++ )
		{
			CvAvgComp r1 = *(CvAvgComp*)cvGetSeqElem( seq2, i );
			int j, flag = 1;

			for( j = 0; j < seq2->total; j++ )
			{
				CvAvgComp r2 = *(CvAvgComp*)cvGetSeqElem( seq2, j );
				int distance = cvRound( r2.rect.width * 0.2 );

				if( i != j &&
					r1.rect.x >= r2.rect.x - distance &&
					r1.rect.y >= r2.rect.y - distance &&
					r1.rect.x + r1.rect.width <= r2.rect.x + r2.rect.width + distance &&
					r1.rect.y + r1.rect.height <= r2.rect.y + r2.rect.height + distance &&
					(r2.neighbors > MAX( 3, r1.neighbors ) || r1.neighbors < 3) )
				{
					flag = 0;
					break;
				}
			}

			if (flag)
			{
				cvSeqPush( result_seq, &r1 );
				/* cvSeqPush( result_seq, &r1.rect ); */
			}
		}
	}

	__END__;

#ifdef _OPENMP
	for( i = 0; i < max_threads; i++ )
	{
		if( seq_thread[i] )
			cvReleaseMemStorage( &seq_thread[i]->storage );
	}
#endif

	cvReleaseMemStorage( &temp_storage );
#if GPUCV_HAAR
	cvgReleaseMat( &sum );
	cvgReleaseMat( &sqsum );
	cvgReleaseMat( &sum32f );
	cvgReleaseMat( &tilted );
	cvgReleaseMat( &temp );
	cvgReleaseMat( &sumcanny );
	cvgReleaseMat( &tempsum );
	cvgReleaseMat( &tempsumcanny );
	cvgReleaseMat( &norm_img );
	cvgReleaseMat( &img_small );
#else
	cvReleaseMat( &sum );
	cvReleaseMat( &sqsum );
	cvReleaseMat( &tilted );
	cvReleaseMat( &temp );
	cvReleaseMat( &sumcanny );
	cvReleaseMat( &norm_img );
	cvReleaseMat( &img_small );
#endif
	cvFree( &comps );

	return result_seq;

}
#endif
#endif//is needed for linux???

