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
#include "StdAfx.h"
#include "mainSampleTest.h"
#include "commands.h"
#include "cxcoreg_test.h"
#include <cxcore_switch/cxcore_switch.h>


#if !_GCV_CUDA_EXTERNAL
#ifdef _GPUCV_SUPPORT_CUDA
	#include <cxcoregcu/cxcoregcu.h>
#endif
#endif

#include <GPUCV/cv_new.h>

using namespace GCV;

//CVGXCORE_OPER_ARRAY_INIT_GRP
//CVGXCORE_OPER_ARRAY_ACCESS_ELEM_GRP
//CVGXCORE_OPER_COPY_FILL_GRP
//CVGXCORE_OPER_TRANSFORM_PERMUT_GRP
//CVGXCORE_OPER_ARITHM_LOGIC_COMP_GRP
//CVGXCORE_OPER_STATS_GRP
//CVGXCORE_OPER_LINEAR_ALGEBRA_GRP
//CVGXCORE_OPER_MATH_FCT_GRP
//CVGXCORE_OPER_RANDOW_NUMBER_GRP
//CVGXCORE_OPER_DISCRETE_TRANS_GRP
//CVGXCORE_DRAWING_CURVES_SHAPES_GRP
//CVGXCORE_DRAWING_TEXT_GRP
//CVGXCORE_DRAWING_POINT_CONTOURS_GRP
bool cxcore_test_enable = true;
void cxcore_test_print()
{
#if TEST_GPUCV_CXCORE
	GPUCV_NOTICE("===<cxcore.h> test operators ========");
	GPUCV_NOTICE(" 'cxcore_all'\t : run all cxcore operators implemented.");
	GPUCV_NOTICE(" 'disable/enable cxcore'\t : enable benchmarking of cxcore* libraries(default=true).");
	GPUCV_NOTICE("-------------------------------------");
	GPUCV_NOTICE(" 'convertscale'\t : cvConvertScale()");
	GPUCV_NOTICE(" 'add'\t : cvAdd()");
	GPUCV_NOTICE(" 'adds'\t : cvAddS()");
	GPUCV_NOTICE(" 'and'\t : cvAnd()");
	GPUCV_NOTICE(" 'avg'\t : cvAvg()");
	GPUCV_NOTICE(" 'clone'\t : cvClone()");
	GPUCV_NOTICE(" 'copy'\t : cvCopy()");
	GPUCV_NOTICE(" 'div %float'\t : cvDiv(float factor)");
	GPUCV_NOTICE(" 'lut'\t : cvLut()");
	GPUCV_NOTICE(" 'matmul %float'\t : cvMatMul(float factor)");
	GPUCV_NOTICE(" 'matmuladd %float %float'\t : cvMatMulAdd(float factor1, float factor2)");
	GPUCV_NOTICE(" 'max'\t : cvMax()");
	GPUCV_NOTICE(" 'maxs'\t : cvMaxs()");
	GPUCV_NOTICE(" 'merge'\t : cvMerge()");
	GPUCV_NOTICE(" 'min'\t : cvMin()");
	GPUCV_NOTICE(" 'mins'\t : cvMinS()");
#if _GPUCV_DEVELOP_BETA
	GPUCV_NOTICE(" 'minmaxloc'\t : cvMinMaxLoc()");
#endif
	GPUCV_NOTICE(" 'mul %float'\t : cvMul(float factor)");
	GPUCV_NOTICE(" 'not'\t : cvNot()");
	GPUCV_NOTICE(" 'or'\t : cvOr()");
	GPUCV_NOTICE(" 'pow %double' \t : cvPow(double factor)");
	GPUCV_NOTICE(" 'scaleadd'\t : cvScaleAdd()");
	GPUCV_NOTICE(" 'set %value'\t : cvSet()");
	GPUCV_NOTICE(" 'split'\t : cvSplit()==cvCvtPixToPlane()");
	GPUCV_NOTICE(" 'sum'\t : cvSum()");
	GPUCV_NOTICE(" 'sub'\t : cvSub()");
	GPUCV_NOTICE(" 'subs'\t : cvSubS()");
	GPUCV_NOTICE(" 'subrs'\t : cvSubS()");
//	GPUCV_NOTICE(" 'subsmask'\t : cvSubS()");
//	GPUCV_NOTICE(" 'subrsmask'\t : cvSubS()");
	GPUCV_NOTICE(" 'xor'\t : cvXor()");
	GPUCV_NOTICE("===============================");
	GPUCV_NOTICE(" 'glxcudacompat'\t : compatibility test between GLSL and CUDA");
	
#endif
}

bool cxcoreg_processCommand(std::string & CurCmd, std::string & nextCmd)
{
	CurLibraryName="cxcore";
	//std::string Arg1;
	float factor=0;
	float factor2=0;
	if (CurCmd=="")
	{
		return false;
	}
	else if (cxcore_test_enable==false)
	{
		return false;
	}
#if TEST_GPUCV_CXCORE
	else if (CurCmd=="cxcore_all")
	{
		cxcoreg_runAll(&GlobSrc1, &GlobSrc2, &GlobMask);
		CmdOperator = false;
	}
	//CVGXCORE_OPER_ARRAY_INIT_GRP
	else if (CurCmd=="clone")	runClone(GlobSrc1);
	//CVGXCORE_OPER_ARRAY_ACCESS_ELEM_GRP
	//CVGXCORE_OPER_COPY_FILL_GRP
	else if (CurCmd=="copy")	runCopy(GlobSrc1,GlobMask);
	else if (CurCmd=="set")
	{
		CvScalar value ;
		GPUCV_NOTICE("\nValue 4 float needed?");
		for(int i=0;i<4;i++)
		{
			SGE::GetNextCommand(nextCmd, value.val[i]);
			printf ("   %lf\n",value.val[i]);
		}
		runSet(GlobSrc1, &value, GlobMask);
	}
	//CVGXCORE_OPER_TRANSFORM_PERMUT_GRP
	else if (CurCmd=="split")
	{
		int Channels;
		GPUCV_NOTICE("\nNbr of channels?");
		SGE::GetNextCommand(nextCmd, Channels);
		runSplit(GlobSrc1,Channels);
	}
	else if (CurCmd=="merge")		runMerge(MaskBackup,0);
	else if (CurCmd=="flip")
	{
		int type;
		GPUCV_NOTICE("\nFlip for Image?");
		GPUCV_NOTICE("\n\t 0- Flip horizontal");
		GPUCV_NOTICE("\n\t 1- Flip vertical");
		GPUCV_NOTICE("\n\t 2- Flip horizontal puis vertical");
		SGE::GetNextCommand(nextCmd,type);
		runFlip(GlobSrc1,type);
	}
	//CVGXCORE_OPER_ARITHM_LOGIC_COMP_GRP
	else if (CurCmd=="lut"){runLut(GlobSrc2);}
	else if (CurCmd=="convertscale")
	{
		float scale=0, shift=0;
		SGE::GetNextCommand(nextCmd, scale);
		SGE::GetNextCommand(nextCmd, shift);
		//SGE::GetNextCommand(nextCmd, Curmd2);
		runConvertScale(GlobSrc1,scale,shift);
	}
	//KEYTAGS: TUTO_TEST_OP_TAG__STP2_1__PARSE_FCT_CALL
	else if (CurCmd=="add")	{runAdd(GlobSrc1,GlobSrc2, GlobMask, global_scalar);}
	else if (CurCmd=="addweighted")
	{
		double s1,s2,gamma;
		GPUCV_NOTICE("\nWeightage for Image 1?");
		SGE::GetNextCommand(nextCmd,s1);
		GPUCV_NOTICE("\nWeightage for Image 2?");
		SGE::GetNextCommand(nextCmd,s2);
		GPUCV_NOTICE("\nScalar value to add?");
		SGE::GetNextCommand(nextCmd,gamma);
		runAddWeighted(GlobSrc1,s1,GlobSrc2,s2,gamma);
		printf ("coefficients %f %f %f",s1,s2,gamma);
	}
	else if (CurCmd=="sub")	runSub(GlobSrc1,GlobSrc2, GlobMask, global_scalar, NULL, 0);
	else if (CurCmd=="subr")runSub(GlobSrc1,GlobSrc2, GlobMask, global_scalar, NULL, 1);
	else if (CurCmd=="and")	runLogics(OPER_AND, GlobSrc1,GlobSrc2,GlobMask, global_scalar);
	else if (CurCmd=="or")	runLogics(OPER_OR, GlobSrc1,GlobSrc2,GlobMask, global_scalar);
	else if (CurCmd=="xor")	runLogics(OPER_XOR, GlobSrc1,GlobSrc2,GlobMask, global_scalar);
	else if (CurCmd=="not")	runLogics(OPER_NOT, MaskBackup,NULL,NULL);
	else if (CurCmd=="cmp")
	{
		GPUCV_NOTICE("\nOperator (0-5)?");
		int i= 0;
		SGE::GetNextCommand(nextCmd, i);
		runCmp(GlobSrc1,GlobSrc2,i);
	}
	else if (CurCmd=="cmps")
	{
		GPUCV_NOTICE("\nOperator (0-5)?");
		int i= 0;
		SGE::GetNextCommand(nextCmd, i);
		GPUCV_NOTICE("Double?");
		double val;
		SGE::GetNextCommand(nextCmd, val);
		runCmp(GlobSrc1,NULL,i, &val);
	}
	else if (CurCmd=="absdiff")	runAbsDiff(GlobSrc1,GlobSrc2,global_scalar);
	else if (CurCmd=="abs")		runAbs(GlobSrc1);
	//else if (CurCmd=="cadd"){runCustomAdd();}
	//else if (CurCmd=="csub"){runCustomSub();}
	else if (CurCmd=="div")
	{
		GPUCV_NOTICE("\nFactor?");
		SGE::GetNextCommand(nextCmd, factor);
		runDiv(GlobSrc1,GlobSrc2, factor);
	}
	else if (CurCmd=="mul")
	{
		GPUCV_NOTICE("\nFactor?");
		SGE::GetNextCommand(nextCmd, factor);
		runMul(GlobSrc1,GlobSrc2, factor);
	}
	else if (CurCmd=="max")			runMinMax(GlobSrc1,GlobSrc2, false, NULL);
	else if (CurCmd=="maxs")
	{
		double value;
		GPUCV_NOTICE("\nValue 1 float needed?");
		SGE::GetNextCommand(nextCmd, value);
		runMinMax(GlobSrc1,NULL, false, &value);
	}
	else if (CurCmd=="min")			runMinMax(GlobSrc1,GlobSrc2, true);
	else if (CurCmd=="mins")
	{
		double value ;
		GPUCV_NOTICE("\nValue 1 float needed?");
		SGE::GetNextCommand(nextCmd, value);
		runMinMax(GlobSrc1,NULL, true, &value);
	}
	//CVGXCORE_OPER_STATS_GRP
	else if (CurCmd=="sum")		runSum(GlobSrc1);
	else if (CurCmd=="avg")		runAvg(GlobSrc1,NULL);
#if _GPUCV_DEVELOP_BETA
	else if (CurCmd=="minmaxloc")	runMinMaxLoc(GlobSrc1);
#endif
	//CVGXCORE_OPER_LINEAR_ALGEBRA_GRP
	else if (CurCmd=="scaleadd")
	{
		CvScalar value ;
		GPUCV_NOTICE("\nValue 4 float needed?");
		for(int i=0;i<4;i++)
		{
			SGE::GetNextCommand(nextCmd, value.val[i]);
			printf ("   %lf\n",value.val[i]);
		}
		runScaleAdd(GlobSrc1,GlobSrc2 ,&value);
	}

#if _GPUCV_SUPPORT_CVMAT
	else if (CurCmd=="matmul")
	{
		GPUCV_NOTICE("\nSize?");
		int val= 0;
		SGE::GetNextCommand(nextCmd, val);
		GPUCV_NOTICE("\nFactor1?");
		SGE::GetNextCommand(nextCmd, factor);
		runGEMM(val,factor);
	}
	else if (CurCmd=="matmuladd")
	{
		GPUCV_NOTICE("\nSize?");
		int val= 0;
		SGE::GetNextCommand(nextCmd, val);
		GPUCV_NOTICE("\nFactor1?");
		SGE::GetNextCommand(nextCmd, factor);
		GPUCV_NOTICE("\nFactor2?");
		SGE::GetNextCommand(nextCmd, factor2);
		runGEMM(val,factor, factor2);
	}
	else if (CurCmd=="transpose") runTranspose(GlobSrc1);
#endif
	//CVGXCORE_OPER_MATH_FCT_GRP
	else if (CurCmd=="pow")
	{
		GPUCV_NOTICE("\nPower?");
		SGE::GetNextCommand(nextCmd, factor);
		runPow(GlobSrc1,factor);
	}
	//CVGXCORE_OPER_RANDOW_NUMBER_GRP
	//CVGXCORE_OPER_DISCRETE_TRANS_GRP
	//CVGXCORE_DRAWING_CURVES_SHAPES_GRP

	//CVGXCORE_DRAWING_TEXT_GRP
	//CVGXCORE_DRAWING_POINT_CONTOURS_GRP
	else if (CurCmd=="line")		runLine(GlobSrc1);
	else if (CurCmd=="rectangle")	runRectangle(GlobSrc1);
	else if (CurCmd=="circle")	    runCircle(GlobSrc1);
	else if (CurCmd=="dft")
	{
		int flag=0;
		GPUCV_NOTICE("Flags?");
		GPUCV_NOTICE("CV_DXT_FORWARD  - 0");
		GPUCV_NOTICE("CV_DXT_INVERSE  - 1");
		//GPUCV_NOTICE("CV_DXT_SCALE    - 2");
		//GPUCV_NOTICE("CV_DXT_ROWS     - 4");
		//GPUCV_NOTICE("CV_DXT_INV_SCALE (CV_DXT_SCALE|CV_DXT_INVERSE)
		//GPUCV_NOTICE("CV_DXT_INVERSE_SCALE CV_DXT_INV_SCALE
		SGE::GetNextCommand(nextCmd, flag);
		int zr=0;
		GPUCV_NOTICE("Non Zero rows?");
		SGE::GetNextCommand(nextCmd, zr);
		runDFT(GlobSrc2,flag,zr);
	}
	else if (CurCmd=="cxcore_test")	runCXCOREBench(GlobSrc1, GlobSrc2, GlobMask);
	else if (CurCmd=="glxcudacompat") runCompat_GLSLxCUDA(GlobSrc1, GlobSrc2);
#endif//TEST_GPUCV_CVCORE
	else
		return false;


	return true;
}
//==========================================
void cxcoreg_runAll(IplImage **src1, IplImage ** src2, IplImage ** mask)
{
	//testing zone
//	runAdd(*src1,*src2,GlobMask, global_scalar);
//	runPow(*src1,2);
//	return;
	//end of testing zone

	if (cxcore_test_enable==false)
	{
		return;
	}
	CurLibraryName="cxcore";
	CvScalar scalarValue;
	scalarValue.val[0] = scalarValue.val[1] = scalarValue.val[2] = scalarValue.val[3] = 2;  

#if TEST_GPUCV_CXCORE
	GPUCV_NOTICE("Benchmarking CXCORE libs");
	//KEYTAGS: TUTO_TEST_OP_TAG__STP2_2__BENCHMARK
	//CVGXCORE_OPER_ARRAY_INIT_GRP
		runClone(*src1);

	//CVGXCORE_OPER_ARRAY_ACCESS_ELEM_GRP
	//CVGXCORE_OPER_COPY_FILL_GRP
	EnableDisableSettings("mask", false);
		runCopy(*src1,*mask);
		runSet(*src1,&scalarValue, *mask);
		runSet(*src1,NULL, *mask);
	EnableDisableSettings("mask", true);
		runCopy(*src1,*mask);
		runSet(*src1,&scalarValue, *mask);
		runSet(*src1,NULL, *mask);
	EnableDisableSettings("mask", false);//restore initial value....

#if 1
	//CVGXCORE_OPER_TRANSFORM_PERMUT_GRP
		for(int i=1; i <=4; i++)
			runSplit(*src1,i);
	//failed on MacBook Pro
	//	for(int i=0; i <3; i++)
	//		runFlip(*src1,i);


		runMerge(MaskBackup,0);
	//CVGXCORE_OPER_ARITHM_LOGIC_COMP_GRP
		runLut(*src1);
#endif


#if 1//working, using masks and scalar... rev 520
	 
	runScaleAdd(*src1,*src2 ,&scalarValue);
	int Opt = 0;
	for(Opt = 0; Opt < 4;Opt++)
	{
		switch(Opt)
		{
		case 0:
			EnableDisableSettings("mask", false);
			EnableDisableSettings("scalar", false);
			break;
		case 1:
			EnableDisableSettings("mask", true);
			EnableDisableSettings("scalar", false);
			break;
		case 2:
			EnableDisableSettings("mask", true);
			use_scalar = true;
			if(global_scalar==NULL)
				global_scalar = new CvScalar;
			for(int i=0;i<4;i++)
				global_scalar->val[i]= 128;
			break;
		case 3:
			EnableDisableSettings("mask", false);
			use_scalar = true;
			if(global_scalar==NULL)
				global_scalar = new CvScalar;
			for(int i=0;i<4;i++)
				global_scalar->val[i]= 128;
			break;
		}
		//KEYTAGS: TUTO_TEST_OP__STP2__RUN_FCT_CALL
		runAdd(*src1,*src2,GlobMask, global_scalar);
		
#if 1
		runSub(*src1,*src2,GlobMask, global_scalar);
		if(global_scalar)
			runSub(*src1,*src2,GlobMask, global_scalar, NULL, 1);
		runMinMax(*src1,*src2, false);
		runMinMax(*src1,*src2, true);
		double Val = 45;
		runMinMax(*src1,NULL, false, &Val);
		runMinMax(*src1,NULL, true, &Val);

		runLogics(OPER_AND,*src1,*src2,GlobMask, global_scalar);
		runLogics(OPER_OR,*src1,*src2,GlobMask, global_scalar);
		runLogics(OPER_XOR,*src1,*src2,GlobMask, global_scalar);
		runLogics(OPER_NOT,MaskBackup,NULL,NULL, NULL);
		runAbsDiff(*src1,*src2,global_scalar);
		runAbs(*src1);
#endif
	}
	use_scalar = false;
	if(global_scalar!=NULL)
	delete global_scalar;
	global_scalar=NULL;
	
	runDiv(*src1,*src2, 10);
	runMul(*src1,*src2, 0.1);
	runAddWeighted(*src1, 0.5, *src2, 0.5, 10);
#endif

#if 1 //working, rev 520
	double val=100;
	for(int Opt = 0; Opt < 5;Opt++)
	{
		runCmp(*src1,*src2,Opt);
		runCmp(*src1,NULL,Opt, &val);
	}
#endif

	//runDFT(*src1,0,0);
	//	runIntegral(*src1);
	//runConvertScale(*src1,2,20);
	//CVGXCORE_OPER_STATS_GRP
	//runSum(*src1);
	//runAvg(*src1,NULL);
#if _GPUCV_DEVELOP_BETA
		runMinMaxLoc(*src1);
#endif
		//CVGXCORE_OPER_LINEAR_ALGEBRA_GRP

#if _GPUCV_SUPPORT_CVMAT
		//runGEMM((*src1)->width, 1);
		//runGEMM((*src1)->width, 1, 2);
		runTranspose(*src1);
#endif


	//CVGXCORE_OPER_MATH_FCT_GRP
		runPow(*src1,2);//..Failed sometime with CUDA.??
	//CVGXCORE_OPER_RANDOW_NUMBER_GRP
	//CVGXCORE_OPER_DISCRETE_TRANS_GRP
	//CVGXCORE_DRAWING_CURVES_SHAPES_GRP

		//CVGXCORE_DRAWING_TEXT_GRP
		//CVGXCORE_DRAWING_POINT_CONTOURS_GRP
		runLine(*src1);
		runRectangle(*src1);
		runCircle(*src1);

	//other
	//failed on MacBook Pro
	//	runCompat_GLSLxCUDA(GlobSrc1, GlobSrc2);
#endif //TEST_GPUCV_CXCORE
}
//==========================================
void runLut(IplImage * src1)
{
	GPUCV_FUNCNAME("LUT");
	if( (src1->depth !=IPL_DEPTH_8U))
	{
		GPUCV_WARNING(FctName << ": Only 8U input image supported.");
		return;
	}


	IplImage * src2 = NULL;
	__CreateImages__(cvGetSize(src1),IPL_DEPTH_8U,src1->nChannels,OperALL-OperGLSL);

	CvSize LutSize;
	LutSize.height = 1;
	int scale = 1;
	LutSize.width = 256;
	IplImage * lut = NULL;
	IplImage * test = cvgCreateImage(cvGetSize(src1),src1->depth,src1->nChannels);
	if(destCV)
		lut = cvgCreateImage(LutSize,destCV->depth,1);//destCV->nChannels);
	else if(destCUDA)
		lut = cvgCreateImage(LutSize,destCUDA->depth,1);//,destCUDA->nChannels);
	else if(destGLSL)
		lut = cvgCreateImage(LutSize,destGLSL->depth,1);//,destGLSL->nChannels);
	else
	{//we can not do any thing, they are not selected

	}

	if(!lut)
	{
		GPUCV_WARNING("Running " << FctName << " with no implementation capable or enabled.");
		return ;
	}
	cvgSetLabel(lut,"lut");


	//cvgSetLocation<DataDsc_CUDA_Buffer>(lut, true);
	//cvgSetLocation<DataDsc_IplImage>(lut, true);

	//__SetInputImage__(lut, "lutTable");

	if (lut->depth == IPL_DEPTH_8U)
		scale = 1 ;
	else
		scale = 256;

	char * Data = lut->imageData;
	for(int i =0; i < 256; i++)
	{
		for(int j =0; j <lut->nChannels ; j++)
		{
			*Data = (255*scale)-i;//-j;
			Data++;
		}
	}


	__CreateWindows__();
	std::string localParams="";
	_SW_benchLoop(cvgswLUT(src1,destSW,lut),localParams);
	_CV_benchLoop(cvLUT(src1,destCV,lut),localParams);
//	_GPU_benchLoop(cvgLUT(src1,destGLSL,lut),destGLSL, localParams);
	_CUDA_benchLoop(cvgCudaLUT(src1,destCUDA,lut),destCUDA,localParams);

	{/*
	 float * Databuff_CV = NULL;
	 float * Databuff_CUDA = NULL;
	 if(destCV)
	 {
	 Databuff_CV = (float*)destCV->imageData;
	 }
	 if(destCUDA)
	 {
	 Databuff_CUDA = (float*)destCUDA->imageData;
	 }
	 if(Databuff_CV && Databuff_CUDA)
	 {
	 //GPUCV_DEBUG("OpenCV destCV==============================");
	 int destCVPitch = destCV->width*destCV->nChannels;
	 for (int j=0; j< 32; j++)
	 {
	 std::cout << std::endl << "\nVal:" << j << "\t";
	 std::cout << Databuff_CV[j]
	 << "\t"
	 <<Databuff_CUDA[j];
	 }
	 }*/
	}

	//_GPU_benchLoop("Lut",cvgLUT(src1,destGLSL, lut),destGLSL , "");
	__ShowImages__();
	__ReleaseImages__();
	cvgReleaseImage(&lut);
	cvgReleaseImage(&test);

}
//==========================================
void runClone(IplImage * src1)
{
	GPUCV_FUNCNAME("Clone");
	__CreateImages__(cvGetSize(src1) ,src1->depth, src1->nChannels, 0);
	__CurrentOperMask = (OperOpenCV|OperGLSL|OperCuda) & GpuCVSelectionMask;

	__SetInputImage__(src1, "Clone");
	//IplImage * src2  = NULL;

	//! \todo Add _SW_benchloop for runClone. Manage the new image release.
	__CreateWindows__();
	std::string strLocalFctName = (CurLibraryName!="" && 0)?CurLibraryName + "::" + FctName:FctName;\
	//opencv version, need to release images
	if(__CurrentOperMask & OperOpenCV)
	{
		if(src1)
			cvgSynchronize(src1);

		for(int i=0;i<NB_ITER_BENCH;i++)
		{
			//bench start
			{
				_PROFILE_BLOCK_INITPARAM(GPUCV_IMPL_OPENCV_STR);
				_PROFILE_BLOCK_SETPARAM_SIZE(src1);
				_PROFILE_BLOCK_GPU(strLocalFctName, _PROFILE_PARAMS);
				destCV	= (IplImage*)cvCloneImage(src1);
			}
			if(i!=NB_ITER_BENCH-1)
				cvgReleaseImage(&destCV);
		}
	}
	
	//GPUCV
	if(__CurrentOperMask & OperGLSL)
	{
		if(src1)
		{
			cvgSetLabel(src1, "CloneImage_Src1");
			cvgSetLocation<DataDsc_GLTex>(src1, true);
		}
			//	time_cvg = (float)glutGet(GLUT_ELAPSED_TIME);
		for(int i=0;i<NB_ITER_BENCH;i++)
		{
			SynchronizeOper(OperGLSL, src1);
			//bench start
			{
				_PROFILE_BLOCK_INITPARAM(GPUCV_IMPL_GLSL_STR);
				_PROFILE_BLOCK_SETPARAM_SIZE(src1);
				_PROFILE_BLOCK_GPU(strLocalFctName, _PROFILE_PARAMS);
				destGLSL	= (IplImage*)cvgCloneImage(src1);
			}
			SynchronizeOper(OperGLSL, destGLSL);
			if(i!=NB_ITER_BENCH-1)
				cvgReleaseImage(&destGLSL);
		}
			//		 time_cvg = ((float)glutGet(GLUT_ELAPSED_TIME)-time_cvg)/NB_ITER_BENCH;
	}
#if !_GCV_CUDA_EXTERNAL
#ifdef _GPUCV_SUPPORT_CUDA
	if(__CurrentOperMask & OperCuda)
	{
		if(src1)\
		{\
			if (AvoidCPUReturn)cvgUnsetCpuReturn(src1);\
				cvgSetLabel(src1, "CloneImage_Src1");\
				cvgSetLocation<DataDsc_CUDA_Buffer>(src1, true);\
			}\
			//		time_cvg = (float)glutGet(GLUT_ELAPSED_TIME);
			for(int i=0;i<NB_ITER_BENCH;i++) \
			{\
			SynchronizeOper(OperCuda, src1);
			//bench start
			{
				_PROFILE_BLOCK_INITPARAM(GPUCV_IMPL_CUDA_STR);
				_PROFILE_BLOCK_SETPARAM_SIZE(src1);
				_PROFILE_BLOCK_GPU(strLocalFctName, _PROFILE_PARAMS);
				destCUDA	= (IplImage*)cvgCloneImage(src1);
				//no need to call CUDA one, the cvg* knows that image is already in CUDA
				SynchronizeOper(OperCuda, destCUDA);//
			}
			if(i!=NB_ITER_BENCH-1)cvgReleaseImage(&destCUDA);\
			}\
	}
#endif
#endif
#if _GPUCV_DEPRECATED
	if(benchmark)	showbenchResult("CloneImage");
#endif
	//_GPU_benchLoop("CloneImage",	destGLSL = (IplImage*)cvgCloneImage(src1),destGLSL,"");
	__ShowImages__();
	__ReleaseImages__();
}
//==========================================
#if 1
void runConvertScale(IplImage * src1, double scale, double shift)
{
	GPUCV_FUNCNAME("ConvertScale");
	std::string localParams="From_";
	localParams+= GetStrCVPixelType(src1->depth);
	
	IplImage * src2 = NULL;
	{
		__CreateImages__(cvGetSize(src1) ,IPL_DEPTH_8U, src1->nChannels, OperOpenCV|OperCuda/*|OperGLSL*/);
		__CreateWindows__();
		_SW_benchLoop(cvgswConvertScale(src1,destSW, scale, shift), localParams);
		_CV_benchLoop(cvConvertScale(src1,destCV, scale, shift), localParams);
		// _GPU_benchLoop("ConvertScale",cvgConvertScale(src1,destGLSL, scale, shift),destGLSL , localParams);
		_CUDA_benchLoop(cvgCudaConvertScale(src1,destCUDA, scale, shift),destCUDA , localParams);
		__ShowImages__();
		__ReleaseImages__();
	}
	{
		__CreateImages__(cvGetSize(src1) ,IPL_DEPTH_16U, src1->nChannels, OperOpenCV|OperCuda/*|OperGLSL*/);
		__CreateWindows__();
		_SW_benchLoop(cvgswConvertScale(src1,destSW, scale, shift), localParams);
		_CV_benchLoop(cvConvertScale(src1,destCV, scale, shift), localParams);
		// _GPU_benchLoop("ConvertScale",cvgConvertScale(src1,destGLSL, scale, shift),destGLSL , localParams);
		_CUDA_benchLoop(cvgCudaConvertScale(src1,destCUDA, scale, shift),destCUDA , localParams);
		__ShowImages__();
		__ReleaseImages__();
	}
	{
		__CreateImages__(cvGetSize(src1) ,IPL_DEPTH_32S, src1->nChannels, OperOpenCV|OperCuda/*|OperGLSL*/);
		__CreateWindows__();
		_SW_benchLoop(cvgswConvertScale(src1,destSW, scale, shift), localParams);
		_CV_benchLoop(cvConvertScale(src1,destCV, scale, shift), localParams);
		// _GPU_benchLoop("ConvertScale",cvgConvertScale(src1,destGLSL, scale, shift),destGLSL , localParams);
		_CUDA_benchLoop(cvgCudaConvertScale(src1,destCUDA, scale, shift),destCUDA , localParams);
		__ShowImages__();
		__ReleaseImages__();
	}
	{
		__CreateImages__(cvGetSize(src1) ,IPL_DEPTH_32F, src1->nChannels, OperOpenCV|OperCuda/*|OperGLSL*/);
		__CreateWindows__();
		_SW_benchLoop(cvgswConvertScale(src1,destSW, scale, shift), localParams);
		_CV_benchLoop(cvConvertScale(src1,destCV, scale, shift), localParams);
		// _GPU_benchLoop("ConvertScale",cvgConvertScale(src1,destGLSL, scale, shift),destGLSL , localParams);
		_CUDA_benchLoop(cvgCudaConvertScale(src1,destCUDA, scale, shift),destCUDA , localParams);
		__ShowImages__();
		__ReleaseImages__();
	}

}
#else
void runConvertScale(IplImage *src1,double s1,double s2,std::string Curmd)
{	return;
/*
GPUCV_FUNCNAME("runConvertScale");
int PixelFormat=0;
if(Curmd=="8u")
{
PixelFormat = IPL_DEPTH_8U;
}
else if(Curmd=="8s")
{
PixelFormat = IPL_DEPTH_8S;
}
else if(Curmd=="16s")
{
PixelFormat = IPL_DEPTH_16S;
}
else if(Curmd=="16u")
{
PixelFormat = IPL_DEPTH_16U;
}
else if(Curmd=="32s")
{
PixelFormat = IPL_DEPTH_32S;
}
else if(Curmd=="32f")
{
PixelFormat = IPL_DEPTH_32F;
}
else if(Curmd=="64f")
{
PixelFormat = IPL_DEPTH_64F;
}

std::string Params = "SrcFormat:"+GetStrCVPixelType(src1->depth);
Params+= "DstFormat:"+ GetStrCVPixelType(PixelFormat);

__CreateImages__(cvGetSize(src1),PixelFormat,src1->nChannels,OperALL-OperGLSL);

__SetInputImage__(src1, "ConvertScaleSrc1");

__CreateWindows__();
IplImage * src2 = NULL ;

_CV_benchLoop("runConvertScale", cvConvertScale(src1,destCV,s1,s2),Params.data());
_GPU_benchLoop("runConvertScale", cvgConvertScale(src1,destGLSL,s1,s2), destGLSL,Params.data());
#if _GPUCV_DEVELOP_BETA
_CUDA_benchLoop("runConvertScale", cvgCudaConvertScale(src1,destCUDA,s1,s2),destCUDA,Params.data());
#endif
__ShowImages__();
__ReleaseImages__();
*/
}

#endif
//==========================================
void runSet(IplImage * src1, CvScalar* _scalar, IplImage * mask)
{
	GPUCV_FUNCNAME("Set");
	IplImage * src2 = NULL ;
	__CreateImages__(cvGetSize(src1) ,src1->depth, src1->nChannels, OperALL-OperGLSL);
	__CreateWindows__();


	CvScalar LocalScalar;
	LocalScalar.val[0]=LocalScalar.val[1]=LocalScalar.val[2]=LocalScalar.val[3] = 50;
	std::string localParams="";
	if (mask != NULL)
	{   //with mask
		localParams = "mask";
	}

	if(!_scalar)
	{
		_SW_benchLoop(	cvgswSet(destSW,LocalScalar,mask),localParams);
		_CV_benchLoop(	cvSet(destCV,LocalScalar,mask),		localParams);
		//_GPU_benchLoop(	cvgSet(destGLSL,LocalScalar,mask),	destGLSL,localParams);
		_CUDA_benchLoop(cvgCudaSet(destCUDA,LocalScalar,mask),	destCUDA,localParams);
	}
	else
	{
		localParams +="_Scalar";
		_SW_benchLoop(	cvgswSet(destSW,*_scalar,mask),localParams);
		_CV_benchLoop	(cvSet (destCV,*_scalar,mask),	 localParams);
		//_GPU_benchLoop	(cvgSet(destGLSL,*_scalar,mask), destGLSL,localParams);
		_CUDA_benchLoop	(cvgCudaSet(destCUDA,*_scalar,mask), destCUDA,localParams);
	}
	__ShowImages__();
	__ReleaseImages__();
}
//=====================================================
void runCompat_GLSLxCUDA(IplImage * src1, IplImage * src2)
{
	GPUCV_FUNCNAME("GLSLxCUDA");
	__CreateImages__(cvGetSize(src1) ,src1->depth, src1->nChannels, OperALL);
	__CreateWindows__();


	//run OpenCV sequence.
	cvAdd(src1, src2, destCV);
	cvDiv(destCV, src2, destCV, 32);
#if !_GCV_CUDA_EXTERNAL
#ifdef  _GPUCV_SUPPORT_CUDA
	//run GLSL+CUDA sequence
	cvgAdd(src1, src2, destGLSL);
	cvgCudaDiv(destGLSL, src2, destCUDA, 32);
#endif
#endif

	float fError = ControlResultsImages(destCV, destCUDA, FctName, "");\
	if(fError < fEpsilonOperTest)\
	{   GPUCV_NOTICE("Testing operator CV >> CUDA '" << FctName << "()'" << green <<" passed!" << white);}\
	else\
	{	GPUCV_NOTICE("Testing operator CV >> CUDA '" << FctName << "()'" << red <<" failed "  << white << "with rate " << fError << "!!!!!!");}\
	

	__ShowImages__();
	__ReleaseImages__();
}
//=====================================================
void runCopy(IplImage * src1,IplImage * mask/* = NULL */)
{
	GPUCV_FUNCNAME("Copy");
	IplImage * src2=NULL;
	__CreateImages__(cvGetSize(src1) ,src1->depth, src1->nChannels, OperOpenCV|OperGLSL|OperCuda);

	std::string localParams="";
	if (mask != NULL)
	{   //with mask
		localParams = "mask";
	}
	__CreateWindows__();

	_SW_benchLoop(	cvgswCopy(src1, destSW,mask),localParams);
	_CV_benchLoop(cvCopy(src1,  destCV ,mask ),localParams);
	_GPU_benchLoop(cvgCopy(src1,destGLSL, mask ), destGLSL,localParams);
	_CUDA_benchLoop(cvgCudaCopy(src1,destCUDA, mask ), destCUDA,localParams);
	__ShowImages__();
	__ReleaseImages__();
}
//==========================================

#if _GPUCV_SUPPORT_CVMAT
#define DEBUG_GEMM 1	//_GPUCV_DEBUG_MODE * 1
void runGEMM(int size, double alpha/*=1.*/, double beta/*=0.*/)
{
	GPUCV_FUNCNAME("GEMM");
#if _GPUCV_DEVELOP
	/*if(size != 4  || size != 8 || size != 16)

	GPUCV_ERROR("runCustomMul()-> size must be 4, 8, or 16 for testing");
	}
	*/
	//#define MAT_SIZE 8
	//IplImage *src3= cvLoadImage("data/pictures/bbc.jpg",1);
	//IplImage *src4= cvLoadImage("data/pictures/bcb.jpg",1);

	float *arr1 = new float[size*size];
	for (int i =0; i < size*size; i++)
	{
		arr1[i]= i+1;
	}

#if 1
	//identiy matrix
	float *arr2 = new float[size*size];
	int offset = 0;
	for (int i =0; i < size*size; i++)
	{
		if (i == offset)
		{
			arr2[i]= 1;
			offset += size +1;
		}
		else
		{
			arr2[i]= 0;
		}
	}
#else
	//incrementale matrix
	float *arr2 = new float[size*size];
	int offset = 0;
	int a = 1;
	for (int i =0; i < MAT_SIZE*MAT_SIZE; i++)
	{
		if (i == offset)
		{
			arr2[i]= a++;
			offset += MAT_SIZE +1;
		}
		else
		{
			arr2[i]= 1;
		}
	}
#endif

	float *arr3 = new float[size*size];
	for (int i =0; i < size*size; i++)
	{
		arr3[i]= 1;
	}

	//for opencv cause GPUCV does not use CvMat yet.
	CvMat mat1 = cvMat(size, size, CV_32FC1, arr1);
	CvMat mat2 = cvMat(size, size, CV_32FC1, arr2);
	CvMat mat3;
	CvMat *src1 = &mat1;
	CvMat *src2 = &mat2;
	CvMat *src3 = NULL;
	if (beta)
	{
		mat3 = cvMat(size, size, CV_32FC1, arr3);
		src3 = &mat3;
	}



#if DEBUG_GEMM
	//DataContainer * cvgScr1 = GPUCV_GET_TEX(src1);

	if (ShowImage)
	{
		//cvgShowMatrix("Array 1", arr1, size, size);
		//cvgShowMatrix("Array 2", arr2, size, size);
		PrintMatrix ("Array 1", 32, 32/*size, size*/, arr1);
		PrintMatrix ("Array 2", 32, 32/*size, size*/, arr2);
	}
	//debug matrix transfer
	//transfer back to CPU
	//cvgScr1->SetLocation(DataContainer::LOC_CPU, true);
	//cvgScr1->Print();
	//PrintMatrix ("Array 1 after transfer", size, size, (float*)*cvgScr1->_GetPixelsData());
#endif


	__CreateMatrixes__(size, size,CV_32FC1, OperALL-OperGLSL);
	__SetInputMatrix__(src1, "GEMM-src1");
	__SetInputMatrix__(src2, "GEMM-src2");
	if(src3)
	{
		__SetInputMatrix__(src3, "GEMM-src3");
	}

	//	__CreateWindows__();
	std::string localParams="";
//	_SW_benchLoop(cvgswGEMM		(src1,src2,alpha, src3,beta, destSW,0),localParams);
	_CV_benchLoop(cvGEMM		(src1,src2,alpha, src3,beta, destCV,0), localParams);
	_GPU_benchLoop(cvgGEMM		(src1,src2,alpha, src3,beta, destGLSL,0),destGLSL , localParams);
	_CUDA_benchLoop(cvgCudaGEMM	(src1,src2,alpha, src3,beta, destCUDA,0),destCUDA , localParams);

#if DEBUG_GEMM
	if(__CurrentOperMask & OperOpenCV && ShowImage)
		//PrintMatrix ("OpenCV Result", size, size, (float *)(destCV->data.fl));
		cvgShowMatrix("OpenCV Result", destCV);

	if(__CurrentOperMask & OperGLSL && ShowImage)
		//PrintMatrix ("GLSL Result", size, size, (float *)(destGLSL->data.fl));
		cvgShowMatrix("GLSL Result", destGLSL);

	if(__CurrentOperMask & OperCuda && ShowImage)
		//PrintMatrix ("CUDA Result", size, size, (float *)(destCUDA->data.fl));
		cvgShowMatrix("CUDA Result", destCUDA);
#endif

	__ReleaseMatrixes__();
#endif
}
#endif
//run logics
void runLogics(GPUCV_ARITHM_LOGIC_OPER _opertype, IplImage * src1,IplImage * src2, IplImage * mask, CvScalar * value2)
{
	if((src1->depth == IPL_DEPTH_32F) || (src1->depth == IPL_DEPTH_64F))
		return;//no logic with floats and double
	std::string FctName;//
	switch(_opertype)
	{
		//logics
		case OPER_AND:	FctName = "And";break;
		case OPER_OR:	FctName = "Or";break;
		case OPER_XOR:	FctName = "Xor";break;
		case OPER_NOT:	FctName = "Not";break;
		default:
			GPUCV_ERROR(FctName << "(): unknown logic operator type");
			return;
			break;
	}

	//format strings
	std::string localParams="";
	if (mask)
		localParams="MASK";
	if (value2)
	{
		FctName+="S";
	}
	//===============
	__CreateImages__(cvGetSize(src1) ,src1->depth, src1->nChannels, OperALL-OperGLSL);
	__CreateWindows__();

	if(!value2 && (src2 || _opertype==OPER_NOT))
	{
		switch(_opertype)
		{
			//logics
		case OPER_AND:
			_SW_benchLoop	(cvgswAnd(src1,src2, destSW, mask),localParams);
			_CV_benchLoop	(cvAnd(src1,src2, destCV, mask), localParams);
			_CUDA_benchLoop	(cvgCudaAnd(src1,src2, destCUDA, mask),destCUDA , localParams);
			break;
		case OPER_OR:
			_SW_benchLoop	(cvgswOr(src1,src2, destSW, mask),localParams);
			_CV_benchLoop	(cvOr(src1,src2, destCV, mask), localParams);
			_CUDA_benchLoop	(cvgCudaOr(src1,src2, destCUDA, mask),destCUDA , localParams);
			break;
		case OPER_XOR:
			_SW_benchLoop	(cvgswXor(src1,src2, destSW, mask),localParams);
			_CV_benchLoop	(cvXor(src1,src2, destCV, mask), localParams);
			_CUDA_benchLoop	(cvgCudaXor(src1,src2, destCUDA, mask),destCUDA , localParams);
			break;
		case OPER_NOT:
			_SW_benchLoop	(cvgswNot(src1,destSW),localParams);
			_CV_benchLoop	(cvNot(src1,destCV), localParams);
			_CUDA_benchLoop	(cvgCudaNot(src1,destCUDA),destCUDA , localParams);
			break;
		}
	}
	else if(value2)
	{
		switch(_opertype)
		{
			//logics
		case OPER_AND:
			_SW_benchLoop	(cvgswAndS(src1,*value2, destSW, mask),localParams);
			_CV_benchLoop	(cvAndS(src1,*value2, destCV, mask), localParams);
			_CUDA_benchLoop	(cvgCudaAndS(src1,*value2, destCUDA, mask),destCUDA , localParams);
			break;
		case OPER_OR:
			_SW_benchLoop	(cvgswOrS(src1,*value2, destSW, mask),localParams);
			_CV_benchLoop	(cvOrS(src1,*value2, destCV, mask), localParams);
			_CUDA_benchLoop	(cvgCudaOrS(src1,*value2, destCUDA, mask),destCUDA , localParams);
			break;
		case OPER_XOR:
			_SW_benchLoop	(cvgswXorS(src1,*value2, destSW, mask),localParams);
			_CV_benchLoop	(cvXorS(src1,*value2, destCV, mask), localParams);
			_CUDA_benchLoop	(cvgCudaXorS(src1,*value2, destCUDA, mask),destCUDA , localParams);
			break;
		}
	}

	__ShowImages__();
	__ReleaseImages__();
}



//==========================================
#if 0
#include <omp.h>
void cvOMPAdd(IplImage * dst,IplImage * src1,IplImage * src2)
{
	IplImage *ImageTable_Src1[8];
	IplImage *ImageTable_Src2[8];
	IplImage *ImageTable_Dst[8];

	 char* buff_src1;
	 char* buff_src2;
	 char* buff_dst;
	 CvSize size = cvGetSize(src1);
	 size.height /=omp_get_max_threads();
#pragma omp parallel shared(ImageTable_Src1,ImageTable_Src2,ImageTable_Dst, src1, src2, dst)
//for(int i=0; i< 4; i++)
	 {
		unsigned char omp_nbr = omp_get_max_threads();
		unsigned char omp_id = omp_get_thread_num();
		//printf("TID: %d / %d", omp_id, omp_nbr);
		
		ImageTable_Src1[omp_id] = cvCreateImage(size, src1->depth, src1->nChannels); 
		ImageTable_Src2[omp_id] = cvCreateImage(size, src2->depth, src2->nChannels);
		ImageTable_Dst[omp_id] = cvCreateImage(size, dst->depth, dst->nChannels);
		

		int offset=dst->imageSize / omp_nbr * omp_id;

		buff_src1 = ( char*)&src1->imageData[offset];
		buff_src2 = ( char*)&src2->imageData[offset];
		buff_dst  = ( char*)&dst->imageData[offset];

		ImageTable_Src1[omp_id]->imageData = buff_src1;
		ImageTable_Src2[omp_id]->imageData = buff_src2;
		ImageTable_Dst[omp_id]->imageData = buff_dst;

		cvAdd(ImageTable_Src1[omp_id],ImageTable_Src2[omp_id], ImageTable_Dst[omp_id]);

		ImageTable_Src1[omp_id]->imageData= NULL;
		ImageTable_Src2[omp_id]->imageData= NULL;
		ImageTable_Dst[omp_id]->imageData = NULL;
		cvReleaseImage(&ImageTable_Src1[omp_id]);
		cvReleaseImage(&ImageTable_Src2[omp_id]);
		cvReleaseImage(&ImageTable_Dst[omp_id]);
	}

}
#endif//add omp test
//KEYTAGS: TUTO_TEST_OP_TAG__STP1_2__WRITE_TEST_FUNCT
void runAdd(IplImage * src1,IplImage * src2, IplImage * mask, CvScalar * value2)
{
	GPUCV_FUNCNAME("Add");

	__CreateImages__(cvGetSize(src1) ,src1->depth, src1->nChannels, OperALL);
	//format strings
	std::string localParams="";
	if (mask)
		localParams="MASK";
	if (value2)
		FctName+="S";



	/* if (mask==NULL && value)
	sprintf(buff, "%f, %f, %f, %f", value->val[0],value->val[1],value->val[2],value->val[3]);
	else if (mask && value)
	sprintf(buff, "MASK %f, %f, %f, %f", value->val[0],value->val[1],value->val[2],value->val[3]);
	else if (mask)
	strcpy(buff, "MASK");
	else
	strcpy(buff, "");
	*/
	__CreateWindows__();

	if(!value2 && src2)
	{//cvAdd
		_SW_benchLoop	(cvgswAdd	(src1,src2, destSW, mask), localParams);
		_CV_benchLoop	(cvAdd		(src1,src2, destCV, mask), localParams);
		_GPU_benchLoop	(cvgAdd		(src1,src2, destGLSL, mask),destGLSL , localParams);
		_CUDA_benchLoop	(cvgNppAdd	(src1,src2, destCUDA, mask),destCUDA , localParams);
		//localParams+="OpenMP";
		//_CV_benchLoop	(cvOMPAdd(destCV,src1,src2), localParams);
	}
	else if(value2)// && !src2)
	{//cvAddS
		_SW_benchLoop	(cvgswAddS	(src1,*value2, destSW, mask), localParams);
		_CV_benchLoop	(cvAddS		(src1,*value2, destCV, mask), localParams);
		_GPU_benchLoop	(cvgAddS	(src1,*value2, destGLSL, mask),destGLSL , localParams);
		_CUDA_benchLoop	(cvgCudaAddS(src1,*value2, destCUDA, mask),destCUDA , localParams);
	}

	__ShowImages__();
	__ReleaseImages__();
}
//==========================================
void runScaleAdd(IplImage * src1,IplImage * src2, CvScalar * _scale)
{
	GPUCV_FUNCNAME("ScaleAdd");
	IplImage * srcGray1 = NULL;
	IplImage * srcGray2 = NULL;
	__CreateImages__(cvGetSize(src1) ,src1->depth, 1, OperALL-OperGLSL-OperOpenCV);
	if (src1->nChannels !=1)
	{
		GPUCV_DEBUG("\nConverting Src image to gray\n");
		srcGray1 = cvgCreateImage(cvGetSize(src1),src1->depth,1);
		cvCvtColor(src1, srcGray1, CV_BGR2GRAY);
		if(src2)
		{
			srcGray2 = cvgCreateImage(cvGetSize(src2),src2->depth,1);
			cvCvtColor(src2, srcGray2, CV_BGR2GRAY);
		}
	}
	else
	{
		srcGray1 = src1;
		srcGray2 = src2;
	}
	//=====================================

	__CreateWindows__();
	__SetInputImage__(srcGray1, "runScaleAdd-gray1");
	__SetInputImage__(srcGray2, "runScaleAdd-gray2");

	__CreateWindows__();

	std::string localParams="";
	if(_scale)
	{
		_SW_benchLoop	(cvgswScaleAdd	(srcGray1,*_scale, srcGray2, destSW),localParams);
	//! \bug format not supported in OpenCV..? do not know why?	cvScaleAdd(srcGray1,*_scale, srcGray2, destCV);
		_CV_benchLoop	(cvScaleAdd(srcGray1,*_scale, srcGray2, destCV), localParams);
	//	_GPU_benchLoop	(cvgScaleAdd(srcGray1,*_scale, srcGray2, destGLSL),destGLSL , localParams);
		_CUDA_benchLoop	(cvgCudaScaleAdd(srcGray1,*_scale, srcGray2, destCUDA),destCUDA , localParams);
	}
	__ShowImages__();
	__ReleaseImages__();
	if(src1!=srcGray1)
		cvgReleaseImage(&srcGray1);
	if(src2!=srcGray2)
		cvgReleaseImage(&srcGray2);
}
//==========================================
void runSub(IplImage * src1,IplImage * src2, IplImage * mask, CvScalar * value2, CvScalar * _scale, bool reversed)
{
	GPUCV_FUNCNAME("Sub");

	__CreateImages__(cvGetSize(src1) ,src1->depth, src1->nChannels, OperALL);

	//format strings
	if(reversed)
		FctName+="R";

	std::string localParams="";
	if (mask)
		localParams = "MASK";
	//if (_scale)
	//	sprintf(buff, "%s Scale %f,%f,%f,%f", buff, _scale->val[0],_scale->val[1],_scale->val[2],_scale->val[3]);

	if(value2)
	{
		FctName+="S";
	}
	if(_scale)
		FctName="Scale"+FctName;
	//===============
	__CreateWindows__();

	if(!value2 && src2)
	{//cvSub
		_SW_benchLoop	(cvgswSub(src1,src2, destSW, mask),localParams);
		_CV_benchLoop	(cvSub(src1,src2, destCV, mask), localParams);
		_GPU_benchLoop	(cvgSub(src1,src2, destGLSL, mask),destGLSL , localParams);
		_CUDA_benchLoop	(cvgCudaSub(src1,src2, destCUDA, mask),destCUDA , localParams);
	}
	else if(value2 && !reversed)// && !src2 
	{//cvSubS
		_SW_benchLoop	(cvgswSubS(src1,*value2, destSW, mask),localParams);
		_CV_benchLoop	(cvSubS(src1,*value2, destCV, mask), localParams);
		_GPU_benchLoop	(cvgSubS(src1,*value2, destGLSL, mask),destGLSL , localParams);
//???		_CUDA_benchLoop	(cvgCudaSubS(src1,*value2, destCUDA, mask),destCUDA , localParams);
	}
	else if(value2 &&reversed)//&& !src2 
	{//cvSubRS
		_SW_benchLoop	(cvgswSubRS(src1,*value2, destSW, mask),localParams);
		_CV_benchLoop	(cvSubRS(src1,*value2, destCV, mask), localParams);
		_GPU_benchLoop	(cvgSubRS(src1,*value2, destGLSL, mask),destGLSL , localParams);
		_CUDA_benchLoop	(cvgCudaSubRS(src1,*value2, destCUDA, mask),destCUDA , localParams);
	}
	else
	{
		GPUCV_WARNING("runSub(): input parameter set incorrect");
	}

	__ShowImages__();
	__ReleaseImages__();
}
//==========================================
void runMinMax(IplImage * src1,IplImage * src2, bool Min, double *value)
{
	GPUCV_FUNCNAME("Max");
	if(Min==true)
		FctName="Min";

	IplImage * srcGray1 = NULL;
	IplImage * srcGray2 = NULL;
	__CreateImages__(cvGetSize(src1) ,src1->depth, 1, OperALL);

	if (src1->nChannels !=1)
	{
		GPUCV_DEBUG("\nConverting Src image to gray\n");
		srcGray1 = cvgCreateImage(cvGetSize(src1),src1->depth,1);
		cvCvtColor(src1, srcGray1, CV_BGR2GRAY);
		if(src2)
		{
			srcGray2 = cvgCreateImage(cvGetSize(src1),src1->depth,1);
			cvCvtColor(src2, srcGray2, CV_BGR2GRAY);
		}
	}
	else
	{
		GPUCV_DEBUG("\nCloning Src images\n");
		srcGray1 = cvCloneImage(src1);
		if(src2)
			srcGray2 = cvCloneImage(src2);
	}
	//=====================================

	if(value)
	{
		FctName+="S";
	}
	__CreateWindows__();
	__SetInputImage__(srcGray1, "Minmax-gray1");
	if(srcGray2!=NULL)
		__SetInputImage__(srcGray2, "Minmax-gray2");
	
	std::string localParams="";
	if(value)
	{
		if(Min == true)
		{
		_SW_benchLoop	(cvgswMinS(srcGray1,*value, destSW),localParams);
		_CV_benchLoop(cvMinS(srcGray1,*value, destCV), localParams);
		_GPU_benchLoop(cvgMinS(srcGray1,*value,destGLSL), destGLSL, localParams);
		_CUDA_benchLoop(cvgCudaMinS(srcGray1,*value,destCUDA), destCUDA, localParams);
		}
		else
		{
		_SW_benchLoop	(cvgswMaxS(srcGray1,*value, destSW),localParams);
		_CV_benchLoop(cvMaxS(srcGray1,*value, destCV), localParams);
		_GPU_benchLoop(cvgMaxS(srcGray1,*value,destGLSL), destGLSL, localParams);
		_CUDA_benchLoop(cvgCudaMaxS(srcGray1,*value,destCUDA), destCUDA, localParams);
		}
	}
	else
	{
		if(Min == true)
		{
			_SW_benchLoop	(cvgswMin(srcGray1,srcGray2, destSW),localParams);
			_CV_benchLoop(cvMin(srcGray1,srcGray2, destCV), localParams);
			_GPU_benchLoop(cvgMin(srcGray1,srcGray2,destGLSL), destGLSL, localParams);
			_CUDA_benchLoop(cvgCudaMin(srcGray1,srcGray2,destCUDA), destCUDA, localParams);
		}
		else
		{
			_SW_benchLoop	(cvgswMax(srcGray1,srcGray2, destSW),localParams);
			_CV_benchLoop(cvMax(srcGray1,srcGray2, destCV), localParams);
			_GPU_benchLoop(cvgMax(srcGray1,srcGray2,destGLSL), destGLSL, localParams);
			_CUDA_benchLoop(cvgCudaMax(srcGray1,srcGray2,destCUDA), destCUDA, localParams);
		}
	}

	__ShowImages__();
	cvgReleaseImage(&srcGray1);
	if(srcGray2)
		cvgReleaseImage(&srcGray2);
	__ReleaseImages__();
}
//==========================================
void runSplit(IplImage * src1, GLuint channel)
{
	if(src1->nChannels==1)
	{
		GPUCV_WARNING("runSplit()->single channel source image");
		return;
	}
	if(channel==2)
	{
		GPUCV_WARNING("runSplit()->can split to 1-3 or 4 channels");
		return;
	}
	else if (src1->nChannels < (int)channel)
	{
		GPUCV_WARNING("runSplit()->to few input channels");
		return;
	}
	GPUCV_FUNCNAME("Split");
	IplImage * src2 = NULL ;
	__CreateImages__(cvGetSize(src1) ,src1->depth, 1, OperALL);
	//__CreateWindows__();

	IplImage ** pDestCV  = NULL;
	IplImage ** pDestGLSL = NULL;
	IplImage ** pDestCUDA = NULL;
	std::string Buff = SGE::ToCharStr(channel);
	Buff+=" channel";

	IplImage * ModelImage = (destCV)? destCV:
		(destGLSL)?destGLSL:
		(destCUDA)?destCUDA:NULL;

	std::string ImgLabel;
	if(__CurrentOperMask & OperOpenCV)
	{
		pDestCV = new IplImage*[4];
		for(GLuint i=0; i < channel; i++)
		{
			pDestCV[i] = cvCloneImage(ModelImage);
			ImgLabel = "CV";
			ImgLabel += SGE::ToCharStr(i);
			__SetOutputImage__(pDestCV[i], ImgLabel);
		}
		for(GLuint i=channel; i < 4; i++)
			pDestCV[i] = NULL;
	}
	if(__CurrentOperMask & OperGLSL)
	{
		pDestGLSL = new IplImage*[4];
		for(GLuint i=0; i < channel; i++)
		{
			pDestGLSL[i] = cvCloneImage(ModelImage);
			ImgLabel = "GLSL";
			ImgLabel += SGE::ToCharStr(i);
			__SetOutputImage__(pDestGLSL[i], ImgLabel);
			cvgSetLocation<DataDsc_GLTex>(pDestGLSL[i], false);
		}
		for(GLuint i=channel; i < 4; i++)
			pDestGLSL[i] = NULL;
	}
#if !_GCV_CUDA_EXTERNAL
#ifdef _GPUCV_SUPPORT_CUDA
	if(__CurrentOperMask & OperCuda)
	{
		pDestCUDA = new IplImage*[4];
		for(GLuint i=0; i < channel; i++)
		{
			pDestCUDA[i] = cvCloneImage(ModelImage);
			ImgLabel = "CUDA";
			ImgLabel += SGE::ToCharStr(i);
			__SetOutputImage__(pDestCUDA[i], ImgLabel);
			cvgSetLocation<DataDsc_CUDA_Buffer>(pDestCUDA[i], false);
		}
		for(GLuint i=channel; i < 4; i++)
			pDestCUDA[i] = NULL;
	}
#endif
#endif
	bool ControlOperators_Backup = ControlOperators;
	ControlOperators = false;//we will perform manual test here cause there are several destination images...

	//!\todo Add _SW_benchloop for run Split. Manage several destination images
	_CV_benchLoop(cvSplit(src1,pDestCV[0],pDestCV[1],pDestCV[2],pDestCV[3]),				Buff);
	_GPU_benchLoop(cvgSplit(src1,pDestGLSL[0],pDestGLSL[1],pDestGLSL[2],pDestGLSL[3]),		pDestGLSL[0],Buff);
		if(ControlOperators_Backup && pDestCV!=NULL && pDestGLSL!=NULL)
		{//perform manual test
			float fError = 0;
			for(GLuint i=0; i < channel; i++)
			{
				fError += ControlResultsImages(pDestCV[i], pDestGLSL[i], FctName, Buff);
			}
			if(fError < fEpsilonOperTest)\
			{   GPUCV_NOTICE("Testing operator CV >> GLSL '" << FctName << "("<< Buff << ")'" << green <<" passed!" << white);}\
			else
			{	GPUCV_NOTICE("Testing operator CV >> GLSL '" << FctName << "("<< Buff << ")'" << red <<" failed "  << white << "with rate " << fError << "!!!!!!");}\
		}
	_CUDA_benchLoop(cvgCudaSplit(src1,pDestCUDA[0],pDestCUDA[1],pDestCUDA[2],pDestCUDA[3]), pDestCUDA[0],Buff);
		if(ControlOperators_Backup && pDestCV!=NULL && pDestCUDA!=NULL)
		{//perform manual test
			float fError = 0;
			for(GLuint i=0; i < channel; i++)
			{
				fError += ControlResultsImages(pDestCV[i], pDestCUDA[i], FctName, Buff);
			}
			if(fError < fEpsilonOperTest)\
			{   GPUCV_NOTICE("Testing operator CV >> CUDA '" << FctName << "("<< Buff << ")'" << green <<" passed!" << white);}\
			else
			{	GPUCV_NOTICE("Testing operator CV >> CUDA '" << FctName << "("<< Buff << ")'" << red <<" failed "  << white << "with rate " << fError << "!!!!!!");}\
		}


	ControlOperators = ControlOperators_Backup;//restore settings

	if(ShowImage)
	{
		if(__CurrentOperMask & OperOpenCV)
		{
			for(GLuint i=0; i < channel; i++)
			{
				ImgLabel = cvgGetLabel(pDestCV[i]);
				cvNamedWindow(ImgLabel.data(),1);
				cvgShowImage(ImgLabel.data(),pDestCV[i]);
			}
		}
		if(__CurrentOperMask & OperGLSL)
		{
			for(GLuint i=0; i < channel; i++)
			{
				ImgLabel = cvgGetLabel(pDestGLSL[i]);
				cvNamedWindow(ImgLabel.data(),1);
				cvgShowImage(ImgLabel.data(),pDestGLSL[i]);
			}
		}
		if(__CurrentOperMask & OperCuda)
		{
			for(GLuint i=0; i < channel; i++)
			{
				if(pDestCUDA[i])
				{
					ImgLabel = cvgGetLabel(pDestCUDA[i]);
					cvNamedWindow(ImgLabel.data(),1);
					cvgShowImage(ImgLabel.data(),pDestCUDA[i]);
				}
			}
		}
		cvWaitKey(0);
	}

	__ReleaseImages__();
	for(GLuint i=0; i < channel; i++)
	{
		if(__CurrentOperMask & OperOpenCV)
			if(pDestCV[i])
			{
				cvDestroyWindow(cvgGetLabel(pDestCV[i]));
				cvgReleaseImage(&pDestCV[i]);
			}
		if(__CurrentOperMask & OperGLSL)
			if(pDestGLSL[i])
			{
				cvDestroyWindow(cvgGetLabel(pDestGLSL[i]));
				cvgReleaseImage(&pDestGLSL[i]);
			}
		if(__CurrentOperMask & OperCuda)
			if(pDestCUDA[i])
			{
				cvDestroyWindow(cvgGetLabel(pDestCUDA[i]));
				cvgReleaseImage(&pDestCUDA[i]);
			}
	}
}
//==========================================
void runMerge(IplImage * src1, GLuint channel)
{
	if(src1->nChannels!=1)
	{
		GPUCV_WARNING("runMerge()->multi channel source image");
		return;
	}
	GPUCV_FUNCNAME("Merge");
	std::string localParams="";
	__CreateImages__(cvGetSize(src1) ,src1->depth, 3, OperALL);
	__CreateWindows__();

	IplImage * src2=cvCloneImage(src1);
	IplImage * src3=cvCloneImage(src1);
	__SetInputImage__(src2, "Src2");
	__SetInputImage__(src3, "Src3");

	_SW_benchLoop(cvgswMerge(src1,src2,src3,0, destSW),localParams);
	_CV_benchLoop(cvMerge(src1,src2,src3,0, destCV),localParams);
	_GPU_benchLoop(cvgMerge(src1,src2,src3,0, destGLSL), destGLSL,localParams);
	_CUDA_benchLoop(cvgCudaMerge(src1,src2,src3,0, destCUDA), destCUDA,localParams);

	__ShowImages__();
	cvgReleaseImage(&src2);
	cvgReleaseImage(&src3);
	__ReleaseImages__();
}
//==========================================
#if _GPUCV_DEVELOP_BETA
void runMinMaxLoc(IplImage * src1)
{
	GPUCV_FUNCNAME("MinMaxLoc");
	IplImage *srcGray1 = NULL;
	IplImage *src2=NULL;

	if (src1->nChannels !=1)
	{
		GPUCV_DEBUG("\nConverting Src image to gray");
		srcGray1 = cvgCreateImage(cvGetSize(src1),src1->depth,1);
		cvCvtColor(src1, srcGray1, CV_BGR2GRAY);
	}
	else
	{
		GPUCV_DEBUG("\nCloning Src images");
		srcGray1 = cvCloneImage(src1);
	}
	//=====================================
	CvPoint min_locCV, max_locCV;
	CvPoint min_locCVG, max_locCVG;
	double min_valCV=0, max_valCV=0;
	double min_valCVG=0, max_valCVG=0;

	_CV_benchLoop(cvMinMaxLoc( srcGray1, &min_valCV, &max_valCV, &min_locCV,&max_locCV,NULL ), "No mask");
	_GPU_benchLoop(cvgMinMaxLoc( srcGray1, &min_valCVG, &max_valCVG, &min_locCVG,&max_locCVG,NULL ), NULL,"No mask");

	if (!benchmark)
	{
		GPUCV_DEBUG("\ncvMinMaxLoc results ==============================");
		GPUCV_DEBUG("\n\t\tCV\tCVG");
		printf("\nMin val :\t%lf\t%lf",  min_valCV  , min_valCVG);
		printf("\nMin Pos X :\t%d\t%d",min_locCV.x, min_locCVG.x);
		printf("\nMin Pos Y :\t%d\t%d",min_locCV.y, min_locCVG.y);
		printf("\nMax val :\t%lf\t%lf",  max_valCV  , max_valCVG);
		printf("\nMax Pos X :\t%d\t%d",max_locCV.x, max_locCVG.x);
		printf("\nMax Pos Y :\t%d\t%d",max_locCV.y, max_locCVG.y);
		GPUCV_DEBUG("\n==================================================");
	}
	cvgReleaseImage(&srcGray1);
}
#endif

//==========================================
void runDiv(IplImage * src1,IplImage * src2, float factor)
{
	GPUCV_FUNCNAME("Div");
	//convert images
	std::string localParams="";
	__CreateImages__(cvGetSize(src1) ,src1->depth, src1->nChannels, OperALL);

	__CreateWindows__();
	
	_SW_benchLoop(cvgswDiv(src1,src2, destSW,factor ),localParams);
	_CV_benchLoop(cvDiv(src1,src2, destCV,factor ),localParams);
	_GPU_benchLoop(cvgDiv(src1,src2,destGLSL,factor), destGLSL,localParams);
	_CUDA_benchLoop(cvgCudaDiv(src1,src2,destCUDA,factor), destCUDA,localParams);
	__ShowImages__();
	__ReleaseImages__();
}
//==========================================
void runMul(IplImage * src1,IplImage * src2, float factor)
{
	GPUCV_FUNCNAME("Mul");
	//convert images
	std::string localParams="";
	__CreateImages__(cvGetSize(src1) ,src1->depth, src1->nChannels, OperALL);
	__CreateWindows__();

	_SW_benchLoop(cvgswMul(src1,src2, destSW,factor ),localParams);
	_CV_benchLoop(cvMul(src1,src2, destCV,factor ),localParams);
	_GPU_benchLoop(cvgMul(src1,src2, destGLSL,factor), destGLSL,localParams);
	_CUDA_benchLoop(cvgCudaMul(src1,src2,destCUDA,factor), destCUDA,localParams);

	__ShowImages__();
	__ReleaseImages__();
}
//==========================================

void runAvg(IplImage * src1, IplImage * mask=NULL)
{
	GPUCV_FUNCNAME("Avg");
	__CreateImages__(cvGetSize(src1) ,IPL_DEPTH_8U, 1, (OperALL-OperCuda) & GpuCVSelectionMask);

	IplImage * src2=NULL;
	CvScalar ResultCv;
	CvScalar ResultCvg;
	CvScalar ResultCuda;
	//=====================================
	//cvThreshold(srcGray1, srcGray1, 128,128, CV_THRESH_BINARY);
	CvScalar Val;
	Val.val[0] = Val.val[1] = Val.val[2] = Val.val[3] = 127;

	ResultCv.val[0] = ResultCv.val[1] = ResultCv.val[2] = ResultCv.val[3] = 0;
	ResultCvg.val[0] = ResultCvg.val[1] = ResultCvg.val[2] = ResultCvg.val[3] = 0;
	ResultCuda.val[0] = ResultCuda.val[1] = ResultCuda.val[2] = ResultCuda.val[3] = 0;

	cvgSynchronize(src1);
	IplImage * _src_clone = cvCloneImage(src1);
	cvgSetOptions(_src_clone, DataContainer::UBIQUITY, true);
	cvgSetLocation<DataDsc_GLTex>(_src_clone);

	std::string Params = (mask)?"Mask": "";

	//!\todo runAvg, manage result with the _SW_benchloop.
	_SW_benchLoop(ResultCv=cvgswAvg(_src_clone,mask), Params);
	_CV_benchLoop(ResultCv=cvAvg(_src_clone,mask), Params);
	_GPU_benchLoop(ResultCvg=cvgAvg(_src_clone, mask);, NULL, Params);
	//_CUDA_benchLoop(ResultCuda=cvgCudaAvg(_src_clone, mask);, NULL, Params);

	if (ShowImage==true)
	{
		IplImage * ControlImage = cvCloneImage(src1);
		cvgSetLocation<DataDsc_GLTex>(src1, true);
#if 0
		__CreateWindows__();
		for (int i =0; i < 9; i++)
		{
			cvgShowImage("cvgXXX()",src1, i);
			cvgShowImage("cvXXX()",ControlImage, i);
			cvWaitKey(0);
			std::cout << std::endl;
		}
#endif
		if (__CurrentOperMask&OperOpenCV)
			printf("\ncvAvg  result : %lf, %lf %lf %lf", ResultCv.val[0], ResultCv.val[1], ResultCv.val[2], ResultCv.val[3]);
		if (__CurrentOperMask&OperGLSL)
			printf("\ncvgAvg result : %lf, %lf %lf %lf", ResultCvg.val[0], ResultCvg.val[1], ResultCvg.val[2], ResultCvg.val[3]);
		if (__CurrentOperMask&OperCuda)
			printf("\ncvgAvg result : %lf, %lf %lf %lf", ResultCuda.val[0], ResultCuda.val[1], ResultCuda.val[2], ResultCuda.val[3]);
	}
	__ReleaseImages__();
	cvgReleaseImage(&_src_clone);
}
//=========================================
void runSum(IplImage * src1)
{
	GPUCV_FUNCNAME("Sum");
	__CreateImages__(cvGetSize(src1) ,IPL_DEPTH_8U, 1, OperALL-OperCuda);
	IplImage * src2=NULL;
	std::string localParams="";
	CvScalar ResultCv;
	CvScalar ResultCvg;
	CvScalar ResultCuda;
	ResultCv.val[0] = ResultCv.val[1] = ResultCv.val[2] = ResultCv.val[3] = 0;
	ResultCvg.val[0] = ResultCvg.val[1] = ResultCvg.val[2] = ResultCvg.val[3] = 0;
	ResultCuda.val[0] = ResultCuda.val[1] = ResultCuda.val[2] = ResultCuda.val[3] = 0;
	//=====================================

	//!\todo Manage result with the _SW_benchloop
	_SW_benchLoop(ResultCv=cvgswSum(MaskBackup), localParams);
	_CV_benchLoop(ResultCv=cvSum(MaskBackup), localParams);
	_GPU_benchLoop(ResultCvg=cvgSum(MaskBackup);, NULL, localParams);
//	_CUDA_benchLoop(ResultCuda=cvgCudaSum(MaskBackup);, NULL, localParams);

	if (ShowImage==true)
	{
		if(__CurrentOperMask&OperOpenCV)
			printf("\ncvSum  result : %lf, %lf %lf %lf", ResultCv.val[0], ResultCv.val[1], ResultCv.val[2], ResultCv.val[3]);
		if(__CurrentOperMask&OperGLSL)
			printf("\ncvgSum result : %lf, %lf %lf %lf", ResultCvg.val[0], ResultCvg.val[1], ResultCvg.val[2], ResultCvg.val[3]);
		if(__CurrentOperMask&OperCuda)
			printf("\ncvgCuda result : %lf, %lf %lf %lf", ResultCuda.val[0], ResultCuda.val[1], ResultCuda.val[2], ResultCuda.val[3]);
	}
	__ReleaseImages__();
}
//=========================================
void runPow(IplImage * src1, double factor)
{
	GPUCV_FUNCNAME("Pow");
	IplImage * src2=NULL;
	int DepthSize = IPL_DEPTH_8U;//32F;
	IplImage * newSrc32f = cvgCreateImage(cvGetSize(src1),DepthSize,src1->nChannels);
	__CreateImages__(cvGetSize(src1) ,DepthSize, src1->nChannels, OperALL);

	if (DepthSize == IPL_DEPTH_32F)
		cvConvertScale(src1, newSrc32f,1./256.);
	else
		cvConvertScale(src1, newSrc32f,1./8);

	std::string localParams="";
	__SetInputImage__(newSrc32f, "Pow-float src")
	__CreateWindows__();
		
	_SW_benchLoop(	cvgswPow	(newSrc32f,destSW,		factor), localParams);
	_CV_benchLoop(	cvPow		(newSrc32f,destCV,		factor), localParams);
	_GPU_benchLoop(	cvgPow		(newSrc32f,destGLSL,	factor),destGLSL , localParams);
	_CUDA_benchLoop(cvgCudaPow	(newSrc32f,destCUDA,	factor),destCUDA, localParams);

	//	cvShowImage("cvXXX()",newSrc32f);
	//	cvWaitKey(0);
	__ShowImages__();
	__ReleaseImages__();
	cvgReleaseImage(&newSrc32f);
}
//=========================================
void runAddWeighted(IplImage * src1, double alpha, IplImage * src2, double beta, double gamma)
{
	GPUCV_FUNCNAME("AddWeighted");
	__CreateImages__(cvGetSize(src1) ,src1->depth, src1->nChannels, OperALL-OperGLSL);
	__CreateWindows__();
	std::string localParams="";
	
	_SW_benchLoop(cvgswAddWeighted(src1, alpha, src2, beta, gamma, destSW), localParams);
	_CV_benchLoop(cvAddWeighted(src1, alpha, src2, beta, gamma, destCV), localParams);
	//_GPU_benchLoop(cvgAddWeighted(src1, alpha, src2, beta, gamma, destGLSL), destGLSL, localParams);
	_CUDA_benchLoop(cvgCudaAddWeighted(src1,alpha,src2,beta,gamma,destCUDA), destCUDA,localParams);

	__ShowImages__();
	__ReleaseImages__();
}
//=========================================
void runAbsDiff(IplImage * src1 , IplImage * src2, CvScalar * _scalar)
{
	GPUCV_FUNCNAME("AbsDiff");
	if(_scalar)
		FctName+="S";
	
	__CreateImages__(cvGetSize(src1) ,src1->depth, src1->nChannels, OperALL);
	__CreateWindows__();
	std::string localParams="";

	if(_scalar)
	{
		//sprintf(buff,"%.1Lf",(long double)value);
		_SW_benchLoop(cvgswAbsDiffS(src1, destSW,*_scalar),localParams);
		_CV_benchLoop(cvAbsDiffS(src1, destCV,*_scalar),localParams);
		_GPU_benchLoop(cvgAbsDiffS(src1, destGLSL,*_scalar), destGLSL,localParams);
		_CUDA_benchLoop(cvgCudaAbsDiffS(src1, destCUDA, *_scalar), destCUDA,localParams);
	}
	else
	{
		_SW_benchLoop(cvgswAbsDiff(src1, src2, destSW  ),localParams);
		_CV_benchLoop(cvAbsDiff(src1, src2, destCV  ),localParams);
		_GPU_benchLoop(cvgAbsDiff(src1, src2, destGLSL), destGLSL,localParams);
		_CUDA_benchLoop(cvgCudaAbsDiff(src1, src2, destCUDA), destCUDA,localParams);
	}
	__ShowImages__();
	__ReleaseImages__();
}
//====================================================================
void runAbs(IplImage * src1)
{
	IplImage *src2=NULL;
	GPUCV_FUNCNAME("AbsDiffS");

	__CreateImages__(cvGetSize(src1) ,src1->depth, src1->nChannels, OperALL);
	__CreateWindows__();
		std::string localParams="";
		_SW_benchLoop(cvgswAbsDiffS(src1, destSW, cvScalar(0)),localParams);
		_CV_benchLoop(cvAbsDiffS(src1, destCV, cvScalar(0)),localParams);
		_GPU_benchLoop(cvgAbsDiffS(src1, destGLSL, cvScalar(0)), destGLSL,localParams);
		_CUDA_benchLoop(cvgCudaAbsDiffS(src1, destCUDA, cvScalar(0)), destCUDA,localParams);
	__ShowImages__();
	__ReleaseImages__();
}
//=======================================================================
void runTranspose(IplImage * src1)
{
	GPUCV_FUNCNAME("Transpose");
	//int i=0;
	IplImage * src2=NULL;
	CvSize srcSize = cvGetSize(src1);
	CvSize dstSize;
	dstSize.height = srcSize.width;
	dstSize.width= srcSize.height;

	__CreateImages__(dstSize,src1->depth, src1->nChannels, OperALL-OperGLSL);
	__CreateWindows__();
	std::string localParams="";
	_SW_benchLoop(cvgswTranspose(src1, destSW),localParams);
	_CV_benchLoop(cvTranspose(src1, destCV),localParams);
	//_GPU_benchLoop("Subs", cvgSubS(src1, value, destGLSL, mask ), destGLSL,localParams);
	_CUDA_benchLoop(cvgCudaTranspose(src1, destCUDA), destCUDA,localParams);

	__ShowImages__();
	__ReleaseImages__();

}
//=======================================================================
void runLine(IplImage * src1)
{
	GPUCV_FUNCNAME("Line");
	IplImage * src2=NULL;
	__CreateImages__(cvGetSize(src1) , src1->depth, src1->nChannels, OperOpenCV|OperGLSL|OperSW);


	CvPoint pt1,pt2,pt3,pt4,pt5,pt6;
	CvScalar color;
	pt1.x = 30;  pt1.y = 30;
	pt2.x = 100; pt2.y = 100;

	color.val[0] = 255.0;
	color.val[1] = 4.0;
	color.val[2] = 4.0;
	color.val[3] = 0.0;

	pt3.x = 30;  pt3.y = 80;
	pt4.x = 100; pt4.y = 150;

	pt5.x = 30;  pt5.y = 130;
	pt6.x = 100; pt6.y = 200;
	
	std::string localParams="";

	if(destCV)
		cvSetZero(destCV);
	if(destGLSL)
		cvSetZero(destGLSL);
	if(destCUDA)
		cvSetZero(destCUDA);

	__CreateWindows__();
	
	_SW_benchLoop(cvgswLine(destSW,pt1,pt2,CV_RGB( 1.0*255, 255.0, 255.0 ),10,8,0), localParams);
	_CV_benchLoop(cvLine(destCV,pt1,pt2,CV_RGB( 1.0*255, 255.0, 255.0 ),10,8,0), localParams);
	_GPU_benchLoop(cvgLine(destGLSL,pt1,pt2,CV_RGB( 1.0*255, 255.0, 255.0 ),10,8,0),destGLSL , localParams);
	__ShowImages__();
	__ReleaseImages__();
}
//=======================================================================

void runRectangle(IplImage * src1)
{
	GPUCV_FUNCNAME("Rectangle");
	IplImage * src2=NULL;
	__CreateImages__(cvGetSize(src1) , src1->depth, src1->nChannels, OperOpenCV|OperGLSL|OperSW);


	CvPoint pt1,pt2,pt3,pt4,pt5,pt6;
	CvScalar color;
	pt1.x = 30;  pt1.y = 30;
	pt2.x = 100; pt2.y = 100;

	color.val[0] = 1.0;
	color.val[1] = 0.0;
	color.val[2] = 0.0;
	color.val[3] = 0.0;

	pt3.x = 30;  pt3.y = 80;
	pt4.x = 100; pt4.y = 150;

	pt5.x = 30;  pt5.y = 130;
	pt6.x = 100; pt6.y = 200;

	std::string localParams="";
	__CreateWindows__();
	_SW_benchLoop	(cvgswRectangle	(destSW,pt1,pt2,CV_RGB( 1.0*255, 255.0, 255.0 ),10,8,0), localParams);
	_CV_benchLoop	(cvRectangle	(destCV,pt1,pt2,CV_RGB( 1.0*255, 255.0, 255.0 ),10,8,0), localParams);
	_GPU_benchLoop	(cvgRectangle	(destGLSL,pt1,pt2,CV_RGB( 1.0*255, 255.0, 255.0 ),10,8,0),destGLSL , localParams);
	__ShowImages__();
	__ReleaseImages__();
}
//=======================================================================
void runCircle(IplImage * src1)
{
	GPUCV_FUNCNAME("Circle");
	IplImage * src2=NULL;
	__CreateImages__(cvGetSize(src1) , src1->depth, src1->nChannels, OperOpenCV|OperGLSL|OperSW);

	CvPoint pt1;
	CvScalar color;
	pt1.x = src1->width/2;
	pt1.y = src1->height/2;
	int radius = pt1.x /5;

	color.val[0] = 1.0;
	color.val[1] = 0.0;
	color.val[2] = 0.0;
	color.val[3] = 0.0;

	std::string localParams="";
	__CreateWindows__();
	_SW_benchLoop	(cvgswCircle(destSW,pt1,radius,CV_RGB( 1.0*255, 255.0, 255.0 ),10,8,0), localParams);
	_CV_benchLoop	(cvCircle	(destCV,pt1,radius,CV_RGB( 1.0*255, 255.0, 255.0 ),10,8,0), localParams);
	_GPU_benchLoop	(cvgCircle	(destGLSL,pt1,radius,CV_RGB( 1.0*255, 255.0, 255.0 ),10,8,0),destGLSL , localParams);
	__ShowImages__();
	__ReleaseImages__();
}
//=======================================================================
/** \brief Quick benchmark of some OpenCV founctions that do not have GPU equivalence.
 */
void runCXCOREBench(IplImage * src1, IplImage * src2, IplImage * mask)
{
	GPUCV_FUNCNAME("");
	__CreateImages__(cvGetSize(src1) , src1->depth, src1->nChannels, OperOpenCV);
	__CreateWindows__();
	std::string localParams="";
	//cvCmp
	if(src1->nChannels==1)
	{
		//cvInRange
		FctName="InRange";
		_CV_benchLoop(cvInRange(src1,src2,src2,destCV), localParams);
	}
	//cvCountNonZero
	//   _CV_benchLoop("cvCountNonZero",cvCountNonZero(src1), "");
	//cvNorm
	FctName="Norm";
	localParams="CV_RELATIVE_L1";
	_CV_benchLoop(cvNorm(src1, src2,CV_RELATIVE_L1), localParams);
	localParams="CV_RELATIVE_L2";
	_CV_benchLoop(cvNorm(src1, src2,CV_RELATIVE_L1), localParams);

	//LINEAR ALGEBRA
	FctName="Normalize";
	localParams="";
	//_CV_benchLoop("cvDotProduct",cvDotProduct(src1, src2),localParams);
	_CV_benchLoop(cvNormalize(src1, destCV ), localParams);
	//_CV_benchLoop("cvCrossProduct",cvCrossProduct(src1, src2, destCV),localParams);
	//_CV_benchLoop("cvMulTransposed",cvMulTransposed(src1, destCV, 0), localParams);

	__ShowImages__();
	__ReleaseImages__();
}
//==========================================
void runFlip(IplImage * src1, int Flip_mode)
{
	IplImage * src2=NULL;
	IplImage * mask=NULL;
	if (Flip_mode==2){
		Flip_mode=-1;
	}

	//Required function name definition, used for debugging
	GPUCV_FUNCNAME("Flip");

	//Create temporary images of the given format and size
	//Last value: OperALL, is a flag that specify that we want to test all implemantation of the given operator, see OperType to know all possible combination.
	__CreateImages__(cvGetSize(src1) ,src1->depth, src1->nChannels, OperALL);

	//format a parameter string that is used to store arguments for the benchmarking tools
	std::string localParams = SGE::ToCharStr(Flip_mode);
	switch(Flip_mode)
	{
		case -1:	localParams = "x_y-axis";break;
		case 0:	localParams = "x-axis";break;
		case 1:	localParams = "y-axis";break;
	}
	//===============

	//create any required windows to see all results images, if global variable (ShowImage==true)
	__CreateWindows__();
	
	//different macros are used for different type of implemantation
	_SW_benchLoop   (cvgswFlip(src1, destSW, Flip_mode), localParams);
	_CV_benchLoop   (cvFlip(src1, destCV, Flip_mode), localParams);
	_GPU_benchLoop  (cvgFlip(src1, destGLSL, Flip_mode), destGLSL, localParams);
	_CUDA_benchLoop (cvgCudaFlip(src1, destCUDA, Flip_mode), destCUDA, localParams);

	//Show image if required (ShowImage==true), wait for a key pressed and destroy windows.
	__ShowImages__();
	//Release temporary ouput images
	__ReleaseImages__();
}
//=======================================================================
void runCmp(IplImage * src1,IplImage * src2,int op, double * value)
{
	GPUCV_FUNCNAME("Cmp");
	IplImage *srcGray1 = NULL;
	IplImage *srcGray2 = NULL;
	__CreateImages__(cvGetSize(src1) , IPL_DEPTH_8U, 1,OperALL);

	if (src1->nChannels !=1)
	{
		srcGray1 = cvgCreateImage(cvGetSize(src1),src1->depth,1);
		cvCvtColor(src1, srcGray1, CV_BGR2GRAY);
	}
	else
	{
		srcGray1 = cvCloneImage(src1);
	}
	__SetInputImage__(srcGray1, "CmpSrc1");
	if (src2)
	{
		if (src2 && src2->nChannels !=1)
		{
			srcGray2 = cvgCreateImage(cvGetSize(src2),src2->depth,1);
			cvCvtColor(src2, srcGray2, CV_BGR2GRAY);
		}
		else
		{
			srcGray2 = cvCloneImage(src2);
		}
		__SetInputImage__(srcGray2, "CmpSrc2");
	}



	std::string localParams;
	switch (op)
	{
		case 0:localParams = "Equal to";break;
		case 1:localParams = "Greater than";break;
		case 2:localParams = "Greater than Equal to";break;
		case 3:localParams = "Less than";break;
		case 4:localParams = "Lesser than Equal to";break;
		case 5:localParams = "NOT Equal to";break;
		default:
			GPUCV_ERROR("Parameter out of range!");
			return;
	}

	if(value)
	{
		FctName="CmpS";
		__CreateWindows__();
		_SW_benchLoop	(cvgswCmpS	(srcGray1,*value,destSW,op),	localParams);
		_CV_benchLoop	(cvCmpS		(srcGray1,*value,destCV,op),	localParams);
		_GPU_benchLoop	(cvgCmpS	(srcGray1,*value, destGLSL, op ), destGLSL,localParams);
		_CUDA_benchLoop	(cvgCudaCmpS(srcGray1,*value,destCUDA,op),	destCUDA,localParams);
	}
	else
	{
		FctName="Cmp";
		__CreateWindows__();
		_SW_benchLoop	(cvgswCmp	(srcGray1,srcGray2,destSW,op),	localParams);
		_CV_benchLoop	(cvCmp		(srcGray1,srcGray2,destCV,op),	localParams);
		_GPU_benchLoop	(cvgCmp		(srcGray1,srcGray2, destGLSL, op ), destGLSL,localParams);
		_CUDA_benchLoop	(cvgCudaCmp	(srcGray1,srcGray2,destCUDA,op), destCUDA,localParams);
	}
	__ShowImages__();
	__ReleaseImages__();
	cvgReleaseImage(&srcGray1);
	cvgReleaseImage(&srcGray2);
}
//=======================================================================
void cvShiftDFT(CvArr * src_arr, CvArr * dst_arr )
{
	CvMat * tmp;
	CvMat q1stub, q2stub;
	CvMat q3stub, q4stub;
	CvMat d1stub, d2stub;
	CvMat d3stub, d4stub;
	CvMat * q1, * q2, * q3, * q4;
	CvMat * d1, * d2, * d3, * d4;

	CvSize size = cvGetSize(src_arr);
	CvSize dst_size = cvGetSize(dst_arr);
	int cx, cy;

	if(dst_size.width != size.width ||
		dst_size.height != size.height){
			cvError( CV_StsUnmatchedSizes, "cvShiftDFT", "Source and Destination arrays must have equal sizes", __FILE__, __LINE__ );
	}

	if(src_arr==dst_arr){
		tmp = cvCreateMat(size.height/2, size.width/2, cvGetElemType(src_arr));
	}

	cx = size.width/2;
	cy = size.height/2; // image center

	q1 = cvGetSubRect( src_arr, &q1stub, cvRect(0,0,cx, cy) );
	q2 = cvGetSubRect( src_arr, &q2stub, cvRect(cx,0,cx,cy) );
	q3 = cvGetSubRect( src_arr, &q3stub, cvRect(cx,cy,cx,cy) );
	q4 = cvGetSubRect( src_arr, &q4stub, cvRect(0,cy,cx,cy) );
	d1 = cvGetSubRect( src_arr, &d1stub, cvRect(0,0,cx,cy) );
	d2 = cvGetSubRect( src_arr, &d2stub, cvRect(cx,0,cx,cy) );
	d3 = cvGetSubRect( src_arr, &d3stub, cvRect(cx,cy,cx,cy) );
	d4 = cvGetSubRect( src_arr, &d4stub, cvRect(0,cy,cx,cy) );

	if(src_arr!=dst_arr){
		if( !CV_ARE_TYPES_EQ( q1, d1 )){
			cvError( CV_StsUnmatchedFormats, "cvShiftDFT", "Source and Destination arrays must have the same format", __FILE__, __LINE__ );
		}
		cvCopy(q3, d1, 0);
		cvCopy(q4, d2, 0);
		cvCopy(q1, d3, 0);
		cvCopy(q2, d4, 0);
	}
	else{
		cvCopy(q3, tmp, 0);
		cvCopy(q1, q3, 0);
		cvCopy(tmp, q1, 0);
		cvCopy(q4, tmp, 0);
		cvCopy(q2, q4, 0);
		cvCopy(tmp, q2, 0);
	}
}
//=======================================================================

/** \brief Test Function for DFT implemented using FFT
Nothing was mentioned in OpenCV documentation about R2C or C2C so the conversion will be decided upon type of input.
*/

void runDFT(IplImage *src1,int flags,int zr)
{
	//! \todo make test on image input=> convert to correct format or ouput mesage!!!!
	GPUCV_FUNCNAME("DFT");
	IplImage * realInput;
	IplImage * imaginaryInput;
	IplImage * complexInput;
	int dft_M, dft_N;
	CvMat* dft_A, tmp;
	IplImage * image_Re;
	IplImage * image_Im;
	IplImage * src2=NULL;//just to be compatible with the macros...
	IplImage * TmpSrc1 = NULL;

	if(src1->width >=_GPUCV_TEXTURE_MAX_SIZE_X || src1->height >=_GPUCV_TEXTURE_MAX_SIZE_Y)
	{
		GPUCV_WARNING("runDFT requires images size to be < to " << _GPUCV_TEXTURE_MAX_SIZE_X);
		return;
	}
	if(src1->nChannels!=1)
	{
		TmpSrc1 = cvCreateImage(cvGetSize(src1), src1->depth, 1);
		cvCvtColor(src1, TmpSrc1, CV_BGR2GRAY);
	}
	else
		TmpSrc1 = src1;

	__CreateImages__(cvGetSize(TmpSrc1) ,IPL_DEPTH_32F/*TmpSrc1->depth*/, /*src1->nChannels*/2,OperALL-OperGLSL);
	//	__CreateWindows__();

	realInput		= cvgCreateImage( cvGetSize(TmpSrc1), IPL_DEPTH_32F, 1);
	imaginaryInput	= cvgCreateImage( cvGetSize(TmpSrc1), IPL_DEPTH_32F, 1);
	complexInput	= cvgCreateImage( cvGetSize(TmpSrc1), IPL_DEPTH_32F, 2);

	__SetInputImage__(realInput, "DTFT_realInput");
	__SetInputImage__(imaginaryInput, "DTFT_imaginaryInput");
	__SetInputImage__(complexInput, "DTFT_complexInput");



	cvScale(TmpSrc1, realInput, 1.0, 0.0);
	cvZero(imaginaryInput);
	cvMerge(realInput, imaginaryInput, NULL, NULL, complexInput);

	dft_M = cvGetOptimalDFTSize( TmpSrc1->height - 1 );
	dft_N = cvGetOptimalDFTSize( TmpSrc1->width - 1 );

	dft_A = cvgCreateMat( dft_M, dft_N, CV_32FC2 );
	image_Re = cvgCreateImage( cvSize(dft_N, dft_M), IPL_DEPTH_32F, 1);
	image_Im = cvgCreateImage( cvSize(dft_N, dft_M), IPL_DEPTH_32F, 1);

	__SetInputImage__(image_Re, "DTFT_image_Re");
	__SetInputImage__(image_Im, "DTFT_image_Im");

	// copy A to dft_A and pad dft_A with zeros
	cvGetSubRect( dft_A, &tmp, cvRect(0,0, TmpSrc1->width, TmpSrc1->height));
	cvCopy( complexInput , &tmp, NULL );
	if( dft_A->cols > TmpSrc1->width )
	{
		cvGetSubRect( dft_A, &tmp, cvRect(src1->width,0, dft_A->cols - TmpSrc1->width, TmpSrc1->height));
		cvZero( &tmp );
	}

	__SetInputImage__(TmpSrc1, "DTFT_TmpSrc1");
	__SetInputMatrix__(dft_A, "DFTSrc");
	cvgShowImageProperties(TmpSrc1);
	cvgShowImageProperties(dft_A);

	std::string localParams="";
	bool backupControllOper = ControlOperators;
	ControlOperators = false;
	
	_SW_benchLoop	(cvgswDFT(/*src1*/dft_A, destSW, 		flags, complexInput->height), localParams);
	_CV_benchLoop	(cvDFT(/*src1*/dft_A, destCV, 		flags, complexInput->height), localParams);
	_CUDA_benchLoop	(cvgCudaDFT(/*src1*/dft_A, destCUDA,flags, complexInput->height), destCUDA,localParams);
	
	ControlOperators =  backupControllOper; 

	if(ShowImage)
	{
		GPUCV_DEBUG("runDFT(): Testing results");
		float * Databuff_CV = NULL;
		float * Databuff_CUDA = NULL;
		if(destCV)
		{
			cvgGetRawData(destCV, (uchar**)&Databuff_CV);
			if(Databuff_CV==NULL)
				GPUCV_ERROR("runDFT(): could not read back image data");
		
		}
		if(destCUDA)
		{
			cvgGetRawData(destCUDA, (uchar**)&Databuff_CUDA);
			if(Databuff_CUDA==NULL)
				GPUCV_ERROR("runDFT(): could not read back image data");
		}
		if(Databuff_CV && Databuff_CUDA)
		{
			//GPUCV_DEBUG("OpenCV destCV==============================");
			int destCVPitch = destCV->width*destCV->nChannels;
			int iEqualValue = 0;
			for (int j=0; j< 32; j++)
			{
				GPUCV_DEBUG("Val:" << j << "\t" << Databuff_CV[j] << "\t" << Databuff_CUDA[j]);
				if (Databuff_CV[j] == Databuff_CUDA[j])
					iEqualValue++;
			}
			//!\todo chek result differences more precisely
			if(iEqualValue != 32)
			{
				//GPUCV_ERROR("runDFT(): results are different");
			}
		}
		else
		{
			GPUCV_ERROR("runDFT(): could not read back image data");
		}
	}

	//__ShowImages__();

	//release all data
	GPUCV_ERROR("runDFT(): release images");
	__ReleaseImages__();
	if(TmpSrc1!=src1)
		cvgReleaseImage(&TmpSrc1);

	cvgReleaseMat(&dft_A);
	cvgReleaseImage(&realInput);
	cvgReleaseImage(&imaginaryInput);
	cvgReleaseImage(&complexInput);
	cvgReleaseImage(&image_Re);
	cvgReleaseImage(&image_Im);
}
//==========================================================
void runLocalSum(IplImage *src1,int h,int w)
{
	GPUCV_FUNCNAME("LocalSum");
	IplImage * src2=NULL;
	IplImage * Sum=NULL;
	CvSize Size = cvGetSize(src1);
	Size.width  +=1;
	Size.height +=1;
	if(Size.width > _GPUCV_TEXTURE_MAX_SIZE_X
		|| Size.height > _GPUCV_TEXTURE_MAX_SIZE_Y)
	{
		GPUCV_NOTICE("runLocalSum() requires image inferior to _GPUCV_TEXTURE_MAX_SIZE_X or _GPUCV_TEXTURE_MAX_SIZE_Y");
		return;
	}
	__CreateImages__(Size, IPL_DEPTH_32S, 1, OperALL-OperGLSL);
	int InternalType= IPL_DEPTH_32S;//IPL_DEPTH_64F;
	Sum=cvgCreateImage(Size, InternalType, src1->nChannels);

	__CreateWindows__();

	CvScalar Val;
	Val.val[0] = Val.val[1] = Val.val[2] = Val.val[3] = 1;
	cvSet(src1,Val);
	__SetInputImage__(src1, "src1");
	std::string localParams="";
	cvIntegral( GlobMask, Sum, NULL, NULL);
	
//	_SW_benchLoop(cvgswLocalSum(Sum,destSW,h,w),localParams);

#if 0//_GPUCV_DEVELOP_BETA
	_CV_benchLoop(cvLocalSum(Sum,destCV,h,w),localParams);
#endif
	_CUDA_benchLoop(cvgCudaLocalSum(Sum,destCUDA,h,w),destCUDA,localParams);

	/*cvgShowImage3("cvgcuda",destCUDA);
	cvgShowImage3("CPU",destCV);*/


	cvgReleaseImage(&Sum);
	__ShowImages__();
	__ReleaseImages__();
}
