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
#include "misc_test.h"
#include "commands.h"


using namespace GCV;
bool transfer_test_enable=true;
bool misc_processCommand(std::string & CurCmd, std::string & nextCmd)
{
	CurLibraryName="misc";
	if (CurCmd=="")
	{
		return false;
	}
	/*====================================================*/
	//testing functions: check that GpuCV is working properly on current computer
#if _GPUCV_DEPRECATED
	else if (CurCmd=="togpu")		runCpuToGpu(GlobSrc1);
	else if (CurCmd=="tocpu")		runGpuToCpu(GlobSrc1);
#endif
	else if (CurCmd=="transfertest")
	{
		bool DataTransfer=true;
		GPUCV_NOTICE("Transfer data(0-1)");
		SGE::GetNextCommand(nextCmd, DataTransfer);
		RunIplImageTransferTest(GlobSrc1, DataTransfer);
	}
	else if (CurCmd=="clonetest")
	{
		runCloneTest(GlobSrc1);
		CmdOperator = false;
	}
	else if (CurCmd=="isdiff")		runIsDiff(GlobSrc1,GlobSrc2);
	else if (CurCmd=="testgeo")		runTestGeometry(GlobSrc1,GlobSrc2);
	else if (CurCmd=="testformat")	testImageFormat(GlobSrc1, 3, 8, 1);
	/*====================================================*/
	else
		return false;
	return true;
}





#define TRANSFER_TEST_OPER cvAnd




template <typename TPLDATA_DSC>
bool TestClone (IplImage * _src)
{
	std::string FctName = "TestClone for '";
	FctName+= typeid(TPLDATA_DSC).name();
	FctName+= "'";

	//copy to temp images
	cvgSynchronize(_src);
	IplImage * ControlImg	 = cvCloneImage(_src);
	IplImage * TestImg		 = cvCloneImage(_src);
	IplImage * ResultImg	 = NULL;//cvCreateImage(cvGetSize(_src),_src->depth, _src->nChannels);
	CvgArr	 * TestImgDC	 = dynamic_cast <CvgArr *>(GPUCV_GET_TEX(TestImg));

	try
	{
		GPUCV_NOTICE(">>>>=======" << FctName << " start test =========");
		LogIndentIncrease();
		GPUCV_NOTICE("Set location to " << typeid(TPLDATA_DSC).name());
		TestImgDC->SetLocation<TPLDATA_DSC>(true);
		TestImgDC->SetLabel("Src Image");

		GPUCV_NOTICE("Clone to " << typeid(TPLDATA_DSC).name());
		//bench start
#if _GCV_CUDA_EXTERNAL
		cvgFlush(TestImg);
#else
		glFlush();
		glFinish();
#ifdef _GPUCV_SUPPORT_CUDA
		cvgCudaThreadSynchronize();
#endif
#endif
		//FctTrcGPU = AppliTracer()->AddRecord(FctName, GPUCV_IMPL_GLSL_STR, "",GetWidth(TestImg), GetHeight(TestImg));

		{
			_PROFILE_BLOCK_INITPARAM(GPUCV_IMPL_GLSL_STR);
			_PROFILE_BLOCK_SETPARAM_SIZE(TestImg);
			_PROFILE_BLOCK_GPU(FctName, _PROFILE_PARAMS);
			ResultImg = cvgCloneImage(TestImg);
#if _GCV_CUDA_EXTERNAL
		cvgFlush(TestImg);
#else
	#ifdef _GPUCV_SUPPORT_CUDA
		cvgCudaThreadSynchronize();
	#endif
#endif
		}
		if(ControlOperators)
		{
			GPUCV_NOTICE("Test results");
			cvgSynchronize(TestImg);
			cvgSynchronize(ResultImg);
			//TRANSFER_TEST_OPER(ResultImg, ControlImg, TestImg);
			cvSub(ControlImg, ResultImg, TestImg);
			CvScalar Avg = cvAvg(TestImg);
			float AvgSum = Avg.val[0] + Avg.val[1] + Avg.val[2] + Avg.val[3];
			if(AvgSum >0)
			{
				cvShowImage("Diff()", TestImg);
				GPUCV_NOTICE("=>failed!! Error rate is " << AvgSum);
				cvNamedWindow("Original", 1);
				cvNamedWindow("Destination", 1);
				cvShowImage("Original", ControlImg);
				cvShowImage("Destination", ResultImg);
				cvWaitKey(0);
				cvDestroyWindow("Original");
				cvDestroyWindow("Destination");
			}
			else
				GPUCV_NOTICE("=>Passed");
		}
		else
			GPUCV_NOTICE("=>test skipped");
		LogIndentDecrease();
		GPUCV_NOTICE("<<<<=======" << FctName << " stop test=========");
	}
	catch(SGE::CAssertException &e)\
	{\
	GPUCV_ERROR("=================== Exception catched Start =================")\
	GPUCV_ERROR("");\
	GPUCV_ERROR("catched in function : "<< GPUCV_GET_FCT_NAME());\
	GPUCV_ERROR("line : "<< __LINE__);\
	GPUCV_ERROR("file : "<< __FILE__);\
	GPUCV_ERROR("_____________________________________________________________");\
	GPUCV_ERROR("Exception description:");\
	GPUCV_ERROR("_____________________________________________________________");\
	GPUCV_ERROR(e.what());\
	if(GetGpuCVSettings()->GetLastExceptionObj())\
	{\
	GPUCV_ERROR("Object description:");\
	GPUCV_ERROR(*GetGpuCVSettings()->GetLastExceptionObj());\
	GPUCV_ERROR(GetGpuCVSettings()->GetLastExceptionObj()->LogException());\
	}\
	GPUCV_NOTICE("Source image seq:" << _src->channelSeq);
	GPUCV_NOTICE("Source image depth:" << _src->depth);
	GPUCV_NOTICE("Source image nChannels:" << _src->nChannels);
	GPUCV_NOTICE("Source image width:" << _src->width);
	GPUCV_NOTICE("Source image height:" << _src->height);
	GPUCV_ERROR("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");\
		GPUCV_NOTICE("Printing Texture manager data=============================");	\
		GetTextureManager()->PrintAllObjects();										\
		GPUCV_NOTICE("==================================================");			\
		if(_GPUCV_DEBUG_MODE)Beep(3000, 2);\
			GPUCV_ERROR("A GpuCV operator failed!");\
			GPUCV_ERROR("=================== Exception catched End =================");\
	}\



	cvgReleaseImage(&TestImg);
	cvgReleaseImage(&ControlImg);
	cvgReleaseImage(&ResultImg);
	//	if(AvgSum>0)
	//	return false;
	//	else
	return true;
}

bool runCloneTest(IplImage * _src1)
{
	float DiffResult = 0.;
	IplImage * TestImg	= cvCloneImage(_src1);
	IplImage * ResultImg= cvCloneImage(_src1);
	cvgSetOptions(_src1, DataContainer::UBIQUITY, true);
	CvgArr * cvgTestImg = dynamic_cast <CvgArr *>(GPUCV_GET_TEX(TestImg));
	cvgTestImg->SetOption(DataContainer::DEST_IMG, true);//force object to have only one location
	cvgTestImg->SetOption(DataContainer::UBIQUITY, true);//force object to have only one location

	//cvNamedWindow("Source",1);
	//cvNamedWindow("Clone",1);


	for (int i = 0; i < NB_ITER_BENCH; i++)
	{
		if(GpuCVSelectionMask & OperOpenCV)
		{
			TestClone<DataDsc_IplImage>	(_src1);
			//need special function...		TestClone<DataDsc_CvMat>	(_src1);
		}
		if(GpuCVSelectionMask & OperGLSL)
		{
			TestClone<DataDsc_GLTex>	(_src1);
			//		TestClone<DataDsc_GLBuff>	(_src1);
		}
#if defined _GPUCV_SUPPORT_CUDA && !_GCV_CUDA_EXTERNAL

		if(GpuCVSelectionMask & OperCuda)
		{
			TestClone<DataDsc_CUDA_Array>(_src1);
			TestClone<DataDsc_CUDA_Buffer>(_src1);
		}
#endif
	}


	//cvDestroyWindow("Source");
	//cvDestroyWindow("Clone");
	cvgReleaseImage(&TestImg);
	cvgReleaseImage(&ResultImg);
	return true;
}




template <typename DATA_DSC_SRC, typename DATA_DSC_DST>
bool TestTransfer (IplImage * _src, bool datatransfer, enum OperType _operType)
{
	bool bResult = false; 
	std::string strDstName, strSrcName;
	strSrcName = DATA_DSC_SRC().GetClassName();
	strDstName = DATA_DSC_DST().GetClassName();

	std::string strOperType;
	switch(_operType)
	{
		case OperOpenCV : strOperType = GPUCV_IMPL_OPENCV_STR;break;
		case OperGLSL : strOperType = GPUCV_IMPL_GLSL_STR;break;
		case OperCuda : strOperType = GPUCV_IMPL_CUDA_STR;break;
	}
	std::string FctName = "Transfer::From_";
	FctName+= strSrcName;
	FctName+= "_TO_";
	FctName+= strDstName;
	float AvgSum=0;

	std::string Format = GetStrCVTextureFormat(_src);
	Format+= "-";
	Format+= GetStrCVPixelType(_src);

	//copy to temp images
	IplImage * ControlImg	 = cvCloneImage(_src);
	IplImage * TestImg		 = cvCloneImage(_src);
	IplImage * ResultImg	 = cvCreateImage(cvGetSize(_src),_src->depth, _src->nChannels);
	CvgArr	 * cvgTestImg	 = dynamic_cast <CvgArr *>(GPUCV_GET_TEX(TestImg));

	cvgSetLabel(ControlImg, "ControlImg");
	cvgSetLabel(TestImg, "TestImg");
	cvgSetLabel(ResultImg, "ResultImg");
	cvgSetOptions(TestImg, DataContainer::UBIQUITY, false);//disable multiple instances
	//cvgSetLabel(ControlImg, "cvgTestImg");

	try{
		GPUCV_NOTICE(">>>>=======" << FctName << " start test =========");
		LogIndentIncrease();
		GPUCV_NOTICE("Prepare source " << typeid(DATA_DSC_SRC).name());
		cvgTestImg->SetLocation<DATA_DSC_SRC>(true);
		GPUCV_NOTICE("transfer to " << typeid(DATA_DSC_DST).name());



		//record to bench from / to
		//bench start
		{
			FctName = "Transfer::From_";
			FctName+= strSrcName;
			//FctName+= strDstName;
			_PROFILE_BLOCK_INITPARAM(strOperType);
			_PROFILE_BLOCK_SETPARAM_SIZE(_src);\
			_PROFILE_PARAMS->AddParam("option", (datatransfer)?"data":"alloc_only");
			_PROFILE_PARAMS->AddParam("to", strDstName);
			_PROFILE_BLOCK_GPU(FctName, _PROFILE_PARAMS);
			cvgTestImg->SetLocation<DATA_DSC_DST>(datatransfer);
			SynchronizeOper((OperType)(OperGLSL|OperCuda), TestImg);
		}
/*		cvgTestImg->RemoveDataDsc<DATA_DSC_DST>();
		SynchronizeOper(_operType);
		{
			FctName = "Transfer::To_";
			//FctName+= strSrcName;
			FctName+= strDstName;
			_PROFILE_BLOCK_INITPARAM(strOperType);
			_PROFILE_BLOCK_SETPARAM_SIZE(TestImg);\
			_PROFILE_PARAMS->AddParam("option", (datatransfer)?"data":"alloc_only");
			_PROFILE_PARAMS->AddParam("from", strSrcName);
			_PROFILE_BLOCK_GPU(FctName, _PROFILE_PARAMS);
			cvgTestImg->SetLocation<DATA_DSC_DST>(datatransfer);
			SynchronizeOper(_operType);
		}
		cvgTestImg->RemoveDataDsc<DATA_DSC_SRC>();
		SynchronizeOper(_operType);
*/

		//perform other way transfer
	
		//cvgTestImg->RemoveDataDsc<DATA_DSC_SRC>();
#if 0
		//bench start
		{
			FctName = "Transfer::To_";
			FctName+= strSrcName;
			//FctName+= strDstName;
			_PROFILE_BLOCK_INITPARAM(strOperType);
			_PROFILE_BLOCK_SETPARAM_SIZE(_src);
			_PROFILE_PARAMS->AddParam("option", (datatransfer)?"data":"alloc_only");
			_PROFILE_PARAMS->AddParam("From", strDstName);
			_PROFILE_BLOCK_GPU(FctName, _PROFILE_PARAMS);
			cvgTestImg->SetLocation<DATA_DSC_SRC>(datatransfer);
			SynchronizeOper(_operType);
		}
#endif

		GPUCV_NOTICE("Test results");
		float fError = ControlResultsImages(_src, TestImg, FctName, (datatransfer)?"opt:data":"opt:alloc_only");
		if(fError < fEpsilonOperTest)\
		{   
			GPUCV_NOTICE("Testing memory transfer "<< strSrcName << " -> " << strDstName << green <<" passed!" << white);
			bResult = true;
		}
		else
		{	
			GPUCV_NOTICE("Testing memory transfer "<< strSrcName << " -> " << strDstName << red <<" failed"  << white << " with rate " << fError << "!!!!!!");
		}
	

/*		cvgTestImg->RemoveDataDsc<DATA_DSC_SRC>();
		SynchronizeOper(_operType);
		{
			FctName = "Transfer::To_";
			FctName+= strSrcName;
			//FctName+= strDstName;
			_PROFILE_BLOCK_INITPARAM(strOperType);
			_PROFILE_BLOCK_SETPARAM_SIZE(TestImg);
			_PROFILE_PARAMS->AddParam("option", (datatransfer)?"data":"alloc_only");
			_PROFILE_PARAMS->AddParam("to", strDstName);
			_PROFILE_BLOCK_GPU(FctName, _PROFILE_PARAMS);
			cvgTestImg->SetLocation<DATA_DSC_SRC>(datatransfer);
			SynchronizeOper(_operType);
		}
		//===============
*/
#if 0
		if(ControlOperators && datatransfer)
		{
			TRANSFER_TEST_OPER(ControlImg, TestImg, ResultImg);

			cvSub(ControlImg, TestImg, ResultImg);
			CvScalar Avg = cvAvg(ResultImg);
			AvgSum = Avg.val[0] + Avg.val[1] + Avg.val[2] + Avg.val[3];
			if(1)//AvgSum >0)
			{
				cvNamedWindow("Diff()",1);
				cvShowImage("Diff()", ResultImg);
				GPUCV_ERROR("=>failed!! Error rate is " << AvgSum);
				cvNamedWindow("Original", 1);
				cvNamedWindow("Destination", 1);
				cvShowImage("Original", ControlImg);
				cvShowImage("Destination", TestImg);
				cvWaitKey(0);
				cvDestroyWindow("Original");
				cvDestroyWindow("Destination");
				cvDestroyWindow("Diff()");
			}
			else if (0)//for visual test
			{
				cvNamedWindow("Destination", 1);
				cvShowImage("Destination", TestImg);
				cvWaitKey(0);
				cvDestroyWindow("Destination");

			}
			else
				GPUCV_NOTICE("=>Passed");
		}
		else
			GPUCV_NOTICE("=>test skipped");
#endif
		LogIndentDecrease();
		GPUCV_NOTICE("<<<<=======" << FctName << " stop test=========");

	}
	catch(SGE::CAssertException &e)\
	{\
	GPUCV_ERROR("=================== Exception catched Start =================")\
	GPUCV_ERROR("");\
	GPUCV_ERROR("catched in function : "<< GPUCV_GET_FCT_NAME());\
	GPUCV_ERROR("line : "<< __LINE__);\
	GPUCV_ERROR("file : "<< __FILE__);\
	GPUCV_ERROR("_____________________________________________________________");\
	GPUCV_ERROR("Exception description:");\
	GPUCV_ERROR("_____________________________________________________________");\
	GPUCV_ERROR(e.what());\
	if(GetGpuCVSettings()->GetLastExceptionObj())\
	{\
	GPUCV_ERROR("Object description:");\
	GPUCV_ERROR(*GetGpuCVSettings()->GetLastExceptionObj());\
	GPUCV_ERROR(GetGpuCVSettings()->GetLastExceptionObj()->LogException());\
	}\
	GPUCV_NOTICE("Source image seq:" << _src->channelSeq);
	GPUCV_NOTICE("Source image depth:" << _src->depth);
	GPUCV_NOTICE("Source image nChannels:" << _src->nChannels);
	GPUCV_NOTICE("Source image width:" << _src->width);
	GPUCV_NOTICE("Source image height:" << _src->height);
	GPUCV_ERROR("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");\
		GPUCV_NOTICE("Printing Texture manager data=============================");	\
		if(0)GetTextureManager()->PrintAllObjects();										\
		GPUCV_NOTICE("==================================================");			\
		if(_GPUCV_DEBUG_MODE)Beep(3000, 2);\
			GPUCV_ERROR("A GpuCV operator failed!");\
			GPUCV_ERROR("=================== Exception catched End =================");\
	}\


	cvgReleaseImage(&TestImg);
	cvgReleaseImage(&ControlImg);
	cvgReleaseImage(&ResultImg);

	
	return bResult;
}

/** \brief Test transfer between different locations(DataDsc_*).
*	\bug Transfer between DataDsc_CUDA_Buffer and DataDsc_CUDA_Array are not working...
*/
bool RunIplImageTransferTest(IplImage * _src1, bool datatransfer)
{
	if(transfer_test_enable==false)
		return true;

	float DiffResult = 0.;
	IplImage * TestImg	= cvCloneImage(_src1);
	cvgSetOptions(_src1, DataContainer::UBIQUITY, true);
	CvgArr * cvgTestImg = dynamic_cast <CvgArr *>(GPUCV_GET_TEX(TestImg));
	cvgTestImg->SetOption(DataContainer::DEST_IMG, true);
	cvgTestImg->SetOption(DataContainer::UBIQUITY, true);//force object to have only one location

	cvgSetLabel(TestImg, "Src1Clone");


	for (int i = 0; i < NB_ITER_BENCH; i++)
	{
		//transfer from IplImage
		if(GpuCVSelectionMask & OperGLSL)
		{
			TestTransfer<DataDsc_IplImage, 	DataDsc_GLTex>(_src1, datatransfer, OperGLSL);
			TestTransfer<DataDsc_GLTex, 	DataDsc_IplImage>(_src1, datatransfer,OperGLSL);
			
			TestTransfer<DataDsc_IplImage, 	DataDsc_GLBuff>(_src1, datatransfer, OperGLSL);
			TestTransfer<DataDsc_GLBuff, 	DataDsc_IplImage>(_src1, datatransfer,OperGLSL);
			TestTransfer<DataDsc_GLTex, 	DataDsc_GLBuff>(_src1, datatransfer, OperGLSL);
			TestTransfer<DataDsc_GLBuff, 	DataDsc_GLBuff>(_src1, datatransfer, OperGLSL);
		}

#if defined _GPUCV_SUPPORT_CUDA && !_GCV_CUDA_EXTERNAL

		if(GpuCVSelectionMask & OperCuda)
		{
			//	TestTransfer<DataDsc_IplImage, DataDsc_CUDA_Array>(_src1, datatransfer, OperCuda);
			TestTransfer<DataDsc_IplImage, DataDsc_CUDA_Buffer>(_src1, datatransfer, OperCuda);
			TestTransfer<DataDsc_CUDA_Buffer, DataDsc_IplImage>(_src1, datatransfer, OperCuda);

			//transfer from DataDsc_GLTex
			if(GpuCVSelectionMask & OperGLSL)
			{
				//transfer from DataDsc_GLBuff
				//TestTransfer<DataDsc_GLBuff, DataDsc_CUDA_Array>(_src1, datatransfer, OperCuda | OperGLSL);
				TestTransfer<DataDsc_GLBuff, DataDsc_CUDA_Buffer>(_src1, datatransfer, OperCuda/* | OperGLSL*/);
				TestTransfer<DataDsc_CUDA_Buffer, DataDsc_GLBuff>(_src1, datatransfer, OperCuda/* | OperGLSL*/);

				//transfer from DataDsc_GLTex
				//TestTransfer<DataDsc_GLTex, DataDsc_CUDA_Array>(_src1, datatransfer, OperCuda | OperGLSL);
//				TestTransfer<DataDsc_GLTex, DataDsc_CUDA_Buffer>(_src1, datatransfer, (OperType)(OperCuda | OperGLSL));
//				TestTransfer<DataDsc_CUDA_Buffer, DataDsc_GLTex>(_src1, datatransfer, (OperType)(OperCuda | OperGLSL));
			}

			//internal transfer
			//TestTransfer<DataDsc_CUDA_Array, DataDsc_CUDA_Buffer>(_src1, datatransfer, OperCuda);
			//TestTransfer<DataDsc_CUDA_Buffer, DataDsc_CUDA_Array>(_src1, datatransfer, OperCuda);
		}
#endif
	}
	cvgReleaseImage(&TestImg);
	return true;
}
