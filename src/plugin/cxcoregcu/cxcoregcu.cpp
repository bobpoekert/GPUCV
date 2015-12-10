#include "StdAfx.h"
#include <cxcoregcu/config.h>
#include <cxcoregcu/cxcoregcu.h>
#include <highguig/highguig.h>
#include <GPUCV/misc.h>
#include <GPUCVCuda/cuda_misc.h>
#include <GPUCVHardware/moduleInfo.h>
#include <GPUCVHardware/moduleDefaultColor.h>

using namespace GCV;

_GPUCV_CXCOREGCU_EXPORT_C LibraryDescriptor *modGetLibraryDescriptor(void)
{
	static LibraryDescriptor * pLibraryDescriptor=NULL;
	if(pLibraryDescriptor ==NULL)//init
	{
		pLibraryDescriptor = new LibraryDescriptor();
		pLibraryDescriptor->SetVersionMajor("1");
		pLibraryDescriptor->SetVersionMinor("0");
		pLibraryDescriptor->SetSvnRev("570");
		pLibraryDescriptor->SetSvnDate("");
		pLibraryDescriptor->SetWebUrl(DLLINFO_DFT_URL);
		pLibraryDescriptor->SetAuthor(DLLINFO_DFT_AUTHOR);
		pLibraryDescriptor->SetDllName("cxcoregcu");
		pLibraryDescriptor->SetImplementationName("CUDA");
		pLibraryDescriptor->SetBaseImplementationID(GPUCV_IMPL_CUDA);
		pLibraryDescriptor->SetUseGpu(true);
		pLibraryDescriptor->SetStartColor(GPUCV_IMPL_CUDA_COLOR_START);
		pLibraryDescriptor->SetStopColor(GPUCV_IMPL_CUDA_COLOR_STOP);
	}
	return pLibraryDescriptor;
}
_GPUCV_CXCOREGCU_EXPORT_C int cvgCudaDLLInit(bool InitGLContext, bool isMultiThread)
{
	return cvgcuInit(InitGLContext, isMultiThread);
}
//___________________________________________________
//___________________________________________________
//___________________________________________________
//===================================================
//=>start $(PRJ_NAME)$_??_GRP
//===================================================

//===================================================
//=>stop  $(PRJ_NAME)$_??_GRP
//===================================================

