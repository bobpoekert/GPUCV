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
/** \brief Contains GpuCV-CUDA correspondance of cv->matrix manipulation.
\author Yannick Allusse
*/
#include <cxcoregcu/config.h>
#include <GPUCVCuda/gpucv_wrapper_c.h>

#if _GPUCV_COMPILE_CUDA

#include <cxcoregcu/oper_array/linear_algebra/gemm.kernel.h>
#include <cublas.h>
#define PROF_HISTO(FCT)//FCT

template <typename TPLType>
void  CudaGEMM(CvArr* src1,
			   CvArr* src2, 
			   double alpha,
			   CvArr* src3, 
			   double beta, 
			   CvArr* dst, 
			   int tABC)
{
#if 1//use CUBLAS
	unsigned int WA = gcuGetWidth(src1);
	unsigned int WB = gcuGetWidth(src2);
	unsigned int WD = gcuGetWidth(dst);
	unsigned int HA = gcuGetHeight(src1);
	unsigned int HB = gcuGetHeight(src2);
	unsigned int HD = gcuGetHeight(dst);

	char tA, tB, tC;
	/* from cxcore.h
	#define CV_GEMM_A_T 1
	#define CV_GEMM_B_T 2
	#define CV_GEMM_C_T 4
	*/
	tA = (/*CV_GEMM_A_T*/ 1 & tABC)?'t':'n';
	tB = (/*CV_GEMM_B_T*/ 2 & tABC)?'t':'n';
	tC = (/*CV_GEMM_C_T*/ 4 & tABC)?'t':'n';

	//GPUCV_Assert(tC=='n', "Third matrix transpose is not yet supported");


	//prepare source
	TPLType * d_src1 = (TPLType *)gcuPreProcess(src1, GCU_INPUT, CU_MEMORYTYPE_DEVICE);
	TPLType * d_src2 = (TPLType *)gcuPreProcess(src2, GCU_INPUT, CU_MEMORYTYPE_DEVICE);
	TPLType * d_src3 = (src3)?(TPLType *)gcuPreProcess(src3, GCU_INPUT, CU_MEMORYTYPE_DEVICE):NULL;
	//prepare ouput========
	TPLType * d_result = (TPLType *)gcuPreProcess(dst, GCU_OUTPUT, CU_MEMORYTYPE_DEVICE);
	//=====================

	cublasStatus stat=cublasGetError();
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf ("data download failed");
		//cublasFree (devPtrA);
		cublasShutdown();
		//return EXIT_FAILURE;
	}

	cublasSgemm(tA, tB, WA, WB, HA, alpha, d_src1,max(WA,HA),d_src2,max(WB,HB), beta, d_result, max(WA,WB));
	
	stat=cublasGetError();
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf ("data download failed");
		//cublasFree (devPtrA);
		cublasShutdown();
		//return EXIT_FAILURE;
	}

#else

	// Matrix dimensions
	// (chosen as multiples of the thread block size for simplicity)
	//#define WA (9*5 * BLOCK_SIZE) // Matrix A width
	//#define HA (15*5 * BLOCK_SIZE) // Matrix A height
	//#define WB (24*5 * BLOCK_SIZE) // Matrix B width
	//#define HB WA  // Matrix B height
	//#define WC WB  // Matrix C width 
	//#define HC HA  // Matrix C height
	unsigned int WA = gcuGetWidth(src1);
	unsigned int WB = gcuGetWidth(src2);
	unsigned int WD = gcuGetWidth(dst);
	unsigned int HA = gcuGetHeight(src1);
	unsigned int HB = gcuGetHeight(src2);
	unsigned int HD = gcuGetHeight(dst);

	//printf("Src1:%d %d\n", WA, HA);
	//printf("Src2:%d %d\n", WB, HB);
	//printf("Dst:%d %d\n", WD, HD);

	//unsigned int channels = src->nChannels;
	//const unsigned int DataN = width * height * channels;
	//const unsigned int DataSize = DataN * sizeof(unsigned char);
	int i;

	//Check inputs is done in the cv_cu.cpp file, to manage exceptions


	//=====================
	//prepare source
	TPLType * d_src1 = (TPLType *)gcuPreProcess(src1, GCU_INPUT, CU_MEMORYTYPE_DEVICE);
	TPLType * d_src2 = (TPLType *)gcuPreProcess(src2, GCU_INPUT, CU_MEMORYTYPE_DEVICE);
	TPLType * d_src3 = (src3)?(TPLType *)gcuPreProcess(src3, GCU_INPUT, CU_MEMORYTYPE_DEVICE):NULL;
	//prepare ouput========
	TPLType * d_result = (TPLType *)gcuPreProcess(dst, GCU_OUTPUT, CU_MEMORYTYPE_DEVICE);
	//=====================

	
	gcudaThreadSynchronize();

	// setup execution parameters
	dim3 threads(CUDA_GEMM_BLOCK_SIZE, CUDA_GEMM_BLOCK_SIZE);
	if(WD<32 && HD<32)
		threads.x = threads.y = 8;

	dim3 blocks = dim3(iDivUp(WD,threads.x), iDivUp(HD,threads.y), 1);

	//execute the kernel
	//if(threads.x == 8)
	//	matrixMul<8, 8> <<< grid, threads >>>(d_result, d_src1, d_src2, WA, WB, alpha,0.);
	//else if (threads.x ==16)
	//<16, 16> 
	matrixMul<<< blocks, threads >>>(d_result, d_src1, d_src2, WA, WB, alpha,0.);

	//check results

	//clean output
	gcuPostProcess(dst);
	//clean source
	gcuPostProcess(src1);
	gcuPostProcess(src2);
	if(src3)
		gcuPostProcess(src3);

#endif//CUBLAS
}
_GPUCV_CXCOREGCU_EXPORT_CU
void  gcuGEMM(CvArr* src1,
				   CvArr* src2, 
				   double alpha,
				   CvArr* src3, 
				   double beta, 
				   CvArr* dst, 
				   int tABC)
{
	//int DstType 
	CudaGEMM<float>(src1, src2, alpha, src3, beta, dst, tABC);
}

#endif//_GPUCV_COMPILE_CUDA
