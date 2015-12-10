//==== Deriche ===========

#include <GPUCVCuda/base_kernels/cuda_macro.kernel.cu>
#define DERICHE_SIMPLE_PROC 0  // one kernel works on one image
#define DERICHE_OPTIMIZED 1 

//=============================================
// lissage 
#if !DERICHE_OPTIMIZED
template <typename T>
__device__ void DericheLissage(
							   T *src, T* dst, T* Ym, T* Yp,    
							   int width, int height, float k, float a0, float a1, float a2, float a3, float b1, float b2,
							   unsigned int xIndex
							   ) {

								   int YcurPos=0;
								   float TmpValFor1=0;
								   float TmpValFor2=0;
								   float TmpAFor1=0;
								   float TmpAFor2=0;
								   float TmpValBack1=0;
								   float TmpValBack2=0;
								   float CurVal;
								   float TmpABack1=0;
								   float TmpABack2=0;
								   float TmpABack3=0;

#if DERICHE_OPTIMIZED//test
								   //__shared__ 
								   T Sh_block[512]; //maximum block size here is 256

								   for(int y = 0; y < height; y += 1) {
									   //move forward
									   YcurPos = y*width;

									   TmpAFor2=TmpAFor1;
									   TmpAFor1=src[YcurPos+xIndex];

									   CurVal = a0*TmpAFor1  
										   + a1*TmpAFor2  
										   - b1*TmpValFor1 
										   - b2*TmpValFor2;

									   Sh_block[height] = CurVal;
									   TmpValFor2=TmpValFor1;
									   TmpValFor1=CurVal;


									   //move backward
#if 0
									   YcurPos= (height-y)*width;

									   TmpABack3=TmpABack2;
									   TmpABack2=TmpABack1;
									   TmpABack1=src[YcurPos+width+xIndex];

									   CurVal = a2*TmpABack1 
										   + a3*TmpABack3
										   - b1*TmpValBack1 
										   - b2*TmpValBack2;
									   Sh_block[height-y]+= CurVal;
									   TmpValBack2=TmpValBack1;
									   TmpValBack1=CurVal;
#endif
								   }
								   for(int y = 0; y < height; y += 1) {
									   YcurPos = y*width + xIndex;
									   dst[YcurPos] = 
										   Sh_block[height];
									   //src[YcurPos];
								   }

#else
								   for(int y = 0; y < height; y += 1) {
									   //move forward
									   YcurPos = y*width;

									   TmpAFor2=TmpAFor1;
									   TmpAFor1=src[YcurPos+xIndex];

									   CurVal = a0*TmpAFor1  
										   + a1*TmpAFor2  
										   - b1*TmpValFor1 
										   - b2*TmpValFor2;

									   Yp[YcurPos + xIndex] = CurVal;
									   TmpValFor2=TmpValFor1;
									   TmpValFor1=CurVal;


									   //move backward
#if 1
									   YcurPos= (height-y)*width;

									   TmpABack3=TmpABack2;
									   TmpABack2=TmpABack1;
									   TmpABack1=src[YcurPos+width+xIndex];

									   CurVal = a2*TmpABack1 
										   + a3*TmpABack3
										   - b1*TmpValBack1 
										   - b2*TmpValBack2;
									   Ym[YcurPos + xIndex] = CurVal;
									   TmpValBack2=TmpValBack1;
									   TmpValBack1=CurVal;
#endif

								   }

								   for(int y = 0; y < height; y += 1) {
									   YcurPos = y*width + xIndex;
									   dst[YcurPos] = Yp[YcurPos]+ Ym[YcurPos];
								   }
#endif
}
#else
//=============================================
template <typename T>
__device__ void DericheLissage2(
								T *src, T* dst, 
								int width, int height, 
								float k, float a0, float a1, float a2, float a3, float b1, float b2, float scaleFactor,
								unsigned int xIndex,
								unsigned int start, unsigned int end, bool forward
								) {
									int mgx, mgy;
									mgx = mgy = 2;
									int Sx = width - 2*mgx;
									int Sy = height - 2*mgy;

									float val_b1=0;
									float val_b2=0;
									float val_a0=0;
									float val_a1=0;
									float val_a2=0;
									float val_a3=0;
									float CurVal;
									int YcurPos=0;

									unsigned int IndexIn = 0;
									unsigned int IndexOut = 0;

									if(!forward)
									{//move backward, using a2, a3
#if 1									
										val_b1= 0;//dst[(y+1)*width+xIndex];
										val_b2= 0;//dst[(y+2)*width+xIndex];
										val_a2= 0;//src[(y+1)*width+xIndex];//*scaleFactor;
										val_a3= 0;//src[(y+2)*width+xIndex];//*scaleFactor;
										int y=0;
										for(y = (Sy+mgy-1); y >=  (Sy+mgy)/2; y -=1) 
										{		
											//val_a1=val_a0;
											YcurPos		= y*width;
											IndexIn		= YcurPos+xIndex;	//+width
											IndexOut	= YcurPos+xIndex;

											//val_a1=src[IndexIn]*scaleFactor;		

											val_a2=src[(y+1)*width+xIndex];//*scaleFactor;
											//val_a3=src[(y+2)*width+xIndex];//*scaleFactor;
											//val_b1= dst[(y+1)*width+xIndex];
											//val_b2= dst[(y+2)*width+xIndex];

											CurVal = a2*val_a2  
												+ a3*val_a3
												- b1*val_b1 
												- b2*val_b2;

											dst[IndexOut] = CurVal;
											val_b2 = val_b1;
											val_b1= CurVal;

											val_a3 = val_a2;


											//val_b2= dst[(y+2)*width+xIndex];

											//val_b2	=	val_b1;
											//TmpValFor1=CurVal;
										}
#endif
										__syncthreads();
#if 1
										for(; y >=mgy  ; y -=1) 
										{		
											//a1=TmpAFor1;
											YcurPos = (y)*width;
											IndexIn = YcurPos+xIndex;//+width
											IndexOut = YcurPos+xIndex;

											val_a2=src[(y+1)*width+xIndex];//*scaleFactor;
											//val_a3=src[(y+2)*width+xIndex];//*scaleFactor;
											//	val_b1= dst[(y+1)*width+xIndex];
											//	val_b2= dst[(y+2)*width+xIndex];

											CurVal = a2*val_a2  
												+ a3*val_a3
												- b1*val_b1 
												- b2*val_b2;

											dst[IndexOut] += CurVal;
											val_b2 = val_b1;
											val_b1= CurVal;
											val_a3 = val_a2;
											//	TmpValFor2=TmpValFor1;
											//	TmpValFor1=CurVal;
										}
#endif
									}
									else 
									{//move forward, using a0, a1
										//							__syncthreads();
#if 1
										//YcurPos = 0;
										//IndexIn = YcurPos+xIndex;
										//a0=src[xIndex+]*scaleFactor;
										//a1=src[IndexIn]*scaleFactor;
										//a1=src[width*(mgy-1)+xIndex]*scaleFactor;
										//dst[xIndex]=0;
										//dst[xIndex+width]=0;
										val_b1=0;
										val_b2=0;
										val_a0=0;//src[width*y+xIndex];//*scaleFactor;		
										val_a1=0;//src[width*(y-1)+xIndex];//*scaleFactor;
										int y=0;
										for(y = mgy; y < (Sy+mgy)/2; y +=1) 
										{
#if 0
											YcurPos = y*width;
											IndexIn = YcurPos+xIndex;
											IndexOut = IndexIn;

											val_a0=src[IndexIn]*scaleFactor;		
											CurVal = a0*val_a0  
												+ a1*val_a1  
												- b1*TmpValFor1 
												- b2*TmpValFor2;

											dst[IndexOut] = CurVal;

											TmpValFor2=TmpValFor1;
											TmpValFor1=CurVal;

											val_a1=val_a0;
#else
											YcurPos = y*width;
											IndexIn = YcurPos+xIndex;
											IndexOut = IndexIn;

											val_a0=src[width*y+xIndex];//*scaleFactor;		
											//val_a1=src[width*(y-1)+xIndex];//*scaleFactor;
											//val_b1=dst[width*(y-1)+xIndex];
											//val_b2=dst[width*(y-2)+xIndex];

											CurVal = a0*val_a0  
												+ a1*val_a1  
												- b1*val_b1 
												- b2*val_b2;

											dst[width*y+xIndex] = CurVal;
											val_b2 = val_b1;
											val_b1= CurVal;

											val_a1 = val_a0;
											//TmpValFor2=TmpValFor1;
											//TmpValFor1=CurVal;

											//a1=a0;

#endif
										}
										__syncthreads();
										for(; y <Sy+mgy; y +=1) 
										{
											YcurPos = y*width;
											IndexIn = YcurPos+xIndex;
											IndexOut = IndexIn;

											val_a0=src[width*y+xIndex];//*scaleFactor;		
											//val_a1=src[width*(y-1)+xIndex];//*scaleFactor;
											//val_b1=dst[width*(y-1)+xIndex];
											//val_b2=dst[width*(y-2)+xIndex];

											CurVal = a0*val_a0  
												+ a1*val_a1  
												- b1*val_b1 
												- b2*val_b2;

											dst[width*y+xIndex] += CurVal;
											val_b2 = val_b1;
											val_b1= CurVal;

											val_a1 = val_a0;


											/*a0=src[IndexIn]*scaleFactor;		
											CurVal = a0*a0  
											+ a1*a1  
											- b1*TmpValFor1 
											- b2*TmpValFor2;

											dst[IndexOut] += CurVal;
											TmpValFor2=TmpValFor1;
											a0=CurVal;
											a1=TmpAFor1;
											*/
										}
#endif
									}

}
#endif
//=============================================
// derivation Y
#if !DERICHE_OPTIMIZED

template <typename T>
__device__ void DericheDerivation(
								  T *src, T* dst, T* Ym, T* Yp,    
								  int width, int height, float k, float a, float b1, float b2,
								  unsigned int xIndex
								  ) {


									  int YcurPos=0;
									  float TmpValFor1=0;
									  float TmpValFor2=0;
									  float TmpValBack1=0;
									  float TmpValBack2=0;
									  float CurVal;
									  for(int y = 0; y < height; y += 1)
									  {
										  //move forward
										  YcurPos= y*width;
										  CurVal = a*(float)src[YcurPos-width+xIndex] 
										  - b1*TmpValFor1 
											  - b2*TmpValFor2;

										  Yp[YcurPos + xIndex] = CurVal;
										  TmpValFor2=TmpValFor1;
										  TmpValFor1=CurVal;

										  //move backward
										  YcurPos= (height-y)*width;
										  CurVal = -a*(float)src[YcurPos+width+xIndex] 
										  - b1*TmpValBack1 
											  - b2*TmpValBack2;
										  Ym[YcurPos + xIndex] = CurVal;
										  TmpValBack2=TmpValBack1;
										  TmpValBack1=CurVal;

									  }


									  for(int y = 0; y < height; y += 1) {
										  YcurPos = y*width + xIndex;
										  dst[YcurPos] = Yp[YcurPos]+ Ym[YcurPos];
									  }
}
#else
//=============================================
template <typename T>
__device__ void DericheDerivation2(
								   T *src, T* dst, 
								   int width, int height,
								   float k, float a, float b1, float b2,
								   unsigned int xIndex,
								   unsigned int start, unsigned int end, bool forward
								   ) {

									   float val_b1=0;
									   float val_b2=0;
									   int YcurPos=0;
									   unsigned int IndexIn = 0;
									   unsigned int IndexOut = 0;
									   float CurVal;

									   int mgx, mgy;
									   mgx = mgy = 2;
									   int Sx = width - 2*mgx;
									   int Sy = height - 2*mgy;
									   int y=0;

									   if(!forward)
									   {//backward
										   //for(int y = start+1; y <end/2+1; y += 1)

										   //val_b1= 0;//dst[(y+1)*width+xIndex];
										   //val_b2= 0;//dst[(y+2)*width+xIndex];
										   for(y=(Sy+mgy-1); y >= (Sy+mgy)/2; y -=1)
										   {
											   //move backward
											   YcurPos	= y*width;

											   CurVal = -a*(float)src[YcurPos+width+xIndex] 
											   - b1*val_b1 
												   - b2*val_b2;

											   dst[YcurPos + xIndex] = CurVal;
											   val_b2=val_b1;
											   val_b1 = CurVal;
										   }
										   __syncthreads();

										   for(; y >= mgy; y -= 1)
										   {
											   //move backward
											   YcurPos= y*width;
											   CurVal = -a*(float)src[YcurPos+width+xIndex] 
											   - b1*val_b1 
												   - b2*val_b2;
											   dst[YcurPos + xIndex] += CurVal; // add to existing value
											   val_b2=val_b1;
											   val_b1 = CurVal;
										   }
									   }
									   else
									   {
										   for(y = mgy; y <(Sy+mgy)/2; y += 1)
										   {
											   //move forward
											   YcurPos= y*width;
											   CurVal = a*(float)src[YcurPos-width+xIndex] 
											   - b1*val_b1 
												   - b2*val_b2;

											   dst[YcurPos + xIndex] = CurVal;
											   val_b2=val_b1;
											   val_b1=CurVal;
										   }
										   __syncthreads();

										   for(; y < (Sy+mgy); y += 1)
										   {
											   //move forward
											   YcurPos= y*width;
											   CurVal = a*(float)src[YcurPos-width+xIndex] 
											   - b1*val_b1 
												   - b2*val_b2;


											   dst[YcurPos + xIndex] += CurVal;// add to existing value
											   val_b2=val_b1;
											   val_b1=CurVal;
										   }
									   }
									   //  __syncthreads();
}
#endif

#if DERICHE_SIMPLE_PROC


//=============================================
template <typename T>
__GCU_FCT_GLOBAL void gcuDericheLissageKernel(
	T *src, T* dst
	//, T* Ym, T* Yp,    
	,int width, int height, float k, float a0, float a1, float a2, float a3, float b1, float b2
	)
{
	unsigned int xBlock = __mul24(blockDim.x, blockIdx.x);
	unsigned int yBlock = __mul24(blockDim.y, blockIdx.y);
	unsigned int xIndex = xBlock + threadIdx.x;
	unsigned int yIndex = yBlock + threadIdx.y;

	if (xIndex < width && yIndex < height)
	{
#if !DERICHE_OPTIMIZED

		DericheLissage(
			src, dst, Ym, Yp,    
			width, height, k, a0, a1, a2, a3, b1, b2,
			xIndex
			);
#else

		if( threadIdx.y == 0)
		{//lissage from top to bottom
			DericheLissage2(
				src, dst/*Ym*/,    
				width, height, 
				k, a0, a1, a2, a3, b1, b2,
				xIndex,
				0, height, true
				);
		}
		else
		{//lissage from bottom to top
			DericheLissage2(
				src, dst/*Yp*/,    
				width, height, 
				k, a0, a1, a2, a3, b1, b2,
				xIndex,
				0, height, false
				);
		}
#endif
	}//width/height check
}


//=============================================
template <typename T>
__GCU_FCT_GLOBAL void gcuDericheDerivationKernel(
	T *src, T* dst
	//, T* Ym, T* Yp,    
	,int width, int height, float k, float a, float b1, float b2
	)
{
	unsigned int xBlock = __mul24(blockDim.x, blockIdx.x);
	unsigned int yBlock = __mul24(blockDim.y, blockIdx.y);
	unsigned int xIndex = xBlock + threadIdx.x;
	unsigned int yIndex = yBlock + threadIdx.y;

	if (xIndex < width && yIndex < height)
	{
#if !DERICHE_OPTIMIZED

		DericheDerivation(
			src, dst, Ym, Yp,    
			width, height, k, a, b1, b2,
			xIndex
			); 
#else

		if( threadIdx.y == 0)
		{//derivation from top to bottom
			DericheDerivation2(
				src, dst,    
				width, height, 
				k, a, b1, b2,
				xIndex,
				0, height, true, false
				);
		}
		else
		{//derivation from bottom to top
			DericheDerivation2(
				src,dst,    
				width, height, 
				k, a, b1, b2,
				xIndex,
				0, height, false, false
				);
		}
#endif
	}//width/height check
}


#else

template <typename T>
__GCU_FCT_GLOBAL void gcuDericheLissageKernel(
	T *src,  T* srcT, T* dst, T* dstT,
	int width, int height, 
	float k, float a0, float a1, float a2, float a3, float b1, float b2, float scaleFactor)
{
	unsigned int xBlock = __mul24(blockDim.x, blockIdx.x);
	unsigned int yBlock = __mul24(blockDim.y, blockIdx.y);
	unsigned int xIndex = xBlock + threadIdx.x;
	unsigned int yIndex = yBlock + threadIdx.y;
#if 1
	//if (xIndex < width && yIndex < height)
	{
		// y = 0 => lissage Src top to bottom
		// y = 1 => lissage Src bottom to top
		// y = 2 => lissage SrcT top to bottom
		// y = 3 => lissage SrcT bottom to top
		if( threadIdx.y == 0)
		{//lissage from top to bottom
			DericheLissage2(
				src, dst/*Ym*/    
				,width, height
				,k, a0, a1, a2, a3, b1, b2, scaleFactor
				,xIndex
				,0, height, true
				);
		}
		else if(threadIdx.y == 1)
		{//lissage from bottom to top
			DericheLissage2(
				src, dst/*Yp*/    
				,width, height
				,k, a0, a1, a2, a3, b1, b2, scaleFactor
				,xIndex
				,0, height, false
				);
		}
		//	else
		//		__syncthreads();
#if 1
		//transpose
		else if( threadIdx.y == 2)
		{//lissage from top to bottom
			DericheLissage2(
				srcT, dstT/*Ym*/    
				,height, width
				,k, a0, a1, a2, a3, b1, b2, scaleFactor
				,xIndex
				,0, width, true
				);
		}
		else if(threadIdx.y == 3)
		{//lissage from bottom to top
			DericheLissage2(
				srcT, dstT/*Yp*/ 
				,height, width
				,k, a0, a1, a2, a3, b1, b2, scaleFactor
				,xIndex
				,0, width, false
				);
		}
#endif
	}
#endif 
}

template <typename T>
__GCU_FCT_GLOBAL void gcuDericheDerivationKernel(
	T *src,  T* srcT, T* dst, T* dstT,
	int width, int height, 
	float k, float a, float b1, float b2)
{
	unsigned int xBlock = __mul24(blockDim.x, blockIdx.x);
	unsigned int yBlock = __mul24(blockDim.y, blockIdx.y);
	unsigned int xIndex = xBlock + threadIdx.x;
	unsigned int yIndex = yBlock + threadIdx.y;

	if (xIndex < width && yIndex < height)
	{
		// y = 0 => lissage Src top to bottom
		// y = 1 => lissage Src bottom to top
		// y = 2 => lissage SrcT top to bottom
		// y = 3 => lissage SrcT bottom to top
		if( threadIdx.y == 0)
		{//lissage from top to bottom
			DericheDerivation2(
				src, dst/*Ym*/,    
				width, height, 
				k, a, b1, b2,
				xIndex,
				0, height, true
				);
		}
		else if(threadIdx.y == 1)
		{//lissage from bottom to top
			DericheDerivation2(
				src, dst/*Yp*/,    
				width, height, 
				k, a, b1, b2,
				xIndex,
				0, height, false
				);
		}
		//transpose
		else if( threadIdx.y == 2)
		{//lissage from top to bottom
			DericheDerivation2(
				srcT, dstT/*Ym*/,    
				height, width, 
				k, a, b1, b2,
				xIndex,
				0, width, true
				);
		}
		else if(threadIdx.y == 3)
		{//lissage from bottom to top
			DericheDerivation2(
				srcT, dstT/*Yp*/,    
				height, width,
				k, a, b1, b2,
				xIndex,
				0, width, false
				);
		}
	}
}
#endif


// compute extrema of grandient
template <typename T>
__GCU_FCT_GLOBAL void gcuDericheExtremaKernel(
	T *Ix, T* Iy, T* Tre, T* Extrema
	,int width, int height, float dist
	,int blockX, int blockY
	)
{
#if 1
	unsigned int xBlock = __mul24(blockDim.x, blockIdx.x);
	unsigned int yBlock = __mul24(blockDim.y, blockIdx.y);
	unsigned int xIndex = (xBlock + threadIdx.x)*blockX;
	unsigned int yIndex = (yBlock + threadIdx.y)*blockY;


	if (xIndex < width && yIndex < height)
	{
		//if((xIndex > (int)dist+blockX) && (yIndex > (int)dist+blockY))
		{
			int dx, dy;
			float N, N1, N2;
			T EleX, EleY;
			float Max=0;
			float CurResult=0;
			//int x = xIndex;
			//????
			int in,jn,im,jm;
			float Ixval, Iyval, IxvalInJn, IyvalInJn, IxvalImJm, IyvalImJm, MulTemp, MulTempN, MulTempM;

			int x = xIndex;
#if 0//new version
			for(int i = yIndex; i < yIndex+blockY; i++)
			{
				//int y = yIndex;	
				for(int j = xIndex; j < xIndex+blockX; j++)
				{
					Iyval = Iy[i*width+j];//(Gy_Data+i*Gystep)[j];  //cvmGet(Gy,i,j); //Gy[i][j];
					Ixval = Ix[i*width+j];//(Gx_Data+i*Gxstep)[j];  //cvmGet(Gx,i,j); //Gx[i][j];

					MulTemp = sqrt(Iyval*Iyval + Ixval*Ixval);
					dx = (int)(Ixval/MulTemp*dist);
					dy = (int)(Iyval/MulTemp*dist);

					in = i - (int) (dy);
					if ((in<0)|| (in>height-1)) in=0;
					im = i + (int ) (dy);
					if ((im>height-1) || (im<0)) im=height-1;
					jn = j - (int ) (dx);
					if ((jn<0) || (jn>width-1)) jn=0;
					jm = j + (int ) (dx);
					if ((jm>width-1) || (jm<0)) jm=width-1;

					IyvalInJn = Iy[in*width+jn];//(Gy_Data+in*Gystep)[jn];  //cvmGet(Gy,in,jn); //Gy[in][jn];
					IxvalInJn = Ix[in*width+jn];//(Gx_Data+in*Gxstep)[jn];  //cvmGet(Gx,in,jn); //Gx[in][jn];
					MulTempN = sqrt(IyvalInJn*IyvalInJn + IxvalInJn*IxvalInJn);

					IyvalImJm = Iy[im*width+jm];//(Gy_Data+im*Gystep)[jm];   //cvmGet(Gy,im,jm); //Gy[in][jn];
					IxvalImJm = Iy[im*width+jm];//(Gx_Data+im*Gxstep)[jm];   //cvmGet(Gx,im,jm); //Gx[in][jn];
					MulTempM = sqrt(IyvalImJm*IyvalImJm + IxvalImJm*IxvalImJm);

					if ((MulTemp>MulTempN) && (MulTemp>=MulTempM))
					{
						Tre[i*width+j]	= MulTemp*256;//CurResult;B_Data[i*width+j]
					}
					else
					{
						Tre[i*width+j]	= 0;//B_Data[i*width+j]=0;
					}	
				}
			}//fin for


#else
			for(int x = xIndex; x < xIndex+blockX; x++)
			{
				int y = yIndex;	
				for(int y = yIndex; y < yIndex+blockY; y++)
				{
#if 1
					EleX = Ix[y*width+x];
					EleY = Iy[y*width+x];

					N = (sqrt( EleX*EleX + EleY*EleY ));
					dx = (int)(EleX/(N*dist));
					dy = (int)(EleY/(N*dist));

					/*
					in = x - (int) (dy);
					if ((in<0)|| (in>height-1)) in=0;
					im = x + (int ) (dy);
					if ((im>height-1) || (im<0)) im=height-1;
					jn = y - (int ) (dx);
					if ((jn<0) || (jn>width-1)) jn=0;
					jm = y + (int ) (dx);
					if ((jm>width-1) || (jm<0)) jm=width-1;
					*/

					in = y - (int) (dy);
					if ((in<0)|| (in>height-1)) in=0;
					im = y + (int ) (dy);
					if ((im>height-1) || (im<0)) im=height-1;
					jn = x - (int ) (dx);
					if ((jn<0) || (jn>width-1)) jn=0;
					jm = x + (int ) (dx);
					if ((jm>width-1) || (jm<0)) jm=width-1;

					IyvalInJn = Iy[in*width+jn];
					IxvalInJn = Ix[in*width+jn];
					N1 = sqrt(IyvalInJn*IyvalInJn + IxvalInJn*IxvalInJn);

					IyvalImJm = Iy[im*width+jm];
					IxvalImJm = Ix[im*width+jm];
					N2 = sqrt(IyvalImJm*IyvalImJm + IxvalImJm*IxvalImJm);
					/*
					N1 = (sqrt(Ix[(y+dy)*width+(x+dx)]*Ix[(y+dy)*width+(x+dx)]
					+ Iy[(y+dy)*width+(x+dx)]*Iy[(y+dy)*width+(x+dx)]));

					N2 = (sqrt(Ix[(y-dy)*width+(x-dx)]*Ix[(y-dy)*width+(x-dx)]
					+ Iy[(y-dy)*width+(x-dx)]*Iy[(y-dy)*width+(x-dx)]));
					*/
					if ((N>=N1)&&(N>=N2))
					{
						//CurResult		= log(N+1);
						CurResult		= N;
					}
					else
					{
						CurResult		= 0;
					}
					Tre[y*width+x]	= CurResult;
#else
					EleX = Ix[x*width+y];
					EleY = Iy[x*width+y];
					N = (sqrt( EleX*EleX + EleY*EleY ));
					dx = (int)(EleX/(N*dist));
					dy = (int)(EleY/(N*dist));

					N1 = (sqrt(Ix[(x+dy)*width+(y+dx)]*Ix[(x+dy)*width+(y+dx)]
					+ Iy[(x+dy)*width+(y+dx)]*Iy[(x+dy)*width+(y+dx)]));

					N2 = (sqrt(Ix[(x-dy)*width+(y-dx)]*Ix[(x-dy)*width+(y-dx)]
					+ Iy[(x-dy)*width+(y-dx)]*Iy[(x-dy)*width+(y-dx)]));

					if ((N>=N1)&&(N>=N2))
					{
						CurResult		= log(N+1);
						Tre[x*width+y]	= CurResult;
					}
#endif
					//else
					//	Tre[y*width+x]	= 0;

					if(Max < CurResult)
						Max = CurResult;	

				}
			}
			Extrema[yIndex*blockDim.x*gridDim.x/blockY+xIndex/blockX] = Max;
#endif
				
		}
	}
#else
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if( (width > x) && (x > (int)dist) && (height > y) && (y > (int)dist) ) 
	{
		int dx, dy;
		float N, N1, N2;
		T EleX, EleY;	
		float Max=0;
		float CurResult=0;
		//int x = xIndex;
#if 1//switch X and Y
		EleX = Ix[y*width+x];
		EleY = Iy[y*width+x];
		N = (sqrt( EleX*EleX + EleY*EleY ));
		dx = (int)(EleX/(N*dist));
		dy = (int)(EleY/(N*dist));

		N1 = (sqrt(Ix[(y+dy)*width+(x+dx)]*Ix[(y+dy)*width+(x+dx)]
		+ Iy[(y+dy)*width+(x+dx)]*Iy[(y+dy)*width+(x+dx)]));

		N2 = (sqrt(Ix[(y-dy)*width+(x-dx)]*Ix[(y-dy)*width+(x-dx)]
		+ Iy[(y-dy)*width+(x-dx)]*Iy[(y-dy)*width+(x-dx)]));

		if ((N>=N1)&&(N>=N2))
			Tre[y*width+x] = log(N+1);
#else
		EleX = Ix[x*width+y];
		EleY = Iy[x*width+y];
		N = (sqrt( EleX*EleX + EleY*EleY ));
		dx = (int)(EleX/(N*dist));
		dy = (int)(EleY/(N*dist));

		N1 = (sqrt(Ix[(x+dy)*width+(y+dx)]*Ix[(x+dy)*width+(y+dx)]
		+ Iy[(x+dy)*width+(y+dx)]*Iy[(x+dy)*width+(y+dx)]));

		N2 = (sqrt(Ix[(x-dy)*width+(y-dx)]*Ix[(x-dy)*width+(y-dx)]
		+ Iy[(x-dy)*width+(y-dx)]*Iy[(x-dy)*width+(y-dx)]));

		if ((N>=N1)&&(N>=N2))
			Tre[x*width+y] = log(N+1);
#endif

	}
#endif
}



// kind of normalisation
template <typename T>
__GCU_FCT_GLOBAL void gcuDericheExtensionKernel(
	T *Tre, T* Dyn,  
	int width, int height,
	T maxVal)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	/*	float Maxx = 0;
	if((width-10 > x) && (x > 10) && (height-10 > y) && (y > 10) ) {
	if (Tre[x*width+y]>Maxx) {
	Maxx = Tre[x*width+y];
	}
	}

	__syncthreads();
	*/
	if((width > x) && (height > y)) {
		float result = Tre[y*width+x];
		//result *= 255.;
		result *= 1./maxVal;//float type, no need to "*255"
		Dyn[y*width+x] = result ;
	}

	/*
	int Sy = height;
	height = Sy+2*mgy;
	int Sx = width;
	width = Sx+2*mgx;

	float Maxx = 0;
	for(int i = threadIdx.y+mgy+10; i <= Sy +mgy - 11; i += blockDim.y) {
	for(int j = threadIdx.x+mgx+10; j <= Sx +mgx - 11; j += blockDim.x)
	if (Tre[i*height+j]>Maxx) { Maxx = Tre[i*height+j]; } 
	}

	for(int i = threadIdx.y+mgy; i <= Sy + mgy - 1; i += blockDim.y) {
	for(int j = threadIdx.x+mgx; j <= Sx + mgx - 1; j += blockDim.x)
	Dyn[i*height+j] = Tre[i*height+j]*255/Maxx;
	}
	*/
}
