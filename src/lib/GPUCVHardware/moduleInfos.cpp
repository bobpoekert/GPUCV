
#include "StdAfx.h"
#include <GPUCVHardware/moduleInfo.h>
using namespace std;

namespace GCV{

//==================================================================================================
ModColor CreateModColor(unsigned int R, unsigned int G, unsigned int B, unsigned int A)
{
	ModColor CurrentColor;
	CurrentColor.R = R;
	CurrentColor.G = G;
	CurrentColor.B = B;
	CurrentColor.A = A;
	return CurrentColor;
}
//todo
//_GPUCV_HARDWARE_EXPORT TiXmlElement* XMLRead(TiXmlElement* _XML_Root ,CreateModColor & _rModColor);
//_GPUCV_HARDWARE_EXPORT TiXmlElement* XMLWrite(TiXmlElement* _XML_Root ,CreateModColor & _rModColor);
//
//==================================================================================================
std::string GetStrImplemtation(const ImplementationDescriptor* _Impl)
{
	if(_Impl)
		return _Impl->m_strImplName;
	else
		return "AUTO";
}
//==================================================================================================
std::string GetStrImplemtationID(BaseImplementation _ID)
{
	switch(_ID)
	{
		case GPUCV_IMPL_AUTO:	return GPUCV_IMPL_AUTO_STR;
		case GPUCV_IMPL_GLSL:	return GPUCV_IMPL_GLSL_STR;
		case GPUCV_IMPL_CUDA:	return GPUCV_IMPL_CUDA_STR;
		case GPUCV_IMPL_OPENCL:	return GPUCV_IMPL_OPENCL_STR;
		case GPUCV_IMPL_OPENCV:	return GPUCV_IMPL_OPENCV_STR;
		default:return "Unknown";
	}
}
//==================================================================================================
LibraryDescriptor::LibraryDescriptor()
:m_UseGpu(false)
,m_ImplementationDescriptor(NULL)
//,m_DynImplID(0)
{
	m_StopColor.R = m_StopColor.G = m_StopColor.B = m_StopColor.A =200;
	m_StartColor.R = m_StartColor.G = m_StartColor.B = m_StartColor.A =20;
}
//==================================================================================================
LibraryDescriptor::~LibraryDescriptor()
{}
//==================================================================================================
SG_TRC::ColorFilter & LibraryDescriptor::GenerateColorFilter(SG_TRC::ColorFilter &_rColorTable, ModColor & _rStartColor, ModColor & _rStopColor, unsigned int _ColorNbr)
{
	ModColor ColorOffset, CurrentColor;
	ColorOffset.R = (_rStopColor.R - _rStartColor.R) / _ColorNbr;
	ColorOffset.G = (_rStopColor.G - _rStartColor.G) / _ColorNbr;
	ColorOffset.B = (_rStopColor.B - _rStartColor.B) / _ColorNbr;
	ColorOffset.A = (_rStopColor.A - _rStartColor.A) / _ColorNbr;

	CurrentColor.R = _rStopColor.R;
	CurrentColor.G = _rStopColor.G;
	CurrentColor.B = _rStopColor.B;
	CurrentColor.A = _rStopColor.A;
		
	//add _ColorNbr to the ModColor filter
	std::string strName, strColor;
	for(unsigned int i = 0; i < _ColorNbr; i++)
	{
		strName = GetImplementationName();
		strName += "_";
		strName += SGE::ToCharStr(i);

		strColor = "rgb(";
		strColor += SGE::ToCharStr(CurrentColor.R);
		strColor += ",";
		strColor += SGE::ToCharStr(CurrentColor.G);
		strColor += ",";
		strColor += SGE::ToCharStr(CurrentColor.B);
		strColor += ")";
		//strColor += SGE::ToCharStr(CurrentColor.A);


		//add new ModColor filter
		_rColorTable.m_InputColors.AddParam(strName.data(), strColor.data());
		
		//update current ModColor
		CurrentColor.R += ColorOffset.R;
		CurrentColor.G += ColorOffset.G;
		CurrentColor.B += ColorOffset.B;
		CurrentColor.A += ColorOffset.A;
	}
	return _rColorTable;
}
//==================================================================================================
SG_TRC::ColorFilter & LibraryDescriptor::GenerateDefaultColorFilter(SG_TRC::ColorFilter &_rColorTable, unsigned int _ColorNbr)
{
	return GenerateColorFilter(_rColorTable, m_StartColor, m_StopColor, _ColorNbr);
}
//==================================================================================================

}//namespace GCV