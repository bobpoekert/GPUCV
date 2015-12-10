rem remove line
/LOAD_CHDL/ d

s/(\n/\(/


s/CV_INLINE/ /

s/CVAPI(void)/void/

s/CVAPI(void\*)/void\*/

s/CVAPI(double)/double/

s/DOUBLE/double/

s/CVAPI(int)/int/

s/CVAPI(int64)/int64/

s/CVAPI(int\*)/int\*/

s/CVAPI(float)/float/

s/CVAPI(float\*)/float\*/

s/CVAPI(char)/char/

s/CVAPI(char\*)/char\*/

s/CVAPI(const char\*)/const char\*/

s/CVAPI(uchar)/uchar/

s/CVAPI(uchar\*)/uchar\*/

s/CVAPI(CvArr)/CvArr/

s/CVAPI(CvArr\*)/CvArr\*/

s/CVAPI(CvBox2D)/CvBox2D/

s/CVAPI(CvStereoGCState\*)/CvStereoGCState\*/
s/CvStereoGCState;//

s/CVAPI(CvStereoBMState\*)/CvStereoBMState\*/
s/CvStereoBMState;//


s/CVAPI(CvConDensation\*)/CvConDensation\*/
s/CVAPI(CvContourTree\*)/CvContourTree\*/
s/CVAPI(CvContourScanner\*)/CvContourScanner\*/
s/CVAPI(CvContourScanner)/CvContourScanner/
s/CVAPI(CvErrorCallback)/CvErrorCallback/
s/CVAPI(CvFileStorage\*)/CvFileStorage\*/

s/CVAPI(CvFileNode\*)/CvFileNode\*/
s/CVAPI(CvKalman\*)/CvKalman\*/
s/CVAPI(CvGraph\*)/CvGraph\*/
s/CVAPI(CvGraphEdge\*)/CvGraphEdge\*/
s/CVAPI(CvGraphScanner\*)/CvGraphScanner\*/
s/CVAPI(CvHistogram\*)/CvHistogram\*/


s/CVAPI(CvHaarClassifierCascade\*)/CvHaarClassifierCascade\*/
s/CVAPI(const CvMat\*)/const CvMat\*/
s/CVAPI(CvMat\*)/CvMat\*/
s/CVAPI(CvMatND\*)/CvMatND\*/
s/CVAPI(CvMemStorage\*)/CvMemStorage\*/
s/CVAPI(CvPoint)/CvPoint/
s/CVAPI(CvPOSITObject\*)/CvPOSITObject\*/
s/CVAPI(CvRect)/CvRect/
s/CVAPI(CvScalar\*)/CvScalar\*/
s/CVAPI(CvScalar)/CvScalar/
s/CVAPI(CvSeq\*)/CvSeq\*/
s/CVAPI(CvSet\*)/CvSet\*/
s/CVAPI(CvSize)/CvSize/
s/CVAPI(CvSparseMat\*)/CvSparseMat\*/
s/CVAPI(CvSparseNode\*)/CvSparseNode\*/
s/CVAPI(CvString)/CvString/
s/CVAPI(CvStringHashNode\*)/CvStringHashNode\*/
s/CVAPI(CvSubdiv2D\*)/CvSubdiv2D\*/
s/CVAPI(CvSubdiv2DPointLocation)/CvSubdiv2DPointLocation/
s/CVAPI(CvSubdiv2DPoint\*)/CvSubdiv2DPoint\*/
s/CVAPI(CvTypeInfo\*)/CvTypeInfo\*/
s/CVAPI(CvTermCriteria)/CvTermCriteria/
s/CVAPI(IplConvKernel\*)/IplConvKernel\*/
s/CVAPI(IplConvKernel\*)/IplConvKernel\*/
s/CVAPI(IplImage\*)/IplImage\*/
s/CVAPI(IplImage)/IplImage/
s/CVAPI(IplImage\*\*)/IplImage\*\*/

rem Highgui
s/CVAPI(CvCapture\*)/CvCapture\*/
s/CVAPI(CvVideoWriter\*)/CvVideoWriter\*/


s/DOUBLE/double/
s/DOUBLE/double/

rem extra
/cvSetAdd(set_header/ d

/^cvReshapeMatND/ d

/else / d

s/else//

/CV_GRAPH_ITEM_VISITED_FLAG/ d

/CvNArrayIterator;/ d

/CvGraphScanner;/ d

/CvFont;/ d

s/void  cvDecRefData( CvArr* arr )/void  cvDecRefData( CvArr* arr );/

rem empty lines
/^$/ d
