echo You should be root to run this
GPUCV_TMP_DIR=/tmp
GPUCV_VERSION=0.4.1.rev.175
GPUCV_SRC_DIR=/usr/src/redhat/SOURCES
GPUCV_BUILD_DIR=/usr/src/redhat/BUILD
GPUCV_FULL_NAME=gpucv-$GPUCV_VERSION
export GPUCV_TMP_DIR
export GPUCV_VERSION
export GPUCV_SRC_DIR
export GPUCV_BUILD_DIR

rm -Rf $GPUCV_TMP_DIR/gpucv*
echo copy local source to temp folder

#cd ../../../
#cp -Rf gpucv_cuda $GPUCV_TMP_DIR/gpucv-$GPUCV_VERSION
#echo clean local folder
#sh 01-CleanSolution.sh
sh 03-BuildSolution.sh
sh 05-MakeDoc.sh
sh 06-MaketempCopyBIN.sh
cd $GPUCV_TMP_DIR/gpucv-$GPUCV_VERSION/
#sh
echo Clean package archive
rm -Rf $GPUCV_SRC_DIR/gpucv*
rm -Rf $GPUCV_BUILD_DIR/gpucv*
echo make package archive
cd $GPUCV_TMP_DIR
tar -jcf $GPUCV_FULL_NAME.tar.bz2 $GPUCV_FULL_NAME
#echo make RPM
#rpmbuild -ba /home/allusse/workspace/gpucv_cuda/etc/RPM/gpucv.spec
