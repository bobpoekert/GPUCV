set BENCHCOLOR=-c OpenCV_0:rgb(255,20,20) -c GpuCV-GLSL_0:rgb(20,255,20)  -c GpuCV-CUDA_0:rgb(20,20,255)
set BENCHCOLOR=%BENCHCOLOR% -c OpenCV_1:rgb(220,20,20) -c GpuCV-GLSL_1:rgb(20,220,20)  -c GpuCV-CUDA_1:rgb(20,20,220)
set BENCHCOLOR=%BENCHCOLOR% -c OpenCV_2:rgb(180,20,20) -c GpuCV-GLSL_2:rgb(20,180,20)  -c GpuCV-CUDA_2:rgb(20,20,180)
set BENCHCOLOR=%BENCHCOLOR% -c OpenCV_3:rgb(140,20,20) -c GpuCV-GLSL_3:rgb(20,140,20)  -c GpuCV-CUDA_3:rgb(20,20,140)

set BENCHCOLOR_GPU=-c geforce gtx 480/pci/sse2_0:rgb(255,20,20) -c ati radeon hd 5800 series_0:rgb(20,255,20)
set BENCHCOLOR_GPU=%BENCHCOLOR_GPU% -c geforce gtx 480/pci/sse2_1:rgb(220,20,20) -c ati radeon hd 5800 series_1:rgb(20,220,20)
set BENCHCOLOR_GPU=%BENCHCOLOR_GPU% -c geforce gtx 480/pci/sse2_1:rgb(180,20,20) -c ati radeon hd 5800 series_1:rgb(20,180,20)
set BENCHCOLOR_GPU=%BENCHCOLOR_GPU% -c geforce gtx 480/pci/sse2_1:rgb(140,20,20) -c ati radeon hd 5800 series_1:rgb(20,140,20)

set BENCHCOLOR_VERSION=-c "2.1.0_0:rgb(255,20,20)" -c "GpuCV-GLSL_0:rgb(20,255,20)"  -c "GpuCV-CUDA_0:rgb(20,20,255)"
set BENCHCOLOR_VERSION=%BENCHCOLOR_VERSION% -c "2.1.0_1:rgb(220,20,20)" -c "GpuCV-GLSL_1:rgb(20,220,20)"  -c "GpuCV-CUDA_1:rgb(20,20,220)"
set BENCHCOLOR_VERSION=%BENCHCOLOR_VERSION% -c "2.1.0_2:rgb(180,20,20)" -c "GpuCV-GLSL_2:rgb(20,180,20)"  -c "GpuCV-CUDA_2:rgb(20,20,180)"
set BENCHCOLOR_VERSION=%BENCHCOLOR_VERSION% -c "2.1.0_3:rgb(140,20,20)" -c "GpuCV-GLSL_3:rgb(20,140,20)"  -c "GpuCV-CUDA_3:rgb(20,20,140)"

echo generating ATi/CPU benchs

benchreport.x86_32.exe --dpath Benchs_OPENCV -i Benchmarks_WINDOWS.xml -i Benchmarks_WINDOWS_I7.V2.xml --colorfield type %BENCHCOLOR% 
rem echo %BENCHCOLOR_VERSION%

benchreport.x86_32.exe --dpath Benchs_OPENCV_VER -i Benchmarks_windows_i7.xml -i Benchmarks_WINDOWS_I7.V2.xml --colorfield "opencv-V" %BENCHCOLOR_VERSION% 
rem echo %BENCHCOLOR_VERSION%

echo generating NVIDIA/CPU benchs

rem benchreport.x86_32.exe --dpath Benchs_NV_CPU -i Benchmarks_windows_i7.xml -i Benchmarks_WINDOWS_nv480.xml %BENCHCOLOR% --colorfield type


echo generating NVIDIA/ATI GLSL

rem benchreport.x86_32.exe --dpath Benchs_GPU_GLSL -i Benchmarks_windows_ati5870.xml -i Benchmarks_WINDOWS_nv480.xml %BENCHCOLOR_GPU% --colorfield GPU
