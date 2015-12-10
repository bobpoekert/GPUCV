set CUDA_PROFILE=1
set CUDA_PROFILE_CONFIG="./cuda_prof_config.txt"
set CUDA_PROFILE_LOG=.\cuda_log_file.log
..\..\CudaVisualProfiler\bin\cudaprof.exe ..\etc\GpuCVConsole_Cuda.cpj
pause