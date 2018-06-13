all: wave_2d_fd_perf_gpu/libvcuda1.so wave_2d_fd_perf_gpu/libvcuda2.so wave_2d_fd_perf_gpu/libvcuda3.so wave_2d_fd_perf_gpu/libvcuda4.so wave_2d_fd_perf_gpu/libvcuda5.so

wave_2d_fd_perf_gpu/libvcuda1.so: wave_2d_fd_perf_gpu/vcuda1.cu
	nvcc -ccbin clang-3.8 --restrict --use_fast_math -arch=sm_37 --shared \
		-O3 -o wave_2d_fd_perf_gpu/libvcuda1.so --compiler-options \
		-fPIC --compiler-options -Wall \
		wave_2d_fd_perf_gpu/vcuda1.cu


wave_2d_fd_perf_gpu/libvcuda2.so: wave_2d_fd_perf_gpu/vcuda2.cu
	nvcc -ccbin clang-3.8 --restrict --use_fast_math -arch=sm_37 --shared \
		-O3 -o wave_2d_fd_perf_gpu/libvcuda2.so --compiler-options \
		-fPIC --compiler-options -Wall \
		wave_2d_fd_perf_gpu/vcuda2.cu


wave_2d_fd_perf_gpu/libvcuda3.so: wave_2d_fd_perf_gpu/vcuda3.cu
	nvcc -ccbin clang-3.8 --restrict --use_fast_math -arch=sm_37 --shared \
		-O3 -o wave_2d_fd_perf_gpu/libvcuda3.so --compiler-options \
		-fPIC --compiler-options -Wall \
		wave_2d_fd_perf_gpu/vcuda3.cu


wave_2d_fd_perf_gpu/libvcuda4.so: wave_2d_fd_perf_gpu/vcuda4.cu
	nvcc -ccbin clang-3.8 --restrict --use_fast_math -arch=sm_37 --shared \
		-O3 -o wave_2d_fd_perf_gpu/libvcuda4.so --compiler-options \
		-fPIC --compiler-options -Wall \
		wave_2d_fd_perf_gpu/vcuda4.cu


wave_2d_fd_perf_gpu/libvcuda5.so: wave_2d_fd_perf_gpu/vcuda5.cu
	nvcc -ccbin clang-3.8 --restrict --use_fast_math -arch=sm_37 --shared \
		-O3 -o wave_2d_fd_perf_gpu/libvcuda5.so --compiler-options \
		-fPIC --compiler-options -Wall \
		wave_2d_fd_perf_gpu/vcuda5.cu
