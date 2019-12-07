all: canny_show_img canny_test_uo cannt_test_o canny_test_analysis_uo canny_test_analysis_o

canny_show_img: canny_show_img.cu
	nvcc canny_show_img.cu -o canny_show_img `pkg-config --cflags --libs opencv`

canny_test_uo: canny_test_dataset_benchmark_unoptimized.cu
	nvcc canny_test_dataset_benchmark_unoptimized.cu -o canny_data_uo `pkg-config --cflags --libs opencv`

cannt_test_o: canny_test_dataset_benchmark_optimized.cu
	nvcc canny_test_dataset_benchmark_optimized.cu -o canny_data_o `pkg-config --cflags --libs opencv`

canny_test_analysis_uo: canny_test_dataset_analysis_unoptimized.cu
	nvcc canny_test_dataset_analysis_unoptimized.cu -o canny_test_analysis_uo `pkg-config --cflags --libs opencv`

canny_test_analysis_o: canny_test_dataset_analysis_optimized.cu
	nvcc canny_test_dataset_analysis_optimized.cu -o canny_test_analysis_o `pkg-config --cflags --libs opencv`
