all: canny_basic canny_test_dataset canny_data

canny_basic: canny_basic.cu
	nvcc canny_basic.cu -o canny `pkg-config --cflags --libs opencv`

canny_test_dataset: canny_test_dataset.cu
	nvcc canny_test_dataset.cu -o canny_test_dataset `pkg-config --cflags --libs opencv`

canny_data: convertOpenCV.cu
	nvcc convertOpenCV.cu -o canny_data `pkg-config --cflags --libs opencv`

# p2_1: p2_1.cu
# 	nvcc p2_1.cu -o p2_1