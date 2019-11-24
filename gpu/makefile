all: canny_basic

canny_basic: canny_basic.cu
	nvcc canny_basic.cu -o canny `pkg-config --cflags --libs opencv`

# p2_1: p2_1.cu
# 	nvcc p2_1.cu -o p2_1