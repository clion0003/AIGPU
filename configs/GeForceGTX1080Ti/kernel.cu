
#include "cuda_runtime.h"

#include <stdio.h>
#include <memory>
#include <stdlib.h>
#include <time.h>

#include <fstream>
#include <iostream>

#define input_channel_width 16
#define output_channel_width (input_channel_width-2)
#define output_channel_num 512
#define input_channel_num 512
#define THREADS 256
#define BLOCKS 200

void RandomInit(float* data, int n)
{
	srand(time(0));
	for (int i = 0; i < n; i++)
	{
		data[i] = rand() / (float)RAND_MAX - 0.5;
	}
}

__global__ void VecConv_cnnacc(const float *input_channel, const float *kernel, float *output_channel)
{
	int tx = threadIdx.x;

	output_channel[tx] = input_channel[tx];
}


__global__ void VecConv( const float *input_channel, const float *kernel, const float *kernel_bias, float *output_channel)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;

	int index = bx * blockDim.x + tx;
	if (index < output_channel_width * output_channel_width)
	{
		int x = index % output_channel_width;
		int y = index / output_channel_width;
		for (int i = 0; i < output_channel_num; i++)
		{
			float result = 0;
			for (int j = 0; j < input_channel_num; j++)
			{
				const float *input_image = input_channel + input_channel_width * input_channel_width * j;
				const float *input_kernel = kernel + 3 * 3 * input_channel_num * i + 3 * 3 * j;
				for (int k = 0; k < 3; k++)
					for (int t = 0; t < 3; t++)
					{
						result += input_image[(y + t) * input_channel_width + (x + k)] * input_kernel[t * 3 + k];
					}
			}
            result += kernel_bias[i];
            if(result < 0)result =0;
			output_channel[output_channel_width * output_channel_width * i + index] = result;
		}
	}
}

int main()
{
	const int input_channel_size = input_channel_width * input_channel_width * input_channel_num * sizeof(float);
	const int kernel_size = 3 * 3 * input_channel_num * output_channel_num * sizeof(float);
    const int bias_size = output_channel_num * sizeof(float);
	const int output_channel_size = (input_channel_width - 2)* (input_channel_width - 2) * output_channel_num * sizeof(float);

	float *input_channel_host, *kernel_host, *kernel_bias, *output_channel_host,*output_cpu;
	float *input_channel_device, *kernel_device, *output_channel_device,*kernel_bias_device;
	float *output_channel_cnn_cpu, *output_channel_cnn_device;

	input_channel_host = (float*)malloc(input_channel_size);
	kernel_host = (float*)malloc(kernel_size);
    kernel_bias = (float*)malloc(bias_size);
	output_channel_host = (float*)malloc(output_channel_size);
	output_cpu = (float*)malloc(output_channel_size);
	output_channel_cnn_cpu = (float*)malloc(output_channel_size);


	cudaMalloc((void**)&input_channel_device, input_channel_size);
	cudaMalloc((void**)&kernel_device, kernel_size);
	cudaMalloc((void**)&output_channel_device, output_channel_size);
	cudaMalloc((void**)&output_channel_cnn_device, output_channel_size);
    cudaMalloc((void**)&kernel_bias_device,kernel_size);
	
	std::ifstream infile1("conv1_1_o.txt"), infile2("conv1_2_w.txt"), infile3("conv1_2_b.txt");

	
	//for (int i = 0; i < input_channel_num; i++)
	//{
	//	for (int j = 0; j < input_channel_width; j++){
	//		for (int k = 0; k < input_channel_width; k++){
	//			infile1 >> input_channel_host[input_channel_width * input_channel_width * i + input_channel_width * j + k];
	//		}
	//	}
	//}

	for (int i = 0; i < input_channel_num; i++)
	{
		for (int j = 0; j < output_channel_width; j++)
        {
			for (int k = 0; k < output_channel_width; k++)
            {
				infile1 >> input_channel_host[input_channel_width * input_channel_width * i + input_channel_width * (j + 1) + k + 1];
			}
		}
	}

	for (int i = 0; i < output_channel_num; i++)
	{
		for (int t = 0; t < input_channel_num; t++)
        {
			for (int j = 0; j < 3; j++)
            {
				for (int k = 0; k < 3; k++)
                {
					infile2 >> kernel_host[3 * 3 * input_channel_num * i + 3 * 3 * t + 3 * j + k];
				}
			}
		}
	}

    for (int i = 0;i < output_channel_num; i++) infile3 >> kernel_bias[i];
	

	//RandomInit(input_channel_host, input_channel_width * input_channel_width * input_channel_num);
	//RandomInit(kernel_host, 3 * 3 * input_channel_num * output_channel_num);

	//padding
	for (int i = 0; i < input_channel_num; i++){
		for (int j = 0; j < input_channel_width; j++)
		{
			input_channel_host[input_channel_width * input_channel_width * i + j] = 0;
			input_channel_host[input_channel_width * input_channel_width * i + input_channel_width * (input_channel_width - 1)+j] = 0;
			input_channel_host[input_channel_width * input_channel_width * i + input_channel_width * j] = 0;
			input_channel_host[input_channel_width * input_channel_width * i + input_channel_width * j + input_channel_width - 1] = 0;
		}
	}

	for (int i = 0; i < (input_channel_width - 2) * (input_channel_width - 2) * output_channel_num; i++)output_cpu[i] = 0;

	for (int i = 0; i < output_channel_num; i++)
	{
		for (int j = 0; j < input_channel_num; j++)
		{
			for (int k = 0; k < input_channel_width - 2; k++)
				for (int t = 0; t < input_channel_width - 2; t++)
				{
					float result = 0;
					for (int p = 0; p < 3; p++)
						for (int q = 0; q < 3; q++)
						{
							result += input_channel_host[input_channel_width * input_channel_width * j + (k + p) * input_channel_width + (t + q)] * kernel_host[3 * 3 * input_channel_num * i + 3 * 3 * j + p * 3 + q];
						}
					output_cpu[(input_channel_width - 2) * (input_channel_width - 2) * i + k * (input_channel_width - 2) + t] += result;

				}
		}
	}

    for(int i = 0; i < output_channel_num; i++)
    {
        for(int k = 0; k < output_channel_width; k++)
        {
            for(int j = 0; j < output_channel_width; j++)
            {
                output_cpu[output_channel_width * output_channel_width * i + output_channel_width * k + j] += kernel_bias[i];
                if(output_cpu[output_channel_width * output_channel_width * i + output_channel_width * k + j] < 0)
                    output_cpu[output_channel_width * output_channel_width * i + output_channel_width * k + j] = 0;

            }
        }
    }

	cudaMemcpy(input_channel_device, input_channel_host, input_channel_size, cudaMemcpyHostToDevice);
	//cudaMemcpy(output_channel_device, output_channel_host, output_channel_size, cudaMemcpyHostToDevice);
	cudaMemcpy(kernel_device, kernel_host, kernel_size, cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_bias_device, kernel_bias, bias_size, cudaMemcpyHostToDevice);

	int threadsPerBlock = THREADS;
	int blocksPerGrid = BLOCKS;
	VecConv << <blocksPerGrid, threadsPerBlock >> >(input_channel_device, kernel_device, kernel_bias_device, output_channel_device);

	cudaMemcpy(output_channel_host, output_channel_device, output_channel_size, cudaMemcpyDeviceToHost);

	int count = 0, count2 = 0;;
	//for (int i = 0; i < 56 * 56 * 256; i++)
	//{
	//	if (abs(output_cpu[i] - output_channel_host[i] > 0.000001)){
	//		count++;
	//		std::cout << i%

	//	}
	//}
	threadsPerBlock = 36;
	blocksPerGrid = 1;

	VecConv_cnnacc << <blocksPerGrid, threadsPerBlock >> >(input_channel_device, kernel_device, output_channel_cnn_device);

	cudaMemcpy(output_channel_cnn_cpu, output_channel_cnn_device, output_channel_size, cudaMemcpyDeviceToHost);

	std::ofstream ofile1("cpu.txt"), ofile2("device.txt"), ofile3("cnn_acc.txt");//, ofile4("input_channel.txt"), ofile5("input_kernel.txt");
    std::ifstream ifile1("conv1_2_o.txt");
    float tmp;
	for (int i = 0; i < output_channel_num; i++)
	{
		for (int j = 0; j < output_channel_width; j++)
		{
			for (int k = 0; k < output_channel_width; k++)
			{
				ofile1 << output_cpu[output_channel_width * output_channel_width * i + output_channel_width * j + k] << " ";
				ofile2 << output_channel_host[output_channel_width * output_channel_width * i + output_channel_width * j + k] << " ";
				ofile3 << output_channel_cnn_cpu[output_channel_width * output_channel_width * i + output_channel_width * j + k] << " ";
				
                if (abs(output_cpu[output_channel_width * output_channel_width * i + output_channel_width * j + k] - output_channel_host[output_channel_width * output_channel_width * i + output_channel_width * j + k] > 0.01))
				{
					count++;
					std::cout << i << " " << j << " " << k << " " << output_cpu[output_channel_width * output_channel_width * i + output_channel_width * j + k] << " " << output_channel_host[output_channel_width * output_channel_width * i + output_channel_width * j + k] << std::endl;
				}
                ifile1 >> tmp;
				if (abs(tmp - output_cpu[output_channel_width * output_channel_width * i + output_channel_width * j + k] > 0.01))
				{
					count2++;
				    std::cout << output_cpu[output_channel_width * output_channel_width * i + output_channel_width * j + k] << " " << tmp  << std::endl;
				}
			}
			ofile1 << std::endl;
			ofile2 << std::endl;
			ofile3 << std::endl;
		}
		ofile1 << std::endl << std::endl << std::endl;
		ofile2 << std::endl << std::endl << std::endl;
		ofile3 << std::endl << std::endl << std::endl;
	}

	//for (int i = 0; i < input_channel_num; i++)
	//{
	//	for (int j = 0; j < input_channel_width; j++){
	//		for (int k = 0; k < input_channel_width; k++){
	//			ofile4 << input_channel_host[input_channel_width * input_channel_width * i + input_channel_width * j + k] << " ";
	//		}
	//		ofile4 << std::endl;
	//	}
	//	ofile4 << std::endl << std::endl << std::endl;
	//}

	//for (int i = 0; i < output_channel_num;i++)
	//{
	//	for (int t = 0; t < input_channel_num; t++){
	//		for (int j = 0; j < 3; j++){
	//			for (int k = 0; k < 3; k++){
	//				ofile5 << kernel_host[3 * 3 * input_channel_num * i + 3 * 3 * t + 3 * j + k] << " ";
	//			}
	//			ofile5 << std::endl;
	//		}
	//		ofile5 << std::endl;
	//	}
	//	ofile5 << std::endl << std::endl << std::endl;
	//}




	printf("CPU&GPU:%d\nCPU&CAFFE:%d\n", count,count2);
	ofile1.close();
	ofile2.close();
	ofile3.close();
	//ofile4.close();
	//ofile5.close();


    return 0;
}
