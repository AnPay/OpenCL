//reference: http://www.es.ele.tue.nl/~mwijtvliet/5KK73/?page=mmopencl
/*
This document describes a matrix multiplication example application using OpenCL for Nvidia GPUs, the focus will be on the code structure for the host application and the OpenCL GPU kernels. For examples of optimization matrix multiplication please refer to the CUDA example documentation, most CUDA kernels will be very similar in a OpenCL implementation. This example can be found here. The source code for the OpenCL matrix multiplication example can be found here.
Host code
The host code initializes the OpenCL capable GPUs, allocates and transfers memory and executed the OpenCL kernel.

The code shown below declares OpenCL memories which will be instantiated on the device, hence the prefix 'd_'. The A and B memories are two input matrices of size 1024x1024, C is the result matrix. Since the memory described above is on the device we also need to declare and allocate memory on the host, in this case the server, and fill the input arrays with values. This is done by the radomInit() function.
*/
// OpenCL device memory for matrices
   cl_mem d_A;
   cl_mem d_B;
   cl_mem d_C;

   // set seed for rand()
   srand(2014);
 
   //Allocate host memory for matrices A and B
   unsigned int size_A = WA * HA;
   unsigned int mem_size_A = sizeof(float) * size_A;
   float* h_A = (float*) malloc(mem_size_A);
 
   unsigned int size_B = WB * HB;
   unsigned int mem_size_B = sizeof(float) * size_B;
   float* h_B = (float*) malloc(mem_size_B);

   //Initialize host memory
   randomInit(h_A, size_A);
   randomInit(h_B, size_B);
 
   //Allocate host memory for the result C
   unsigned int size_C = WC * HC;
   unsigned int mem_size_C = sizeof(float) * size_C;
   float* h_C = (float*) malloc(mem_size_C);

//The output memory on the host will be allocated but only written with the result after execution of the OpenCL kernel.
//The function clCreateCommandQueue creates a OpenCL command queue. The OpenCL functions that are submitted to a command-queue are enqueued in the order the calls are made but can be configured to execute in-order or out-of-order. The properties argument in clCreateCommandQueue can be used to specify the execution order.

   cl_uint dev_cnt = 0;
   clGetPlatformIDs(0, 0, &dev_cnt);
    
   cl_platform_id platform_ids[100];
   clGetPlatformIDs(dev_cnt, platform_ids, NULL);
    
   // Connect to a compute device
   int gpu = 1;
   err = clGetDeviceIDs(platform_ids[0], gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1,
   &device_id, NULL);

   if (err != CL_SUCCESS)
   {
       printf("Error: Failed to create a device group!\n");
       return EXIT_FAILURE;
   }
  
   // Create a compute context 
   context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
   if (!context)
   {
       printf("Error: Failed to create a compute context!\n");
       return EXIT_FAILURE;
   }

   // Create a command commands
   commands = clCreateCommandQueue(context, device_id, 0, &err);
   if (!commands)
   {
       printf("Error: Failed to create a command commands!\n");
       return EXIT_FAILURE;
   }


//Once a OpenCL context and command queue are defined the OpenCL kernel can be loaded. In OpenCL kernels are typically loaded are runtime and compiled by the function clBuildProgram. In order to do this the actual kernel is loaded by the function LoadOpenCLKernel and transformed into an OpenCL program description with the clCreateProgramWithSource function. The built kernel description will then be made ready for execution by the clCreateKernel function. Be aware that the second argument should match the name of the kernel as descibed in the .cl file.

   // Create the compute program from the source file
   char *KernelSource;
   long lFileSize;

   lFileSize = LoadOpenCLKernel("matrixmul_kernel.cl", &KernelSource, false);
   if( lFileSize < 0L ) {
       perror("File read failed");
       return 1;
   }

   program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
   if (!program)
   {
       printf("Error: Failed to create compute program!\n");
       return EXIT_FAILURE;
   }

   // Build the program executable
   err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
   if (err != CL_SUCCESS)
   {
       size_t len;
       char buffer[2048];
       printf("Error: Failed to build program executable!\n");
       clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), 
       buffer, &len);

       printf("%s\n", buffer);
       exit(1);
   }

   // Create the compute kernel in the program we wish to run
   //
   kernel = clCreateKernel(program, "matrixMul", &err);
   if (!kernel || err != CL_SUCCESS)
   {
       printf("Error: Failed to create compute kernel!\n");
       exit(1);
   }

//Now the kernel is ready for execution the buffers on the compute device (in our case the GPU) should be allocated, this is done with the clCreateBuffer function, the arguments of this function can be used to describe if a memory is read-only, write-only or read-write. Specifying this correct can help to increase performance. The function clSetKernelArg links the allocated memory space in the GPU to the arguments of the kernel, in our case the A,B and C matrices and two integers specifying the width of the matrices.


   // Create the input and output arrays in device memory for our calculation
   d_C = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_A, NULL, &err);
   d_A = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_A, h_A,
   &err);
   d_B = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_B, h_B, 
   &err);

   if (!d_A || !d_B || !d_C)
   {
       printf("Error: Failed to allocate device memory!\n");
       exit(1);
   }    
    
   printf("Running matrix multiplication for matrices A (%dx%d) and B (%dx%d) ...\n", 
   WA,HA,WB,HB); 

   //Launch OpenCL kernel
   size_t localWorkSize[2], globalWorkSize[2];
 
   int wA = WA;
   int wC = WC;
   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_C);
   err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_A);
   err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_B);
   err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&wA);
   err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&wC);

   if (err != CL_SUCCESS)
   {
       printf("Error: Failed to set kernel arguments! %d\n", err);
       exit(1);
   }

/*The function clEnqueueNDRangeKernel enqueues a command to execute a kernel on a device. Some important parameters are the global work size and the local work size. These are explained as follows by the OpenCL documentation:

Global work size
Points to an array of work_dim unsigned values that describe the number of global work-items in work_dim dimensions that will execute the kernel function. The total number of global work-items is computed as global_work_size[0] *...* global_work_size[work_dim - 1]. The values specified in global_work_size cannot exceed the range given by the sizeof(size_t) for the device on which the kernel execution will be enqueued. The sizeof(size_t) for a device can be determined using CL_DEVICE_ADDRESS_BITS in the table of OpenCL Device Queries for clGetDeviceInfo. If, for example, CL_DEVICE_ADDRESS_BITS = 32, i.e. the device uses a 32-bit address space, size_t is a 32-bit unsigned integer and global_work_size values must be in the range 1 .. 2^32 - 1. Values outside this range return a CL_OUT_OF_RESOURCES error.

Local work size
Points to an array of work_dim unsigned values that describe the number of work-items that make up a work-group (also referred to as the size of the work-group) that will execute the kernel specified by kernel. The total number of work-items in a work-group is computed as local_work_size[0] *... * local_work_size[work_dim - 1]. The total number of work-items in the work-group must be less than or equal to the CL_DEVICE_MAX_WORK_GROUP_SIZE value specified in table of OpenCL Device Queries for clGetDeviceInfo and the number of work-items specified in local_work_size[0],... local_work_size[work_dim - 1] must be less than or equal to the corresponding values specified by CL_DEVICE_MAX_WORK_ITEM_SIZES[0],.... CL_DEVICE_MAX_WORK_ITEM_SIZES[work_dim - 1]. The explicitly specified local_work_size will be used to determine how to break the global work-items specified by global_work_size into appropriate work-group instances. If local_work_size is specified, the values specified in global_work_size[0],... global_work_size[work_dim - 1] must be evenly divisable by the corresponding values specified in local_work_size[0],... local_work_size[work_dim - 1].

In effect, these parameters describe something similar to the CUDA block sizes.
*/
   localWorkSize[0] = 16;
   localWorkSize[1] = 16;
   globalWorkSize[0] = 1024;
   globalWorkSize[1] = 1024;
 
   err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, globalWorkSize, localWorkSize, 
   0, NULL, NULL);

   if (err != CL_SUCCESS)
   {
       printf("Error: Failed to execute kernel! %d\n", err);
       exit(1);
   }

//After execution of the kernel the clEnqueueReadBuffer is used to read the result memory on the device and copy it to the memory on the host.

   //Retrieve result from device
   err = clEnqueueReadBuffer(commands, d_C, CL_TRUE, 0, mem_size_C, h_C, 0, NULL, NULL);

   if (err != CL_SUCCESS)
   {
       printf("Error: Failed to read output array! %d\n", err);
       exit(1);
   }

/*GPU code
The OpenCL kernel is very similar in structure to a CUDA kernel, with some small differences. The external memory is described with __global and shared memory is described with __local, whereas this would be called shared memory in CUDA. Additionally a similar structure to CUDA is used for determining the thread id. This can be done via the get_global_id function which works for multiple dimensions. The return values of this function can be used to determine the matrix location to read for calculation. Due to the similar structure between CUDA and OpenCL many of the optimizations described in the CUDA matrix multiplication example can be applied to the OpenCL version without too many modifications.
*/

/* kernel.cl 
 * Matrix multiplication: C = A * B.
 * Device code.
 */
 
// OpenCL Kernel
__kernel void
matrixMul(__global float* C, 
          __global float* A, 
          __global float* B, 
          int wA, int wB)
{
  
   int tx = get_global_id(0); 
   int ty = get_global_id(1);
 
   // value stores the element that is 
   // computed by the thread
   float value = 0;
   for (int k = 0; k < wA; ++k)
   {
      float elementA = A[ty * wA + k];
      float elementB = B[k * wB + tx];
      value += elementA * elementB;
   }
 
   // Write the matrix to device memory each 
   // thread writes one element
   C[ty * wA + tx] = value;
}
