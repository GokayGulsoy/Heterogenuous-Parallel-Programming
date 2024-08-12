#include <stdlib.h>
#include <stdio.h>
#include <time.h> // required for srand(unsigned int) function

// defining CUDA kernel
__global__ void parallel_saxpy(int size,float* dev_vec1,float* dev_vec2,float scalar) {
           // determining the thread for execution
           int index = blockIdx.x * blockDim.x + threadIdx.x;

           /* checking if the thread number 
              is less than the size of vectors
           */
           if (index < size) {
              dev_vec2[index] = scalar*dev_vec1[index] + dev_vec2[index];  
           } 
}

// utility function to display vector
void display_vector(float* vec,int size) {
      
    printf("\n");
    printf("[");
    for (int i = 0; i < size ; i++) {
        if (i == 0) { 
           printf("%.2f",vec[i]);
        }

        else {
            printf(",%.2f",vec[i]);
        }
    }
    printf("]"); 
}

// main function
int main() {
    /* printing the device name 
       and maximum number of threads 
       per block
    */

    int device_no;
    cudaGetDevice(&device_no);
 
    cudaDeviceProp device_prop; 
    cudaGetDeviceProperties(&device_prop,device_no);

    printf("Device name is: %s\n",device_prop.name);
    printf("Maximum number of threads per block is: %d\n",device_prop.maxThreadsPerBlock);

    // declaring vectors
    float* vec1;
    float* vec2;

    // declaring size of vectors and scalar
    int vec1_size;
    int vec2_size;
    float scalar;


    /* taking the array sizes 
       and scalar value as an
       input from the user
    */

    printf("Enter size of first vector: ");
    scanf("%d",&vec1_size);
    printf("Enter the size of second vector: ");
    scanf("%d",&vec2_size);
    printf("Enter the value of scalar: ");
    scanf("%f",&scalar);

    // for two vectors to be added
    // they should have same sizes
    if (vec1_size != vec2_size) {
       printf("Sizes of two vectors should be same!!");
       exit(1); 
    }


    // dynamically allocating memory for vectors
    vec1 = (float *)malloc(vec1_size*sizeof(float));
    vec2 = (float *)malloc(vec2_size*sizeof(float));

    // assigning random values to vector entries
    srand(time(0));
    for (int i = 0; i < vec1_size; i++) {
        float random_entry1 = (float)(rand() % 100 + 1);
        float random_entry2 = (float)(rand() % 100 + 1);

        vec1[i] = random_entry1;
        vec2[i] = random_entry2;   
    }
    
    /* 
     printing vec1 and vec2 before
     launching the parallel_saxpy kernel
   */
   
   printf("\nvec1 and vec2 before saxpy kernel is launched: \n");
   printf("vec1 is as follows: ");
   display_vector(vec1,vec1_size);
   printf("\nvec2 is as follows: ");
   display_vector(vec2,vec2_size);

   // Alocate memory on device
   float *dev_vec1, *dev_vec2; 
   cudaMalloc(&dev_vec1,vec1_size*sizeof(float));
   cudaMalloc(&dev_vec2,vec2_size*sizeof(float));

   // copying vectors from CPU to GPU
   cudaMemcpy(dev_vec1,vec1,vec1_size*sizeof(float),cudaMemcpyHostToDevice); 
   cudaMemcpy(dev_vec2,vec2,vec2_size*sizeof(float),cudaMemcpyHostToDevice);

   // Launching CUDA kernels
   // launching with 2 blocks and 32 threads per block
   parallel_saxpy<<<32,256>>>(vec1_size,dev_vec1,dev_vec2,scalar);

   // launching with 4 blocks and 512 threads per block
   // we can uncomment the following kernel and comment the above
   // kernel to see the difference on how long does it take for
   // the saxpy operation on different kernel launch configurations 
   /*parallel_saxpy<<<4,512>>>(vec1_size,dev_vec1,dev_vec2,scalar);*/

   // copying computed vec2 from GPU to CPU     
   cudaMemcpy(vec2,dev_vec2,vec2_size*sizeof(float),cudaMemcpyDeviceToHost); 

   // printing vec2 after saxpy operation is applied
   printf("\n\nvec2 after saxpy kernel is launched: ");
   display_vector(vec2,vec2_size);
   printf("\n");

   // deallocating memory on CPU 
   free(vec1);
   free(vec2);

   // deallocating memory on GPU
   cudaFree(dev_vec1);
   cudaFree(dev_vec2); 

   return 0;
}
