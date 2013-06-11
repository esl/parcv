#include "imgproc.h"

//------------------------------------------------------------------------------
static cl_uint nplatforms;
static cl_uint ndevices;
static cl_platform_id* platforms;
static cl_platform_id  platform;
static cl_device_id* devices;
static cl_device_id dev;
static cl_context ctx;
static cl_command_queue cmdq;
static cl_program prg;
static cl_kernel krn;

//------------------------------------------------------------------------------
int cl_check_errors(const char* label, int err) 
{
  if (err != CL_SUCCESS) {
    switch (err) {
    case CL_INVALID_PROGRAM:
      fprintf(stderr, "OpenCL error (%s): CL_INVALID_PROGRAM\n", label);
      return -1;
    case CL_INVALID_BINARY:
      fprintf(stderr, "OpenCL error (%s): CL_INVALID_BINARY\n", label);
      return -1;
    case CL_BUILD_PROGRAM_FAILURE:
      fprintf(stderr, "OpenCL error (%s): CL_BUILD_PROGRAM_FAILURE\n", label);
      return -1;
    case CL_INVALID_DEVICE:
      fprintf(stderr, "OpenCL error (%s): CL_INVALID_DEVICE\n", label);
      return -1;
    case CL_OUT_OF_HOST_MEMORY:
      fprintf(stderr, "OpenCL error (%s): CL_OUT_OF_HOST_MEMORY\n", label);
      return -1;
    case CL_INVALID_PROGRAM_EXECUTABLE:
      fprintf(stderr, "OpenCL error (%s): CL_INVALID_PROGRAM_EXECUTABLE\n", label);
      return -1;
    case CL_INVALID_COMMAND_QUEUE:
      fprintf(stderr, "OpenCL error (%s): CL_INVALID_COMMAND_QUEUE\n", label);
      return -1;
    case CL_INVALID_KERNEL:
      fprintf(stderr, "OpenCL error (%s): CL_INVALID_KERNEL\n", label);
      return -1;
    case CL_INVALID_CONTEXT:
      fprintf(stderr, "OpenCL error (%s): CL_INVALID_CONTEXT\n", label);
      return -1;
    case CL_INVALID_KERNEL_ARGS:
      fprintf(stderr, "OpenCL error (%s): CL_INVALID_KERNEL_ARGS\n", label);
      return -1;
    case CL_INVALID_WORK_DIMENSION:
      fprintf(stderr, "OpenCL error (%s): CL_INVALID_WORK_DIMENSION\n", label);
      return -1;
    case CL_INVALID_WORK_GROUP_SIZE:
      fprintf(stderr, "OpenCL error (%s): CL_INVALID_WORK_GROUP_SIZE\n", label);
      return -1;
    case CL_INVALID_WORK_ITEM_SIZE:
      fprintf(stderr, "OpenCL error (%s): CL_INVALID_WORK_ITEM_SIZE\n", label);
      return -1;
    case CL_INVALID_GLOBAL_OFFSET:
      fprintf(stderr, "OpenCL error (%s): CL_INVALID_GLOBAL_OFFSET\n", label);
      return -1;
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
      fprintf(stderr, "OpenCL error (%s): CL_MEM_OBJECT_ALLOCATION_FAILURE\n", label);
      return -1;
    case CL_INVALID_VALUE:
      fprintf(stderr, "OpenCL error (%s): CL_INVALID_VALUE\n", label);
      return -1;
    default:
      fprintf(stderr, "OpenCL error (%s): UNKNOWN\n", label);
      return -1;
    }
  }
  return 0;
}

//------------------------------------------------------------------------------
int cl_initialize(const char* src_path)
{
  int i, err;

  // Initialize platform
  clGetPlatformIDs(0,0,&nplatforms);
  platforms = (cl_platform_id*) malloc(nplatforms*sizeof(cl_platform_id));
  clGetPlatformIDs(nplatforms, platforms, 0);
  
  char buffer[256];
  for (i = 0; i < nplatforms; i++) {
    platform = platforms[i];
    clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 256, buffer, 0);
    if (!strcmp(buffer, "coprthr-e"))
      break;
  }

  if (i < nplatforms)
    platform = platforms[i];
  else
    exit(1);

  // Initialize devices
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_ACCELERATOR, 0, 0, &ndevices);
  devices = (cl_device_id*) malloc(ndevices * sizeof(cl_device_id));
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_ACCELERATOR, ndevices, devices, 0);
  
  if (ndevices)
    dev = devices[0];
  else
    exit(1);

  size_t sizes[3];
  clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t)*3, &sizes, NULL);
  printf("%d, %d, %d\n", sizes[0], sizes[1], sizes[2]);

  // Initialize context
  cl_context_properties ctxprop[3] = {
    (cl_context_properties) CL_CONTEXT_PLATFORM,
    (cl_context_properties) platform,
    (cl_context_properties) 0
  };

  ctx = clCreateContext(ctxprop, 1, &dev, 0, 0, &err);

  FILE* programHandle = fopen(src_path, "r");
  fseek(programHandle, 0, SEEK_END);
  size_t programSize = ftell(programHandle);
  rewind(programHandle);

  char* programBuffer; 
  char* kernelSource;
  programBuffer = (char*) malloc(programSize + 1);
  programBuffer[programSize] = '\0';
  fread(programBuffer, sizeof(char), programSize, programHandle);
  fclose(programHandle);

  prg = clCreateProgramWithSource(ctx, 1, (const char**) &programBuffer, &programSize, &err);
  cl_check_errors("clCreateProgramWithSource", err);
  free(programBuffer);
  
  err = clBuildProgram(prg, 1, &dev, 0, 0, 0);
  cl_check_errors("clBuildProgram", err);

  char buf[1024];
  int ret;
  clGetProgramBuildInfo(prg, dev, CL_PROGRAM_BUILD_LOG, 1024, buf, &ret);

  printf("ret = %d\n", ret);
  for (i = 0; i < 1024; i++)
    printf("%c", buf[i]);
  printf("\n");

  return 0;
}

int cl_teardown()
{
  clReleaseProgram(prg);
  clReleaseContext(ctx);

  free(devices);
  free(platforms);

  return 0;
}

static int sobel_convolution_in_x[] = { -1, 0, 1, -2, 0, 2, -1, 0, 1  };
static int sobel_convolution_in_y[] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };

//------------------------------------------------------------------------------
// Sobel row pass
//------------------------------------------------------------------------------
int convolve(byte* src, int* filter, int* output, int src_w, int src_h, int filter_w) 
{
  int err;
  cmdq = clCreateCommandQueue(ctx, dev, 0, &err);
  cl_check_errors("clCreateCommandQueue", err);

  int nm = src_w * src_h;
  size_t src_sz = nm * sizeof(byte);
  size_t filter_sz = 3 * 3 * sizeof(int);
  size_t output_sz = nm * sizeof(int);

  printf("[canny-pass-1] Allocating memory on the Parallela\n");

  cl_mem src_buf = clCreateBuffer(ctx, CL_MEM_USE_HOST_PTR, src_sz, src, &err);
  cl_check_errors("clCreateBuffer", err);
  cl_mem filter_buf = clCreateBuffer(ctx, CL_MEM_USE_HOST_PTR, filter_sz, filter, &err);
  cl_check_errors("clCreateBuffer", err);
  cl_mem output_buf = clCreateBuffer(ctx, CL_MEM_USE_HOST_PTR, output_sz, output, &err);
  cl_check_errors("clCreateBuffer", err);

  printf("[canny-pass-1] Creating kernel\n");

  krn = clCreateKernel(prg, "convolve_uc_i", &err);
  cl_check_errors("clCreateKernel", err);

  int step = 1;
  int offset = 0;

  printf("[canny-pass-1] Setting kernel args\n");

  clSetKernelArg(krn, 0, sizeof(cl_mem), &src_buf);
  clSetKernelArg(krn, 1, sizeof(cl_mem), &filter_buf);
  clSetKernelArg(krn, 2, sizeof(cl_mem), &output_buf);
  clSetKernelArg(krn, 3, sizeof(int), &src_w);
  clSetKernelArg(krn, 4, sizeof(int), &filter_w);
  
  printf("[canny-pass-1] Initiating NDRange query\n");

  size_t gtdsz[] = { src_w, src_h, 1 };
  size_t ltdsz[] = { 4, 4, 1 };

  cl_event ev[3];

  err = clEnqueueNDRangeKernel(cmdq, krn, 2, 0, gtdsz, ltdsz, 0, NULL, &ev[0]);
  cl_check_errors("clEnqueueNDRangeKernel", err);

  printf("[canny-pass-1] Enqueuing read buffers\n");
  printf("[canny-pass-1] dx_sz = %d\n", output_sz);

  clEnqueueReadBuffer(cmdq, output_buf, CL_TRUE, 0, output_sz, output, 0, 0, &ev[1]);

  printf("[canny-pass-1] Waiting on events\n");

  err = clWaitForEvents(2, ev);
  cl_check_errors("clWaitForEvents", err);

  printf("[canny-pass-1] Completed canny pass, releasing events\n");

  clReleaseEvent(ev[1]);
  clReleaseEvent(ev[0]);

  clReleaseKernel(krn);
  clReleaseCommandQueue(cmdq);

  return 0;
}

//------------------------------------------------------------------------------
// magnitude
//------------------------------------------------------------------------------
int magnitude(int* dx, int* dy, float* mag, int src_w, int src_h)
{
  int err;

  cmdq = clCreateCommandQueue(ctx, dev, 0, &err);
  cl_check_errors("clCreateCommandQueue", err);
 
  int nm = src_w * src_h;
  size_t dx_sz = nm * sizeof(int);
  size_t dy_sz = nm * sizeof(int);
  size_t mag_sz = nm * sizeof(float);

  printf("[magnitude] Allocating OpenCL buffers\n"); 

  cl_mem dx_buf = clCreateBuffer(ctx, CL_MEM_USE_HOST_PTR, dx_sz, dx, &err);
  cl_check_errors("clCreateBuffer", err);
  cl_mem dy_buf = clCreateBuffer(ctx, CL_MEM_USE_HOST_PTR, dy_sz, dy, &err);
  cl_check_errors("clCreateBuffer", err);
  cl_mem mag_buf = clCreateBuffer(ctx, CL_MEM_USE_HOST_PTR, mag_sz, mag, &err);
  cl_check_errors("clCreateBuffer", err);

  printf("[magnitude] Creating kernel\n"); 
  krn = clCreateKernel(prg, "magnitude", &err);
  cl_check_errors("clCreateKernel", err);

  clSetKernelArg(krn, 0, sizeof(cl_mem), &dx_buf);
  clSetKernelArg(krn, 1, sizeof(cl_mem), &dy_buf);
  clSetKernelArg(krn, 2, sizeof(cl_mem), &mag_buf);
  
  size_t gtdsz[] = { src_w, src_h, 1 };
  size_t ltdsz[] = { 4, 4, 1 };

  cl_event ev[4];

  printf("[magnitude] Enqueueing NDRange \n"); 

  err = clEnqueueNDRangeKernel(cmdq, krn, 2, 0, gtdsz, ltdsz, 0, 0, &ev[0]);
  cl_check_errors("clEnqueueNDRangeKernel", err);

  printf("[magnitude] Enqueueing read buffers\n");

  clEnqueueReadBuffer(cmdq, mag_buf, CL_TRUE, 0, mag_sz, mag, 0, 0, &ev[1]);

  printf("[magnitude] Waiting on events \n"); 

  err = clWaitForEvents(2, ev);
  cl_check_errors("clEnqueueNDRangeKernel", err);

  clReleaseEvent(ev[1]);
  clReleaseEvent(ev[0]);
  clReleaseKernel(krn);
  clReleaseCommandQueue(cmdq);

  return 0;
}

//------------------------------------------------------------------------------0
// Normalised output
//------------------------------------------------------------------------------
int normalize(uchar* input, float* mag, float low_thresh, float high_thresh,
	      uchar* output, int src_w, int src_h)
{
  int err;

  cmdq = clCreateCommandQueue(ctx, dev, 0, &err);
  cl_check_errors("clCreateCommandQueue", err);
 
  int nm = src_w * src_h;
  size_t input_sz = nm * sizeof(byte);
  size_t mag_sz = nm * sizeof(float);
  size_t output_sz = nm * sizeof(byte);

  printf("[normalize_output] Allocating OpenCL buffers\n"); 

  cl_mem input_buf = clCreateBuffer(ctx, CL_MEM_USE_HOST_PTR, input_sz, input, &err);
  cl_check_errors("clCreateBuffer", err);
  cl_mem mag_buf = clCreateBuffer(ctx, CL_MEM_USE_HOST_PTR, mag_sz, mag, &err);
  cl_check_errors("clCreateBuffer", err);
  cl_mem output_buf = clCreateBuffer(ctx, CL_MEM_USE_HOST_PTR, output_sz, output, &err);
  cl_check_errors("clCreateBuffer", err);

  printf("[normalize_output] Creating kernel\n"); 
  krn = clCreateKernel(prg, "normalized_output", &err);
  cl_check_errors("clCreateKernel", err);

  clSetKernelArg(krn, 0, sizeof(cl_mem), &input_buf);
  clSetKernelArg(krn, 1, sizeof(cl_mem), &mag_buf);
  clSetKernelArg(krn, 2, sizeof(cl_mem), &low_thresh);
  clSetKernelArg(krn, 3, sizeof(cl_mem), &high_thresh);
  clSetKernelArg(krn, 4, sizeof(cl_mem), &output_buf);
  
  size_t gtdsz[] = { src_w, src_h, 1 };
  size_t ltdsz[] = { 4, 4, 1 };

  cl_event ev[4];

  printf("[normalize_output] Enqueueing NDRange \n"); 

  err = clEnqueueNDRangeKernel(cmdq, krn, 2, 0, gtdsz, ltdsz, 0, 0, &ev[0]);
  cl_check_errors("clEnqueueNDRangeKernel", err);

  printf("[normalize_output] Enqueueing read buffers\n");

  clEnqueueReadBuffer(cmdq, output_buf, CL_TRUE, 0, output_sz, output, 0, 0, &ev[1]);

  printf("[normalize_output] Waiting on events \n"); 

  err = clWaitForEvents(2, ev);
  cl_check_errors("clEnqueueNDRangeKernel", err);

  clReleaseEvent(ev[1]);
  clReleaseEvent(ev[0]);
  clReleaseKernel(krn);
  clReleaseCommandQueue(cmdq);

  return 0;
}

int cl_transform(byte* img, int img_w, int img_h) {
  int i, j;
  int nm = img_w * img_h;

  //----------------------------------------------------------------------------
  // Edge Detection
  //----------------------------------------------------------------------------

  // Convolve a sobel filter over the image for both x and y directions
  int* dx = (int*) calloc(nm, sizeof(int));
  int* dy = (int*) calloc(nm, sizeof(int));
  convolve(img, sobel_convolution_in_x, dx, img_w, img_h, 3);
  convolve(img, sobel_convolution_in_y, dy, img_w, img_h, 3);

  // Calculate pixel magnitude
  float* mag = (float*) malloc(img_w * img_h * sizeof(float));

  magnitude(dx, dy, mag, img_w, img_h);
  
  FILE* magnitude_out = fopen("magnitude.out", "wt");
  for (i = 0; i < nm; i++)
    fprintf(magnitude_out, "mag[%d] = %f\n", i, mag[i]);

  // Normalise mag & write image
  byte* output = (byte*) malloc(nm * sizeof(byte));

  float low_thresh = 5.0f;
  float high_thresh = 15.0f;
  
  normalize(img, mag, low_thresh, high_thresh, output, img_w, img_h);

  png_write("norm.png", output, img_w, img_h);

  free(dx);
  free(dy);
  free(mag);
  free(img);

  return 0;
}

byte* png_read(const char* filename, int src_w, int src_h) {
  byte* out = (byte*) malloc(sizeof(byte) * src_w * src_h);
  
  int w, h;

  lodepng_decode_file(&out, &w, &h, filename, LCT_GREY, 8);

  printf("w = %d\n", w);
  printf("h = %d\n", h);

  return out;
}

void png_write(const char* filename, const byte* pixels, int src_w, int src_h) {
  lodepng_encode_file(filename, pixels, src_w, src_h, LCT_GREY, 8);
}
