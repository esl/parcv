#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <errno.h>
#include <CL/cl.h>
#include "lodepng.h"
#include "erl_nif.h"

//------------------------------------------------------------------------------
#define PI 3.14159265359

typedef unsigned char byte;
typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;

//------------------------------------------------------------------------------
void write_png(const char* filename, const byte* pixels, int src_w, int src_h);
void draw_circle(byte* output, byte pixel, int x_center, int y_center, int r, int w);

//------------------------------------------------------------------------------
static ErlNifResourceType* image_r = NULL;

static void 
image_cleanup (ErlNifEnv* e, void* arg) {
  free(arg);
}

static int
load(ErlNifEnv* e, void** priv, ERL_NIF_TERM load_info) {
  ErlNifResourceFlags flags = ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER;
  image_r = enif_open_resource_type(e, "imgproc_nif", "image", 
				    &image_cleanup, flags, 0);
  return 0;
}

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

int check_errors(const char* label, int err) 
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
int initialize(const char* src_path)
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
  check_errors("clCreateProgramWithSource", err);
  free(programBuffer);
  
  err = clBuildProgram(prg, 1, &dev, 0, 0, 0);
  check_errors("clBuildProgram", err);

  char buf[1024];
  int ret;
  clGetProgramBuildInfo(prg, dev, CL_PROGRAM_BUILD_LOG, 1024, buf, &ret);

  printf("ret = %d\n", ret);
  for (i = 0; i < 1024; i++)
    printf("%c", buf[i]);
  printf("\n");

  return 0;
}

int teardown()
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
  check_errors("clCreateCommandQueue", err);

  int nm = src_w * src_h;
  size_t src_sz = nm * sizeof(byte);
  size_t filter_sz = 3 * 3 * sizeof(int);
  size_t output_sz = nm * sizeof(int);

  printf("[canny-pass-1] Allocating memory on the Parallela\n");

  cl_mem src_buf = clCreateBuffer(ctx, CL_MEM_USE_HOST_PTR, src_sz, src, &err);
  check_errors("clCreateBuffer", err);
  cl_mem filter_buf = clCreateBuffer(ctx, CL_MEM_USE_HOST_PTR, filter_sz, filter, &err);
  check_errors("clCreateBuffer", err);
  cl_mem output_buf = clCreateBuffer(ctx, CL_MEM_USE_HOST_PTR, output_sz, output, &err);
  check_errors("clCreateBuffer", err);

  printf("[canny-pass-1] Creating kernel\n");

  krn = clCreateKernel(prg, "convolve_uc_i", &err);
  check_errors("clCreateKernel", err);

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
  check_errors("clEnqueueNDRangeKernel", err);

  printf("[canny-pass-1] Enqueuing read buffers\n");
  printf("[canny-pass-1] dx_sz = %d\n", output_sz);

  clEnqueueReadBuffer(cmdq, output_buf, CL_TRUE, 0, output_sz, output, 0, 0, &ev[1]);

  printf("[canny-pass-1] Waiting on events\n");

  err = clWaitForEvents(2, ev);
  check_errors("clWaitForEvents", err);

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
  check_errors("clCreateCommandQueue", err);
 
  int nm = src_w * src_h;
  size_t dx_sz = nm * sizeof(int);
  size_t dy_sz = nm * sizeof(int);
  size_t mag_sz = nm * sizeof(float);

  printf("[magnitude] Allocating OpenCL buffers\n"); 

  cl_mem dx_buf = clCreateBuffer(ctx, CL_MEM_USE_HOST_PTR, dx_sz, dx, &err);
  check_errors("clCreateBuffer", err);
  cl_mem dy_buf = clCreateBuffer(ctx, CL_MEM_USE_HOST_PTR, dy_sz, dy, &err);
  check_errors("clCreateBuffer", err);
  cl_mem mag_buf = clCreateBuffer(ctx, CL_MEM_USE_HOST_PTR, mag_sz, mag, &err);
  check_errors("clCreateBuffer", err);

  printf("[magnitude] Creating kernel\n"); 
  krn = clCreateKernel(prg, "magnitude", &err);
  check_errors("clCreateKernel", err);

  clSetKernelArg(krn, 0, sizeof(cl_mem), &dx_buf);
  clSetKernelArg(krn, 1, sizeof(cl_mem), &dy_buf);
  clSetKernelArg(krn, 2, sizeof(cl_mem), &mag_buf);
  
  size_t gtdsz[] = { src_w, src_h, 1 };
  size_t ltdsz[] = { 4, 4, 1 };

  cl_event ev[4];

  printf("[magnitude] Enqueueing NDRange \n"); 

  err = clEnqueueNDRangeKernel(cmdq, krn, 2, 0, gtdsz, ltdsz, 0, 0, &ev[0]);
  check_errors("clEnqueueNDRangeKernel", err);

  printf("[magnitude] Enqueueing read buffers\n");

  clEnqueueReadBuffer(cmdq, mag_buf, CL_TRUE, 0, mag_sz, mag, 0, 0, &ev[1]);

  printf("[magnitude] Waiting on events \n"); 

  err = clWaitForEvents(2, ev);
  check_errors("clEnqueueNDRangeKernel", err);

  clReleaseEvent(ev[1]);
  clReleaseEvent(ev[0]);
  clReleaseKernel(krn);
  clReleaseCommandQueue(cmdq);

  return 0;
}

//------------------------------------------------------------------------------0
// Normalised output
//------------------------------------------------------------------------------
int normalize_output (uchar* input, float* mag, float low_thresh, float high_thresh,
		      uchar* output, int src_w, int src_h)
{
  int err;

  cmdq = clCreateCommandQueue(ctx, dev, 0, &err);
  check_errors("clCreateCommandQueue", err);
 
  int nm = src_w * src_h;
  size_t input_sz = nm * sizeof(byte);
  size_t mag_sz = nm * sizeof(float);
  size_t output_sz = nm * sizeof(byte);

  printf("[normalize_output] Allocating OpenCL buffers\n"); 

  cl_mem input_buf = clCreateBuffer(ctx, CL_MEM_USE_HOST_PTR, input_sz, input, &err);
  check_errors("clCreateBuffer", err);
  cl_mem mag_buf = clCreateBuffer(ctx, CL_MEM_USE_HOST_PTR, mag_sz, mag, &err);
  check_errors("clCreateBuffer", err);
  cl_mem output_buf = clCreateBuffer(ctx, CL_MEM_USE_HOST_PTR, output_sz, output, &err);
  check_errors("clCreateBuffer", err);

  printf("[normalize_output] Creating kernel\n"); 
  krn = clCreateKernel(prg, "normalized_output", &err);
  check_errors("clCreateKernel", err);

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
  check_errors("clEnqueueNDRangeKernel", err);

  printf("[normalize_output] Enqueueing read buffers\n");

  clEnqueueReadBuffer(cmdq, output_buf, CL_TRUE, 0, output_sz, output, 0, 0, &ev[1]);

  printf("[normalize_output] Waiting on events \n"); 

  err = clWaitForEvents(2, ev);
  check_errors("clEnqueueNDRangeKernel", err);

  clReleaseEvent(ev[1]);
  clReleaseEvent(ev[0]);
  clReleaseKernel(krn);
  clReleaseCommandQueue(cmdq);

  return 0;
}

int accumulate(byte* pixels, int* acc, int img_w, int img_h)
{
  int err;

  cmdq = clCreateCommandQueue(ctx, dev, 0, &err);
  check_errors("clCreateCommandQueue", err);
 
  int nm = img_w * img_h;
  size_t pixels_sz = nm * sizeof(byte);
  size_t acc_sz = nm * sizeof(int);

  printf("[build_accumulator] Allocating OpenCL buffers\n"); 

  cl_mem pixels_buf = clCreateBuffer(ctx, CL_MEM_USE_HOST_PTR, pixels_sz, pixels, &err);
  check_errors("clCreateBuffer", err);
  cl_mem acc_buf = clCreateBuffer(ctx, CL_MEM_USE_HOST_PTR, acc_sz, acc, &err);
  check_errors("clCreateBuffer", err);

  printf("[build_accumulator] Creating kernel\n"); 
  krn = clCreateKernel(prg, "accumulate", &err);
  check_errors("clCreateKernel", err);

  clSetKernelArg(krn, 0, sizeof(cl_mem), &pixels_buf);
  clSetKernelArg(krn, 1, sizeof(cl_mem), &acc_buf);
  //  clSetKernelArg(krn, 2, sizeof(int), &img_w);
  //  clSetKernelArg(krn, 3, sizeof(int), &img_h);
  
  size_t gtdsz[] = { img_w, img_h, 1 };
  size_t ltdsz[] = { 4, 4, 1 };

  cl_event ev[4];

  printf("[build_accumulator] Enqueueing NDRange \n"); 

  err = clEnqueueNDRangeKernel(cmdq, krn, 2, 0, gtdsz, ltdsz, 0, 0, &ev[0]);
  check_errors("clEnqueueNDRangeKernel", err);

  printf("[build_accumulator] Enqueueing read buffers\n");

  clEnqueueReadBuffer(cmdq, acc_buf, CL_TRUE, 0, acc_sz, acc, 0, 0, &ev[1]);

  printf("[build_accumulator] Waiting on events \n"); 

  err = clWaitForEvents(2, ev);
  check_errors("clEnqueueNDRangeKernel", err);

  clReleaseEvent(ev[1]);
  clReleaseEvent(ev[0]);
  clReleaseKernel(krn);
  clReleaseCommandQueue(cmdq);

  return 0;
}

int normalise(int* acc, float max, int img_w, int img_h)
{
  int err;

  cmdq = clCreateCommandQueue(ctx, dev, 0, &err);
  check_errors("clCreateCommandQueue", err);
 
  int nm = img_w * img_h;
  size_t pixels_sz = nm * sizeof(byte);
  size_t acc_sz = nm * sizeof(int);

  printf("[normalise_accumulator] Allocating OpenCL buffers\n"); 

  cl_mem acc_buf = clCreateBuffer(ctx, CL_MEM_USE_HOST_PTR, acc_sz, acc, &err);
  check_errors("clCreateBuffer", err);

  printf("[normalise_accumulator] Creating kernel\n"); 
  krn = clCreateKernel(prg, "accumulate", &err);
  check_errors("clCreateKernel", err);

  clSetKernelArg(krn, 0, sizeof(cl_mem), &acc_buf);
  clSetKernelArg(krn, 1, sizeof(float), &max);
  clSetKernelArg(krn, 2, sizeof(int), &img_w);
  clSetKernelArg(krn, 3, sizeof(int), &img_h);
  
  size_t gtdsz[] = { img_w, img_h, 1 };
  size_t ltdsz[] = { 4, 4, 1 };

  cl_event ev[4];

  printf("[normalise_accumulator] Enqueueing NDRange \n"); 

  err = clEnqueueNDRangeKernel(cmdq, krn, 2, 0, gtdsz, ltdsz, 0, 0, &ev[0]);
  check_errors("clEnqueueNDRangeKernel", err);

  printf("[normalise_accumulator] Enqueueing read buffers\n");

  clEnqueueReadBuffer(cmdq, acc_buf, CL_TRUE, 0, acc_sz, acc, 0, 0, &ev[1]);

  printf("[normalise_accumulator] Waiting on events \n"); 

  err = clWaitForEvents(2, ev);
  check_errors("clEnqueueNDRangeKernel", err);

  clReleaseEvent(ev[1]);
  clReleaseEvent(ev[0]);
  clReleaseKernel(krn);
  clReleaseCommandQueue(cmdq);

  return 0;
}

int transform(byte* img, int img_w, int img_h) {
  int x, y, i, j;
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
  
  normalize_output(img, mag, low_thresh, high_thresh, output, img_w, img_h);

  write_png("norm.png", output, img_w, img_h);

  free(dx);
  free(dy);
  free(mag);
  free(img);

  //----------------------------------------------------------------------------
  // Hough Transform
  //----------------------------------------------------------------------------
  /* int* acc = (int*) calloc(nm, sizeof(int)); */

  /* const float r = 1.0; */

  /* printf("Accumulating values\n"); */
  
  /* int theta;  */
  /* for (x = 0; x < img_w; x++) { */
  /*   for (y = 0; y < img_h; y++) {  */
  /*     if (output[x + y * img_w] == 255) { */
  /* 	for (theta = 0; theta < 360; theta++) {  */
  /* 	  const float t = (theta * PI) / 180; */
  /* 	  const int x0 = (int) round(x - r * cos(t)); */
  /* 	  const int y0 = (int) round(y - r * sin(t)); */
  /* 	  if (x0 < img_w && x0 > 0 && y0 < img_h && y0 > 0) { */
  /* 	    acc[x0 + (y0 * img_w)] += 1; */
  /* 	  } */
  /* 	} */
  /*     } */
  /*   } */
  /* } */

  /* printf("Finding max\n"); */

  /* // find max acc value */
  /* int max = 0; */
  /* for (x = 0; x < img_w; x++) { */
  /*   for (y = 0; y < img_h; y++) { */
  /*     if (acc[x + y * img_w] > max) { */
  /* 	max = acc[x + y * img_w]; */
  /*     } */
  /*   } */
  /* } */

  /* printf("Max = %d\n", max); */

  /* // normalize  */
  /* for (x = 0; x < img_w; x++) { */
  /*   for (y = 0; y < img_h; y++) { */
  /*     const int value = (int) (acc[x + y * img_w] / max) * 255.0f; */
  /*     acc[x + y * img_w] = 0xFF000000 | (value << 16 | value << 8 | value); */
  /*   } */
  /* } */

  /* printf("Values normalized\n"); */

  /* // find maxima */
  /* int acc_sz = nm; */
  /* int* results = (int*) calloc(acc_sz * 3, sizeof(int)); */
  
  /* printf("Finding maxima\n"); */

  /* for (x = 0; x < img_w; x++) { */
  /*   for (y = 0; y < img_h; y++) { */
  /*     int value = acc[x + y * img_w] & 0xFF; */
  /*     if (value > results[(acc_sz-1) * 3]) { */
  /* 	results[(acc_sz-1)*3] = value; */
  /* 	results[(acc_sz-1)*3+1] = x; */
  /* 	results[(acc_sz-1)*3+2] = y; */

  /* 	i = (acc_sz-2)*3; */
  /* 	while ((i >= 0) && (results[i+3] > results[i])) { */
  /* 	  for (j = 0; j < 3; j++) { */
  /* 	    int tmp = results[i+j]; */
  /* 	    results[i+j] = results[i+3+j]; */
  /* 	    results[i+3+j] = tmp; */
  /* 	  } */
  /* 	  i = i - 3; */
  /* 	  if (i < 0) { */
  /* 	    break; */
  /* 	  } */
  /* 	} */
  /*     } */
  /*   } */
  /* } */

  /* printf("Found maxima, drawing circle\n"); */

  /* for (i = (acc_sz-1); i >= 0; i--) { */
  /*   draw_circle(output, results[i*3], results[i*3+1], results[i*3+2], img_w); */
  /* } */

  /* printf("Circles drawn, writing output\n"); */

  /* write_png("output.png", output, img_w, img_h); */

  return 0;
}

void set_pixel(byte* output, byte value, int x, int y, int w) {
  output[x + y * w] = value;
}

void draw_circle(byte* output, byte pixel, int x_center, int y_center, int r, int w) {
  pixel = 255;
  int x, y, r2;
  r2 = r * r;

  set_pixel(output, pixel, x_center, y_center + r, w);
  set_pixel(output, pixel, x_center, y_center - r, w);
  set_pixel(output, pixel, x_center + r, y_center, w);
  set_pixel(output, pixel, x_center - r, y_center, w);

  y = r;
  x = 1;
  y = (int) (sqrt(r2 - 1) + 0.5);
  while (x < y) {
    set_pixel(output, pixel, x_center + x, y_center + y, w);
    set_pixel(output, pixel, x_center + x, y_center - y, w);
    set_pixel(output, pixel, x_center - x, y_center + y, w);
    set_pixel(output, pixel, x_center - x, y_center - y, w);
    set_pixel(output, pixel, x_center + y, y_center + x, w);
    set_pixel(output, pixel, x_center + y, y_center - x, w);
    set_pixel(output, pixel, x_center - y, y_center + x, w);
    set_pixel(output, pixel, x_center - y, y_center - x, w);
    x += 1;
    y = (int) (sqrt(r2 - x*x) + 0.5);
  }
  if (x == y) {
    set_pixel(output, pixel, x_center + x, y_center + y, w);
    set_pixel(output, pixel, x_center + x, y_center - y, w);
    set_pixel(output, pixel, x_center - x, y_center + y, w);
    set_pixel(output, pixel, x_center - x, y_center - y, w);
  }
}

int read_file(const char* path, int* sz, char* data)
{
  struct stat st;
  if (stat(path, &st) != 0) {
    printf("stat failed: %s\n", strerror(errno));
    return -1;
  } else {
    *sz = st.st_size;
  }

  FILE* in;
  if ((in = fopen(path, "rt")) == NULL) {
    printf("fopen failed: %s\n", strerror(errno));
  }

  printf("reading file of length %d\n", st.st_size);

  data = malloc(sizeof(char) * (st.st_size + 1));
  fread(data, 1, st.st_size, in);
  data[st.st_size] = '\0';

  printf("file read successfully\n");

  fclose(in);
  
  return 0;
}

byte* read_png(const char* filename, int src_w, int src_h) {
  byte* out = (byte*) malloc(sizeof(byte) * src_w * src_h);
  
  int w, h;

  lodepng_decode_file(&out, &w, &h, filename, LCT_GREY, 8);

  printf("w = %d\n", w);
  printf("h = %d\n", h);

  return out;
}

void write_png(const char* filename, const byte* pixels, int src_w, int src_h) {
  lodepng_encode_file(filename, pixels, src_w, src_h, LCT_GREY, 8);
}

//------------------------------------------------------------------------------
// NIF
//------------------------------------------------------------------------------
static ERL_NIF_TERM
clinitialize(ErlNifEnv* e, int argc, const ERL_NIF_TERM argv[]) {
  // get string (main kernel)
  unsigned len;
  if (!enif_get_list_length(e, argv[0], &len)) {
    return enif_make_badarg(e);
  }
  char* str = (char*) malloc(sizeof(char) * len);
  if (!enif_get_string(e, argv[0], str, len, ERL_NIF_LATIN1)) {
    return enif_make_badarg(e);
  }
  str[len-1] = '\0';

  printf("initialising NIF with %s\n", str);

  initialize(str);
  return enif_make_atom(e, "ok");
}

static ERL_NIF_TERM
clteardown(ErlNifEnv* e, int argc, const ERL_NIF_TERM argv[]) {
  teardown();
  return enif_make_atom(e, "ok");
}

static ERL_NIF_TERM
cltransform(ErlNifEnv* e, int argc, const ERL_NIF_TERM argv[]) {
  // get byte* handle
  byte* image;
  if (!enif_get_resource(e, argv[0], image_r, (void**) image)) {
    return enif_make_badarg(e);
  }

  // get image_width
  int image_w, image_h;
  if (!enif_get_int(e, argv[1], &image_w)) {
    return enif_make_badarg(e);
  }

  // get image_height
  if (!enif_get_int(e, argv[2], &image_h)) {
    return enif_make_badarg(e);
  }

  transform(image, image_w, image_h);

  return enif_make_atom(e, "ok");
}

static ERL_NIF_TERM
clread_png(ErlNifEnv* e, int argc, const ERL_NIF_TERM argv[]) {
  // get filename
  unsigned len;
  if (!enif_get_list_length(e, argv[0], &len)) {
    return enif_make_badarg(e);
  }
  char str[len];
  if (!enif_get_string(e, argv[0], str, len, ERL_NIF_LATIN1)) {
    return enif_make_badarg(e);
  }

  // get image_width
  int image_w, image_h;
  if (!enif_get_int(e, argv[1], &image_w)) {
    return enif_make_badarg(e);
  }

  // get image_height
  if (!enif_get_int(e, argv[2], &image_h)) {
    return enif_make_badarg(e);
  }

  byte* image = read_png(str, image_w, image_h);

  // ret byte* handle
  return enif_make_tuple(e, enif_make_resource(e, image));
}

static ERL_NIF_TERM
clwrite_png(ErlNifEnv* e, int argc, const ERL_NIF_TERM argv[]) {
  // get filename
  unsigned len;
  if (!enif_get_list_length(e, argv[0], &len)) {
    return enif_make_badarg(e);
  }
  char buf[len];
  if (!enif_get_string(e, argv[0], buf, len, ERL_NIF_LATIN1)) {
    return enif_make_badarg(e);
  }

  // get byte* handle
  byte* image;
  if (!enif_get_resource(e, argv[1], image_r, (void**) image)) {
    return enif_make_badarg(e);
  }

  // get image_width
  int image_w, image_h;
  if (!enif_get_int(e, argv[2], &image_w)) {
    return enif_make_badarg(e);
  }
  // get image_height
  if (!enif_get_int(e, argv[3], &image_h)) {
    return enif_make_badarg(e);
  }

  write_png(buf, image, image_w, image_h);

  return enif_make_atom(e, "ok");
}

static ErlNifFunc nif_funcs[] = {
  {"clinitialize", 1, clinitialize},
  {"clteardown", 0, clteardown},
  {"cltransform", 3, cltransform},
  {"clread_png", 3, clread_png},
  {"clwrite_png", 4, clwrite_png}
};

ERL_NIF_INIT(imgproc_nif, nif_funcs, load, NULL, NULL, NULL);

/* int main() */
/* { */
/*   int i, j; */

/*   int img_w = 512; */
/*   int img_h = 512; */
/*   byte* img_pixels = read_png("images/lena_512.png", img_w, img_h); */

/*   initialize("kernels/edge_detect.cl"); */

/*   transform(img_pixels, img_w, img_h); */

/*   teardown(); */

/*   return 0; */
/* } */


