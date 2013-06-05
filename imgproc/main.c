#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <errno.h>
#include <CL/cl.h>

#include "lodepng.h"

typedef unsigned char byte;
typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;

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

  // Initialize program
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
// calcMagnitude
//------------------------------------------------------------------------------
int magnitude(const int* dx, const int* dy, float* mag, int src_w, int src_h)
{
  int err;

  cmdq = clCreateCommandQueue(ctx, dev, 0, &err);
  check_errors("clCreateCommandQueue", err);
 
  int nm = src_w * src_h;
  size_t dx_sz = nm * sizeof(int);
  size_t dy_sz = nm * sizeof(int);
  size_t mag_sz = nm * sizeof(float);

  printf("[calcMagnitude] Allocating OpenCL buffers\n"); 

  cl_mem dx_buf = clCreateBuffer(ctx, CL_MEM_USE_HOST_PTR, dx_sz, dx, &err);
  check_errors("clCreateBuffer", err);
  cl_mem dy_buf = clCreateBuffer(ctx, CL_MEM_USE_HOST_PTR, dy_sz, dy, &err);
  check_errors("clCreateBuffer", err);
  cl_mem mag_buf = clCreateBuffer(ctx, CL_MEM_USE_HOST_PTR, mag_sz, mag, &err);
  check_errors("clCreateBuffer", err);

  printf("[calcMagnitude] Creating kernel\n"); 
  krn = clCreateKernel(prg, "magnitude", &err);
  check_errors("clCreateKernel", err);

  clSetKernelArg(krn, 0, sizeof(cl_mem), &dx_buf);
  clSetKernelArg(krn, 1, sizeof(cl_mem), &dy_buf);
  clSetKernelArg(krn, 2, sizeof(cl_mem), &mag_buf);
  
  size_t gtdsz[] = { src_w, src_h, 1 };
  size_t ltdsz[] = { 4, 4, 1 };

  cl_event ev[4];

  printf("[calcMagnitude] Enqueueing NDRange \n"); 

  err = clEnqueueNDRangeKernel(cmdq, krn, 2, 0, gtdsz, ltdsz, 0, 0, &ev[0]);
  check_errors("clEnqueueNDRangeKernel", err);

  printf("[calcMagnitude] Enqueueing read buffers\n");

  clEnqueueReadBuffer(cmdq, mag_buf, CL_TRUE, 0, mag_sz, mag, 0, 0, &ev[1]);

  printf("[calcMagnitude] Waiting on events \n"); 

  err = clWaitForEvents(2, ev);
  check_errors("clEnqueueNDRangeKernel", err);

  clReleaseEvent(ev[1]);
  clReleaseEvent(ev[0]);
  clReleaseKernel(krn);
  clReleaseCommandQueue(cmdq);

  return 0;
}

int edgeDir(int* dx, int* dy, int* dir, int src_w, int src_h)
{
  int err;

  cmdq = clCreateCommandQueue(ctx, dev, 0, &err);
  check_errors("clCreateCommandQueue", err);

  int nm = src_w * src_h;
  size_t dx_sz = nm * sizeof(int);
  size_t dy_sz = nm * sizeof(float);
  size_t dir_sz = nm * sizeof(byte);
  
  printf("[edgeDir] Allocating OpenCL buffers\n");

  cl_mem dx_buf = clCreateBuffer(ctx, CL_MEM_USE_HOST_PTR, dx_sz, dx, &err);
  check_errors("clCreateBuffer", err);
  cl_mem dy_buf = clCreateBuffer(ctx, CL_MEM_USE_HOST_PTR, dy_sz, dy, &err);
  check_errors("clCreateBuffer", err);
  cl_mem dir_buf = clCreateBuffer(ctx, CL_MEM_USE_HOST_PTR, dir_sz, dir, &err);
  check_errors("clCreateBuffer", err);

  printf("[edgeDir] Creating kernel\n");

  krn = clCreateKernel(prg, "edgeDir", &err);
  check_errors("clCreateKernel", err);

  clSetKernelArg(krn, 0, sizeof(cl_mem), &dx_buf);
  clSetKernelArg(krn, 1, sizeof(cl_mem), &dy_buf);
  clSetKernelArg(krn, 2, sizeof(cl_mem), &dir_buf);

  size_t gtdsz[] = { src_w, src_h, 1 };
  size_t ltdsz[] = { 4, 4, 1 };

  cl_event ev[2];

  printf("[edgeDir] Enqueueing NDRange\n");

  clEnqueueNDRangeKernel(cmdq, krn, 2, 0, gtdsz, ltdsz, 0, 0, &ev[0]);
  clEnqueueReadBuffer(cmdq, dir_buf, CL_TRUE, 0, dir_sz, dir, 0, 0, &ev[1]);

  printf("[edgeDir] Waiting on events\n");

  err = clWaitForEvents(2, ev);

  clReleaseEvent(ev[1]);
  clReleaseEvent(ev[0]);
  clReleaseKernel(krn);
  clReleaseCommandQueue(cmdq);

  return 0;
}

int edgeMap(float* mag, int* dir, int* map, float low_thresh, float high_thresh, 
	    int src_w, int src_h)
{
  int err;

  cmdq = clCreateCommandQueue(ctx, dev, 0, &err);
  check_errors("clCreateCommandQueue", err);

  int nm = src_w * src_h;
  size_t mag_sz = nm * sizeof(float);
  size_t dir_sz = nm * sizeof(int);
  size_t map_sz = nm * sizeof(int);
  
  printf("[edgeMap] Allocating OpenCL buffers\n");

  cl_mem mag_buf = clCreateBuffer(ctx, CL_MEM_USE_HOST_PTR, mag_sz, mag, &err);
  check_errors("clCreateBuffer", err);
  cl_mem dir_buf = clCreateBuffer(ctx, CL_MEM_USE_HOST_PTR, dir_sz, dir, &err);
  check_errors("clCreateBuffer", err);
  cl_mem map_buf = clCreateBuffer(ctx, CL_MEM_USE_HOST_PTR, map_sz, map, &err);
  check_errors("clCreateBuffer", err);

  printf("[edgeMap] Creating kernel\n");

  krn = clCreateKernel(prg, "edgeMap", &err);
  check_errors("clCreateKernel", err);

  clSetKernelArg(krn, 0, sizeof(cl_mem), &mag_buf);
  clSetKernelArg(krn, 1, sizeof(cl_mem), &dir_buf);
  clSetKernelArg(krn, 2, sizeof(cl_mem), &map_buf);
  clSetKernelArg(krn, 3, sizeof(float), &low_thresh);
  clSetKernelArg(krn, 4, sizeof(float), &high_thresh);

  size_t gtdsz[] = { src_w, src_h, 1 };
  size_t ltdsz[] = { 4, 4, 1 };

  cl_event ev[2];

  printf("[calcMap] Enqueueing NDRange\n");

  clEnqueueNDRangeKernel(cmdq, krn, 2, 0, gtdsz, ltdsz, 0, 0, &ev[0]);
  clEnqueueReadBuffer(cmdq, map_buf, CL_TRUE, 0, map_sz, map, 0, 0, &ev[1]);

  printf("[calcMap] Waiting on events\n");

  err = clWaitForEvents(2, ev);

  clReleaseEvent(ev[1]);
  clReleaseEvent(ev[0]);
  clReleaseKernel(krn);
  clReleaseCommandQueue(cmdq);

  return 0;
}

int edgesHysteresis(int* map, ushort* st, uint* counter, int img_w, int img_h)
{
  int err;

  cmdq = clCreateCommandQueue(ctx, dev, 0, &err);
  check_errors("clCreateCommandQueue", err);

  int nm = img_w * img_h;
  size_t map_sz = nm * sizeof(int);
  size_t st_sz = nm * (sizeof(ushort) * 2);
  size_t counter_sz = nm * sizeof(uint);

  cl_mem map_buf = clCreateBuffer(ctx, CL_MEM_USE_HOST_PTR, map_sz, map, &err);
  check_errors("clCreateBuffer", err);
  cl_mem st_buf = clCreateBuffer(ctx, CL_MEM_USE_HOST_PTR, st_sz, st, &err); 
  check_errors("clCreateBuffer", err);
  cl_mem counter_buf = clCreateBuffer(ctx, CL_MEM_USE_HOST_PTR, counter_sz, counter, &err);
  check_errors("clCreateBuffer", err);
  
  krn = clCreateKernel(prg, "edgesHysteresisLocal", &err);
  check_errors("clCreateKernel", err);

  int step = 1;
  int offset = 0;

  clSetKernelArg(krn, 0, sizeof(cl_mem), &map_buf);
  clSetKernelArg(krn, 1, sizeof(cl_mem), &st_buf);
  clSetKernelArg(krn, 2, sizeof(cl_mem), &counter_buf);
  clSetKernelArg(krn, 3, sizeof(cl_int), &img_w);
  clSetKernelArg(krn, 4, sizeof(cl_int), &img_h);
  clSetKernelArg(krn, 5, sizeof(cl_int), &step);
  clSetKernelArg(krn, 6, sizeof(cl_int), &offset);

  size_t gtdsz[] = { img_w, img_h, 1 };
  size_t ltdsz[] = { 4, 4, 1 };

  cl_event ev[3];

  err = clEnqueueNDRangeKernel(cmdq, krn, 2, 0, gtdsz, ltdsz, 0, 0, &ev[0]);
  check_errors("clCreateKernel", err);

  clEnqueueReadBuffer(cmdq, st_buf, CL_TRUE, 0, st_sz, st, 0, 0, &ev[1]);
  clEnqueueReadBuffer(cmdq, counter_buf, CL_TRUE, 0, counter_sz, counter, 0, 0, &ev[2]);

  err = clWaitForEvents(2, ev);
  check_errors("clWaitForEvents", err);

  clReleaseEvent(ev[2]);
  clReleaseEvent(ev[1]);
  clReleaseEvent(ev[0]);
  clReleaseKernel(krn);
  clReleaseCommandQueue(cmdq);

  return 0;
}

int getEdges(int* map, byte* dst, int img_w, int img_h) 
{
  int err;
  cmdq = clCreateCommandQueue(ctx, dev, 0, &err);
  check_errors("clCreateCommandQueue", err);

  int step = 1;
  int offset = 0;

  int nm = img_w * img_h;
  size_t map_sz = nm * sizeof(int);
  size_t dst_sz = nm * sizeof(byte);

  printf("[getEdges] Allocating OpenCL buffers\n");

  cl_mem map_buf = clCreateBuffer(ctx, CL_MEM_USE_HOST_PTR, map_sz, map, &err);
  check_errors("clCreateBuffer", err);
  cl_mem dst_buf = clCreateBuffer(ctx, CL_MEM_USE_HOST_PTR, dst_sz, dst, &err);
  check_errors("clCreateBuffer", err);
  
  printf("[getEdges] Creating kernel\n");

  krn = clCreateKernel(prg, "getEdges", &err);
  check_errors("clCreateKernel", err);

  clSetKernelArg(krn, 0, sizeof(cl_mem), &map_buf);
  clSetKernelArg(krn, 1, sizeof(cl_mem), &dst_buf);
  clSetKernelArg(krn, 2, sizeof(cl_mem), &img_w);
  clSetKernelArg(krn, 3, sizeof(cl_int), &img_h);
  clSetKernelArg(krn, 4, sizeof(cl_int), &step);
  clSetKernelArg(krn, 5, sizeof(cl_int), &offset);
  clSetKernelArg(krn, 6, sizeof(cl_int), &step);
  clSetKernelArg(krn, 7, sizeof(cl_int), &offset);

  size_t gtdsz[] = { img_w, img_h, 1 };
  size_t ltdsz[] = { 4, 4, 1 };

  cl_event ev[2];

  printf("[getEdges] Enqueueing NDRange\n");

  err = clEnqueueNDRangeKernel(cmdq, krn, 1, 0, gtdsz, ltdsz, 0, 0, &ev[0]);
  check_errors("clEnqueueNDRangeKernel", err);
  clEnqueueReadBuffer(cmdq, dst_buf, CL_TRUE, 0, dst_sz, dst, 0, 0, &ev[1]);

  printf("[getEdges] Waiting for events\n");

  err = clWaitForEvents(2, ev);
  check_errors("clWaitForEvents", err);

  clReleaseEvent(ev[1]);
  clReleaseEvent(ev[0]);
  clReleaseKernel(krn);
  clReleaseCommandQueue(cmdq);

  return 0;
  
}

//------------------------------------------------------------------------------
// GPU canny edge detection
//------------------------------------------------------------------------------
int canny(byte* img, int img_w, int img_h) {
  int i, j;
  int nm = img_w * img_h;

  byte out_pixel[1];
  FILE* out_pixels = fopen("pixels.out", "wt");
  for (i = 0; i < nm; i++) {
    sprintf(out_pixel, "%d", img[i]);
    fwrite(out_pixel, sizeof(byte), 1, out_pixels);
    fwrite(" ", sizeof(byte), 1, out_pixels);
  }

  write_png("img.png", img, img_w, img_h);

  //------------------------------------------------------------------------------
  // Sobel convolution
  //------------------------------------------------------------------------------
  int* dx = (int*) calloc(nm, sizeof(int));
  int* dy = (int*) calloc(nm, sizeof(int));
  convolve(img, sobel_convolution_in_x, dx, img_w, img_h, 3);
  convolve(img, sobel_convolution_in_y, dy, img_w, img_h, 3);

  FILE* dx_sobel_out = fopen("dxsobel.out", "wt");
  for (i = 0; i < nm; i++)
    fprintf(dx_sobel_out, "dx[%d] = %d\n", i, dx[i]);

  FILE* dy_sobel_out = fopen("dysobel.out", "wt");
  for (i = 0; i < nm; i++)
    fprintf(dy_sobel_out, "dy[%d] = %d\n", i, dy[i]);

  //----------------------------------------------------------------------------
  // Sobel Row Pass
  //----------------------------------------------------------------------------
  //sobelRowPass(img, img_w, img_h, dx, dy);

  //byte out_dx[256];
  //byte out_dy[256];
  //FILE* sobel_out = fopen("sobel.out", "wt");
  //for (i = 0; i < nm; i++)
  //fprintf(sobel_out, "dx[%d] = %d, dy[%d] = %d\n", i, dx[i], i, dy[i]);

  //----------------------------------------------------------------------------
  // Calculate Magnitudes
  //----------------------------------------------------------------------------
  //int* dxd = (int*) malloc(img_w * img_h * sizeof(int));
  //int* dyd = (int*) malloc(img_w * img_h * sizeof(int));
  //float* mag = (float*) malloc(img_w * img_h * sizeof(float));
  
  //calcMagnitude(img_w, img_h, dx, dy, dxd, dyd, mag);
  float* mag = (float*) malloc(img_w * img_h * sizeof(float));

  magnitude(dx, dy, mag, img_w, img_h);
  
  FILE* magnitude_out = fopen("magnitude.out", "wt");
  for (i = 0; i < nm; i++)
    fprintf(magnitude_out, "mag[%d] = %f\n", i, mag[i]);

  //------------------------------------------------------------------------------
  // Calculate Edge Direction
  //------------------------------------------------------------------------------
  int* dir = (int*) malloc(nm * sizeof(int));
  
  edgeDir(dx, dy, dir, img_w, img_h);

  FILE* edge_dir_out = fopen("edge_dir.out", "wt");
  for (i = 0; i < nm; i++)
    fprintf(edge_dir_out, "dir[%d] = %d\n", i, dir[i]);

  //------------------------------------------------------------------------------
  // Calculate Map
  //------------------------------------------------------------------------------
  int* map = (int*) malloc(nm * sizeof(int));

  float threshold_low = 2.5f;
  float threshold_high = 7.5f;

  edgeMap(mag, dir, map, threshold_low, threshold_high, img_w, img_h);

  FILE* map_out = fopen("map.out", "wt");
  for (i = 0; i < nm; i++)
  fprintf(map_out, "map[%d] = %d\n", i, map[i]);

  byte* map_png = (byte*) malloc(nm * sizeof(byte));
  for (i = 0; i < nm; i++)
    if (map[i] == 0) 
      map_png[i] = 0;
    else if (map[i] == 1)
      map_png[i] = 125;
    else if (map[i] == 2)
      map_png[i] = 255;

  write_png("map.png", map_png, img_w, img_h);

  //------------------------------------------------------------------------------
  // Nonmaximum suppression
  //------------------------------------------------------------------------------
  

  //------------------------------------------------------------------------------
  // Calculate Edge Binary Image relative to Map / Dir
  //------------------------------------------------------------------------------

  //------------------------------------------------------------------------------
  // Hysteresis
  //------------------------------------------------------------------------------
  //unsigned short* st = (unsigned short*) malloc(img_w * img_h * (sizeof(unsigned short) * 2));
  //unsigned int* counter = (unsigned int*) malloc(img_w * img_h * sizeof(unsigned int));
  //edgesHysteresis(map, st, counter, img_w, img_h);

  //------------------------------------------------------------------------------
  // Get Edges
  //------------------------------------------------------------------------------
  /* byte* pixels;     */
  /* pixels = malloc(nm * sizeof(byte)); */

  /* getEdges(map, pixels, img_w, img_h); */

  /* write_png("output.png", pixels, img_w, img_h, 8); */

  return 0;
}

//------------------------------------------------------------------------------
// GPU hough transform
//------------------------------------------------------------------------------

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

  //  int i;
  /* for (i = 0; i < st.st_size; i++) */
  /*   printf("%c", data[i]); */

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

int main()
{
  int i, j;

  // load bitmap
  printf("reading bitmap\n");

  int img_w = 512;
  int img_h = 512;
  byte* img_pixels = read_png("images/lena_512.png", img_w, img_h);

  // load opencl source
  printf("reading opencl source\n");

  //char* ocl_src;
  //int ocl_src_sz;
  //if (read_file("kernels/canny.cl", &ocl_src_sz, ocl_src) != 0)
  //return -1;

  printf("executing canny\n");

  initialize("kernels/_canny.cl");

  canny(img_pixels, img_w, img_h);
  // hough(src, src_w, src_h);

  teardown();

  return 0;
}
