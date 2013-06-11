#ifndef IMGPROC_H
#define IMGPROC_H

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

//------------------------------------------------------------------------------
#define PI 3.14159265359

typedef unsigned char byte;
typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;

//------------------------------------------------------------------------------
int cl_check_errors(const char* label, int err);
int cl_initialize(const char* src_path);
int cl_teardown();
int cl_read_file(const char* path, int* sz, char* data);
static int convolve(byte* src, int* filter, int* output, int src_w, int src_h, int filter_w);
static int magnitude(int* dx, int* dy, float* mag, int src_w, int src_h);
static int normalize(uchar* input, float* mag, float low_thresh, float high_thresh,
		     uchar* output, int src_w, int src_h);
int cl_transform(byte* img, int img_w, int img_h);

byte* png_read(const char* filename, int src_w, int src_h);
void png_write(const char* filename, const byte* pixels, int src_w, int src_h);

#endif
