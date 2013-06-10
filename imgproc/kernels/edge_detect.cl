#define PI 3.14159265359

//------------------------------------------------------------------------------
// Edge detection
//------------------------------------------------------------------------------
__kernel
void convolve_uc_i
(
 const __global uchar* src,
 __constant int* filter,
 __global int* output,
 const int src_w,
 const int filter_w
 )
{
  const int w = get_global_size(0);
  const int x_out = get_global_id(0);
  const int y_out = get_global_id(1);
  const int x_topleft = x_out;
  const int y_topleft = y_out;

  int sum = 0;
  for (int r = 0; r < filter_w; r++) {
    const int tmp_idx_f = r * filter_w;
    const int y_in = y_topleft + r;
    const int tmp_idx_in = y_in * src_w + x_topleft;
    
    for (int c = 0; c < filter_w; c++) {
      const int idx_f = tmp_idx_f + c;
      const int idx_in = tmp_idx_in + c;
      sum += filter[idx_f] * src[idx_in];
    }
  }

  const int idx_out = y_out * w + x_out;
  output[idx_out] = sum;
}

__kernel
void magnitude
(
 __global const int* dx,
 __global const int* dy,
 __global float* mag
 )
{
  const int w = get_global_size(0);
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  const int i = x + y * w;
  const int xi = i;
  const int yi = i;

  const float dxsq = (float) dx[xi] * dx[xi];
  const float dysq = (float) dy[yi] * dy[yi];

  mag[i] = sqrt(dxsq + dysq);
}

__kernel
void normalized_output
(
 __global const uchar* input,
 __global const float* mag,
 __global const float low_thresh,
 __global const float high_thresh,
 __global uchar* output
 ) 
{
  const int w = get_global_size(0);
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int i = x + y * w;
  
  const int pixel = (int) input[i];
  const float norm = (pixel / mag[i]);
  if (norm > low_thresh && norm < high_thresh) {
    output[i] = 255;
  } else {
    output[i] = 0;
  }
}

//------------------------------------------------------------------------------
// Hough transform
//------------------------------------------------------------------------------
/* __kernel */
/* void accumulate */
/* ( */
/*  __global const uchar* image, */
/*  __global int* acc, */
/*  __global mutex_t* mtx */
/*  ) */
/* { */
/*   const int w = get_global_size(0); */
/*   const int h = get_global_size(1); */
/*   const int x = get_global_id(0); */
/*   const int y = get_global_id(1); */
/*   const int gix = x + y * w; */

/*   const float r = 1.0; */
/*   if (image[gix] == 255) { */
/*     for (int theta = 0; theta < 360; theta++) { */
/*       const float t = (theta * PI) / 180; */
/*       const int x0 = (int) round(x - r * cos(t)); */
/*       const int y0 = (int) round(y - r * sin(t)); */
/*       if (x0 < w && x0 > 0 && y0 < h && y0 > 0) { */
/* 	//lacc[theta] = (int2) (x0 + y0 * w, 1); */
/* 	mutex_lock(mtx); */
/* 	acc[x0 + (y0 * w)] += 1; */
/* 	mutex_unlock(mtx); */
/*       } */
/*     } */
/*   } */
/* } */

__kernel
void normalise
( 
 __global int* acc,
 __global const float maximum,
 __global const int image_w,
 __global const int image_h
 )
{
  const int w = get_global_size(0);
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int gix = x + y * w;

  const int value = (int) (acc[gix] / maximum) * 255.0f;
  acc[gix] = 0xFF000000 | (value << 16 | value << 8 | value);
}

__kernel
void find_maxima
(
 __global const int* acc,
 __global int*       results,
 __global const int  acc_sz
 )
{
  const int w = get_global_size(0);
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int gix = x + y * w;

  int value = acc[gix] & 0xFF;
  
  if (value > results[(acc_sz-1)*3]) {
    results[(acc_sz-1)*3] = value;
    results[(acc_sz-1)*3+1] = x;
    results[(acc_sz-1)*3+2] = y;

    int i = (acc_sz-2)*3;
    while ((i >= 0) && (results[i+3] > results[i])) {
      for (int j = 0; j < 3; j++) {
	int temp = results[i+j];
	results[i+j] = results[i+3+j];
	results[i+3+j] = temp;
      }
      i = i - 3;
      if (i < 0) break;
    }
  }
}
