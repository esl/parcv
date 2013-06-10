//------------------------------------------------------------------------------
#define PI 3.14159265359

inline float calc_mag(int x, int y)
{
    return sqrt((float)(x * x + y * y));
}

inline int calc_dir(float theta)
{
  if (theta < 22.5 || theta > 157.5) {
    return 0;
  } else if (theta > 22.5 && theta < 67.5) {
    return 45;
  } else if (theta > 67.5 && theta < 112.5) {
    return 90;
  } else {
    return 135;
  }
}

inline int calc_edge_type(float m, float low_thresh, float high_thresh)
{
  if (m < low_thresh) {
    return 0;
  }
  
  if (m > high_thresh) {
    return 2;
  }

  return 1;
}

#define CANNY_SHIFT 15
#define TG22        (int)(0.4142135623730950488016887242097*(1<<CANNY_SHIFT) + 0.5)

//------------------------------------------------------------------------------
// Smoothing perpendicular to the derivative direction with a triangle filter
// only support 3x3 Sobel kernel
// h (-1) =  1, h (0) =  2, h (1) =  1
// h'(-1) = -1, h'(0) =  0, h'(1) =  1
// thus sobel 2D operator can be calculated as:
// h'(x, y) = h'(x)h(y) for x direction
//
// src		input 8bit single channel image data
// dx_buf	output dx buffer
// dy_buf	output dy buffer
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

/* __kernel */
/* void edgeDir */
/* ( */
/*  __global const int* dx, */
/*  __global const int* dy, */
/*  __global int* dir */
/*  ) */
/* { */
/*   const int w = get_global_size(0); */
/*   const int x = get_global_id(0); */
/*   const int y = get_global_id(1); */

/*   const int i = x + y * w; */
/*   const int xi = i; */
/*   const int yi = i; */

/*   if (dx[xi] == 0) { */
/*     dir[i] = 0; */
/*   } else { */
/*     float gx = (float) dx[xi]; */
/*     float gy = (float) dy[yi]; */
/*     float r = atan2(gy, gx); */
/*     float d = r * (180.0f / PI); */
/*     dir[i] = calc_dir(d); */
/*   } */
/* } */

/* __kernel */
/* void edgeMap2 */
/* ( */
/*  __global const float* mag, */
/*  __global const int* dir, */
/*  __global int* map, */
/*  const int low_thresh, */
/*  const int high_thresh */
/*  ) */
/* { */
/*   const int w = get_global_size(0); */
/*   const int x = get_global_id(0); */
/*   const int y = get_global_id(1); */
/*   const int i = x + y * w; */

/*   if (mag[i] > high_thresh) { */
/*     if (dir[i] == 0) { */
/*       const int west = (x - 1) + y * w; */
/*       const int east = (x + 1) + y * w; */
/*       if (dir[west] == dir[i] && dir[west] > low_thresh) { */
/* 	map[west] = 255; */
/*       } */
/*       if (dir[east] == dir[i] && dir[east] > low_thresh) { */
/* 	map[east] = 255; */
/*       } */
/*     } else if (dir[i] == 45) { */
/*       const int north_east = (x + 1) + ((y + 1) * w); */
/*       const int south_west = (x - 1) + ((y - 1) * w); */
/*       if (dir[north_east] == dir[i] && dir[north_east] > low_thresh) { */
/* 	map[north_east] = 255; */
/*       } */
/*       if (dir[south_west] == dir[i] && dir[south_west] > low_thresh) { */
/* 	map[south_west] = 255; */
/*       } */
/*     } else if (dir[i] == 90) { */
/*       const int north = x + ((y + 1) * w); */
/*       const int south = x + ((y - 1) * w); */
/*       if (dir[north] == dir[i] && dir[north] > low_thresh) { */
/* 	map[north] = 255; */
/*       } */
/*       if (dir[south] == dir[i] && dir[south] > low_thresh) { */
/* 	map[south] = 255; */
/*       } */
/*     } else if (dir[i] == 135) { */
/*       const int north_west = (x - 1) + ((y + 1) * w); */
/*       const int south_east = (x + 1) + ((y - 1) * w); */
/*       if (dir[north_west] == dir[i] && dir[north_west] > low_thresh) { */
/* 	map[north_west] = 255; */
/*       }       */
/*       if (dir[south_east] == dir[i] && dir[south_east] > low_thresh) { */
/* 	map[south_east] = 255; */
/*       } */
/*     } */
/*   } */
/* } */

//const int2 idx4 = { 
//  {-1, 0}, {-1, 1}, {0, 1}, {1, 1},
//  {1, 0}, {1, -1}, {0, -1}, {-1, -1}
//};

/* __kernel */
/* void edgeMap */
/* ( */
/*  __global const float* mag, */
/*  __global const int*   dir, */
/*  __global const uchar* map, */
/*  __global const int    niter, // how many passes */
/*  __global float low_thresh, */
/*  __global float high_thresh */
/*  ) */
/* { */
/*   // There are 8 surrounding pixels & directions */
/*   // ldir = 0: w, 1: nw, 2: n, 3: ne, 4: e, 5: se, 6: s, 7: sw  */
/*   const int lidx_w  = 0; */
/*   const int lidx_nw = 1; */
/*   const int lidx_n  = 2; */
/*   const int lidx_ne = 3; */
/*   const int lidx_e  = 4; */
/*   const int lidx_se = 5; */
/*   const int lidx_s  = 6; */
/*   const int lidx_sw = 7; */

/*   __local int ldir[8 * niter]; */
/*   __local int lmag[8 * niter]; */

/*   for (int i = 0; i < niter; i++) { */
/*     // initialise the local memory with surrounding ldir & lmag values */
/*     for (int j = 0; j < 8; j++) { */
/*       const int2 dir_idx = idxs[j]; */
/*       if ((ix + dir_idx.x) < 0) { */
/* 	// set west to ldir & lmag at other side of img */
/* 	const int gidx_nw = (w - 1) + (iy + 1) * w; */
/* 	const int gidx_w  = (w - 1) + iy * w; */
/* 	const int gidx_sw = (w - 1) + (iy - 1) * w; */
/* 	ldir[nw] = dir[gidx_nw];  */
/* 	lmag[nw] = mag[gidx_nw]; */
/* 	ldir[w]  = dir[gidx_w];       */
/* 	lmag[w]  = mag[gidx_w]; */
/* 	ldir[sw] = dir[gidx_sw];      */
/* 	lmag[sw] = mag[gidx_sw]; */
/*       } else if ((ix + dir_idx.x) > w) { */
/* 	const int gidx_ne = 0 + (iy + 1) * w; */
/* 	const int gidx_e  = 0 + iy * w; */
/* 	const int gidx_se = 0 + (iy - 1) * w; */
/* 	ldir[ne] = dir[gidx_ne];  */
/* 	lmag[ne] = mag[gidx_ne]; */
/* 	ldir[e]  = dir[gidx_e];       */
/* 	lmag[e]  = mag[gidx_e]; */
/* 	ldir[se] = dir[gidx_se];      */
/* 	lmag[se] = mag[gidx_se]; */
/*       } else if ((iy + dir_idx.y) < 0) { */
/* 	const int gidx_se = (xi + 1) + (h - 1) * w; */
/* 	const int gidx_s  = xi + (h - 1) * w; */
/* 	const int gidx_sw = (xi - 1) + (h - 1) * w; */
/* 	ldir[se] = dir[gidx_se];  */
/* 	lmag[se] = mag[gidx_se]; */
/* 	ldir[s]  = dir[gidx_s];       */
/* 	lmag[s]  = mag[gidx_s]; */
/* 	ldir[sw] = dir[gidx_sw];      */
/* 	lmag[sw] = mag[gidx_sw]; */
/*       } else if ((iy + dir_idx.y) > h) { */
/* 	const int gidx_nw = (xi - 1) + 0 * w; */
/* 	const int gidx_n  = xi + 0 * w; */
/* 	const int gidx_ne = (xi + 1) + 0 * w; */
/* 	ldir[nw] = dir[gidx_nw];  */
/* 	lmag[nw] = mag[gidx_nw]; */
/* 	ldir[n]  = dir[gidx_n];       */
/* 	lmag[n]  = mag[gidx_n]; */
/* 	ldir[ne] = dir[gidx_ne];      */
/* 	lmag[ne] = mag[gidx_ne]; */
/*       } */
/*       barrier(CLK_LOCAL_MEM_FENCE); */

/*     if (mag[gix] > high_thresh) { */
/*       if (dir[gix] == 0) { */
/* 	if (mag[w] && mag[e]) { */
/* 	} */
/*       } else if (dir[gix] == 45) { */
/* 	if (lmag[ne] && lmag[sw]) { */
/* 	} */
/*       } else if (dir[gix] == 90) { */
/* 	if (lmag[n] && lmag[s]) { */
/* 	} */
/*       } else if (dir[gix] == 135) { */
/* 	if (lmag[nw] && lmag[se]) { */
/* 	} */
/*       } */
/*     } */
/*   } */
/* } */

__kernel
void edgeMap
(
 __global const float* mag,
 __global const int* dir,
 __global int* map,
 float low_thresh,
 float high_thresh
 )
{
  const int w = get_global_size(0);
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  const int i = x + (y * w);
  const int xyi = x + (y * w);

  const int edge_type = calc_edge_type(mag[xyi], low_thresh, high_thresh);

  if (edge_type == 1) {
    if (dir[i] == 0) {
      const int west = (x - 1) + y * w;
      const int east = (x + 1) + y * w;
      if (mag[i] > mag[west] && mag[i] > mag[east]) {
	map[i] = 2;
      } else {
	map[i] = 0;
      }
    } else if (dir[i] == 45) {
      const int north_east = (x + 1) + ((y + 1) * w);
      const int south_west = (x - 1) + ((y - 1) * w);
      if (mag[i] > mag[north_east] && mag[i] > mag[south_west]) {
	map[i] = 2;
      } else {
	map[i] = 0;
      }
    } else if (dir[i] == 90) {
      const int north = x + ((y + 1) * w);
      const int south = x + ((y - 1) * w);
      if (mag[i] > mag[north] && mag[i] > mag[south]) {
	map[i] = 2;
      } else {
	map[i] = 0;
      }
    } else if (dir[i] == 135) {
      const int north_west = (x - 1) + ((y + 1) * w);
      const int south_east = (x + 1) + ((y - 1) * w);
      if (mag[i] > mag[north_west] && mag[i] > mag[south_east]) {
	map[i] = 2;
      } else {
	map[i] = 0;
      }
    }
  }
}

