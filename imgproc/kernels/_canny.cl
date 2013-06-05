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

  mag[i] = sqrt(dxsq + dxsq);
}

__kernel
void edgeDir
(
 __global const int* dx,
 __global const int* dy,
 __global int* dir
 )
{
  const int w = get_global_size(0);
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  const int i = x + y * w;
  const int xi = i;
  const int yi = i;

  if (dx[xi] == 0) {
    dir[i] = 0;
  } else {
    float gx = (float) dx[xi];
    float gy = (float) dy[yi];
    float r = atan2(gy, gx);
    float d = r * (180.0f / PI);
    dir[i] = calc_dir(d);
  }
}

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

//------------------------------------------------------------------------------
// calculate the magnitude of the filter pass combining both x and y directions
// This is the buffered version(3x3 sobel)
//
// dx_buf		dx buffer, calculated from calcSobelRowPass
// dy_buf		dy buffer, calculated from calcSobelRowPass
// dx			direvitive in x direction output
// dy			direvitive in y direction output
// mag			magnitude direvitive of xy output
__kernel
    void calcMagnitude_buf
    (
    __global const int * dx_buf,
    __global const int * dy_buf,
    __global int * dx,
    __global int * dy,
    __global float * mag,
    int rows,
    int cols,
    int dx_buf_step,
    int dx_buf_offset,
    int dy_buf_step,
    int dy_buf_offset,
    int dx_step,
    int dx_offset,
    int dy_step,
    int dy_offset,
    int mag_step,
    int mag_offset
    )
{
    dx_buf_step    /= sizeof(*dx_buf);
    dx_buf_offset  /= sizeof(*dx_buf);
    dy_buf_step    /= sizeof(*dy_buf);
    dy_buf_offset  /= sizeof(*dy_buf);
    dx_step    /= sizeof(*dx);
    dx_offset  /= sizeof(*dx);
    dy_step    /= sizeof(*dy);
    dy_offset  /= sizeof(*dy);
    mag_step   /= sizeof(*mag);
    mag_offset /= sizeof(*mag);

    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    int lidx = get_local_id(0);
    int lidy = get_local_id(1);

    __local int sdx[18][16];
    __local int sdy[18][16];

    sdx[lidy + 1][lidx] = dx_buf[gidx + gidy * dx_buf_step + dx_buf_offset];
    sdy[lidy + 1][lidx] = dy_buf[gidx + gidy * dy_buf_step + dy_buf_offset];
    if(lidy == 0)
    {
        sdx[0][lidx]  = dx_buf[gidx + max(gidy - 1,  0)        * dx_buf_step + dx_buf_offset];
        sdx[17][lidx] = dx_buf[gidx + min(gidy + 16, rows - 1) * dx_buf_step + dx_buf_offset];

        sdy[0][lidx]  = dy_buf[gidx + max(gidy - 1,  0)        * dy_buf_step + dy_buf_offset];
        sdy[17][lidx] = dy_buf[gidx + min(gidy + 16, rows - 1) * dy_buf_step + dy_buf_offset];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(gidx < cols)
    {
        if(gidy < rows)
        {
            int x =  sdx[lidy][lidx] + 2 * sdx[lidy + 1][lidx] + sdx[lidy + 2][lidx];
            int y = -sdy[lidy][lidx] + sdy[lidy + 2][lidx];

            dx[gidx + gidy * dx_step + dx_offset] = x;
            dy[gidx + gidy * dy_step + dy_offset] = y;

            mag[(gidx + 1) + (gidy + 1) * mag_step + mag_offset] = calc_mag(x, y);
        }
    }
}

//------------------------------------------------------------------------------
//First pass of edge detection and non-maximum suppression
// edgetype is set to for each pixel:
// 0 - below low thres, not an edge
// 1 - maybe an edge
// 2 - is an edge, either magnitude is greater than high thres, or
//     Given estimates of the image gradients, a search is then carried out
//     to determine if the gradient magnitude assumes a local maximum in the gradient direction.
//     if the rounded gradient angle is zero degrees (i.e. the edge is in the north-south direction) the point will be considered to be on the edge if its gradient magnitude is greater than the magnitudes in the west and east directions,
//     if the rounded gradient angle is 90 degrees (i.e. the edge is in the east-west direction) the point will be considered to be on the edge if its gradient magnitude is greater than the magnitudes in the north and south directions,
//     if the rounded gradient angle is 135 degrees (i.e. the edge is in the north east-south west direction) the point will be considered to be on the edge if its gradient magnitude is greater than the magnitudes in the north west and south east directions,
//     if the rounded gradient angle is 45 degrees (i.e. the edge is in the north west-south east direction)the point will be considered to be on the edge if its gradient magnitude is greater than the magnitudes in the north east and south west directions.
//
// dx, dy		direvitives of x and y direction
// mag			magnitudes calculated from calcMagnitude function
// map			output containing raw edge types
__kernel
    void calcMap
    (
    __global const int * dx,
    __global const int * dy,
    __global const float * mag,
    __global int * map,
    int rows,
    int cols,
    float low_thresh,
    float high_thresh,
    int dx_step,
    int dx_offset,
    int dy_step,
    int dy_offset,
    int mag_step,
    int mag_offset,
    int map_step,
    int map_offset
    )
{
    dx_step    /= sizeof(*dx);
    dx_offset  /= sizeof(*dx);
    dy_step    /= sizeof(*dy);
    dy_offset  /= sizeof(*dy);
    mag_step   /= sizeof(*mag);
    mag_offset /= sizeof(*mag);
    map_step   /= sizeof(*map);
    map_offset /= sizeof(*map);

    __local float smem[18][18];

    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    int lidx = get_local_id(0);
    int lidy = get_local_id(1);

    int grp_idx = get_global_id(0) & 0xFFFFF0;
    int grp_idy = get_global_id(1) & 0xFFFFF0;

    int tid = lidx + lidy * 16;
    int lx = tid % 18;
    int ly = tid / 18;
    if(ly < 14)
    {
        smem[ly][lx] = mag[grp_idx + lx + (grp_idy + ly) * mag_step];
    }
    if(ly < 4 && grp_idy + ly + 14 <= rows && grp_idx + lx <= cols)
    {
        smem[ly + 14][lx] = mag[grp_idx + lx + (grp_idy + ly + 14) * mag_step];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if(gidy < rows && gidx < cols)
    {
        int x = dx[gidx + gidy * dx_step];
        int y = dy[gidx + gidy * dy_step];
        const int s = (x ^ y) < 0 ? -1 : 1;
        const float m = smem[lidy + 1][lidx + 1];
        x = abs(x);
        y = abs(y);

        // 0 - the pixel can not belong to an edge
        // 1 - the pixel might belong to an edge
        // 2 - the pixel does belong to an edge
        int edge_type = 0;
        if(m > low_thresh)
        {
            const int tg22x = x * TG22;
            const int tg67x = tg22x + (x << (1 + CANNY_SHIFT));
            y <<= CANNY_SHIFT;
            if(y < tg22x)
            {
                if(m > smem[lidy + 1][lidx] && m >= smem[lidy + 1][lidx + 2])
                {
                    edge_type = 1 + (int)(m > high_thresh);
                }
            }
            else if (y > tg67x)
            {
                if(m > smem[lidy][lidx + 1]&& m >= smem[lidy + 2][lidx + 1])
                {
                    edge_type = 1 + (int)(m > high_thresh);
                }
            }
            else
            {
                if(m > smem[lidy][lidx + 1 - s]&& m > smem[lidy + 2][lidx + 1 + s])
                {
                    edge_type = 1 + (int)(m > high_thresh);
                }
            }
        }
        map[gidx + 1 + (gidy + 1) * map_step] = edge_type;
    }
}

//------------------------------------------------------------------------------
// do Hysteresis for pixel whose edge type is 1
//
// If candidate pixel (edge type is 1) has a neighbour pixel (in 3x3 area) with type 2, it is believed to be part of an edge and
// marked as edge. Each thread will iterate for 16 times to connect local edges.
// Candidate pixel being identified as edge will then be tested if there is nearby potiential edge points. If there is, counter will
// be incremented by 1 and the point location is stored. These potiential candidates will be processed further in next kernel.
//
// map		raw edge type results calculated from calcMap.
// st		the potiential edge points found in this kernel call
// counter	the number of potiential edge points
//------------------------------------------------------------------------------
/* __kernel */
/*     void edgesHysteresisLocal */
/*     ( */
/*     __global int * map, */
/*     __global ushort * st, */
/*     __global unsigned int * counter, */
/*     int rows, */
/*     int cols, */
/*     int map_step, */
/*     int map_offset */
/*     ) */
/* { */
/*     map_step   /= sizeof(*map); */
/*     map_offset /= sizeof(*map); */

/*     __local int smem[18][18]; */

/*     int gidx = get_global_id(0); */
/*     int gidy = get_global_id(1); */

/*     int lidx = get_local_id(0); */
/*     int lidy = get_local_id(1); */

/*     int grp_idx = get_global_id(0) & 0xFFFFF0; */
/*     int grp_idy = get_global_id(1) & 0xFFFFF0; */

/*     int tid = lidx + lidy * 16; */
/*     int lx = tid % 18; */
/*     int ly = tid / 18; */
/*     if(ly < 14) */
/*     { */
/*         smem[ly][lx] = map[grp_idx + lx + (grp_idy + ly) * map_step + map_offset]; */
/*     } */
/*     if(ly < 4 && grp_idy + ly + 14 <= rows && grp_idx + lx <= cols) */
/*     { */
/*         smem[ly + 14][lx] = map[grp_idx + lx + (grp_idy + ly + 14) * map_step + map_offset]; */
/*     } */

/*     barrier(CLK_LOCAL_MEM_FENCE); */

/*     if(gidy < rows && gidx < cols) */
/*     { */
/*         int n; */

/*         #pragma unroll */
/*         for (int k = 0; k < 16; ++k) */
/*         { */
/*             n = 0; */

/*             if (smem[lidy + 1][lidx + 1] == 1) */
/*             { */
/*                 n += smem[lidy    ][lidx    ] == 2; */
/*                 n += smem[lidy    ][lidx + 1] == 2; */
/*                 n += smem[lidy    ][lidx + 2] == 2; */

/*                 n += smem[lidy + 1][lidx    ] == 2; */
/*                 n += smem[lidy + 1][lidx + 2] == 2; */

/*                 n += smem[lidy + 2][lidx    ] == 2; */
/*                 n += smem[lidy + 2][lidx + 1] == 2; */
/*                 n += smem[lidy + 2][lidx + 2] == 2; */
/*             } */

/*             if (n > 0) */
/*                 smem[lidy + 1][lidx + 1] = 2; */
/*         } */

/*         const int e = smem[lidy + 1][lidx + 1]; */
/*         map[gidx + 1 + (gidy + 1) * map_step] = e; */

/*         n = 0; */
/*         if(e == 2) */
/*         { */
/*             n += smem[lidy    ][lidx    ] == 1; */
/*             n += smem[lidy    ][lidx + 1] == 1; */
/*             n += smem[lidy    ][lidx + 2] == 1; */

/*             n += smem[lidy + 1][lidx    ] == 1; */
/*             n += smem[lidy + 1][lidx + 2] == 1; */

/*             n += smem[lidy + 2][lidx    ] == 1; */
/*             n += smem[lidy + 2][lidx + 1] == 1; */
/*             n += smem[lidy + 2][lidx + 2] == 1; */
/*         } */

/*         if(n > 0) */
/*         { */
/*           unsigned int ind = tid; */
/* 	  counter[ind] = 1; */
/* 	  st[ind*2]   = (ushort) gidx + 1; */
/* 	  st[ind*2+1] = (ushort) gidy + 1; */
/*         } */
/*     } */
/* } */

//Get the edge result. egde type of value 2 will be marked as an edge point and set to 255. Otherwise 0.
// map		edge type mappings
// dst		edge output
__kernel
    void getEdges
    (
    __global const int * map,
    __global uchar * dst,
    int rows,
    int cols,
    int map_step,
    int map_offset,
    int dst_step,
    int dst_offset
    )
{
    map_step   /= sizeof(*map);
    map_offset /= sizeof(*map);
    //dst_step   /= sizeof(*dst);
    //dst_offset /= sizeof(*dst);

    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    if(gidy < rows && gidx < cols)
    {
        //dst[gidx + gidy * dst_step] = map[gidx + 1 + (gidy + 1) * map_step] == 2 ? 255: 0;
        dst[gidx + gidy * dst_step] = (uchar)(-(map[gidx + 1 + (gidy + 1) * map_step] / 2));
    }
}
