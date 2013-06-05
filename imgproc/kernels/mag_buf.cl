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

            mag[(gidx + 1) + (gidy + 1) * mag_step + mag_offset] = calc(x, y);
        }
    }
}
