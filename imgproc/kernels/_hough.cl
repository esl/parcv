////////////////////////////////////////////////////////////////////////
// buildPointList

#define PIXELS_PER_THREAD 16

// TODO: add offset to support ROI
__kernel void buildPointList(__global const uchar* src,
                             int cols,
                             int rows,
                             int step,
                             __global unsigned int* list,
                             __global int* counter)
{
    __local unsigned int s_queues[4][32 * PIXELS_PER_THREAD];
    __local int s_qsize[4];
    __local int s_globStart[4];

    const int x = get_group_id(0) * get_local_size(0) * PIXELS_PER_THREAD + get_local_id(0);
    const int y = get_global_id(1);

    if (get_local_id(0) == 0)
        s_qsize[get_local_id(1)] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
        
    if (y < rows)
    {
        // fill the queue
        __global const uchar* srcRow = &src[y * step];
        for (int i = 0, xx = x; i < PIXELS_PER_THREAD && xx < cols; ++i, xx += get_local_size(0))
        {
            if (srcRow[xx])
            {
                const unsigned int val = (y << 16) | xx;
                const int qidx = atomic_add(&s_qsize[get_local_id(1)], 1);
                s_queues[get_local_id(1)][qidx] = val;
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // let one work-item reserve the space required in the global list
    if (get_local_id(0) == 0 && get_local_id(1) == 0)
    {
        // find how many items are stored in each list
        int totalSize = 0;
        for (int i = 0; i < get_local_size(1); ++i)
        {
            s_globStart[i] = totalSize;
            totalSize += s_qsize[i];
        }

        // calculate the offset in the global list
        const int globalOffset = atomic_add(counter, totalSize);
        for (int i = 0; i < get_local_size(1); ++i)
            s_globStart[i] += globalOffset;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);
    
    // copy local queues to global queue
    const int qsize = s_qsize[get_local_id(1)];
    int gidx = s_globStart[get_local_id(1)] + get_local_id(0);
    for(int i = get_local_id(0); i < qsize; i += get_local_size(0), gidx += get_local_size(0))
        list[gidx] = s_queues[get_local_id(1)][i];
}
