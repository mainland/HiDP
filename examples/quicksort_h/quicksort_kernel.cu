template <class T>
__device__ void memcpy_p(T *dst, const T *src, int size)
{
  for (int idx = threadIdx.x; idx < size; idx += blockDim.x)
    dst[idx] = src[idx];
}

__global__ void
reverse_kernel(SORT_T *buffer, int size, int sub_size, int block_per_outer)
{
  GET_GID
  int blockId = blockIdx.x/block_per_outer;
  int totalWarps = (step_size >> 5);
  int totalBlocks = gridDim.x/block_per_outer;
  int totalSubs = size/sub_size;
  int blockIndex = blockIdx.x % block_per_outer;
  int warpIdx = blockIndex * blockDim.x + threadIdx.x;
  int inner_step = blockDim.x * block_per_outer;
  for (int sub_id = blockId; sub_id < totalSubs; sub_id += totalBlocks)
    {
      // sub_id is the current 
      for (int index = warpIdx; index < (sub_size>>2); index += inner_step)
        {
          int new_index = sub_id * sub_size + (sub_size>>1) + index;
          int new_index_mirror = sub_id * sub_size + sub_size - index-1;
          //          int new_index_mirror = sub_id * sub_size + (sub_size>>1) + index;

          SORT_T tmp = buffer[new_index];
          buffer[new_index] = buffer[new_index_mirror];
          buffer[new_index_mirror] = tmp;
        }
    }
  
}


__global__ void
reverse_warp(SORT_T *buffer, int size, int sub_size)
{
  GET_GID
  int warpId = (gid >> 5);
  int totalWarps = (step_size >> 5);
  int totalSubs = size/sub_size;
  int warpIdx = threadIdx.x & 0x1F;
  for (int sub_id = warpId; sub_id < totalSubs; sub_id += totalWarps)
    {
      // sub_id is the current 
      for (int index = warpIdx; index < sub_size/4; index += 32)
        {
          int new_index = sub_id * sub_size + sub_size/2 + index;
          int new_index_mirror = sub_id * sub_size + sub_size - index-1;
          //int new_index_mirror = sub_id * sub_size + (sub_size>>1) + index;
          SORT_T tmp = buffer[new_index];
          buffer[new_index] = buffer[new_index_mirror];
          buffer[new_index_mirror] = tmp;
        }
    }
  
}


__global__ void
reverse_thread(SORT_T *buffer, int size, int sub_size)
{
  GET_GID
  int totalSubs = size/sub_size;
  for (int sub_id = gid; sub_id < totalSubs; sub_id += step_size)
    {
      // sub_id is the current 
      for (int index = 0; index < sub_size/4; index++)
        {
          int new_index = sub_id * sub_size + sub_size/2 + index;
          int new_index_mirror = sub_id * sub_size + sub_size - index-1;
          //int new_index_mirror = sub_id * sub_size + (sub_size>>1) + index;
          SORT_T tmp = buffer[new_index];
          buffer[new_index] = buffer[new_index_mirror];
          buffer[new_index_mirror] = tmp;
        }
    }
}




__global__ void
bitonic_kernel(SORT_T *buffer, int size, int sub_size, int block_per_outer, int sub_size_log)
{
  GET_GID
  int blockId = blockIdx.x/block_per_outer;
  int totalBlocks = gridDim.x/block_per_outer;
  int totalSubs = (size >> sub_size_log);
  int blockIndex = blockIdx.x % block_per_outer;
  int warpIdx = blockIndex * blockDim.x + threadIdx.x;
  int inner_step = blockDim.x * block_per_outer;
  for (int sub_id = blockId; sub_id < totalSubs; sub_id += totalBlocks)
    {
      // sub_id is the current 
      
      for (int index = warpIdx; index < (sub_size>>1); index += inner_step)
        {
          int new_index = (sub_id << sub_size_log) + index;
          //          int new_index_mirror = (sub_id <<sub_size_log) + sub_size - index-1;
          int new_index_mirror = (sub_id <<sub_size_log) + (sub_size>>1) +index;
          SORT_T left = buffer[new_index];
          SORT_T right = buffer[new_index_mirror];
          if (left > right)
            {
              buffer[new_index] = right;
              buffer[new_index_mirror] = left;
            }
        }
    }
  
}


__global__ void
bitonic_subwarp(SORT_T *buffer, int size, int sub_size, int sub_size_log)
{
  GET_GID
  int warpId = (gid >> 3);
  int totalWarps = (step_size >> 3);
  int totalSubs = (size>>sub_size_log);
  int warpIdx = threadIdx.x & 0x7;
  for (int sub_id = warpId; sub_id < totalSubs; sub_id += totalWarps)
    {
      // sub_id is the current 
      for (int index = warpIdx; index < (sub_size>>1); index += 8)
        {
          int new_index = (sub_id << sub_size_log) + index;
          //          int new_index_mirror = (sub_id << sub_size_log) + sub_size - index-1;
          int new_index_mirror = (sub_id << sub_size_log) + (sub_size>>1) + index;
          SORT_T left = buffer[new_index];
          SORT_T right = buffer[new_index_mirror];
          if (left > right)
            {
              buffer[new_index] = right;
              buffer[new_index_mirror] = left;
            }
        }
    }
}

__global__ void
bitonic_warp(SORT_T *buffer, int size, int sub_size, int sub_size_log)
{
  GET_GID
  int warpId = (gid >> 5);
  int totalWarps = (step_size >> 5);
  int totalSubs = (size >> sub_size_log);
  int warpIdx = threadIdx.x & 0x1F;
  for (int sub_id = warpId; sub_id < totalSubs; sub_id += totalWarps)
    {
      // sub_id is the current 
      for (int index = warpIdx; index < (sub_size>>1); index += 32)
        {
          int new_index = (sub_id <<sub_size_log) + index;
          //          int new_index_mirror = (sub_id <<sub_size_log) + sub_size - index-1;
          int new_index_mirror = (sub_id <<sub_size_log) + (sub_size>>1) + index;
          SORT_T left = buffer[new_index];
          SORT_T right = buffer[new_index_mirror];
          if (left > right)
            {
              buffer[new_index] = right;
              buffer[new_index_mirror] = left;
            }
        }
    }
}


__global__ void
bitonic_thread(SORT_T *buffer, int size, int sub_size, int sub_size_log)
{
  GET_GID
  int totalSubs = (size >> sub_size_log);
  for (int sub_id = gid; sub_id < totalSubs; sub_id += step_size)
    {
      // sub_id is the current 
      for (int index = 0; index < (sub_size>>1); index++)
        {
          int new_index = (sub_id << sub_size_log) + index;
          //          int new_index_mirror = (sub_id << sub_size_log) + sub_size - index-1;
          int new_index_mirror = (sub_id << sub_size_log) + (sub_size>>1) + index;

          SORT_T left = buffer[new_index];
          SORT_T right = buffer[new_index_mirror];
          //          printf("thread %d read left %d right %d. sub_size %d\n", threadIdx.x, left, right, sub_size);
          if (left > right)
            {
              //              printf("swapping \n");
              buffer[new_index] = right;
              buffer[new_index_mirror] = left;
            }
        }
    }
}

template <int SUB_SIZE, int LOG_SUB_SIZE>
__global__ void
bitonic_block(SORT_T *buffer, int size)
{
  __shared__ SORT_T s_array[SUB_SIZE];
  int numbs = size >> LOG_SUB_SIZE;
  for (int bid = blockIdx.x; bid <  numbs; bid += gridDim.x)
    {
     s_array[threadIdx.x] = buffer[(bid << LOG_SUB_SIZE) + threadIdx.x];
     s_array[blockDim.x + threadIdx.x] = buffer[(bid << LOG_SUB_SIZE) + blockDim.x + threadIdx.x];
      __syncthreads();
      int innersize = (SUB_SIZE>>1);
      int innersize_log = LOG_SUB_SIZE-1;
      while (innersize >= 1)
        {
          //          if (threadIdx.x == 0 && blockIdx.x == 0 && (SUB_SIZE>>2) != blockDim.x) printf("here.\n");
          //     for (int idx = threadIdx.x; idx < (SUB_SIZE>>1); idx += blockDim.x)
            {
                  int idx = threadIdx.x;
          int subid = (idx>>innersize_log); ///innersize;
          int subindex = idx & (innersize-1); //% innersize;

          int index = (subid << (innersize_log+1)) + subindex;
          //          int index_mirror = (subid << (innersize_log+1)) + (innersize<<1) - subindex -1;
          int index_mirror = (subid << (innersize_log+1)) + innersize + subindex;

      //      printf("subid %d. subindex %d.index %d mirror %d threadIdx %d.\n", subid, subindex, index, index_mirror, idx);
          SORT_T left = s_array[index];
          SORT_T right = s_array[index_mirror];
          if (left > right)
            {
              s_array[index] = right;
              s_array[index_mirror] = left;
            }
            }
          innersize >>= 1;
          innersize_log -= 1;
          __syncthreads();
        }
      __syncthreads();
      buffer[(bid << LOG_SUB_SIZE) + threadIdx.x] = s_array[threadIdx.x];
      buffer[(bid << LOG_SUB_SIZE) + blockDim.x + threadIdx.x] = s_array[blockDim.x + threadIdx.x];
    }
}


__global__ void
bitonic_block2_unroll(SORT_T *buffer, int size)
{
  __shared__ SORT_T s_array[2048];
  int numbs = size >> 11;
  for (int bid = blockIdx.x; bid <  numbs; bid += gridDim.x)
    {
      /*      s_array[threadIdx.x] = buffer[(bid << 11) + threadIdx.x];
      s_array[blockDim.x + threadIdx.x] = buffer[(bid << 11) + blockDim.x + threadIdx.x];
      */
      //      memcpy_p(s_array, &buffer[bid << 11], 2048);
      SORT_T *dst = s_array+threadIdx.x;
      SORT_T *src = &buffer[(bid<<11)+ threadIdx.x];
      *dst = *src;  dst += blockDim.x;  src += blockDim.x;
      *dst = *src;  dst += blockDim.x;  src += blockDim.x;
      *dst = *src;  dst += blockDim.x;  src += blockDim.x;
      *dst = *src;  dst += blockDim.x;  src += blockDim.x;
      *dst = *src;  dst += blockDim.x;  src += blockDim.x;
      *dst = *src;  dst += blockDim.x;  src += blockDim.x;
      *dst = *src;  dst += blockDim.x;  src += blockDim.x;
      *dst = *src;

      __syncthreads();
      int innersize = (2048>>1);
      int innersize_log = 11-1;
      while (innersize >= 1)
        {
          //          for (int idx = threadIdx.x; idx < 1024; idx += blockDim.x)
             int idx = threadIdx.x;
            {          

          int subid = (idx>>innersize_log); ///innersize;
          int subindex = idx & (innersize-1); //% innersize;


          int index = (subid << (innersize_log+1)) + subindex;
          //          int index_mirror = (subid << (innersize_log+1)) + (innersize<<1) - subindex -1;
          int index_mirror = (subid << (innersize_log+1)) + innersize + subindex;

      //      printf("subid %d. subindex %d.index %d mirror %d threadIdx %d.\n", subid, subindex, index, index_mirror, threadIdx.x);
          SORT_T left = s_array[index];
          SORT_T right = s_array[index_mirror];
          if (left > right)
            {
              s_array[index] = right;
              s_array[index_mirror] = left;
            }
            }
            idx += blockDim.x;
            {          

          int subid = (idx>>innersize_log); ///innersize;
          int subindex = idx & (innersize-1); //% innersize;


          int index = (subid << (innersize_log+1)) + subindex;
          //          int index_mirror = (subid << (innersize_log+1)) + (innersize<<1) - subindex -1;
          int index_mirror = (subid << (innersize_log+1)) + innersize + subindex;

      //      printf("subid %d. subindex %d.index %d mirror %d threadIdx %d.\n", subid, subindex, index, index_mirror, threadIdx.x);
          SORT_T left = s_array[index];
          SORT_T right = s_array[index_mirror];
          if (left > right)
            {
              s_array[index] = right;
              s_array[index_mirror] = left;
            }
            }
            idx += blockDim.x;
            {          

          int subid = (idx>>innersize_log); ///innersize;
          int subindex = idx & (innersize-1); //% innersize;


          int index = (subid << (innersize_log+1)) + subindex;
          //          int index_mirror = (subid << (innersize_log+1)) + (innersize<<1) - subindex -1;
          int index_mirror = (subid << (innersize_log+1)) + innersize + subindex;

      //      printf("subid %d. subindex %d.index %d mirror %d threadIdx %d.\n", subid, subindex, index, index_mirror, threadIdx.x);
          SORT_T left = s_array[index];
          SORT_T right = s_array[index_mirror];
          if (left > right)
            {
              s_array[index] = right;
              s_array[index_mirror] = left;
            }
            }
            idx += blockDim.x;
            {          

          int subid = (idx>>innersize_log); ///innersize;
          int subindex = idx & (innersize-1); //% innersize;


          int index = (subid << (innersize_log+1)) + subindex;
          //          int index_mirror = (subid << (innersize_log+1)) + (innersize<<1) - subindex -1;
          int index_mirror = (subid << (innersize_log+1)) + innersize + subindex;

      //      printf("subid %d. subindex %d.index %d mirror %d threadIdx %d.\n", subid, subindex, index, index_mirror, threadIdx.x);
          SORT_T left = s_array[index];
          SORT_T right = s_array[index_mirror];
          if (left > right)
            {
              s_array[index] = right;
              s_array[index_mirror] = left;
            }

            }

          innersize >>= 1;
          innersize_log -= 1;
          __syncthreads();
        }
      __syncthreads();
          /*      buffer[(bid << 11) + threadIdx.x] = s_array[threadIdx.x];
      buffer[(bid << 11) + blockDim.x + threadIdx.x] = s_array[blockDim.x + threadIdx.x];
          */
      //       memcpy_p(&buffer[bid << 11], s_array, 2048);
              {
      SORT_T *dst = &buffer[(bid<<11)+ threadIdx.x];
      SORT_T *src = s_array+threadIdx.x;
      *dst = *src;  dst += blockDim.x;  src += blockDim.x;
      *dst = *src;  dst += blockDim.x;  src += blockDim.x;
      *dst = *src;  dst += blockDim.x;  src += blockDim.x;
      *dst = *src;  dst += blockDim.x;  src += blockDim.x;
      *dst = *src;  dst += blockDim.x;  src += blockDim.x;
      *dst = *src;  dst += blockDim.x;  src += blockDim.x;
      *dst = *src;  dst += blockDim.x;  src += blockDim.x;
      *dst = *src;
              }
    }

}



__global__ void
bitonic_block2(SORT_T *buffer, int size)
{
  __shared__ SORT_T s_array[2048];
  int numbs = size >> 11;
  for (int bid = blockIdx.x; bid <  numbs; bid += gridDim.x)
    {
      /*      s_array[threadIdx.x] = buffer[(bid << 11) + threadIdx.x];
      s_array[blockDim.x + threadIdx.x] = buffer[(bid << 11) + blockDim.x + threadIdx.x];
      */
      memcpy_p(s_array, &buffer[bid << 11], 2048);
      /*
      SORT_T *dst = s_array+threadIdx.x;
      SORT_T *src = &buffer[(bid<<11)+ threadIdx.x];
      *dst = *src;  dst += blockDim.x;  src += blockDim.x;
      *dst = *src;  dst += blockDim.x;  src += blockDim.x;
      *dst = *src;  dst += blockDim.x;  src += blockDim.x;
      *dst = *src;  dst += blockDim.x;  src += blockDim.x;
      *dst = *src;  dst += blockDim.x;  src += blockDim.x;
      *dst = *src;  dst += blockDim.x;  src += blockDim.x;
      *dst = *src;  dst += blockDim.x;  src += blockDim.x;
      *dst = *src;*/

      __syncthreads();
      int innersize = (2048>>1);
      int innersize_log = 11-1;
      while (innersize >= 1)
        {
           for (int idx = threadIdx.x; idx < 1024; idx += blockDim.x)
            {          
          int subid = (idx>>innersize_log); ///innersize;
          int subindex = idx & (innersize-1); //% innersize;


          int index = (subid << (innersize_log+1)) + subindex;
          //          int index_mirror = (subid << (innersize_log+1)) + (innersize<<1) - subindex -1;
          int index_mirror = (subid << (innersize_log+1)) + innersize + subindex;

      //      printf("subid %d. subindex %d.index %d mirror %d threadIdx %d.\n", subid, subindex, index, index_mirror, threadIdx.x);
          SORT_T left = s_array[index];
          SORT_T right = s_array[index_mirror];
          if (left > right)
            {
              s_array[index] = right;
              s_array[index_mirror] = left;
            }

            }

          innersize >>= 1;
          innersize_log -= 1;
          __syncthreads();
        }
      __syncthreads();
          /*      buffer[(bid << 11) + threadIdx.x] = s_array[threadIdx.x];
      buffer[(bid << 11) + blockDim.x + threadIdx.x] = s_array[blockDim.x + threadIdx.x];
          */
           memcpy_p(&buffer[bid << 11], s_array, 2048);
           /* {
      SORT_T *dst = &buffer[(bid<<11)+ threadIdx.x];
      SORT_T *src = s_array+threadIdx.x;
      *dst = *src;  dst += blockDim.x;  src += blockDim.x;
      *dst = *src;  dst += blockDim.x;  src += blockDim.x;
      *dst = *src;  dst += blockDim.x;  src += blockDim.x;
      *dst = *src;  dst += blockDim.x;  src += blockDim.x;
      *dst = *src;  dst += blockDim.x;  src += blockDim.x;
      *dst = *src;  dst += blockDim.x;  src += blockDim.x;
      *dst = *src;  dst += blockDim.x;  src += blockDim.x;
      *dst = *src;
      }*/
    }

}
