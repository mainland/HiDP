#pragma once

/* Matrix size */
#define WARPS_PER_BLOCK 8
#define WARP_SIZE 32
#define BLOCK_SIZE (WARPS_PER_BLOCK * WARP_SIZE)
#define BLOCK_SIZE_SQROOT 16
