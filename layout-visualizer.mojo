from math import ceildiv
from sys import argv, size_of

from buffer.dimlist import DimList
from gpu import WARP_SIZE, barrier
from gpu import lane_id as get_lane_id, warp_id
from gpu.cluster import block_rank_in_cluster
from gpu.host import DeviceContext, FuncAttribute
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu import block_idx, lane_id, thread_idx
from gpu.memory import external_memory, fence_async_view_proxy
from gpu.mma import st_matrix
from gpu.tcgen05 import *

# Additional imports for testing
from layout import (
    UNKNOWN_VALUE,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    RuntimeTuple,
)
from layout._fillers import arange
from layout._utils import ManagedLayoutTensor
from layout.int_tuple import IntTuple
from layout.swizzle import make_ldmatrix_swizzle, make_swizzle
from layout.tensor_core_async import (
    st_matrix_n_layout,
    tile_layout_k_major,
    tile_layout_mn_major,
    tile_to_descriptor,
)
from layout.tma_async import SharedMemBarrier, TMATensorTile, create_tma_tile

from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple


@__llvm_metadata(`nvvm.cluster_dim`=(1, 1, 1))
@__llvm_arg_metadata(a_tma_tile, `nvvm.grid_constant`)
fn visualize_tma[
    dtype: DType,
    a_layout: Layout,
    a_desc_layout: Layout,
    a_swizzle: TensorMapSwizzle,
    M: Int,
    N: Int,
    K: Int,
    BM: Int,
    BN: Int,
    BK: Int,
](a_tma_tile: TMATensorTile[dtype, a_layout, a_desc_layout]):
    comptime a_smem_layout = tile_layout_k_major[
        dtype, BM, BK, swizzle_mode=a_swizzle
    ]()
    comptime a_size = a_smem_layout.size()
    comptime a_expected_bytes = a_size * size_of[dtype]()

    comptime a_smem_size = BM * BK

    var start_smem = external_memory[
            Scalar[dtype],
            address_space = AddressSpace.SHARED,
            alignment=128,
            name="tmem_dynamic_shared_memory",
        ]()

    var smem_a = LayoutTensor[
        dtype,
        Layout.row_major(BM, BK),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ](start_smem)

    var tma_mbar = (smem_a.ptr + a_smem_size).bitcast[SharedMemBarrier]()

def main():
    comptime M = 256
    comptime N = 128
    comptime K = 128

    comptime BM = 128
    comptime BN = 64
    comptime BK = 64

    comptime block_tile_shape = Index(BM, BN, BK)

    # 1SM Shape for UMMA
    comptime UMMA_1SM = Index(128, 64, 16)
    # 2SM Shape for UMMA
    comptime UMMA_2SM = Index(256, 128, 16)

    comptime dtype = DType.bfloat16

    comptime a_swizzle = TensorMapSwizzle.SWIZZLE_NONE

    var ctx = DeviceContext()
    var a_h = ctx.enqueue_create_host_buffer[dtype](M * K)
    var a_d = ctx.enqueue_create_buffer[dtype](M * K)

    comptime a_layout = Layout.row_major(M, K)
    var a_tensor = LayoutTensor[dtype, a_layout](a_d)

    var a_tma_tile = create_tma_tile[Index(BM, BK), swizzle_mode=a_swizzle](
        ctx, a_tensor
    )

    comptime kernel = visualize_tma[
        dtype,
        a_layout,
        a_tma_tile.desc_layout,
        a_swizzle,
        M,
        N,
        K,
        BM,
        BN,
        BK,
    ]

    comptime smem_use = (
        BM * size_of[dtype]()
    ) * BK + 24

    ctx.enqueue_function_checked[kernel, kernel](
        a_tma_tile,
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
        block_dim=(1,),
        shared_mem_bytes=Int(smem_use),
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(smem_use),
    )