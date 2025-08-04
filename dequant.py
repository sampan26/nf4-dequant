from triton import jit
import triton
import triton.language as tl

@triton.jit
def dequantize_nf4_kernel(w_ptr, w_out, abs_ptrs,
                                offset_ptr, 
                                abs2_ptrs, code2,
                                block_size2,gsize,code,blocks:tl.constexpr, Br:tl.constexpr):
    

    pid = tl.program_id(0)
    
    if pid < gsize:

        absmax_group = pid*blocks + tl.arange(0,blocks) # can be coalesced
        absmax = (tl.load(abs_ptrs + absmax_group,  eviction_policy= "evict_first"))

        lz = tl.inline_asm_elementwise(
        # efficient log2(blockwise2)
        asm="""
        {
            clz.b32 $0, $1;
        }
        """,
        constraints=(
            "=r,r"),
        args=[block_size2],
        dtype=(tl.int32),
        is_pure=True,
        pack=1,
        )

        absmax_group2 = (absmax_group)>>(31-lz)
        real_absmax = tl.load(code2 + absmax,  eviction_policy= "evict_last")
        absmax2 = tl.load(abs2_ptrs + absmax_group2,  eviction_policy= "evict_last")
        offset = tl.load(offset_ptr,  eviction_policy= "evict_last") 
        final_absmax = absmax2 * real_absmax + offset

        w_off = pid*(Br//2) + tl.arange(0, blocks)[:, None]*(Br//(2*blocks)) + tl.arange(0, Br//(2*blocks))[None, :]

        w_packed = tl.load(w_ptr + w_off,  eviction_policy= "evict_first")
        w_packed2 = tl.interleave(w_packed,w_packed)
        

        shift_sh = tl.arange(0, blocks)[:, None]*(Br//(blocks)) + tl.arange(0, Br//(blocks))[None, :]
        shift = tl.where(shift_sh % 2 == 0, 4, 0)

        shifted_w = (w_packed2 >> shift) & 0xF
        
        real_w = tl.load(shifted_w + code, eviction_policy= "evict_last")

        scaled_w = (real_w* final_absmax[:, None])

        out_off = pid*Br + tl.arange(0, blocks)[:, None]*(Br//blocks) + tl.arange(0, Br//blocks)[None, :]
        
        tl.store(w_out + out_off, scaled_w, eviction_policy= "evict_first")
    return

def dequantize_nf4(weight, quant_state):
    device = 'cuda:0'
    out_dtype = quant_state.dtype
    code = quant_state.code

    absmax = quant_state.absmax #uint8
    real_shape = quant_state.shape
    block_size = quant_state.blocksize

    absmax2 = quant_state.state2.absmax #f32
    code2 = quant_state.state2.code #f32
    block_size2 = quant_state.state2.blocksize

    size = weight.shape[0]
    offset = quant_state.offset
    out_size = size*2

    Br = 8192
    blocks = Br//block_size
    
    gsize = (triton.cdiv(out_size, Br))

    DEVICE = torch.device(device)
    props = torch.cuda.get_device_properties(DEVICE)
    if props.major ==8:
        if props.minor == 9: #Ada
            max_th = 24*props.multi_processor_count
        elif props.minor == 0: #Ampere
            max_th = 32*props.multi_processor_count
    elif props.major==7:
        max_th = 16*props.multi_processor_count # Turing
    resto = gsize%max_th
    if resto !=0:
        wave_sze =gsize+ (max_th-resto)

    w_out = torch.empty((real_shape), device=device, dtype = out_dtype 
                        if out_dtype == torch.float16 else torch.bfloat16, requires_grad=False)
    
    grid = lambda META: ((wave_sze,))
    out = dequantize_nf4_kernel[grid](weight, w_out,
                                             absmax, offset,
                                             absmax2, code2,
                                             block_size2, gsize, code, blocks, Br,
                                             num_warps = 16, num_stages=1, maxnreg=8192,
                                             )


    return w_out if out_dtype == torch.float16 else w_out

def dequantize_nf4(weight):
    return dequantize_nf4(weight.weight.data, weight.weight.quant_state)
