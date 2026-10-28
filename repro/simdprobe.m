#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static const char * SRC =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"kernel void probe(device uint * out [[buffer(0)]],\n"
"                  uint lane   [[thread_index_in_threadgroup]],\n"
"                  uint slane  [[thread_index_in_simdgroup]],\n"
"                  uint ssize  [[threads_per_simdgroup]],\n"
"                  uint sid    [[simdgroup_index_in_threadgroup]]) {\n"
"    uint sum   = simd_sum(lane);\n"
"    uint bcast = simd_broadcast(lane, 0u);\n"
"    uint shuf  = simd_shuffle(lane, 5u % ssize);\n"
"    uint pref  = simd_prefix_inclusive_sum(lane);\n"
"    uint mx    = simd_max(lane);\n"
"    if (slane == 0) {\n"
"        uint b = (sid) * 8u;\n"
"        out[b+0] = ssize; out[b+1] = sum;  out[b+2] = bcast;\n"
"        out[b+3] = shuf;  out[b+4] = pref; out[b+5] = mx;\n"
"        out[b+6] = sid;   out[b+7] = lane;\n"
"    }\n"
"}\n";

int main(void) {
    @autoreleasepool {
        NSArray<id<MTLDevice>> * devs = MTLCopyAllDevices();
        for (NSUInteger i = 0; i < devs.count; i++) {
            id<MTLDevice> d = devs[i];
            printf("\n[%lu] %s  lowPower=%d removable=%d unified=%d maxWG=%lu\n",
                   (unsigned long)i, [[d name] UTF8String],
                   (int)[d isLowPower], (int)[d isRemovable], (int)[d hasUnifiedMemory],
                   (unsigned long)[d maxThreadsPerThreadgroup].width);

            NSError * err = nil;
            id<MTLLibrary> lib = [d newLibraryWithSource:[NSString stringWithUTF8String:SRC]
                                                 options:nil error:&err];
            if (!lib) { printf("    compile FAILED: %s\n", [[err description] UTF8String]); continue; }
            id<MTLFunction> fn = [lib newFunctionWithName:@"probe"];
            id<MTLComputePipelineState> pso = [d newComputePipelineStateWithFunction:fn error:&err];
            if (!pso) { printf("    pipeline FAILED: %s\n", [[err description] UTF8String]); continue; }

            const int WG = 256;              // 256 threads => 4 or 8 simdgroups
            id<MTLBuffer> out = [d newBufferWithLength:WG*8*sizeof(uint32_t)
                                               options:MTLResourceStorageModeShared];
            memset(out.contents, 0, out.length);
            id<MTLCommandQueue> q = [d newCommandQueue];
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:pso];
            [e setBuffer:out offset:0 atIndex:0];
            [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(WG,1,1)];
            [e endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.error) { printf("    kernel error: %s\n", [[[cb error] description] UTF8String]); continue; }

            uint32_t * o = (uint32_t *) out.contents;
            int nsimd = 0;
            for (int s = 0; s < WG/8; s++) {
                uint32_t ssize = o[s*8+0];
                if (ssize == 0) break;
                nsimd++;
                uint32_t sum = o[s*8+1], bcast = o[s*8+2], shuf = o[s*8+3];
                uint32_t pref = o[s*8+4], mx = o[s*8+5], lane0 = o[s*8+7];
                uint32_t expect_sum = ssize*(ssize-1)/2;     // sum of lane ids in the simdgroup
                printf("    simdgroup %d: width=%u  sum=%u(exp %u)%s  bcast=%u  shuffle5=%u(exp 5)%s  prefix0=%u  max=%u(exp %u)%s  lane0=%u\n",
                       s, ssize, sum, expect_sum, sum==expect_sum?" OK":" BAD",
                       bcast, shuf, shuf==5?" OK":" BAD",
                       pref, mx, ssize-1, mx==ssize-1?" OK":" BAD", lane0);
                if (ssize > 8) break;
            }
            printf("    -> simdgroups detected: %d (expected %d if width %u)\n",
                   nsimd, nsimd ? WG/nsimd : 0, nsimd ? WG/nsimd : 0);
        }
    }
    return 0;
}
