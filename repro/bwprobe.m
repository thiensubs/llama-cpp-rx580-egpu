#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// read-only streaming bandwidth: sums a big float buffer, the same access pattern
// a decode-step matvec has (read every weight once, write almost nothing).

static const char * SRC =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"kernel void sumbuf(device const float4 * in [[buffer(0)]],\n"
"                   device float * out [[buffer(1)]],\n"
"                   constant uint & n4 [[buffer(2)]],\n"
"                   uint gid [[thread_position_in_grid]],\n"
"                   uint gsz [[threads_per_grid]],\n"
"                   uint ti [[thread_index_in_threadgroup]],\n"
"                   uint ss [[threads_per_simdgroup]]) {\n"
"    float4 acc = 0.0f;\n"
"    for (uint i = gid; i < n4; i += gsz) acc += in[i];\n"
"    float s = acc.x + acc.y + acc.z + acc.w;\n"
"    s = simd_sum(s);\n"
"    if (ti % ss == 0) out[gid / ss] = s;\n"
"}\n";

int main(int argc, char ** argv) {
    @autoreleasepool {
        size_t gb = argc > 1 ? (size_t) atoi(argv[1]) : 1;
        size_t bytes = gb * 1024ull*1024ull*1024ull;
        NSArray<id<MTLDevice>> * devs = MTLCopyAllDevices();
        NSUInteger di = argc > 2 ? (NSUInteger) atoi(argv[2]) : 0;
        id<MTLDevice> d = devs[di];
        printf("device[%lu] = %s\n", (unsigned long)di, [[d name] UTF8String]);

        NSError * err = nil;
        id<MTLLibrary> lib = [d newLibraryWithSource:[NSString stringWithUTF8String:SRC] options:nil error:&err];
        if (!lib) { printf("compile failed: %s\n", [[err description] UTF8String]); return 1; }
        id<MTLComputePipelineState> pso = [d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"sumbuf"] error:&err];
        if (!pso) { printf("pso failed: %s\n", [[err description] UTF8String]); return 1; }

        id<MTLBuffer> in = [d newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
        id<MTLBuffer> out = [d newBufferWithLength:64*1024 options:MTLResourceStorageModePrivate];
        if (!in) { printf("alloc failed\n"); return 1; }

        id<MTLCommandQueue> q = [d newCommandQueue];
        const uint32_t n4 = (uint32_t)(bytes / 16);
        const int TPB = 256;
        const uint32_t nthreads = 512u * 1024u;      // fixed grid, each thread strides

        for (int it = 0; it < 3; it++) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:pso];
            [e setBuffer:in  offset:0 atIndex:0];
            [e setBuffer:out offset:0 atIndex:1];
            [e setBytes:&n4 length:8 atIndex:2];
            [e dispatchThreadgroups:MTLSizeMake(nthreads/TPB,1,1) threadsPerThreadgroup:MTLSizeMake(TPB,1,1)];
            [e endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.error) { printf("kernel error: %s\n", [[[cb error] description] UTF8String]); return 1; }
            double sec = cb.GPUEndTime - cb.GPUStartTime;
            printf("  iter %d: %.1f MB streamed in %.3f ms -> %.1f GB/s\n",
                   it, bytes/1e6, sec*1e3, bytes/sec/1e9);
        }
    }
    return 0;
}
