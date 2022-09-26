#define CUB_STDERR
#include <stdio.h>
#include <sys/time.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/reduce.h>

#include <cub/device/device_reduce.cuh>
#include <cub/util_allocator.cuh>
using namespace cub;
//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------
bool g_verbose = false;                    // Whether to display input/output to console
CachingDeviceAllocator g_allocator(true);  // Caching allocator for device memory
//---------------------------------------------------------------------
// Test generation
//---------------------------------------------------------------------
void Initialize(
    int *h_in,
    int num_items) {
    for (int i = 0; i < num_items; ++i)
        h_in[i] = i;
    if (g_verbose) {
        printf("Input:\n");
        // DisplayResults(h_in, num_items);
        printf("\n\n");
    }
}
void Solve(
    int *h_in,
    int &h_reference,
    int num_items) {
    for (int i = 0; i < num_items; ++i) {
        if (i == 0)
            h_reference = h_in[0];
        else
            h_reference += h_in[i];
    }
}

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}
//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------
int main(int argc, char **argv) {
    int num_items = 2 << 16;
    // Initialize command line
    // CommandLineArgs args(argc, argv);
    // g_verbose = args.CheckCmdLineFlag("v");
    // args.GetCmdLineArgument("n", num_items);
    // // Print usage
    // if (args.CheckCmdLineFlag("help")) {
    //     printf(
    //         "%s "
    //         "[--n=<input items> "
    //         "[--device=<device-id>] "
    //         "[--v] "
    //         "\n",
    //         argv[0]);
    //     exit(0);
    // }
    // Initialize device
    // CubDebugExit(args.DeviceInit());
    printf("cub::DeviceReduce::Sum() %d items (%d-byte elements)\n",
           num_items, (int)sizeof(int));
    fflush(stdout);
    // Allocate host arrays
    int *h_in = new int[num_items];
    int h_reference;
    // Initialize problem and solution
    Initialize(h_in, num_items);
    Solve(h_in, h_reference, num_items);
    // Allocate problem device arrays
    int *d_in = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void **)&d_in, sizeof(int) * num_items));
    // Initialize device input
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(int) * num_items, cudaMemcpyHostToDevice));
    // Allocate device output array
    int *d_out = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void **)&d_out, sizeof(int) * 1));
    // Request and allocate temporary storage
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
    // Run
    CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items));
    // Check for correctness (and display results, if specified)
    // int compare = CompareDeviceResults(&h_reference, d_out, 1, g_verbose, g_verbose);
    // printf("\t%s", compare ? "FAIL" : "PASS");
    // AssertEquals(0, compare);

    double istart = cpuSecond();
    int sum = thrust::reduce(d_in, d_in + (2 << 16));
    printf("thrust: %lf\n", cpuSecond() - istart);
    // Cleanup
    if (h_in) delete[] h_in;
    if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
    if (d_out) CubDebugExit(g_allocator.DeviceFree(d_out));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
    printf("\n\n");
    return 0;
}
