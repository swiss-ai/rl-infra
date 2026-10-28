// Single-GPU reproduction of the UCCL combine fault.
//
// UCCL (ep/src/uccl_ep.cc) keeps its 32 D2H queue handles (40 bytes each) and
// the table of pointers to them in cudaMallocManaged memory, initialised on the
// host and prefetched to the GPU. This program builds the same structure, reads
// it from a kernel the way combine does (table[i], then the handle), fills the
// GPU with cudaMalloc as PyTorch's caching allocator does at the logprob peak,
// and reads it again.
//
// Modes:
//   managed              UCCL's allocation, unchanged
//   managed-accessed-by  same, plus cudaMemAdvise preferred location and
//                        accessed-by for the GPU
//   device               table and handles in cudaMalloc memory (the fix)
//
// Appending -concurrent (e.g. managed-concurrent) fills the GPU while a kernel
// keeps reading the table for 20 s, as in training, where PyTorch allocates
// while UCCL's kernels run.

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CK(call)                                                           \
  do {                                                                     \
    cudaError_t err_ = (call);                                             \
    if (err_ != cudaSuccess) {                                             \
      std::printf("FAILED %s: %s (%d)\n", #call, cudaGetErrorString(err_), \
                  static_cast<int>(err_));                                 \
      std::exit(2);                                                        \
    }                                                                      \
  } while (0)

constexpr int kNumHandles = 32;

struct Handle {
  unsigned long long words[5];
};

__global__ void read_table(unsigned long long const* table, int n,
                           unsigned long long* out) {
  int i = threadIdx.x;
  if (i < n) {
    auto const* handle = reinterpret_cast<Handle const*>(table[i]);
    unsigned long long sum = 0;
    for (int k = 0; k < 5; ++k) sum += handle->words[k];
    out[i] = sum;
  }
}

__device__ unsigned long long global_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__global__ void spin_read(unsigned long long const* table, int n,
                          unsigned long long* out,
                          unsigned long long duration_ns) {
  int i = threadIdx.x;
  if (i >= n) return;
  auto const volatile* entries =
      reinterpret_cast<unsigned long long const volatile*>(table);
  unsigned long long mismatches = 0, reads = 0;
  unsigned long long const end = global_ns() + duration_ns;
  while (global_ns() < end) {
    auto const volatile* words =
        reinterpret_cast<unsigned long long const volatile*>(entries[i]);
    if (words[0] != 5ull * i + 1) ++mismatches;
    ++reads;
  }
  out[i] = mismatches;
  out[n + i] = reads;
}

static double gib(size_t bytes) { return bytes / double(1ull << 30); }

static void report_mem(char const* tag) {
  size_t free_bytes = 0, total_bytes = 0;
  CK(cudaMemGetInfo(&free_bytes, &total_bytes));
  std::printf("%-30s free %.3f of %.3f GiB\n", tag, gib(free_bytes),
              gib(total_bytes));
}

static bool timed_read(char const* tag, unsigned long long const* table,
                       unsigned long long* out,
                       unsigned long long const* expected) {
  auto start = std::chrono::steady_clock::now();
  read_table<<<1, kNumHandles>>>(table, kNumHandles, out);
  cudaError_t err = cudaDeviceSynchronize();
  double us = std::chrono::duration<double, std::micro>(
                  std::chrono::steady_clock::now() - start)
                  .count();
  if (err != cudaSuccess) {
    std::printf("%-30s ERROR after %.0f us: %s (%d)\n", tag, us,
                cudaGetErrorString(err), static_cast<int>(err));
    return false;
  }
  unsigned long long got[kNumHandles];
  CK(cudaMemcpy(got, out, sizeof(got), cudaMemcpyDeviceToHost));
  bool correct = true;
  for (int i = 0; i < kNumHandles; ++i) correct &= got[i] == expected[i];
  std::printf("%-30s ok in %.0f us, values %s\n", tag, us,
              correct ? "correct" : "WRONG");
  return true;
}

static void advise(void const* ptr, size_t bytes, cudaMemoryAdvise advice,
                   int dev) {
#if CUDART_VERSION >= 13000
  cudaMemLocation loc;
  loc.type = cudaMemLocationTypeDevice;
  loc.id = dev;
  CK(cudaMemAdvise(ptr, bytes, advice, loc));
#else
  CK(cudaMemAdvise(ptr, bytes, advice, dev));
#endif
}

static void prefetch(void const* ptr, size_t bytes, int dev) {
#if CUDART_VERSION >= 13000
  cudaMemLocation loc;
  loc.type = cudaMemLocationTypeDevice;
  loc.id = dev;
  CK(cudaMemPrefetchAsync(ptr, bytes, loc, 0));
#else
  CK(cudaMemPrefetchAsync(ptr, bytes, dev));
#endif
}

int main(int argc, char** argv) {
  std::string mode = argc > 1 ? argv[1] : "managed";
  std::string const suffix = "-concurrent";
  bool const concurrent =
      mode.size() > suffix.size() &&
      mode.compare(mode.size() - suffix.size(), suffix.size(), suffix) == 0;
  std::string const run_name = mode;
  if (concurrent) mode.resize(mode.size() - suffix.size());
  int const dev = 0;
  CK(cudaSetDevice(dev));

  int driver = 0, runtime = 0, concurrent_access = 0, host_page_tables = 0;
  CK(cudaDriverGetVersion(&driver));
  CK(cudaRuntimeGetVersion(&runtime));
  CK(cudaDeviceGetAttribute(&concurrent_access,
                            cudaDevAttrConcurrentManagedAccess, dev));
  CK(cudaDeviceGetAttribute(&host_page_tables,
                            cudaDevAttrPageableMemoryAccessUsesHostPageTables,
                            dev));
  char const* force = std::getenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC");
  std::printf(
      "mode %s, driver %d, runtime %d, concurrentManagedAccess %d, "
      "pageableMemoryAccessUsesHostPageTables %d, "
      "CUDA_MANAGED_FORCE_DEVICE_ALLOC=%s\n",
      run_name.c_str(), driver, runtime, concurrent_access, host_page_tables,
      force ? force : "(unset)");

  Handle host_handles[kNumHandles];
  unsigned long long expected[kNumHandles];
  for (int i = 0; i < kNumHandles; ++i) {
    expected[i] = 0;
    for (int k = 0; k < 5; ++k) {
      host_handles[i].words[k] = 5ull * i + k + 1;
      expected[i] += host_handles[i].words[k];
    }
  }

  size_t const handle_bytes = kNumHandles * sizeof(Handle);
  size_t const table_bytes = kNumHandles * sizeof(unsigned long long);
  Handle* handles = nullptr;
  unsigned long long* table = nullptr;
  if (mode == "device") {
    CK(cudaMalloc(&handles, handle_bytes));
    CK(cudaMalloc(&table, table_bytes));
    unsigned long long host_table[kNumHandles];
    for (int i = 0; i < kNumHandles; ++i)
      host_table[i] = reinterpret_cast<unsigned long long>(handles + i);
    CK(cudaMemcpy(handles, host_handles, handle_bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(table, host_table, table_bytes, cudaMemcpyHostToDevice));
  } else if (mode == "managed" || mode == "managed-accessed-by") {
    CK(cudaMallocManaged(&handles, handle_bytes));
    CK(cudaMallocManaged(&table, table_bytes));
    for (int i = 0; i < kNumHandles; ++i) {
      handles[i] = host_handles[i];
      table[i] = reinterpret_cast<unsigned long long>(handles + i);
    }
    if (mode == "managed-accessed-by") {
      advise(handles, handle_bytes, cudaMemAdviseSetPreferredLocation, dev);
      advise(table, table_bytes, cudaMemAdviseSetPreferredLocation, dev);
      advise(handles, handle_bytes, cudaMemAdviseSetAccessedBy, dev);
      advise(table, table_bytes, cudaMemAdviseSetAccessedBy, dev);
    }
    prefetch(handles, handle_bytes, dev);
    prefetch(table, table_bytes, dev);
    CK(cudaDeviceSynchronize());
  } else {
    std::printf("unknown mode %s\n", mode.c_str());
    return 2;
  }
  std::printf("table at %p, handles at %p\n", static_cast<void*>(table),
              static_cast<void*>(handles));

  unsigned long long* out = nullptr;
  CK(cudaMalloc(&out, 2 * table_bytes));
  report_mem("before fill:");
  if (!timed_read("read before fill:", table, out, expected)) return 1;
  if (!timed_read("read before fill, again:", table, out, expected)) return 1;

  if (concurrent) {
    cudaStream_t stream;
    CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    auto start = std::chrono::steady_clock::now();
    auto seconds = [&start] {
      return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                           start)
          .count();
    };
    spin_read<<<1, kNumHandles, 0, stream>>>(table, kNumHandles, out,
                                             20ull * 1000 * 1000 * 1000);
    CK(cudaGetLastError());
    std::vector<void*> blocks;
    size_t filled = 0;
    for (size_t chunk = 1ull << 30; chunk >= (64ull << 10);) {
      void* block = nullptr;
      if (cudaMalloc(&block, chunk) == cudaSuccess) {
        blocks.push_back(block);
        filled += chunk;
      } else {
        cudaGetLastError();
        chunk /= 2;
      }
    }
    cudaError_t state = cudaStreamQuery(stream);
    std::printf(
        "filled %.3f GiB in %zu cudaMalloc blocks by %.1f s, reader %s\n",
        gib(filled), blocks.size(), seconds(),
        state == cudaErrorNotReady ? "still running"
        : state == cudaSuccess     ? "already finished"
                                   : cudaGetErrorString(state));
    report_mem("after fill:");
    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
      std::printf("%-30s ERROR at %.1f s: %s (%d)\n", "reader:", seconds(),
                  cudaGetErrorString(err), static_cast<int>(err));
      std::printf("RESULT %s: fault reproduced\n", run_name.c_str());
      return 1;
    }
    unsigned long long counts[2 * kNumHandles];
    CK(cudaMemcpy(counts, out, sizeof(counts), cudaMemcpyDeviceToHost));
    unsigned long long mismatches = 0, reads = 0;
    for (int i = 0; i < kNumHandles; ++i) {
      mismatches += counts[i];
      reads += counts[kNumHandles + i];
    }
    std::printf("%-30s ok at %.1f s, %llu reads, %llu wrong values\n",
                "reader:", seconds(), reads, mismatches);
    for (void* block : blocks) CK(cudaFree(block));
    std::printf("RESULT %s: no fault\n", run_name.c_str());
    return 0;
  }

  std::vector<void*> blocks;
  size_t filled = 0;
  for (size_t chunk = 1ull << 30; chunk >= (64ull << 10);) {
    void* block = nullptr;
    if (cudaMalloc(&block, chunk) == cudaSuccess) {
      CK(cudaMemset(block, 0, chunk));
      blocks.push_back(block);
      filled += chunk;
    } else {
      cudaGetLastError();
      chunk /= 2;
    }
  }
  CK(cudaDeviceSynchronize());
  std::printf("filled %.3f GiB in %zu cudaMalloc blocks\n", gib(filled),
              blocks.size());
  report_mem("after fill:");

  bool ok = timed_read("read with HBM full:", table, out, expected) &&
            timed_read("read with HBM full, again:", table, out, expected);
  if (!ok) {
    std::printf("RESULT %s: fault reproduced\n", run_name.c_str());
    return 1;
  }

  for (void* block : blocks) CK(cudaFree(block));
  report_mem("after freeing the fill:");
  timed_read("read after freeing the fill:", table, out, expected);
  std::printf("RESULT %s: no fault\n", run_name.c_str());
  return 0;
}
