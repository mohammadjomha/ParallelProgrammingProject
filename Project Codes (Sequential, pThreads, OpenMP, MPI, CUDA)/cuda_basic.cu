#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define LAT_BINS 180
#define LON_BINS 360
#define TIME_BINS 120

typedef struct {
    long timestamp;
    float latitude;
    float longitude;
    float magnitude;
} EarthquakeEvent;

__device__ void compute_bin_indices(EarthquakeEvent event, long min_time, long max_time,
                                    int *lat_idx, int *lon_idx, int *time_idx) {
    *lat_idx = (int)((event.latitude + 90.0f) * LAT_BINS / 180.0f);
    if (*lat_idx >= LAT_BINS) *lat_idx = LAT_BINS - 1;
    if (*lat_idx < 0) *lat_idx = 0;

    *lon_idx = (int)((event.longitude + 180.0f) * LON_BINS / 360.0f);
    if (*lon_idx >= LON_BINS) *lon_idx = LON_BINS - 1;
    if (*lon_idx < 0) *lon_idx = 0;

    *time_idx = (int)((event.timestamp - min_time) * TIME_BINS /
                     (double)(max_time - min_time + 1));
    if (*time_idx >= TIME_BINS) *time_idx = TIME_BINS - 1;
    if (*time_idx < 0) *time_idx = 0;
}

__device__ float atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                       __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void histogram_kernel(EarthquakeEvent *events, int n_events,
                                 int *counts, float *mags, float *maxs,
                                 long min_time, long max_time) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_events) {
        int lat_idx, lon_idx, time_idx;
        compute_bin_indices(events[idx], min_time, max_time,
                          &lat_idx, &lon_idx, &time_idx);

        int cell_idx = lat_idx * LON_BINS * TIME_BINS + lon_idx * TIME_BINS + time_idx;

        atomicAdd(&counts[cell_idx], 1);
        atomicAdd(&mags[cell_idx], events[idx].magnitude);
        atomicMaxFloat(&maxs[cell_idx], events[idx].magnitude);
    }
}

int load_binary_data(const char *filename, EarthquakeEvent **events) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        perror("Error opening file");
        return -1;
    }

    int n_events;
    fread(&n_events, sizeof(int), 1, fp);

    *events = (EarthquakeEvent *)malloc(n_events * sizeof(EarthquakeEvent));
    if (!*events) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(fp);
        return -1;
    }

    for (int i = 0; i < n_events; i++) {
        fread(&(*events)[i], sizeof(EarthquakeEvent), 1, fp);
    }

    fclose(fp);
    return n_events;
}

void write_histogram_csv(const char *filename, int *counts, float *mags, float *maxs) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("Error opening output file");
        return;
    }

    fprintf(fp, "lat_bin,lon_bin,time_bin,count,total_magnitude,max_magnitude\n");

    for (int lat = 0; lat < LAT_BINS; lat++) {
        for (int lon = 0; lon < LON_BINS; lon++) {
            for (int time = 0; time < TIME_BINS; time++) {
                int idx = lat * LON_BINS * TIME_BINS + lon * TIME_BINS + time;
                if (counts[idx] > 0) {
                    fprintf(fp, "%d,%d,%d,%d,%.2f,%.2f\n",
                           lat, lon, time, counts[idx], mags[idx], maxs[idx]);
                }
            }
        }
    }

    fclose(fp);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <binary_data_file> <block_size>\n", argv[0]);
        return 1;
    }

    int block_size = atoi(argv[2]);
    if (block_size < 1) block_size = 256;

    EarthquakeEvent *h_events;
    int n_events = load_binary_data(argv[1], &h_events);

    if (n_events <= 0) {
        fprintf(stderr, "Failed to load data\n");
        return 1;
    }

    printf("=== CUDA Basic Implementation (Block Size %d) ===\n", block_size);
    printf("Loaded %d earthquake events\n", n_events);

    long min_time = h_events[0].timestamp;
    long max_time = h_events[0].timestamp;
    for (int i = 1; i < n_events; i++) {
        if (h_events[i].timestamp < min_time) min_time = h_events[i].timestamp;
        if (h_events[i].timestamp > max_time) max_time = h_events[i].timestamp;
    }

    int total_cells = LAT_BINS * LON_BINS * TIME_BINS;
    printf("Grid dimensions: %d x %d x %d = %d cells\n",
           LAT_BINS, LON_BINS, TIME_BINS, total_cells);

    EarthquakeEvent *d_events;
    int *d_counts;
    float *d_mags, *d_maxs;

    cudaMalloc(&d_events, n_events * sizeof(EarthquakeEvent));
    cudaMalloc(&d_counts, total_cells * sizeof(int));
    cudaMalloc(&d_mags, total_cells * sizeof(float));
    cudaMalloc(&d_maxs, total_cells * sizeof(float));

    cudaMemset(d_counts, 0, total_cells * sizeof(int));
    cudaMemset(d_mags, 0, total_cells * sizeof(float));
    cudaMemset(d_maxs, 0, total_cells * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cudaMemcpy(d_events, h_events, n_events * sizeof(EarthquakeEvent), cudaMemcpyHostToDevice);

    int grid_size = (n_events + block_size - 1) / block_size;
    histogram_kernel<<<grid_size, block_size>>>(d_events, n_events, d_counts, d_mags, d_maxs,
                                                min_time, max_time);

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("\nExecution time: %.5f ms\n", milliseconds);

    int *h_counts = (int *)malloc(total_cells * sizeof(int));
    float *h_mags = (float *)malloc(total_cells * sizeof(float));
    float *h_maxs = (float *)malloc(total_cells * sizeof(float));

    cudaMemcpy(h_counts, d_counts, total_cells * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mags, d_mags, total_cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_maxs, d_maxs, total_cells * sizeof(float), cudaMemcpyDeviceToHost);

    int non_empty = 0;
    int max_count = 0;
    long long total_events = 0;
    for (int i = 0; i < total_cells; i++) {
        if (h_counts[i] > 0) {
            non_empty++;
            total_events += h_counts[i];
            if (h_counts[i] > max_count) max_count = h_counts[i];
        }
    }

    printf("Non-empty cells: %d (%.2f%%)\n",
           non_empty, 100.0 * non_empty / total_cells);
    printf("Max events in single cell: %d\n", max_count);
    printf("Total events processed: %lld\n\n", total_events);

    char output_filename[256];
    sprintf(output_filename, "histogram_cuda_%d.csv", block_size);
    write_histogram_csv(output_filename, h_counts, h_mags, h_maxs);
    printf("Histogram saved to %s\n", output_filename);

    free(h_events);
    free(h_counts);
    free(h_mags);
    free(h_maxs);
    cudaFree(d_events);
    cudaFree(d_counts);
    cudaFree(d_mags);
    cudaFree(d_maxs);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
