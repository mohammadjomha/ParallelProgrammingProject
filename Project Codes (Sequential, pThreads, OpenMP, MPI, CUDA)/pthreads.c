#include "earthquake_analysis.h"
#include <sys/time.h>
#include <pthread.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>

typedef struct {
    const int *idx;
    int idx_count;
    int thread_id;
    int lat_lo;
    int lat_hi;

    const EarthquakeEvent *events;
    const int *lat_bin;
    const int *lon_bin;
    const int *time_bin;

    GridCell ***global_histogram;
} ThreadData;

double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000.0) + (tv.tv_usec / 1000.0);
}

int load_binary_data(const char *filename, EarthquakeEvent **events) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) { perror("fopen"); return -1; }

    int n_events = 0;
    if (fread(&n_events, sizeof(int), 1, fp) != 1) {
        perror("fread n_events");
        fclose(fp);
        return -1;
    }

    *events = (EarthquakeEvent *)malloc((size_t)n_events * sizeof(EarthquakeEvent));
    if (!*events) { perror("malloc events"); fclose(fp); return -1; }

    for (int i = 0; i < n_events; i++) {
        if (fread(&(*events)[i], sizeof(EarthquakeEvent), 1, fp) != 1) {
            perror("fread event");
            free(*events);
            fclose(fp);
            return -1;
        }
    }
    fclose(fp);
    return n_events;
}

static inline int lat_to_bin_clamped(double lat) {
    int v = (int)((lat + 90.0) * LAT_BINS / 180.0);
    if (v < 0) v = 0;
    else if (v >= LAT_BINS) v = LAT_BINS - 1;
    return v;
}

static inline int lon_to_bin_clamped(double lon) {
    int v = (int)((lon + 180.0) * LON_BINS / 360.0);
    if (v < 0) v = 0;
    else if (v >= LON_BINS) v = LON_BINS - 1;
    return v;
}

static inline int time_to_bin_clamped(long ts, long min_time, long max_time) {
    double denom = (double)(max_time - min_time + 1);
    int v = (int)((ts - min_time) * TIME_BINS / denom);
    if (v < 0) v = 0;
    else if (v >= TIME_BINS) v = TIME_BINS - 1;
    return v;
}

GridCell ***allocate_histogram(void) {
    GridCell ***hist = (GridCell ***)malloc((size_t)LAT_BINS * sizeof(GridCell **));
    for (int i = 0; i < LAT_BINS; i++) {
        hist[i] = (GridCell **)malloc((size_t)LON_BINS * sizeof(GridCell *));
        for (int j = 0; j < LON_BINS; j++) {
            hist[i][j] = (GridCell *)calloc((size_t)TIME_BINS, sizeof(GridCell));
        }
    }
    return hist;
}

void free_histogram(GridCell ***hist) {
    for (int i = 0; i < LAT_BINS; i++) {
        for (int j = 0; j < LON_BINS; j++) {
            free(hist[i][j]);
        }
        free(hist[i]);
    }
    free(hist);
}

void save_histogram_csv(GridCell ***histogram, const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("Error opening output file");
        return;
    }

    fprintf(fp, "lat_bin,lon_bin,time_bin,count,total_magnitude,max_magnitude\n");

    for (int i = 0; i < LAT_BINS; i++) {
        for (int j = 0; j < LON_BINS; j++) {
            for (int k = 0; k < TIME_BINS; k++) {
                if (histogram[i][j][k].count > 0) {
                    fprintf(fp, "%d,%d,%d,%d,%.2f,%.2f\n",
                           i, j, k,
                           histogram[i][j][k].count,
                           histogram[i][j][k].total_magnitude,
                           histogram[i][j][k].max_magnitude);
                }
            }
        }
    }

    fclose(fp);
    printf("Histogram saved to %s\n", filename);
}

static void *thread_histogram(void *arg) {
    ThreadData *d = (ThreadData *)arg;

    for (int k = 0; k < d->idx_count; k++) {
        int i = d->idx[k];
        int li = d->lat_bin[i];
        int lj = d->lon_bin[i];
        int tk = d->time_bin[i];

        GridCell *cell = &d->global_histogram[li][lj][tk];
        cell->count += 1;
        cell->total_magnitude += d->events[i].magnitude;
        if (d->events[i].magnitude > cell->max_magnitude) {
            cell->max_magnitude = d->events[i].magnitude;
        }
    }
    return NULL;
}

void pthread_histogram(const EarthquakeEvent *events, int n_events,
                       GridCell ***hist, long min_time, long max_time,
                       int num_threads, double *bucket_ms, double *compute_ms) {

    double t0 = get_time_ms();

    int *lat_bin = (int *)malloc((size_t)n_events * sizeof(int));
    int *lon_bin = (int *)malloc((size_t)n_events * sizeof(int));
    int *time_bin = (int *)malloc((size_t)n_events * sizeof(int));

    for (int i = 0; i < n_events; i++) {
        lat_bin[i] = lat_to_bin_clamped(events[i].latitude);
        lon_bin[i] = lon_to_bin_clamped(events[i].longitude);
        time_bin[i] = time_to_bin_clamped(events[i].timestamp, min_time, max_time);
    }

    int *lat_lo = (int *)malloc(num_threads * sizeof(int));
    int *lat_hi = (int *)malloc(num_threads * sizeof(int));
    for (int t = 0; t < num_threads; t++) {
        lat_lo[t] = (LAT_BINS * t) / num_threads;
        lat_hi[t] = (LAT_BINS * (t + 1)) / num_threads - 1;
    }

    int *counts = (int *)calloc(num_threads, sizeof(int));
    for (int i = 0; i < n_events; i++) {
        int t = (lat_bin[i] * num_threads) / LAT_BINS;
        if (t >= num_threads) t = num_threads - 1;
        counts[t]++;
    }

    int **buckets = (int **)malloc(num_threads * sizeof(int *));
    for (int t = 0; t < num_threads; t++) {
        buckets[t] = (int *)malloc((size_t)counts[t] * sizeof(int));
        counts[t] = 0;
    }

    for (int i = 0; i < n_events; i++) {
        int t = (lat_bin[i] * num_threads) / LAT_BINS;
        if (t >= num_threads) t = num_threads - 1;
        buckets[t][counts[t]++] = i;
    }

    double t1 = get_time_ms();
    *bucket_ms = t1 - t0;

    pthread_t *ths = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    ThreadData *td = (ThreadData *)malloc(num_threads * sizeof(ThreadData));

    double c0 = get_time_ms();
    for (int t = 0; t < num_threads; t++) {
        td[t].idx = buckets[t];
        td[t].idx_count = counts[t];
        td[t].thread_id = t;
        td[t].lat_lo = lat_lo[t];
        td[t].lat_hi = lat_hi[t];
        td[t].events = events;
        td[t].lat_bin = lat_bin;
        td[t].lon_bin = lon_bin;
        td[t].time_bin = time_bin;
        td[t].global_histogram = hist;

        pthread_create(&ths[t], NULL, thread_histogram, &td[t]);
    }
    for (int t = 0; t < num_threads; t++) {
        pthread_join(ths[t], NULL);
    }
    double c1 = get_time_ms();
    *compute_ms = c1 - c0;

    for (int t = 0; t < num_threads; t++) free(buckets[t]);
    free(buckets);
    free(counts);
    free(lat_lo);
    free(lat_hi);
    free(lat_bin);
    free(lon_bin);
    free(time_bin);
    free(ths);
    free(td);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <binary_data_file> <num_threads>\n", argv[0]);
        return 1;
    }

    int num_threads = atoi(argv[2]);
    if (num_threads < 1) {
        fprintf(stderr, "num_threads must be >= 1\n");
        return 1;
    }

    EarthquakeEvent *events = NULL;
    int n = load_binary_data(argv[1], &events);
    if (n <= 0) {
        fprintf(stderr, "Failed to load data\n");
        return 1;
    }

    printf("=== Pthreads Implementation (Optimized) ===\n");
    printf("Loaded %d earthquake events\n", n);
    printf("Number of threads: %d\n", num_threads);
    printf("Grid: %d x %d x %d = %d cells\n",
           LAT_BINS, LON_BINS, TIME_BINS, LAT_BINS * LON_BINS * TIME_BINS);

    long min_time = events[0].timestamp;
    long max_time = events[0].timestamp;
    for (int i = 1; i < n; i++) {
        if (events[i].timestamp < min_time) min_time = events[i].timestamp;
        if (events[i].timestamp > max_time) max_time = events[i].timestamp;
    }

    GridCell ***hist = allocate_histogram();

    double bucket_ms = 0.0, compute_ms = 0.0;
    double t0 = get_time_ms();
    pthread_histogram(events, n, hist, min_time, max_time,
                      num_threads, &bucket_ms, &compute_ms);
    double t1 = get_time_ms();

    printf("Pthreads total time: %.2f ms\n", t1 - t0);
    printf("  - Bucketing: %.2f ms\n", bucket_ms);
    printf("  - Computation: %.2f ms\n", compute_ms);
    printf("  - Assembly: 0.00 ms (direct write)\n");

    int non_empty = 0, maxc = 0;
    for (int i = 0; i < LAT_BINS; i++)
        for (int j = 0; j < LON_BINS; j++)
            for (int k = 0; k < TIME_BINS; k++)
                if (hist[i][j][k].count > 0) {
                    non_empty++;
                    if (hist[i][j][k].count > maxc)
                        maxc = hist[i][j][k].count;
                }

    printf("Non-empty cells: %d (%.2f%%)\n",
           non_empty, 100.0 * non_empty / (LAT_BINS * LON_BINS * TIME_BINS));
    printf("Max events in single cell: %d\n", maxc);

    //filename has size of thread count
    char filename[256];
    snprintf(filename, sizeof(filename), "histogram_pthreads_%d.csv", num_threads);
    save_histogram_csv(hist, filename);

    free_histogram(hist);
    free(events);
    return 0;
}
