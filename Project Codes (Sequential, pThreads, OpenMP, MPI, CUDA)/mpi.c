
#include "earthquake_analysis.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

double get_time_ms(void){
    struct timeval tv; gettimeofday(&tv, NULL);
    return (tv.tv_sec*1000.0) + (tv.tv_usec/1000.0);
}

int load_binary_data(const char *filename, EarthquakeEvent **events){
    FILE *fp = fopen(filename, "rb");
    if (!fp) { perror("fopen"); return -1; }
    int n = 0;
    if (fread(&n, sizeof(int), 1, fp) != 1) { perror("fread n"); fclose(fp); return -1; }
    *events = (EarthquakeEvent*)malloc((size_t)n * sizeof(EarthquakeEvent));
    if (!*events) { perror("malloc events"); fclose(fp); return -1; }
    for (int i = 0; i < n; ++i){
        if (fread(&(*events)[i], sizeof(EarthquakeEvent), 1, fp) != 1){
            perror("fread event"); free(*events); fclose(fp); return -1;
        }
    }
    fclose(fp);
    return n;
}

static inline int lat_to_bin_clamped(double lat){
    int v = (int)((lat + 90.0) * LAT_BINS / 180.0);
    if (v < 0) v = 0; else if (v >= LAT_BINS) v = LAT_BINS - 1;
    return v;
}

static inline int lon_to_bin_clamped(double lon){
    int v = (int)((lon + 180.0) * LON_BINS / 360.0);
    if (v < 0) v = 0; else if (v >= LON_BINS) v = LON_BINS - 1;
    return v;
}

static inline int time_to_bin_clamped(long ts, long min_time, long max_time){
    double denom = (double)(max_time - min_time + 1);
    int v = (int)((ts - min_time) * TIME_BINS / denom);
    if (v < 0) v = 0; else if (v >= TIME_BINS) v = TIME_BINS - 1;
    return v;
}

GridCell*** allocate_histogram_stripe(int lat_count) {
    GridCell ***hist = (GridCell ***)malloc((size_t)lat_count * sizeof(GridCell **));
    for (int i = 0; i < lat_count; i++) {
        hist[i] = (GridCell **)malloc((size_t)LON_BINS * sizeof(GridCell *));
        for (int j = 0; j < LON_BINS; j++) {
            hist[i][j] = (GridCell *)calloc((size_t)TIME_BINS, sizeof(GridCell));
        }
    }
    return hist;
}

void free_histogram_stripe(GridCell ***hist, int lat_count) {
    for (int i = 0; i < lat_count; i++) {
        for (int j = 0; j < LON_BINS; j++) {
            free(hist[i][j]);
        }
        free(hist[i]);
    }
    free(hist);
}

void save_histogram_csv(GridCell ***histogram, int lat_start, int lat_count, const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("Error opening output file");
        return;
    }

    fprintf(fp, "lat_bin,lon_bin,time_bin,count,total_magnitude,max_magnitude\n");

    for (int i = 0; i < lat_count; i++) {
        for (int j = 0; j < LON_BINS; j++) {
            for (int k = 0; k < TIME_BINS; k++) {
                if (histogram[i][j][k].count > 0) {
                    fprintf(fp, "%d,%d,%d,%d,%.2f,%.2f\n",
                           i + lat_start, j, k,
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

int main(int argc, char **argv){
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (argc != 2){
        if (rank == 0) fprintf(stderr, "Usage: %s <binary_data_file>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    // each rank loads the data
    EarthquakeEvent *events = NULL;
    int n_events = load_binary_data(argv[1], &events);
    if (n_events <= 0){
        if (rank == 0) fprintf(stderr, "Failed to load data\n");
        MPI_Finalize();
        return 1;
    }

    // time range
    long min_time = events[0].timestamp;
    long max_time = events[0].timestamp;
    for (int i = 1; i < n_events; i++){
        if (events[i].timestamp < min_time) min_time = events[i].timestamp;
        if (events[i].timestamp > max_time) max_time = events[i].timestamp;
    }

    //determine the rank latitude stripe
    int lat_per_proc = LAT_BINS / num_procs;
    int lat_start = rank * lat_per_proc;
    int lat_end = (rank == num_procs - 1) ? LAT_BINS : (rank + 1) * lat_per_proc;
    int lat_count = lat_end - lat_start;

    if (rank == 0){
        printf("=== MPI Implementation (Stripe + Bucket) ===\n");
        printf("Loaded %d earthquake events\n", n_events);
        printf("Number of processes: %d\n", num_procs);
        printf("Grid: %d x %d x %d = %d cells\n",
               LAT_BINS, LON_BINS, TIME_BINS, LAT_BINS * LON_BINS * TIME_BINS);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    //precompute bins
    int *lat_bin = (int *)malloc((size_t)n_events * sizeof(int));
    int *lon_bin = (int *)malloc((size_t)n_events * sizeof(int));
    int *time_bin = (int *)malloc((size_t)n_events * sizeof(int));

    for (int i = 0; i < n_events; i++){
        lat_bin[i] = lat_to_bin_clamped(events[i].latitude);
        lon_bin[i] = lon_to_bin_clamped(events[i].longitude);
        time_bin[i] = time_to_bin_clamped(events[i].timestamp, min_time, max_time);
    }

    //allocate only this rank's stripe
    GridCell ***histogram = allocate_histogram_stripe(lat_count);

    //  process events in the stripe
    for (int i = 0; i < n_events; i++){
        int li = lat_bin[i];

        //   if not in stripe then skip
        if (li < lat_start || li >= lat_end) continue;

        int lj = lon_bin[i];
        int tk = time_bin[i];

        //convert indices to local indices
        int local_lat = li - lat_start;
        GridCell *cell = &histogram[local_lat][lj][tk];

        cell->count += 1;
        cell->total_magnitude += events[i].magnitude;
        if (events[i].magnitude > cell->max_magnitude){
            cell->max_magnitude = events[i].magnitude;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    //gather stats
    int local_non_empty = 0;
    int local_maxc = 0;
    for (int i = 0; i < lat_count; i++){
        for (int j = 0; j < LON_BINS; j++){
            for (int k = 0; k < TIME_BINS; k++){
                if (histogram[i][j][k].count > 0){
                    local_non_empty++;
                    if (histogram[i][j][k].count > local_maxc){
                        local_maxc = histogram[i][j][k].count;
                    }
                }
            }
        }
    }

    int *all_non_empty = NULL;
    int *all_maxc = NULL;
    if (rank == 0){
        all_non_empty = (int *)malloc(num_procs * sizeof(int));
        all_maxc = (int *)malloc(num_procs * sizeof(int));
    }

    MPI_Gather(&local_non_empty, 1, MPI_INT, all_non_empty, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_maxc, 1, MPI_INT, all_maxc, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0){
        int total_non_empty = 0;
        int global_maxc = 0;
        for (int p = 0; p < num_procs; p++){
            total_non_empty += all_non_empty[p];
            if (all_maxc[p] > global_maxc) global_maxc = all_maxc[p];
        }

        printf("MPI total time: %.2f ms\n", (t1 - t0) * 1000.0);
        printf("Non-empty cells: %d (%.2f%%)\n",
               total_non_empty, 100.0 * total_non_empty / (LAT_BINS * LON_BINS * TIME_BINS));
        printf("Max events in single cell: %d\n", global_maxc);

        char filename[256];
        snprintf(filename, sizeof(filename), "histogram_mpi_%d.csv", num_procs);
        save_histogram_csv(histogram, lat_start, lat_count, filename);

        free(all_non_empty);
        free(all_maxc);
    }
    free(lat_bin);
    free(lon_bin);
    free(time_bin);
    free_histogram_stripe(histogram, lat_count);
    free(events);

    MPI_Finalize();
    return 0;
}
