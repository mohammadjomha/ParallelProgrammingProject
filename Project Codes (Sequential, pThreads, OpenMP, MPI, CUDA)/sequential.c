#include "earthquake_analysis.h"
#include <sys/time.h>
#include <string.h>

double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000.0) + (tv.tv_usec / 1000.0);
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

void compute_bin_indices(EarthquakeEvent event, long min_time, long max_time,
                        int *lat_idx, int *lon_idx, int *time_idx) {
    // Latitude: -90 to 90 -> 0 to LAT_BINS-1
    *lat_idx = (int)((event.latitude + 90.0) * LAT_BINS / 180.0);
    if (*lat_idx >= LAT_BINS) *lat_idx = LAT_BINS - 1;
    if (*lat_idx < 0) *lat_idx = 0;

    // Longitude: -180 to 180 -> 0 to LON_BINS-1
    *lon_idx = (int)((event.longitude + 180.0) * LON_BINS / 360.0);
    if (*lon_idx >= LON_BINS) *lon_idx = LON_BINS - 1;
    if (*lon_idx < 0) *lon_idx = 0;

    // Time: min_time to max_time -> 0 to TIME_BINS-1
    *time_idx = (int)((event.timestamp - min_time) * TIME_BINS /
                     (double)(max_time - min_time + 1));
    if (*time_idx >= TIME_BINS) *time_idx = TIME_BINS - 1;
    if (*time_idx < 0) *time_idx = 0;
}

GridCell*** allocate_histogram() {
    GridCell ***histogram = (GridCell ***)malloc(LAT_BINS * sizeof(GridCell **));
    for (int i = 0; i < LAT_BINS; i++) {
        histogram[i] = (GridCell **)malloc(LON_BINS * sizeof(GridCell *));
        for (int j = 0; j < LON_BINS; j++) {
            histogram[i][j] = (GridCell *)calloc(TIME_BINS, sizeof(GridCell));
        }
    }
    return histogram;
}

void sequential_histogram(EarthquakeEvent *events, int n_events,
                         GridCell ***histogram, long min_time, long max_time) {
    for (int i = 0; i < n_events; i++) {
        int lat_idx, lon_idx, time_idx;
        compute_bin_indices(events[i], min_time, max_time,
                          &lat_idx, &lon_idx, &time_idx);

        GridCell *cell = &histogram[lat_idx][lon_idx][time_idx];
        cell->count++;
        cell->total_magnitude += events[i].magnitude;
        if (events[i].magnitude > cell->max_magnitude) {
            cell->max_magnitude = events[i].magnitude;
        }
    }
}

void save_histogram_csv(GridCell ***histogram, const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("Error opening output file");
        return;
    }

    //write csv header
    fprintf(fp, "lat_bin,lon_bin,time_bin,count,total_magnitude,max_magnitude\n");

    //non-empty cells only
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

void free_histogram(GridCell ***histogram) {
    for (int i = 0; i < LAT_BINS; i++) {
        for (int j = 0; j < LON_BINS; j++) {
            free(histogram[i][j]);
        }
        free(histogram[i]);
    }
    free(histogram);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <binary_data_file>\n", argv[0]);
        return 1;
    }

    EarthquakeEvent *events;
    int n_events = load_binary_data(argv[1], &events);

    if (n_events <= 0) {
        fprintf(stderr, "Failed to load data\n");
        return 1;
    }

    printf("=== Sequential Implementation ===\n");
    printf("Loaded %d earthquake events\n", n_events);

    //time range
    long min_time = events[0].timestamp;
    long max_time = events[0].timestamp;
    for (int i = 1; i < n_events; i++) {
        if (events[i].timestamp < min_time) min_time = events[i].timestamp;
        if (events[i].timestamp > max_time) max_time = events[i].timestamp;
    }

    printf("Time range: %ld to %ld seconds\n", min_time, max_time);
    printf("Grid dimensions: %d x %d x %d = %d cells\n",
           LAT_BINS, LON_BINS, TIME_BINS, LAT_BINS * LON_BINS * TIME_BINS);

    //allocate histogram
    GridCell ***histogram = allocate_histogram();

    //run computation
    double start = get_time_ms();
    sequential_histogram(events, n_events, histogram, min_time, max_time);
    double end = get_time_ms();

    printf("Sequential execution time: %.2f ms\n", end - start);


    int non_empty = 0;
    int max_count = 0;
    float total_events = 0;

    for (int i = 0; i < LAT_BINS; i++) {
        for (int j = 0; j < LON_BINS; j++) {
            for (int k = 0; k < TIME_BINS; k++) {
                if (histogram[i][j][k].count > 0) {
                    non_empty++;
                    total_events += histogram[i][j][k].count;
                    if (histogram[i][j][k].count > max_count) {
                        max_count = histogram[i][j][k].count;
                    }
                }
            }
        }
    }

    printf("Non-empty cells: %d out of %d (%.2f%%)\n",
           non_empty, LAT_BINS * LON_BINS * TIME_BINS,
           100.0 * non_empty / (LAT_BINS * LON_BINS * TIME_BINS));
    printf("Max events in single cell: %d\n", max_count);
    printf("Average events per non-empty cell: %.2f\n", total_events / non_empty);

    save_histogram_csv(histogram, "histogram_sequential.csv");

    free_histogram(histogram);
    free(events);

    return 0;
}
