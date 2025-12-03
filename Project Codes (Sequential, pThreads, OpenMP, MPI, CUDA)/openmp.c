
#include "earthquake_analysis.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

/* ---- local helpers so we can compile standalone ---- */
double get_time_ms(void){
    struct timeval tv; gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000.0) + (tv.tv_usec / 1000.0);
}
int load_binary_data(const char *filename, EarthquakeEvent **events){
    FILE *fp = fopen(filename, "rb");
    if(!fp){ perror("fopen"); return -1; }
    int n = 0;
    if(fread(&n, sizeof(int), 1, fp) != 1){ perror("fread n"); fclose(fp); return -1; }
    *events = (EarthquakeEvent*) malloc((size_t)n * sizeof(EarthquakeEvent));
    if(!*events){ perror("malloc events"); fclose(fp); return -1; }
    for(int i=0;i<n;i++){
        if(fread(&(*events)[i], sizeof(EarthquakeEvent), 1, fp) != 1){
            perror("fread event"); free(*events); fclose(fp); return -1;
        }
    }
    fclose(fp);
    return n;
}
/* ---------------------------------------------------- */

static inline int lat_to_bin_clamped(double lat){
    int v = (int)((lat + 90.0) * LAT_BINS / 180.0);
    if (v < 0) v = 0; else if (v >= LAT_BINS) v = LAT_BINS - 1; return v;
}
static inline int lon_to_bin_clamped(double lon){
    int v = (int)((lon + 180.0) * LON_BINS / 360.0);
    if (v < 0) v = 0; else if (v >= LON_BINS) v = LON_BINS - 1; return v;
}
static inline int time_to_bin_clamped(long ts,long mn,long mx){
    double d = (double)(mx - mn + 1);
    int v = (int)((ts - mn) * TIME_BINS / d);
    if (v < 0) v = 0; else if (v >= TIME_BINS) v = TIME_BINS - 1; return v;
}

static GridCell*** alloc_hist(void){
    GridCell ***h = (GridCell ***)malloc((size_t)LAT_BINS * sizeof(GridCell **));
    for(int i=0;i<LAT_BINS;i++){
        h[i] = (GridCell **)malloc((size_t)LON_BINS * sizeof(GridCell *));
        for(int j=0;j<LON_BINS;j++){
            h[i][j] = (GridCell *)calloc((size_t)TIME_BINS, sizeof(GridCell));
        }
    }
    return h;
}
static void free_hist(GridCell ***h){
    for(int i=0;i<LAT_BINS;i++){ for(int j=0;j<LON_BINS;j++) free(h[i][j]); free(h[i]); }
    free(h);
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

int main(int argc,char**argv){
    if(argc!=3){ fprintf(stderr,"Usage: %s <binary_data_file> <num_threads>\n",argv[0]); return 1; }
    int T = atoi(argv[2]); if(T<1){ fprintf(stderr,"num_threads must be >=1\n"); return 1; }
    omp_set_num_threads(T);

    EarthquakeEvent *ev=NULL; int N=load_binary_data(argv[1], &ev);
    if(N<=0){ fprintf(stderr,"Failed to load data\n"); return 1; }

    long tmin=ev[0].timestamp, tmax=ev[0].timestamp;
    for(int i=1;i<N;i++){ if(ev[i].timestamp<tmin)tmin=ev[i].timestamp; if(ev[i].timestamp>tmax)tmax=ev[i].timestamp; }

    printf("=== OpenMP Implementation (Stripe + Bucket) ===\n");
    printf("Loaded %d earthquake events\n",N);
    printf("Number of threads: %d\n",T);
    printf("Grid dimensions: %d x %d x %d = %d cells\n", LAT_BINS,LON_BINS,TIME_BINS, LAT_BINS*LON_BINS*TIME_BINS);

    GridCell ***H = alloc_hist();

    //1 precompute bins
    double t0=get_time_ms();
    int *latb=(int*)malloc((size_t)N*sizeof(int));
    int *lonb=(int*)malloc((size_t)N*sizeof(int));
    int *timeb=(int*)malloc((size_t)N*sizeof(int));
    #pragma omp parallel for schedule(static)
    for(int i=0;i<N;i++){
        latb[i]=lat_to_bin_clamped(ev[i].latitude);
        lonb[i]=lon_to_bin_clamped(ev[i].longitude);
        timeb[i]=time_to_bin_clamped(ev[i].timestamp,tmin,tmax);
    }

    //2 bucket counts per stripe
    int *counts=(int*)calloc((size_t)T,sizeof(int));
    for(int i=0;i<N;i++){ int t=(latb[i]*T)/LAT_BINS; if(t>=T)t=T-1; counts[t]++; }

    //3 buckets of indices
    int **bucket=(int**)malloc((size_t)T*sizeof(int*));
    for(int t=0;t<T;t++){ bucket[t]=(int*)malloc((size_t)counts[t]*sizeof(int)); counts[t]=0; }
    for(int i=0;i<N;i++){ int t=(latb[i]*T)/LAT_BINS; if(t>=T)t=T-1; bucket[t][counts[t]++]=i; }
    double t1=get_time_ms(); double bucketing_ms=t1-t0;

    //4 parallel compute / direct writes
    double c0=get_time_ms();
    #pragma omp parallel
    {
        int tid=omp_get_thread_num();
        int *lst=bucket[tid]; int cnt=counts[tid];
        for(int k=0;k<cnt;k++){
            int i=lst[k]; int li=latb[i], lj=lonb[i], tk=timeb[i];
            GridCell *cell=&H[li][lj][tk];
            cell->count += 1;
            cell->total_magnitude += ev[i].magnitude;
            if(ev[i].magnitude > cell->max_magnitude) cell->max_magnitude = ev[i].magnitude;
        }
    }
    double c1=get_time_ms(); double compute_ms=c1-c0;

    //validate results
    int non_empty=0, maxc=0;
    for(int i=0;i<LAT_BINS;i++)
        for(int j=0;j<LON_BINS;j++)
            for(int k=0;k<TIME_BINS;k++)
                if(H[i][j][k].count>0){ non_empty++; if(H[i][j][k].count>maxc) maxc=H[i][j][k].count; }

    printf("OpenMP total time: %.2f ms\n", bucketing_ms + compute_ms);
    printf("  - Bucketing/Indexing: %.2f ms\n", bucketing_ms);
    printf("  - Computation: %.2f ms\n", compute_ms);
    printf("  - Assembly: 0.00 ms (direct write)\n");
    printf("Non-empty cells: %d (%.2f%%)\n", non_empty, 100.0*non_empty/(LAT_BINS*LON_BINS*TIME_BINS));
    printf("Max events in single cell: %d\n", maxc);

    char filename[256];
    snprintf(filename, sizeof(filename), "histogram_openmp_%d.csv", T);
    save_histogram_csv(H, filename);

    for(int t=0;t<T;t++) free(bucket[t]);
    free(bucket); free(counts);
    free(latb); free(lonb); free(timeb);
    free_hist(H); free(ev);
    return 0;
}
