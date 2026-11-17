/* ecg_demo.c - ECG arrhythmia screening demo for QNX 8.0 (and POSIX hosts).
 *
 * Replays a MIT-BIH record from disk, extracts a 180-sample window around
 * each annotated beat, classifies it with the on-target CNN (ecgnet), and
 * prints the prediction vs. the cardiologist annotation plus measured
 * inference latency (CLOCK_MONOTONIC). Beats are streamed at the record's
 * real 360 Hz cadence unless --fast is given.
 *
 * Research/education demo only. Not a medical device; no diagnostic claims.
 * Dataset: MIT-BIH Arrhythmia Database (PhysioNet, ODC-BY 1.0).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include "ecgnet.h"

int ecgnet_load(ecgnet_t *, const char *, const float[5]);
void ecgnet_free(ecgnet_t *);

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

int main(int argc, char **argv) {
    const char *weights = "weights.bin", *recfile = "rec200.bin";
    int fast = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--fast")) fast = 1;
        else if (!strcmp(argv[i], "--record") && i + 1 < argc) recfile = argv[++i];
        else if (!strcmp(argv[i], "--weights") && i + 1 < argc) weights = argv[++i];
        else { fprintf(stderr, "usage: %s [--fast] [--record f] [--weights f]\n", argv[0]); return 2; }
    }

    ecgnet_t net;
    float bias[5] = {0.5f, 0, 0, 0, 0};  /* decision threshold tuned on validation records only */
    if (ecgnet_load(&net, weights, bias)) return 1;

    FILE *f = fopen(recfile, "rb");
    if (!f) { perror(recfile); return 1; }
    uint32_t nsig, nbeats;
    if (fread(&nsig, 4, 1, f) != 1) return 1;
    float *sig = malloc(nsig * sizeof(float));
    if (fread(sig, 4, nsig, f) != nsig) return 1;
    if (fread(&nbeats, 4, 1, f) != 1) return 1;
    uint32_t *pos = malloc(nbeats * 4); uint8_t *lab = malloc(nbeats);
    if (fread(pos, 4, nbeats, f) != nbeats || fread(lab, 1, nbeats, f) != nbeats) return 1;
    fclose(f);

    const char *names[5] = {"N (normal)", "S (supraventricular)", "V (ventricular)", "F (fusion)", "Q (paced/other)"};
    printf("QNX ECG beat classification research demo - replaying %u beats (360 Hz)%s\n", nbeats, fast ? " [fast]" : "");
    printf("%-6s %-6s %-22s %-22s %-6s %-10s\n", "beat", "time", "annotation", "prediction", "ok?", "infer us");

    int correct = 0, per[5] = {0}, pert[5] = {0};
    double tmin = 1e18, tmax = 0, tsum = 0;
    for (uint32_t i = 0; i < nbeats; i++) {
        float beat[ECG_IN], m = 0, sd = 0;
        for (int k = 0; k < ECG_IN; k++) { beat[k] = sig[pos[i] - 90 + k]; m += beat[k]; }
        m /= ECG_IN;
        for (int k = 0; k < ECG_IN; k++) { beat[k] -= m; sd += beat[k] * beat[k]; }
        sd = sqrtf(sd / ECG_IN);
        if (sd > 1e-6f) for (int k = 0; k < ECG_IN; k++) beat[k] /= sd;

        double t0 = now_us();
        int pred = ecgnet_run(&net, beat);
        double dt = now_us() - t0;
        if (dt < tmin) tmin = dt;
        if (dt > tmax) tmax = dt;
        tsum += dt;

        int ok = pred == lab[i];
        correct += ok; pert[lab[i]]++; per[lab[i]] += ok;
        if (i < 30 || i % 200 == 0 || !ok)
            printf("%-6u %-6.1f %-22s %-22s %-6s %-10.1f\n", i, pos[i] / 360.0,
                   names[lab[i]], names[pred], ok ? "" : "MISS", dt);
        if (!fast && i + 1 < nbeats) {
            int64_t wait = ((int64_t)pos[i + 1] - (int64_t)pos[i]) * 1000000 / 360;
            if (wait > 0 && wait < 2000000) usleep(wait);
        }
    }
    printf("\n=== summary ===\n");
    printf("beats: %u  correct: %d (%.1f%%)\n", nbeats, correct, 100.0 * correct / nbeats);
    for (int c = 0; c < 5; c++)
        if (pert[c]) printf("  %s: %d/%d (%.1f%%)\n", names[c], per[c], pert[c], 100.0 * per[c] / pert[c]);
    printf("inference latency: avg %.1f us  min %.1f us  max %.1f us\n",
           tsum / nbeats, tmin, tmax);
    printf("Model: MIT-BIH DS1-trained tiny CNN, patient-independent DS2 overall 80.2%%, balanced 48.6%%, macro-F1 0.40.\n");
    printf("Research demo only - not a medical device, no diagnostic use.\n");
    free(sig); free(pos); free(lab); ecgnet_free(&net);
    return 0;
}
