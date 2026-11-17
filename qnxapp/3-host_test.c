/* host_test.c - parity check: classify beats_test.bin, print accuracy.
 * Format: u32 n; n x {u8 label, float32 beat[180] LE} */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "ecgnet.h"
int ecgnet_load(ecgnet_t *, const char *, const float[5]);
void ecgnet_free(ecgnet_t *);

int main(int argc, char **argv) {
    ecgnet_t net;
    float bias[5] = {0.5f, 0, 0, 0, 0};
    if (ecgnet_load(&net, argv[1], bias)) return 1;
    FILE *f = fopen(argv[2], "rb");
    uint32_t n; fread(&n, 4, 1, f);
    int correct = 0, per[5] = {0}, pert[5] = {0};
    for (uint32_t i = 0; i < n; i++) {
        uint8_t lab; float beat[ECG_IN];
        fread(&lab, 1, 1, f); fread(beat, 4, ECG_IN, f);
        int pred = ecgnet_run(&net, beat);
        pert[lab]++; if (pred == lab) { correct++; per[lab]++; }
    }
    printf("C engine: %d/%u = %.4f\n", correct, n, (double)correct / n);
    const char *nm = "NSVFQ";
    for (int c = 0; c < 5; c++)
        if (pert[c]) printf("  %c: %d/%d = %.3f\n", nm[c], per[c], pert[c], (double)per[c] / pert[c]);
    ecgnet_free(&net);
    return 0;
}
