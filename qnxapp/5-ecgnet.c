/* ecgnet.c - loader + host/QNX shared glue for ecgnet.h */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ecgnet.h"

int ecgnet_load(ecgnet_t *n, const char *weights_path, const float bias[5]) {
    FILE *f = fopen(weights_path, "rb");
    if (!f) { perror("weights.bin"); return -1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); rewind(f);
    float *w = malloc(sz);
    if (!w || fread(w, 1, sz, f) != (size_t)sz) { fclose(f); return -1; }
    fclose(f);
    /* manifest order: c1.w c1.b c2.w c2.b c3.w c3.b f1.w f1.b f2.w f2.b
     * shapes: [8,1,5] [8] [16,8,5] [16] [32,16,5] [32] [48,608] [48] [5,48] [5] */
    float *p = w;
    n->c1 = (conv1d_t){p, p + 8 * 1 * 5, 1, 8, 5};        p += 8 * 1 * 5 + 8;
    n->c2 = (conv1d_t){p, p + 16 * 8 * 5, 8, 16, 5};      p += 16 * 8 * 5 + 16;
    n->c3 = (conv1d_t){p, p + 32 * 16 * 5, 16, 32, 5};    p += 32 * 16 * 5 + 32;
    n->f1 = (linear_t){p, p + 48 * 608, 608, 48};         p += 48 * 608 + 48;
    n->f2 = (linear_t){p, p + 5 * 48, 48, 5};             p += 5 * 48 + 5;
    memcpy(n->bias, bias, sizeof n->bias);
    n->buf_sz = 16 * 84 > 8 * 176 ? 16 * 84 : 8 * 176;   /* max intermediate */
    if ((size_t)(32 * 38) > n->buf_sz) n->buf_sz = 32 * 38;
    n->buf1 = malloc(n->buf_sz * sizeof(float));
    n->buf2 = malloc(n->buf_sz * sizeof(float));
    return n->buf1 && n->buf2 ? 0 : -1;
}
void ecgnet_free(ecgnet_t *n) {
    free((void *)n->c1.w);  /* the single weights blob */
    free(n->buf1); free(n->buf2);
}
