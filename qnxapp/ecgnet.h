/* ecgnet.h - tiny 1D CNN ECG beat classifier, portable C99.
 * Ops match the PyTorch exporter exactly: Conv1d valid + ReLU + MaxPool2,
 * Linear, logit bias, argmax. Weights loaded from weights.bin (float32 LE)
 * in the fixed order of the manifest. No dynamic allocation after init. */
#ifndef ECGNET_H
#define ECGNET_H
#include <stdint.h>
#include <stddef.h>
#include <math.h>

#define ECG_IN 180
typedef struct { const float *w; const float *b; int cin, cout, k; } conv1d_t;
typedef struct { const float *w; const float *b; int in, out; } linear_t;

typedef struct {
    conv1d_t c1, c2, c3;
    linear_t f1, f2;
    float bias[5];          /* decision logit bias (chosen on validation split) */
    float *buf1, *buf2;     /* scratch, allocated once */
    size_t buf_sz;
} ecgnet_t;

static void conv1d_valid(const conv1d_t *c, const float *in, float *out, int len) {
    int olen = len - c->k + 1;
    for (int o = 0; o < c->cout; o++)
        for (int i = 0; i < olen; i++) {
            float s = c->b[o];
            for (int j = 0; j < c->cin; j++)
                for (int k = 0; k < c->k; k++)
                    s += in[j * len + i + k] * c->w[(o * c->cin + j) * c->k + k];
            out[o * olen + i] = s > 0.f ? s : 0.f;  /* fused ReLU */
        }
}
static void maxpool2(const float *in, float *out, int ch, int len) {
    int olen = len / 2;
    for (int c = 0; c < ch; c++)
        for (int i = 0; i < olen; i++) {
            float a = in[c * len + 2 * i], b = in[c * len + 2 * i + 1];
            out[c * olen + i] = a > b ? a : b;
        }
}
static void linear_relu(const linear_t *l, const float *in, float *out, int relu) {
    for (int o = 0; o < l->out; o++) {
        float s = l->b[o];
        for (int i = 0; i < l->in; i++) s += in[i] * l->w[o * l->in + i];
        out[o] = relu ? (s > 0.f ? s : 0.f) : s;
    }
}
/* returns class index 0..4 */
static int ecgnet_run(ecgnet_t *n, const float beat[ECG_IN]) {
    /* c1: 1->8 k5: 180->176, mp: 88; c2: 8->16: 84, mp: 42; c3: 16->32: 38, mp: 19 */
    float *a = n->buf1, *b = n->buf2;
    conv1d_valid(&n->c1, beat, a, ECG_IN);            /* a: 8*176 */
    maxpool2(a, b, 8, 176);                            /* b: 8*88 */
    conv1d_valid(&n->c2, b, a, 88);                    /* a: 16*84 */
    maxpool2(a, b, 16, 84);                            /* b: 16*42 */
    conv1d_valid(&n->c3, b, a, 42);                    /* a: 32*38 */
    maxpool2(a, b, 32, 38);                            /* b: 32*19 = 608 */
    linear_relu(&n->f1, b, a, 1);                      /* a: 48 */
    float logit[5];
    linear_t f2 = n->f2; linear_relu(&f2, a, logit, 0);
    int best = 0; float bs = logit[0] + n->bias[0];
    for (int c = 1; c < 5; c++) { float s = logit[c] + n->bias[c]; if (s > bs) { bs = s; best = c; } }
    return best;
}
#endif
