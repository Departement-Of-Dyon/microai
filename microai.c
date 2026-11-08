/* ==========================================================================
 * MicroMind — a tiny language model built from scratch (no dependencies)
 *
 * Character-level n-gram model (order 3) with Katz-style backoff and
 * temperature sampling. Train it on any text file, then generate.
 *
 * Commands:
 *   microai train <corpus.txt>        train on a text file
 *   microai gen <chars> [seed]        generate text
 *   microai repl                      interactive generation loop
 *   microai save <model.bin>          save trained model
 *   microai load <model.bin>          load model
 *
 * Build: gcc -O2 -o microai microai.c
 * ========================================================================== */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ---- vocabulary: 34 chars -------------------------------------------- */
static const char CHARSET[] =
    "abcdefghijklmnopqrstuvwxyz .,!?'-\n";
#define V 34   /* exact length of CHARSET below — do not exceed */

static int c2i[256];
static char i2c[V];

static void init_vocab(void) {
    memset(c2i, -1, sizeof(c2i));
    int n = 0;
    for (const char *p = CHARSET; *p; p++, n++) {
        if (n >= V) { fprintf(stderr, "CHARSET too long for V\n"); exit(1); }
        c2i[(unsigned char)*p] = n;
        i2c[n] = *p;
    }
}

static int byte2id(int ch) {
    int id = c2i[ch & 0xff];
    return id >= 0 ? id : c2i[' '];   /* unknown -> space */
}

/* ---- model: counts[context][next] ------------------------------------
 * order-3 context id fits in V^3 ; order-2 in V^2 ; unigram in V.
 * Stored flat: counts[ctx * V + next] as uint32.                    */
static unsigned int *tri, *bi, *uni;
static const size_t N_TRI = (size_t)V * V * V;
static const size_t N_BI  = (size_t)V * V;

static void model_alloc(void) {
    tri = calloc(N_TRI * V, sizeof(unsigned int));
    bi  = calloc(N_BI  * V, sizeof(unsigned int));
    uni = calloc(V * V,      sizeof(unsigned int));
    if (!tri || !bi || !uni) { fprintf(stderr, "oom\n"); exit(1); }
}

static void train_text(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    /* sliding window of last 3 ids */
    int h1 = byte2id('\n'), h2 = byte2id('\n'), h3 = byte2id('\n');
    int c, total = 0;
    while ((c = fgetc(f)) != EOF) {
        int id = byte2id(c);
        size_t ctx3 = ((size_t)h1 * V + h2) * V + h3;
        tri[ctx3 * V + id]++;
        size_t ctx2 = (size_t)h2 * V + h3;
        bi[ctx2 * V + id]++;
        uni[id]++;
        h1 = h2; h2 = h3; h3 = id;
        if (++total % 1000000 == 0)
            fprintf(stderr, "  ... %dM chars\n", total / 1000000);
    }
    fclose(f);
    fprintf(stderr, "trained on %d chars\n", total);
}

/* ---- sampling --------------------------------------------------------- */
static double frand(void) { return (double)rand() / ((double)RAND_MAX + 1); }

/* pick next char given counts row; temperature >0; returns id */
static int sample_row(const unsigned int *row, double temp) {
    double w[V], tot = 0;
    int inv = temp > 0 ? 1 : 1;
    double exponent = temp > 0.001 ? 1.0 / temp : 1000.0;
    for (int i = 0; i < V; i++) {
        double c = row[i] + 0.05;          /* add-k smoothing */
        w[i] = pow(c, exponent);
        tot += w[i];
        (void)inv;
    }
    double r = frand() * tot, acc = 0;
    for (int i = 0; i < V; i++) { acc += w[i]; if (r <= acc) return i; }
    return V - 1;
}

static int predict(int h1, int h2, int h3, double temp) {
    size_t ctx3 = ((size_t)h1 * V + h2) * V + h3;
    const unsigned int *r3 = tri + ctx3 * V;
    unsigned long long s3 = 0;
    for (int i = 0; i < V; i++) s3 += r3[i];
    if (s3 > 0) return sample_row(r3, temp);
    size_t ctx2 = (size_t)h2 * V + h3;
    const unsigned int *r2 = bi + ctx2 * V;
    unsigned long long s2 = 0;
    for (int i = 0; i < V; i++) s2 += r2[i];
    if (s2 > 0) return sample_row(r2, temp);
    return sample_row(uni, temp);
}

static void generate(int nchars, const char *seed, double temp) {
    int h1 = byte2id('\n'), h2 = byte2id('\n'), h3 = byte2id('\n');
    if (seed && *seed) {           /* prime context with seed chars */
        for (const char *p = seed; *p; p++) {
            int id = byte2id((unsigned char)*p);
            h1 = h2; h2 = h3; h3 = id;
        }
    }
    for (int i = 0; i < nchars; i++) {
        int id = predict(h1, h2, h3, temp);
        putchar(i2c[id]);
        h1 = h2; h2 = h3; h3 = id;
    }
    putchar('\n');
}

static void repl(void) {
    char line[512];
    printf("MicroMind REPL — type a seed, get a continuation. 'q' quits.\n");
    while (printf("seed> ") && fgets(line, sizeof(line), stdin)) {
        line[strcspn(line, "\r\n")] = 0;
        if (!strcmp(line, "q")) break;
        if (!*line) continue;
        generate(300, line, 0.8);
    }
}

static void save_model(const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror("save"); return; }
    fwrite("MAM1", 1, 4, f);
    fwrite(tri, sizeof(unsigned int), N_TRI * V, f);
    fwrite(bi,  sizeof(unsigned int), N_BI  * V, f);
    fwrite(uni, sizeof(unsigned int), V * V, f);
    fclose(f);
    fprintf(stderr, "saved -> %s\n", path);
}

static int load_model(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror("load"); return 0; }
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "MAM1", 4)) {
        fprintf(stderr, "bad model file\n"); fclose(f); return 0;
    }
    if (fread(tri, sizeof(unsigned int), N_TRI * V, f) != N_TRI * V ||
        fread(bi,  sizeof(unsigned int), N_BI  * V, f) != N_BI  * V ||
        fread(uni, sizeof(unsigned int), V * V,      f) != V * V) {
        fprintf(stderr, "truncated model file\n"); fclose(f); return 0;
    }
    fclose(f);
    fprintf(stderr, "loaded <- %s\n", path);
    return 1;
}

static void usage(void) {
    printf("MicroMind — tiny language model from scratch\n\n"
           "  microai train <corpus.txt>   train, then auto-saves model.bin\n"
           "  microai gen <chars> [seed]   generate (loads model.bin)\n"
           "  microai gen <chars> <seed> <temp>\n"
           "  microai repl                 interactive loop\n"
           "  microai save <model.bin>     save model\n"
           "  microai load <model.bin>     load model\n");
}

int main(int argc, char **argv) {
    srand((unsigned)(time(NULL) ^ (uintptr_t)&argc));
    init_vocab();
    model_alloc();
    if (argc < 2) { usage(); return 0; }

    if (!strcmp(argv[1], "train") && argc >= 3) {
        train_text(argv[2]);
        save_model("model.bin");
    } else if (!strcmp(argv[1], "gen") && argc >= 3) {
        if (!load_model("model.bin")) { fprintf(stderr, "no model.bin — run train first\n"); return 1; }
        int n = atoi(argv[2]);
        const char *seed = argc >= 4 ? argv[3] : "";
        double temp = argc >= 5 ? atof(argv[4]) : 0.8;
        generate(n, seed, temp);
    } else if (!strcmp(argv[1], "repl")) {
        if (!load_model("model.bin")) { fprintf(stderr, "no model.bin — run train first\n"); return 1; }
        repl();
    } else if (!strcmp(argv[1], "save") && argc >= 3) {
        save_model(argv[2]);
    } else if (!strcmp(argv[1], "load") && argc >= 3) {
        if (!load_model(argv[2])) return 1;
    } else usage();
    return 0;
}
