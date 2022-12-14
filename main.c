#include <zlib.h>
#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include "ketopt.h"
#include "nasw.h"
#include "kseq.h"
KSEQ_INIT(gzFile, gzread)

int main(int argc, char *argv[])
{
	gzFile fpn, fpa;
	kseq_t *ksn, *ksa;
	ketopt_t o = KETOPT_INIT;
	ns_opt_t opt;
	int32_t c, no_sse = 0, use_32 = 0;

	ns_make_tables(0);
	ns_opt_init(&opt);
	opt.flag |= NS_F_CIGAR;
	while ((c = ketopt(&o, argc, argv, 1, "pswlrx:j:J:", 0)) >= 0) {
		if (c == 'p') no_sse = 1;
		else if (c == 's') opt.flag &= ~NS_F_CIGAR;
		else if (c == 'l') opt.flag |= NS_F_EXT_LEFT;
		else if (c == 'r') opt.flag |= NS_F_EXT_RIGHT;
		else if (c == 'x') opt.xdrop = atoi(o.arg);
		else if (c == 'j') ns_opt_set_sp(&opt, atoi(o.arg));
		else if (c == 'J') opt.io = atoi(o.arg);
		else if (c == 'w') use_32 = 1;
	}
	if (argc - o.ind < 2) {
		fprintf(stderr, "Usage: nasw [options] <nt.fa> <aa.fa>\n");
		fprintf(stderr, "Options:\n");
		fprintf(stderr, "  -w       use 32-bit integers for scoring (16-bit by default)\n");
		fprintf(stderr, "  -s       compute score only without nasw-CIGAR\n");
		fprintf(stderr, "  -l       left extension\n");
		fprintf(stderr, "  -r       right extension\n");
		fprintf(stderr, "  -x INT   x-drop [%d]\n", opt.xdrop);
		fprintf(stderr, "  -p       non-SSE mode\n");
		return 1;
	}

	fpn = gzopen(argv[o.ind], "r");
	assert(fpn);
	fpa = gzopen(argv[o.ind+1], "r");
	assert(fpa);
	ksn = kseq_init(fpn);
	ksa = kseq_init(fpa);

	while (kseq_read(ksn) >= 0 && kseq_read(ksa) >= 0) {
		ns_rst_t r;
		int32_t i;
		ns_rst_init(&r);
		if (no_sse) ns_splice_s1(0, ksn->seq.s, ksn->seq.l, ksa->seq.s, ksa->seq.l, &opt, &r);
		else if (use_32) ns_global_gs32(0, ksn->seq.s, ksn->seq.l, ksa->seq.s, ksa->seq.l, &opt, &r);
		else ns_global_gs16(0, ksn->seq.s, ksn->seq.l, ksa->seq.s, ksa->seq.l, &opt, &r);
		if (opt.flag & NS_F_EXT_LEFT) {
			printf("%s\t%ld\t%ld\t%ld\t+\t%s\t%ld\t%ld\t%ld\t%d\t", ksn->name.s, ksn->seq.l, ksn->seq.l - r.nt_len, ksn->seq.l, ksa->name.s, ksa->seq.l, ksa->seq.l - r.aa_len, ksa->seq.l, r.score);
		} else {
			printf("%s\t%ld\t0\t%d\t+\t%s\t%ld\t0\t%d\t%d\t", ksn->name.s, ksn->seq.l, r.nt_len, ksa->name.s, ksa->seq.l, r.aa_len, r.score);
		}
		for (i = 0; i < r.n_cigar; ++i)
			printf("%d%c", r.cigar[i]>>4, NS_CIGAR_STR[r.cigar[i]&0xf]);
		putchar('\n');
	}

	kseq_destroy(ksn);
	kseq_destroy(ksa);
	gzclose(fpn);
	gzclose(fpa);
	return 0;
}
