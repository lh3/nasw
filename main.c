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
	int32_t c;

	ns_make_tables(0);
	ns_opt_init(&opt);
	while ((c = ketopt(&o, argc, argv, 1, "", 0)) >= 0) {
	}
	if (argc - o.ind < 2) {
		fprintf(stderr, "Usage: nasw [options] <nt.fa> <aa.fa>\n");
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
		ns_splice_s1(0, ksn->seq.s, ksn->seq.l, ksa->seq.s, ksa->seq.l, &opt, &r);
		printf("%s\t%ld\t%s\t%ld\t%d\t", ksn->name.s, ksn->seq.l, ksa->name.s, ksa->seq.l, r.score);
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
