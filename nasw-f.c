#include <string.h>
#include "nasw.h"
#include "kalloc.h"

#ifdef __SSE4_1__
#include <smmintrin.h>
#endif

#define NS_NEG_INF16 (-0x7000)

typedef int16_t ns_int_t;

void ns_splice_i16(void *km, const char *ns, int32_t nl, const char *as, int32_t al, const ns_opt_t *opt, ns_rst_t *r)
{
	int32_t p = 8; // number of values per SSE vector
	int32_t slen = (al + p - 1) / p;
	int32_t i, j;
	uint8_t *nas, *aas, *ap0;
	int8_t *donor, *acceptor;
	__m128i *ap;

	{ // generate nas[], aas[], donor[] and acceptor[]
		int32_t l;
		uint8_t codon;
		nas = Kmalloc(km, uint8_t, nl + al + (nl + 1) * 2);
		aas = nas + nl;
		donor = (int8_t*)aas + al, acceptor = donor + nl + 1;
		for (j = 0; j < al; ++j) // generate aas[]
			aas[j] = opt->aa20[(uint8_t)as[j]];
		for (i = 0; i < nl; ++i) // nt4 encoding of ns[] for computing donor[] and acceptor[]
			nas[i] = opt->nt4[(uint8_t)ns[i]];
		for (i = 0; i < nl + 1; ++i)
			donor[i] = acceptor[i] = opt->nc;
		for (i = 0; i < nl - 3; ++i) { // generate donor[]
			int32_t t = 0;
			if (nas[i] == 2 && nas[i+2] == 3) t = 1;
			if (t && i + 3 < nl && (nas[i+3] == 0 || nas[i+3] == 2)) t = 2;
			donor[i] = t == 2? 0 : t == 1? opt->nc/2 : opt->nc;
		}
		for (i = 1; i < nl; ++i) { // generate acceptor[]
			int32_t t = 0;
			if (nas[i-1] == 0 && nas[i] == 2) t = 1;
			if (t && i > 0 && (nas[i-2] == 1 || nas[i-2] == 3)) t = 2;
			acceptor[i] = t == 2? 0 : t == 1? opt->nc/2 : opt->nc;
		}
		memset(nas, opt->aa20['X'], nl);
		for (i = l = 0, codon = 0; i < nl; ++i) { // generate the real nas[]
			uint8_t c = opt->nt4[(uint8_t)ns[i]];
			if (c < 4) {
				codon = (codon << 2 | c) & 0x3f;
				if (++l >= 3)
					nas[i] = opt->codon[codon];
			} else codon = 0, l = 0;
		}
	}

	{ // generate protein profile
		ns_int_t *t;
		int32_t a;
		ap0 = Kmalloc(km, uint8_t, (slen * opt->asize + 31) / 16 * 16);
		ap = (__m128i*)(((size_t)ap0 + 15) / 16 * 16); // make ap 16-byte aligned
		t = (ns_int_t*)ap;
		for (a = 0; a < opt->asize; ++a) {
			int i, k, nlen = slen * p;
			const int8_t *ma = opt->sc + a * opt->asize;
			for (i = 0; i < slen; ++i)
				for (k = i; k < nlen; k += slen) // p iterations
					*t++ = (k >= al? 0 : ma[aas[k]]);
		}
	}

	kfree(km, ap0);
	kfree(km, nas);
}
