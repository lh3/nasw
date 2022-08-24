#include <string.h>
#include "nasw.h"
#include "kalloc.h"

#ifdef __SSE4_1__
#include <smmintrin.h>
#endif

#define NS_SSE_P       (8)
#define NS_SSE_INT     int16_t
#define NS_SSE_NEG_INF (-0x7000)

void ns_splice_i16(void *km, const char *ns, int32_t nl, const char *as, int32_t al, const ns_opt_t *opt, ns_rst_t *r)
{
	int32_t i, j, slen = (al + NS_SSE_P - 1) / NS_SSE_P; // segment length
	uint8_t *nas, *aas, *ap0;
	int8_t *donor, *acceptor;
	__m128i *ap, *H0;

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
		NS_SSE_INT *t;
		int32_t a;
		ap0 = Kmalloc(km, uint8_t, (slen * opt->asize + 31) / 16 * 16);
		ap = (__m128i*)(((size_t)ap0 + 15) / 16 * 16); // make ap 16-byte aligned
		t = (NS_SSE_INT*)ap;
		for (a = 0; a < opt->asize; ++a) {
			int i, k, nlen = slen * NS_SSE_P;
			const int8_t *ma = opt->sc + a * opt->asize;
			for (i = 0; i < slen; ++i)
				for (k = i; k < nlen; k += slen) // p iterations
					*t++ = (k >= al? NS_SSE_NEG_INF : ma[aas[k]]);
		}
	}

	/*
	 * I(i,j) = max{ H(i,j-1) - q, I(i,j-1) } - e
	 * D(i,j) = max{ H(i-3,j) - q, D(i-3,j) } - e
	 * A(i,j) = max{ H(i-1,j)   - r - d(i-1), A(i-1,j) }
	 * B(i,j) = max{ H(i-1,j-1) - r - d(i),   B(i-1,j) }
	 * C(i,j) = max{ H(i-1,j-1) - r - d(i+1), C(i-1,j) }
	 * H(i,j) = max{ H(i-3,j-1) + s(i,j), I(i,j), D(i,j), H(i-1,j)-f, H(i-2,j)-f, A(i,j)-a(i), B(i,j)-a(i-2), C(i,j)-a(i-1) }
	 */
	{
		__m128i *H, *G[3], *D[3], *A, *B, *C, go, ge, io, fs;

		H0 = H = Kmalloc(km, __m128i, slen * 10);
		G[0] = H + slen, G[1] = G[0] + slen, G[2] = G[1] + slen;
		D[0] = G[2] + slen, D[1] = D[0] + slen, D[2] = D[1] + slen;
		A = D[2] + slen, B = A + slen, C = B + slen;

		go = _mm_set1_epi16(opt->go);
		ge = _mm_set1_epi16(opt->ge);
		io = _mm_set1_epi16(opt->io);
		fs = _mm_set1_epi16(opt->fs);

		for (i = 2; i < nl; ++i) {
			__m128i *tmp, *D3 = D[0], *H3 = G[0], *H2 = G[1], *H1 = G[2];
			__m128i *S = ap + nas[i] * slen;
			for (j = 0; j < slen; ++j) {
				__m128i h, t, u;
				h = _mm_add_epi16(H3[j], S[j]);

				t = _mm_sub_epi16(H3[j], go);
				t = _mm_max_epi16(t, D3[j]);
				t = _mm_sub_epi16(t, ge);
				_mm_store_si128(D3 + j, t);
				h = _mm_max_epi16(h, t);

				u = _mm_sub_epi16(H1[j], io);
				t = _mm_max_epi16(u, A[j]);
				_mm_store_si128(A + j, t);
				h = _mm_max_epi16(h, t);

				t = _mm_max_epi16(u, B[j]);
				_mm_store_si128(B + j, t);
				h = _mm_max_epi16(h, t);

				t = _mm_max_epi16(u, C[j]);
				_mm_store_si128(C + j, t);
				h = _mm_max_epi16(h, t);

				t = _mm_sub_epi16(H1[j], fs);
				h = _mm_max_epi16(h, t);

				t = _mm_sub_epi16(H2[j], fs);
				h = _mm_max_epi16(h, t);
			}
		}
	}

	kfree(km, H0);
	kfree(km, ap0);
	kfree(km, nas);
}
