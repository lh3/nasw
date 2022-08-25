#include <string.h>
#include <stdio.h>
#include "nasw.h"
#include "kalloc.h"

#ifdef __SSE4_1__
#include <smmintrin.h>
#endif

/*
 * I(i,j) = max{ H(i,j-1) - q, I(i,j-1) } - e
 * D(i,j) = max{ H(i-3,j) - q, D(i-3,j) } - e
 * A(i,j) = max{ H(i-1,j)   - r - d(i-1), A(i-1,j) }
 * B(i,j) = max{ H(i-1,j-1) - r - d(i),   B(i-1,j) }
 * C(i,j) = max{ H(i-1,j-1) - r - d(i+1), C(i-1,j) }
 * H(i,j) = max{ H(i-3,j-1) + s(i,j), I(i,j), D(i,j), H(i-1,j-1)-f, H(i-2,j-1)-f, H(i-1,j)-f, H(i-2,j)-f, A(i,j)-a(i), B(i,j)-a(i-2), C(i,j)-a(i-1) }
 */

static uint8_t *ns_prep_seq(void *km, const char *ns, int32_t nl, const char *as, int32_t al, const ns_opt_t *opt, uint8_t **aas_, int8_t **donor_, int8_t **acceptor_)
{
	int32_t i, j, l;
	uint8_t codon, *nas, *aas;
	int8_t *donor, *acceptor;
	nas = Kmalloc(km, uint8_t, nl + al + (nl + 1) * 2);
	*aas_ = aas = nas + nl;
	*donor_ = donor = (int8_t*)aas + al, *acceptor_ = acceptor = donor + nl + 1;
	for (j = 0; j < al; ++j) // generate aas[]
		aas[j] = opt->aa20[(uint8_t)as[j]];
	for (i = 0; i < nl; ++i) // nt4 encoding of ns[] for computing donor[] and acceptor[]
		nas[i] = opt->nt4[(uint8_t)ns[i]];
	for (i = 0; i < nl + 1; ++i)
		donor[i] = acceptor[i] = opt->nc;
	for (i = 0; i < nl - 3; ++i) { // generate donor[]
		int32_t t = 0;
		if (nas[i+1] == 2 && nas[i+2] == 3) t = 1;
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
	return nas;
}

#define ns_gen_prof(INT_TYPE, _km, aas, al, opt, neg_inf, _mem_ap, _ap) do { \
	INT_TYPE *t; \
	int32_t a, p = 16 / sizeof(INT_TYPE), slen = (al + p - 1) / p; \
	*(_mem_ap) = Kmalloc(_km, uint8_t, (16 * slen * opt->asize + 31) / 16 * 16); \
	*(_ap) = (__m128i*)(((size_t)*(_mem_ap) + 15) / 16 * 16); /* 16-byte aligned */ \
	t = (INT_TYPE*)*(_ap); \
	for (a = 0; a < opt->asize; ++a) { \
		int32_t i, k, nlen = slen * p; \
		const int8_t *ma = opt->sc + a * opt->asize;\
		for (i = 0; i < slen; ++i) \
			for (k = i; k < nlen; k += slen) \
				*t++ = (k >= al? neg_inf : ma[aas[k]]); \
	} \
} while (0)

#define sse_gen(func, suf) _mm_##func##_##suf

#define NS_GEN_DEF(INT_TYPE) \
	typedef INT_TYPE ns_int_t; \
	const int32_t ssize = sizeof(ns_int_t), vsize = 16 / ssize; \
	const INT_TYPE neg_inf = (INT_TYPE)(1 << (8*ssize - 1));

#define NS_GEN_VAR(_suf) \
	int32_t i, j, slen = (al + vsize - 1) / vsize; /* segment length */ \
	uint8_t *nas, *aas, *mem_ap, *mem_H; \
	int8_t *donor, *acceptor; \
	__m128i *ap, *H0, *H, *H1, *H2, *H3, *D, *D1, *D2, *D3, *A, *B, *C; \
	__m128i go, ge, goe, io, fs; \

#define NS_GEN_PREPARE(_suf) \
	r->n_cigar = 0; \
	nas = ns_prep_seq(km, ns, nl, as, al, opt, &aas, &donor, &acceptor); \
	ns_gen_prof(ns_int_t, km, aas, al, opt, neg_inf, &mem_ap, &ap); \
	go = sse_gen(set1, _suf)(opt->go); \
	ge = sse_gen(set1, _suf)(opt->ge); \
	goe= sse_gen(set1, _suf)(opt->go + opt->ge); \
	io = sse_gen(set1, _suf)(opt->io); \
	fs = sse_gen(set1, _suf)(opt->fs); \
	mem_H = Kmalloc(km, uint8_t, (sizeof(__m128i) * ((slen + 1) * 4 + slen * 7) + 31) / 16 * 16); \
	H0 = (__m128i*)(((size_t)mem_H + 15) / 16 * 16); /* 16-byte aligned */ \
	H = H0 + 1, H1 = H0 + (slen + 1) + 1, H2 = H0 + (slen + 1) * 2 + 1, H3 = H0 + (slen + 1) * 3 + 1; \
	D = H3 + slen, D1 = D + slen, D2 = D1 + slen, D3 = D2 + slen; \
	A = D3 + slen, B = A + slen, C = B + slen;

#define NS_GEN_INIT1(_suf) \
	for (i = 0; i < (slen + 1) * 4 + slen * 7; ++i) \
		H0[i] = sse_gen(set1, _suf)(neg_inf); \
	H3[-1] = sse_gen(insert, _suf)(H3[-1], 0, 0); \
	H2[-1] = sse_gen(insert, _suf)(H2[-1], -opt->fs, 0); \
	H1[-1] = sse_gen(insert, _suf)(H1[-1], -opt->fs, 0);

#define NS_GEN_INIT2(_suf) \
		int32_t k; \
		__m128i *tmp, I, *S = ap + nas[i] * slen, dim1, di, dip1, ai, aim1, aim2, last_h; \
		dim1 = sse_gen(set1, _suf)(donor[i-1]), di = sse_gen(set1, _suf)(donor[i]), dip1 = sse_gen(set1, _suf)(donor[i+1]); \
		ai = sse_gen(set1, _suf)(acceptor[i]), aim1 = sse_gen(set1, _suf)(acceptor[i-1]), aim2 = sse_gen(set1, _suf)(acceptor[i-2]); \
		I = last_h = sse_gen(set1, _suf)(neg_inf); \
		if (i > 2) { /* FIXME: this is close but not correct */ \
			H3[-1] = sse_gen(insert, _suf)(_mm_slli_si128(H3[slen - 1], sizeof(ns_int_t)), neg_inf, 0); \
			H2[-1] = sse_gen(insert, _suf)(_mm_slli_si128(H2[slen - 1], sizeof(ns_int_t)), neg_inf, 0); \
			H1[-1] = sse_gen(insert, _suf)(_mm_slli_si128(H1[slen - 1], sizeof(ns_int_t)), neg_inf, 0); \
		}

void ns_gs16(void *km, const char *ns, int32_t nl, const char *as, int32_t al, const ns_opt_t *opt, ns_rst_t *r)
{
	NS_GEN_DEF(int16_t)
	NS_GEN_VAR(epi16)
	NS_GEN_PREPARE(epi16)
	NS_GEN_INIT1(epi16)

	for (i = 2; i < nl; ++i) {
		NS_GEN_INIT2(epi16)
		for (j = 0; j < slen; ++j) {
			__m128i h, t, u, v;
			u = _mm_load_si128(H3 + j - 1);
			v = _mm_load_si128(S + j);
			h = _mm_adds_epi16(u, v);

			t = _mm_subs_epi16(last_h, go);
			t = _mm_max_epi16(t, I);
			I = _mm_subs_epi16(t, ge);
			h = _mm_max_epi16(h, I);

			u = _mm_load_si128(H3 + j);
			v = _mm_load_si128(D3 + j);
			t = _mm_max_epi16(_mm_subs_epi16(u, go), v);
			t = _mm_subs_epi16(t, ge);
			_mm_store_si128(D + j, t);
			h = _mm_max_epi16(h, t);

			u = _mm_subs_epi16(_mm_load_si128(H1 + j), io);
			v = _mm_load_si128(A + j);
			t = _mm_subs_epi16(u, dim1);
			t = _mm_max_epi16(t, v);
			_mm_store_si128(A + j, t);
			h = _mm_max_epi16(h, _mm_subs_epi16(t, ai));

			u = _mm_subs_epi16(_mm_load_si128(H1 + j - 1), io);
			v = _mm_load_si128(B + j);
			t = _mm_subs_epi16(u, di);
			t = _mm_max_epi16(t, v);
			_mm_store_si128(B + j, t);
			h = _mm_max_epi16(h, _mm_subs_epi16(t, aim2));

			v = _mm_load_si128(C + j);
			t = _mm_subs_epi16(u, dip1);
			t = _mm_max_epi16(t, v);
			_mm_store_si128(C + j, t);
			h = _mm_max_epi16(h, _mm_subs_epi16(t, aim1));

			t = _mm_subs_epi16(_mm_load_si128(H1 + j), fs);
			h = _mm_max_epi16(h, t);

			t = _mm_subs_epi16(_mm_load_si128(H2 + j), fs);
			h = _mm_max_epi16(h, t);

			t = _mm_subs_epi16(_mm_load_si128(H1 + j - 1), fs);
			h = _mm_max_epi16(h, t);

			t = _mm_subs_epi16(_mm_load_si128(H2 + j - 1), fs);
			h = _mm_max_epi16(h, t);

			_mm_store_si128(H + j, h);
			last_h = h;
		}
		for (k = 0; k < vsize; ++k) {
			I = _mm_insert_epi16(_mm_slli_si128(I, sizeof(ns_int_t)), neg_inf, 0);
			for (j = 0; j < slen; ++j) {
				__m128i h;
				h = _mm_load_si128(H + j);
				h = _mm_max_epi16(h, I);
				_mm_store_si128(H + j, h);
				h = _mm_subs_epi16(h, goe);
				I = _mm_subs_epi16(I, ge);
				if (!_mm_movemask_epi8(_mm_cmpgt_epi16(I, h))) break;
			}
			if (k < vsize) break;
		}
		tmp = H3, H3 = H2, H2 = H1, H1 = H, H = tmp;
		tmp = D3, D3 = D2, D2 = D1, D1 = D, D = tmp;
	}
	r->score = *((ns_int_t*)&H1[(al-1)%slen] + (al < vsize? al : vsize) - 1);
	kfree(km, mem_H);
	kfree(km, mem_ap);
	kfree(km, nas);
}
