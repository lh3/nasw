CC=			gcc
CXX=		g++
CFLAGS=		-g -Wall -std=c99 -O3
CXXFLAGS=	$(CFLAGS)
CPPFLAGS=
INCLUDES=
OBJS=		kalloc.o nasw-tab.o nasw-s.o nasw-sse.o
PROG=		nasw
LIBS=		-lpthread -lz

ifneq ($(asan),)
	CFLAGS+=-fsanitize=address
	LIBS+=-fsanitize=address -ldl -lm
endif

ifneq ($(sse4),)
	CFLAGS+=-msse4
endif

.SUFFIXES:.c .cpp .o
.PHONY:all clean depend

.c.o:
		$(CC) -c $(CFLAGS) $(CPPFLAGS) $(INCLUDES) $< -o $@

.cpp.o:
		$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) $(INCLUDES) $< -o $@

all:$(PROG)

nasw:$(OBJS) main.o
		$(CC) $(CFLAGS) $^ -o $@ $(LIBS)

clean:
		rm -fr gmon.out *.o a.out $(PROG) fmd-occ *~ *.a *.dSYM

depend:
		(LC_ALL=C; export LC_ALL; makedepend -Y -- $(CFLAGS) $(DFLAGS) -- *.c *.cpp)

# DO NOT DELETE

kalloc.o: kalloc.h
main.o: ketopt.h nasw.h kalloc.h kseq.h
nasw-s.o: nasw.h kalloc.h
nasw-sse.o: nasw.h kalloc.h s2n-lite.h
nasw-tab.o: nasw.h kalloc.h
