CC  	:= gcc 
CPP 	:= g++
ARCH	:= -arch x86_64

CFLAGS 	:= -Wall -O2
LIBCFLAGS := -Wall -O3 \
		-current_version 1.0 \
		-compatibility_version 1.0 \
		-fvisibility=hidden \
		-dynamiclib \
		-std=gnu99

INCINSTALLDIR := /usr/local/include
LIBINSTALLDIR := /usr/local/lib
VERSION := 1

KTRLIBS  := -lknitro
BLASLIBS := -framework Accelerate
SPLIBS   := -lcholmod -lamd -lcolamd -lcamd -lccolamd -lmetis -lspqr -lma57 -lgfortran
CPDTLIBS := -lcpdt

default:

	# compiling object files
	$(CC) $(ARCH) $(CFLAGS) -c ktrderchecks.c -o ktrderchecks.o

	# $(CC) $(ARCH) $(CFLAGS) -c ktrsosc.c -o ktrsosc.o

	# creating static library
	libtool *.o -o libktrextras.a -static -s -v
	
	# creating dynamic library
	libtool *.o -o libktrextras.dylib \
		# -lm $(BLASLIBS) $(SPLIBS) \
		-dynamic \
		-install_name $(LIBINSTALLDIR)/ \
		-current_version $(VERSION) \
		-v

install:

	# copying files to their installed location
	cp ktrextras.h $(INCINSTALLDIR)/
	cp libktrextras* $(LIBINSTALLDIR)/