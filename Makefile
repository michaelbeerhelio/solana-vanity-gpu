OS := $(shell uname)

all:
ifeq ($(OS),Darwin)
SO=dylib
else
SO=so
all: cuda_crypt
endif

V=release

.PHONY:cuda_crypt
cuda_crypt:
	$(MAKE) V=$(V) -C src

DESTDIR ?= dist
install:
	mkdir -p $(DESTDIR)
ifneq ($(OS),Darwin)
	cp -f src/$(V)/libcuda-crypt.so $(DESTDIR)
endif
	ls -lh $(DESTDIR)

.PHONY:clean
clean:
	$(MAKE) V=$(V) -C src clean

cuda_ed25519_vanity_lib:
	$(MAKE) V=$(V) -C src cuda_ed25519_vanity_shared

install_lib: cuda_ed25519_vanity_lib
	mkdir -p $(DESTDIR)
	cp -f src/$(V)/libcuda-ed25519-vanity.so $(DESTDIR)
