#brotli/tests

BROTLI = ..

all: test

test: deps
	./compatibility_test.sh
	./roundtrip_test.sh

deps :
	$(MAKE) -C $(BROTLI) brotli

clean :
	rm -f testdata/*.{br,unbr,uncompressed}
	rm -f $(BROTLI)/{enc,dec,tools}/*.{un,}br
	$(MAKE) -C $(BROTLI)/tools clean
