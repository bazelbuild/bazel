CC = g++
CFLAGS += -O2
CPPFLAGS += -std=c++11
SOURCES = $(wildcard *.cc)
EXECUTABLES = $(SOURCES:.cc=)
BINDIR = bin

all: $(EXECUTABLES)

$(BINDIR):
	mkdir -p $@

$(EXECUTABLES): $(BINDIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(addsuffix .cc, $@) -o $(BINDIR)/$@ -lgflags_nothreads

clean:
	rm -rf $(BINDIR)
