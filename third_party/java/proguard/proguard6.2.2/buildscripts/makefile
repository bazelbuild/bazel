# GNU/Linux makefile for ProGuard.

BASIC_MODULES    = core retrace gui annotations
OPTIONAL_MODULES = gradle ant wtk

MODULES = $(BASIC_MODULES) $(OPTIONAL_MODULES)

LIB = ../lib

# The main targets.

all:      basic optional
basic:    $(BASIC_MODULES)
optional: $(OPTIONAL_MODULES)

$(MODULES):
	cd ../$@ && $(MAKE)

clean:
	rm -fr $(LIB)

.PHONY: all basic optional $(MODULES) clean
