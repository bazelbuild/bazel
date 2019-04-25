AUTOMAKE_OPTIONS = foreign nostdinc subdir-objects

ACLOCAL_AMFLAGS = -I m4

# Actual ABI version is substituted by bootstrap
LIBBROTLI_VERSION_INFO = -version-info 0:0:0

bin_PROGRAMS = brotli
lib_LTLIBRARIES = libbrotlicommon.la libbrotlidec.la libbrotlienc.la

include scripts/sources.lst

brotliincludedir = $(includedir)/brotli
brotliinclude_HEADERS = $(BROTLI_INCLUDE)

AM_CFLAGS = -I$(top_srcdir)/c/include

brotli_SOURCES = $(BROTLI_CLI_C)
brotli_LDADD = libbrotlidec.la libbrotlienc.la libbrotlicommon.la -lm
#brotli_LDFLAGS = -static

libbrotlicommon_la_SOURCES = $(BROTLI_COMMON_C) $(BROTLI_COMMON_H)
libbrotlicommon_la_LDFLAGS = $(AM_LDFLAGS) $(LIBBROTLI_VERSION_INFO) $(LDFLAGS)
libbrotlidec_la_SOURCES = $(BROTLI_DEC_C) $(BROTLI_DEC_H)
libbrotlidec_la_LDFLAGS = $(AM_LDFLAGS) $(LIBBROTLI_VERSION_INFO) $(LDFLAGS)
libbrotlidec_la_LIBADD = libbrotlicommon.la -lm
libbrotlienc_la_SOURCES = $(BROTLI_ENC_C) $(BROTLI_ENC_H)
libbrotlienc_la_LDFLAGS = $(AM_LDFLAGS) $(LIBBROTLI_VERSION_INFO) $(LDFLAGS)
libbrotlienc_la_LIBADD = libbrotlicommon.la -lm

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = \
  scripts/libbrotlicommon.pc \
  scripts/libbrotlidec.pc \
  scripts/libbrotlienc.pc
pkgincludedir= $(brotliincludedir)

dist_doc_DATA = README
