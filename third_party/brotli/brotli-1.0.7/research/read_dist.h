/* Copyright 2016 Google Inc. All Rights Reserved.
   Author: zip753@gmail.com (Ivan Nikulin)

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

/* API for reading distances from *.dist file.
   The format of *.dist file is as follows: for each backward reference there is
   a position-distance pair, also a copy length may be specified. Copy length is
   prefixed with flag byte 0, position-distance pair is prefixed with flag
   byte 1. Each number is a 32-bit integer. Copy length always comes before
   position-distance pair. Standalone copy length is allowed, in this case it is
   ignored. */

#ifndef BROTLI_RESEARCH_READ_DIST_H_
#define BROTLI_RESEARCH_READ_DIST_H_

#include <cstdio>
#include <cstdlib>  /* exit, EXIT_FAILURE */

#if !defined(CHECK)
#define CHECK(X) if (!(X)) exit(EXIT_FAILURE);
#endif

/* Reads backwards reference from .dist file. Sets all missing fields to -1.
   Returns false when EOF is met or input is corrupt. */
bool ReadBackwardReference(FILE* fin, int* copy, int* pos, int* dist) {
  int c = getc(fin);
  if (c == EOF) return false;
  if (c == 0) {
    CHECK(fread(copy, sizeof(int), 1, fin) == 1);
    if ((c = getc(fin)) != 1) {
      ungetc(c, fin);
      *pos = *dist = -1;
    } else {
      CHECK(fread(pos, sizeof(int), 1, fin) == 1);
      CHECK(fread(dist, sizeof(int), 1, fin) == 1);
    }
  } else if (c != 1) {
    return false;
  } else {
    CHECK(fread(pos, sizeof(int), 1, fin) == 1);
    CHECK(fread(dist, sizeof(int), 1, fin) == 1);
    *copy = -1;
  }
  return true;
}

#endif  /* BROTLI_RESEARCH_READ_DIST_H_ */
