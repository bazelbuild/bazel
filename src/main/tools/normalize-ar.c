// Copyright 2014 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Removes unnecessary variation in archive libraries.  We walk through
// the archive, rewriting each file header to set the timestamp, user
// id, group id, and mode to fixed values.

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define const_streq(str, pat) \
  (!strncmp(str, pat, sizeof ("" pat "")-1))

int member_is_special(const char *name) {
  return const_streq(name, "/               ") ||
         const_streq(name, "//              ") ||
         const_streq(name, "/SYM64/         ") ||
         const_streq(name, "ARFILENAMES/    ") ||
         const_streq(name, "__.SYMDEF       ") ||
         const_streq(name, "__.SYMDEF/      ");
}

const char *filename = "NONE";
FILE *file = NULL;
int thin = 0;

void die(const char *msg) {
  fprintf(stderr, "%s: file=%s errno=%s\n", msg, filename, strerror(errno));
  exit(1);
}

struct header {
  char name[16];
  char date[12];
  char uid[6];
  char gid[6];
  char mode[8];
  char size[10];
  char magic[2];
} __attribute__((packed));

void fix_header() {
  struct header hdr;
  if (sizeof hdr != 60) die("struct header has wrong size");
  if (fread(&hdr, sizeof hdr, 1, file) < 1) die("fread");
#if 0
  printf("%.16s %.12s %.6s %.6s %.8s %.10s\n",
         hdr.name, hdr.date, hdr.uid, hdr.gid, hdr.mode, hdr.size);
#endif
  if (!const_streq(hdr.magic, "`\012")) die("bad header magic");
  long skip = atol(hdr.size);  // last digit always followed by [ `]
  if (skip < 0) die("atol");
  if (thin && !member_is_special(hdr.name)) skip = 0;
  skip += skip % 2;
  memcpy(hdr.date, "0           ", 12);
  memcpy(hdr.uid, "0     ", 6);
  memcpy(hdr.gid, "0     ", 6);
  memcpy(hdr.mode, "100555  ", 8);
  if (fseek(file, -sizeof hdr, SEEK_CUR)) die("fseek");
  if (fwrite(&hdr, sizeof hdr, 1, file) < 1) die("fwrite");
  if (fseek(file, skip, SEEK_CUR)) die("fseek");
}

void fix_file() {
  if (fseek(file, 0, SEEK_END)) die("fseek");
  long size = ftell(file);
  if (size < 0) die("ftell");
  rewind(file);
  char buf[8];
  if (fread(buf, 8, 1, file) < 1) die("fread");
  if (const_streq(buf, "!<thin>\n")) {
    thin = 1;
  } else if (const_streq(buf, "!<arch>\n")) {
    thin = 0;
  } else {
    die("bad file magic");
  }
  for (;;) {
    long pos = ftell(file);
    if (pos < 0) die("ftell");
    if (pos >= size) break;
    fix_header();
  }
}

int main(int argc, char **argv) {
  int i;
  for (i = 1; i < argc; ++i) {
    file = fopen(filename = argv[i], "r+");
    if (!file) die("fopen for update");
    fix_file();
    if (fclose(file)) die("fclose");
  }
  return 0;
}
