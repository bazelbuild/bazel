// Copyright 2001,2007 Alan Donovan. All rights reserved.
//
// Author: Alan Donovan <adonovan@google.com>
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
//
// ijar.cpp -- .jar -> _interface.jar tool.
//

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>

#include "common.h"

static void usage() {
  fprintf(stderr, "Usage: ijar [-v] x.jar [x_interface.jar>]\n");
  fprintf(stderr, "Creates an interface jar from the specified jar file.\n");
  exit(1);
}

int main(int argc, char **argv) {
  const char *filename_in = NULL;
  const char *filename_out = NULL;

  for (int ii = 1; ii < argc; ++ii) {
    if (strcmp(argv[ii], "-v") == 0) {
      devtools_ijar::verbose = true;
    } else if (filename_in == NULL) {
      filename_in = argv[ii];
    } else if (filename_out == NULL) {
      filename_out = argv[ii];
    } else {
      usage();
    }
  }

  if (filename_in == NULL) {
    usage();
  }

  // Guess output filename from input:
  char filename_out_buf[PATH_MAX];
  if (filename_out == NULL) {
    size_t len = strlen(filename_in);
    if (len > 4 && strncmp(filename_in + len - 4, ".jar", 4) == 0) {
      strcpy(filename_out_buf, filename_in);
      strcpy(filename_out_buf + len - 4, "-interface.jar");
      filename_out = filename_out_buf;
    } else {
      fprintf(stderr, "Can't determine output filename since input filename "
              "doesn't end with '.jar'.\n");
      return 1;
    }
  }

  if (devtools_ijar::verbose) {
    fprintf(stderr, "INFO: writing to '%s'.\n", filename_out);
  }

  return devtools_ijar::OpenFilesAndProcessJar(filename_out, filename_in);
}
