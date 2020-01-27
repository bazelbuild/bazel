// Copyright 2015 The Bazel Authors. All rights reserved.
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
//  realpath.c
//
//  A simple implementation of realpath for Mac OS X.
//  This implementation follows gnu/linux conventions and allows the last
//  component to not exist:
//  http://www.gnu.org/software/coreutils/manual/html_node/realpath-invocation.html
//  Debian requires all components to exist.
//

#include <errno.h>
#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// Print a simple error message and exit.
static void PrintError(const char *argv[]) {
  fprintf(stderr, "%s: %s\n", argv[1], strerror(errno));
  exit(1);
}

// Concatenate two paths together adding a '/' if appropriate.
// Returned pointer is owned by client and should be freed.
static char *JoinPaths(const char *path1, const char* path2) {
  size_t len1 = strlen(path1);
  size_t len2 = strlen(path2);
  // +1 for '/' and +1 for '\0'
  size_t totalSize = len1 + 1 + len2 + 1;
  char *outPath = malloc(totalSize);
  if (outPath == NULL) {
    return NULL;
  }
  strlcpy(outPath, path1, totalSize);
  if (len1 > 0 && len2 > 0) {
    if (path1[len1 - 1] != '/' && path2[0] != '/') {
      strlcat(outPath, "/", totalSize);
    }
  }
  strlcat(outPath, path2, totalSize);
  return outPath;
}

// Since this is a simple utility that quits immediately, we are not worrying
// about making the code more complex by freeing up any memory allocations.
int main(int argc, const char *argv[]) {
  if (argc != 2) {
    fprintf(stderr, "realpath <path>\n");
    return 1;
  }
  const char *path = argv[1];
  char *goodPath = realpath(path, NULL);
  if (goodPath == NULL) {
    if ((errno != ENOENT) || (strlen(path) == 0)) {
      PrintError(argv);
    }

    // If only the last element is missing, then call realpath on the parent
    // dir and append the basename back onto it.

    // Technically the strdup is not required on Mac OS X, but this
    // keeps things compatible with other basename/dirname implementations
    // that do require a string they can modify.
    char *dirCopy = strdup(path);
    char *baseCopy = strdup(path);
    if (dirCopy == NULL || baseCopy == NULL) {
      PrintError(argv);
    }
    char *dir = dirname(dirCopy);
    if (dir == NULL) {
      PrintError(argv);
    }
    char *base = basename(baseCopy);
    if (base == NULL) {
      PrintError(argv);
    }
    char *realdir = realpath(dir, NULL);
    if (realdir == NULL) {
      PrintError(argv);
    }
    goodPath = JoinPaths(realdir, base);
    if (goodPath == NULL) {
      PrintError(argv);
    }
  }
  fprintf(stdout, "%s\n", goodPath);
  return 0;
}
