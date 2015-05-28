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
#include <errno.h>
#include <memory>

#include "third_party/ijar/zip.h"

namespace devtools_ijar {

bool verbose = false;

// Reads a JVM class from classdata_in (of the specified length), and
// writes out a simplified class to classdata_out, advancing the
// pointer.
void StripClass(u1 *&classdata_out, const u1 *classdata_in, size_t in_length);

const char* CLASS_EXTENSION = ".class";
const size_t CLASS_EXTENSION_LENGTH = strlen(CLASS_EXTENSION);

// ZipExtractorProcessor that select only .class file and use
// StripClass to generate an interface class, storing as a new file
// in the specified ZipBuilder.
class JarStripperProcessor : public ZipExtractorProcessor {
 public:
  JarStripperProcessor() {}
  virtual ~JarStripperProcessor() {}

  virtual void Process(const char* filename, const u4 attr,
                       const u1* data, const size_t size);
  virtual bool Accept(const char* filename, const u4 attr);

 private:
  // Not owned by JarStripperProcessor, see SetZipBuilder().
  ZipBuilder* builder;

 public:
  // Set the ZipBuilder to add the ijar class to the output zip file.
  // This pointer should not be deleted while this class is still in use and
  // it should be set before any call to the Process() method.
  void SetZipBuilder(ZipBuilder* builder) {
    this->builder = builder;
  }
};

bool JarStripperProcessor::Accept(const char* filename, const u4 attr) {
  ssize_t offset = strlen(filename) - CLASS_EXTENSION_LENGTH;
  if (offset >= 0) {
    return strcmp(filename + offset, CLASS_EXTENSION) == 0;
  }
  return false;
}

void JarStripperProcessor::Process(const char* filename, const u4 attr,
                                   const u1* data, const size_t size) {
  if (verbose) {
    fprintf(stderr, "INFO: StripClass: %s\n", filename);
  }
  u1 *q = builder->NewFile(filename, 0);
  u1 *classdata_out = q;
  StripClass(q, data, size);  // actually process it
  size_t out_length = q - classdata_out;
  builder->FinishFile(out_length);
}

// Opens "file_in" (a .jar file) for reading, and writes an interface
// .jar to "file_out".
void OpenFilesAndProcessJar(const char *file_out, const char *file_in) {
  JarStripperProcessor processor;
  std::unique_ptr<ZipExtractor> in(ZipExtractor::Create(file_in, &processor));
  if (in.get() == NULL) {
    fprintf(stderr, "Unable to open Zip file %s: %s\n", file_in,
            strerror(errno));
    abort();
  }
  u8 output_length = in->CalculateOutputLength();
  std::unique_ptr<ZipBuilder> out(ZipBuilder::Create(file_out, output_length));
  if (out.get() == NULL) {
    fprintf(stderr, "Unable to open output file %s: %s\n", file_out,
            strerror(errno));
    abort();
  }
  processor.SetZipBuilder(out.get());

  // Process all files in the zip
  if (in->ProcessAll() < 0) {
    fprintf(stderr, "%s\n", in->GetError());
    abort();
  }

  // Add dummy file, since javac doesn't like truly empty jars.
  if (out->GetNumberFiles() == 0) {
    out->WriteEmptyFile("dummy");
  }
  // Finish writing the output file
  if (out->Finish() < 0) {
    fprintf(stderr, "%s\n", out->GetError());
    abort();
  }
  // Get all file size
  size_t in_length = in->GetSize();
  size_t out_length = out->GetSize();
  if (verbose) {
    fprintf(stderr, "INFO: produced interface jar: %s -> %s (%d%%).\n",
            file_in, file_out,
            static_cast<int>(100.0 * out_length / in_length));
  }
}

}  // namespace devtools_ijar

//
// main method
//
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

  devtools_ijar::OpenFilesAndProcessJar(filename_out, filename_in);
  return 0;
}
