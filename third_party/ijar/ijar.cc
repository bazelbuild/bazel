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
// pointer. Returns true if the class should be kept.
bool StripClass(u1*& classdata_out, const u1* classdata_in, size_t in_length);

const char* CLASS_EXTENSION = ".class";
const size_t CLASS_EXTENSION_LENGTH = strlen(CLASS_EXTENSION);

const char *MANIFEST_DIR_PATH = "META-INF/";
const size_t MANIFEST_DIR_PATH_LENGTH = strlen(MANIFEST_DIR_PATH);
const char *MANIFEST_PATH = "META-INF/MANIFEST.MF";
const size_t MANIFEST_PATH_LENGTH = strlen(MANIFEST_PATH);
const char *MANIFEST_HEADER =
    "Manifest-Version: 1.0\r\n"
    "Created-By: bazel\r\n";
const size_t MANIFEST_HEADER_LENGTH = strlen(MANIFEST_HEADER);
// These attributes are used by JavaBuilder, Turbine, and ijar.
// They must all be kept in sync.
const char *TARGET_LABEL_KEY = "Target-Label: ";
const size_t TARGET_LABEL_KEY_LENGTH = strlen(TARGET_LABEL_KEY);
const char *INJECTING_RULE_KIND_KEY = "Injecting-Rule-Kind: ";
const size_t INJECTING_RULE_KIND_KEY_LENGTH = strlen(INJECTING_RULE_KIND_KEY);

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
  const size_t filename_len = strlen(filename);
  if (filename_len < CLASS_EXTENSION_LENGTH ||
      strcmp(filename + filename_len - CLASS_EXTENSION_LENGTH,
             CLASS_EXTENSION) != 0) {
    return false;
  }
  return true;
}

static bool IsModuleInfo(const char* filename) {
  const char* slash = strrchr(filename, '/');
  if (slash == NULL) {
    slash = filename;
  } else {
    slash++;
  }
  return strcmp(slash, "module-info.class") == 0;
}

void JarStripperProcessor::Process(const char* filename, const u4 attr,
                                   const u1* data, const size_t size) {
  if (verbose) {
    fprintf(stderr, "INFO: StripClass: %s\n", filename);
  }
  if (IsModuleInfo(filename)) {
    u1* q = builder->NewFile(filename, 0);
    memcpy(q, data, size);
    builder->FinishFile(size, false, true);
  } else {
    u1* buf = reinterpret_cast<u1*>(malloc(size));
    u1* classdata_out = buf;
    if (!StripClass(buf, data, size)) {
      free(classdata_out);
      return;
    }
    u1* q = builder->NewFile(filename, 0);
    size_t out_length = buf - classdata_out;
    memcpy(q, classdata_out, out_length);
    builder->FinishFile(out_length, false, true);
    free(classdata_out);
  }
}

// Copies the string into the buffer without the null terminator, returns length
static size_t WriteStr(u1 *buf, const char *str) {
  size_t len = strlen(str);
  memcpy(buf, str, len);
  return len;
}

// Computes the size of zip file content for the manifest created by
// WriteManifest, including zip file format overhead.
static size_t EstimateManifestOutputSize(const char *target_label,
                                         const char *injecting_rule_kind) {
  if (target_label == NULL) {
    return 0;
  }
  // local headers
  size_t length = 30 * 2 + MANIFEST_DIR_PATH_LENGTH + MANIFEST_PATH_LENGTH;
  // central directory
  length += 46 * 2 + MANIFEST_DIR_PATH_LENGTH + MANIFEST_PATH_LENGTH;
  // zip64 EOCD entries
  length += 56 * 2;

  // manifest content
  length += MANIFEST_HEADER_LENGTH;
  // target label manifest entry, including newline
  length += TARGET_LABEL_KEY_LENGTH + strlen(target_label) + 2;
  if (injecting_rule_kind) {
    // injecting rule kind manifest entry, including newline
    length += INJECTING_RULE_KIND_KEY_LENGTH + strlen(injecting_rule_kind) + 2;
  }
  return length;
}

static void WriteManifest(ZipBuilder *out, const char *target_label,
                          const char *injecting_rule_kind) {
  if (target_label == NULL) {
    return;
  }
  out->WriteEmptyFile(MANIFEST_DIR_PATH);
  u1 *start = out->NewFile(MANIFEST_PATH, 0);
  u1 *buf = start;
  buf += WriteStr(buf, MANIFEST_HEADER);
  buf += WriteStr(buf, TARGET_LABEL_KEY);
  buf += WriteStr(buf, target_label);
  *buf++ = '\r';
  *buf++ = '\n';
  if (injecting_rule_kind) {
    buf += WriteStr(buf, INJECTING_RULE_KIND_KEY);
    buf += WriteStr(buf, injecting_rule_kind);
    *buf++ = '\r';
    *buf++ = '\n';
  }
  size_t total_len = buf - start;
  out->FinishFile(total_len);
}

// Opens "file_in" (a .jar file) for reading, and writes an interface
// .jar to "file_out".
static void OpenFilesAndProcessJar(const char *file_out, const char *file_in,
                                   const char *target_label,
                                   const char *injecting_rule_kind) {
  JarStripperProcessor processor;
  std::unique_ptr<ZipExtractor> in(ZipExtractor::Create(file_in, &processor));
  if (in.get() == NULL) {
    fprintf(stderr, "Unable to open Zip file %s: %s\n", file_in,
            strerror(errno));
    abort();
  }
  u8 output_length =
      in->CalculateOutputLength() +
      EstimateManifestOutputSize(target_label, injecting_rule_kind);
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
  WriteManifest(out.get(), target_label, injecting_rule_kind);

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
  fprintf(stderr,
          "Usage: ijar "
          "[-v] [--target label label] [--injecting_rule_kind kind] "
          "x.jar [x_interface.jar>]\n");
  fprintf(stderr, "Creates an interface jar from the specified jar file.\n");
  exit(1);
}

int main(int argc, char **argv) {
  const char *target_label = NULL;
  const char *injecting_rule_kind = NULL;
  const char *filename_in = NULL;
  const char *filename_out = NULL;

  for (int ii = 1; ii < argc; ++ii) {
    if (strcmp(argv[ii], "-v") == 0) {
      devtools_ijar::verbose = true;
    } else if (strcmp(argv[ii], "--target_label") == 0) {
      if (++ii >= argc) {
        usage();
      }
      target_label = argv[ii];
    } else if (strcmp(argv[ii], "--injecting_rule_kind") == 0) {
      if (++ii >= argc) {
        usage();
      }
      injecting_rule_kind = argv[ii];
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

  devtools_ijar::OpenFilesAndProcessJar(filename_out, filename_in, target_label,
                                        injecting_rule_kind);
  return 0;
}
