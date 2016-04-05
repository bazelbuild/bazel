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
// Zip / Unzip file using ijar zip implementation.
//
// Note that this Zip implementation intentionally don't compute CRC-32
// because it is useless computation for jar because Java doesn't care.
// CRC-32 of all files in the zip file will be set to 0.
//

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <memory>

#include "third_party/ijar/zip.h"

namespace devtools_ijar {

#define SYSCALL(expr)  do { \
                         if ((expr) < 0) { \
                           perror(#expr); \
                           abort(); \
                         } \
                       } while (0)

//
// A ZipExtractorProcessor that extract all files in the ZIP file.
//
class UnzipProcessor : public ZipExtractorProcessor {
 public:
  // Create a processor who will extract the files into output_root
  // if "extract" is set to true and will print the list of files and
  // their unix modes if "verbose" is set to true.
  UnzipProcessor(const char *output_root, bool verbose, bool extract)
    : output_root_(output_root), verbose_(verbose), extract_(extract) {}
  virtual ~UnzipProcessor() {}

  virtual void Process(const char* filename, const u4 attr,
                       const u1* data, const size_t size);
  virtual bool Accept(const char* filename, const u4 attr) {
    return true;
  }

 private:
  const char *output_root_;
  const bool verbose_;
  const bool extract_;
};

// Concatene 2 path, path1 and path2, using / as a directory separator and
// puting the result in "out". "size" specify the size of the output buffer
void concat_path(char* out, const size_t size,
                 const char *path1, const char *path2) {
  int len1 = strlen(path1);
  size_t l = len1;
  strncpy(out, path1, size - 1);
  out[size-1] = 0;
  if (l < size - 1 && path1[len1] != '/' && path2[0] != '/') {
    out[l] = '/';
    l++;
    out[l] = 0;
  }
  if (l < size - 1) {
    strncat(out, path2, size - 1 - l);
  }
}

// Do a recursive mkdir of all folders of path except the last path
// segment (if path ends with a / then the last path segment is empty).
// All folders are created using "mode" for creation mode.
void mkdirs(const char *path, mode_t mode) {
  char path_[PATH_MAX];
  struct stat statst;
  strncpy(path_, path, PATH_MAX);
  path_[PATH_MAX-1] = 0;
  char *pointer = path_;
  while ((pointer = strchr(pointer, '/')) != NULL) {
    if (path_ != pointer) {  // skip leading slash
      *pointer = 0;
      if (stat(path_, &statst) != 0) {
        if (mkdir(path_, mode) < 0) {
          fprintf(stderr, "Cannot create folder %s: %s\n",
                  path_, strerror(errno));
          abort();
        }
      }
      *pointer = '/';
    }
    pointer++;
  }
}

void UnzipProcessor::Process(const char* filename, const u4 attr,
                             const u1* data, const size_t size) {
  mode_t mode = zipattr_to_mode(attr);
  mode_t perm = mode & 0777;
  bool isdir = (mode & S_IFDIR) != 0;
  if (attr == 0) {
    // Fallback when the external attribute is not set.
    isdir = filename[strlen(filename)-1] == '/';
    perm = 0777;
  }
  if (verbose_) {
    printf("%c %o %s\n", isdir ? 'd' : 'f', perm, filename);
  }
  if (extract_) {
    char path[PATH_MAX];
    int fd;
    concat_path(path, PATH_MAX, output_root_, filename);
    mkdirs(path, perm);
    if (!isdir) {
      fd = open(path, O_CREAT | O_WRONLY, perm);
      if (fd < 0) {
        fprintf(stderr, "Cannot open file %s for writing: %s\n",
                path, strerror(errno));
        abort();
      }
      SYSCALL(write(fd, data, size));
      SYSCALL(close(fd));
    }
  }
}

// Get the basename of path and store it in output. output_size
// is the size of the output buffer.
void basename(const char *path, char *output, size_t output_size) {
  const char *pointer = strrchr(path, '/');
  if (pointer == NULL) {
    pointer = path;
  } else {
    pointer++;  // Skip the leading slash.
  }
  strncpy(output, pointer, output_size);
  output[output_size-1] = 0;
}

// copy size bytes from file descriptor fd into buffer.
int copy_file_to_buffer(int fd, size_t size, void *buffer) {
  size_t nb_read = 0;
  while (nb_read < size) {
    size_t to_read = size - nb_read;
    if (to_read > 16384 /* 16K */) {
      to_read = 16384;
    }
    ssize_t r = read(fd, static_cast<uint8_t *>(buffer) + nb_read, to_read);
    if (r < 0) {
      return -1;
    }
    nb_read += r;
  }
  return 0;
}

// Execute the extraction (or just listing if just v is provided)
int extract(char *zipfile, bool verbose, bool extract) {
  char output_root[PATH_MAX];
  if (getcwd(output_root, PATH_MAX) == NULL) {
    fprintf(stderr, "getcwd() failed: %s.\n", strerror(errno));
    return -1;
  }

  UnzipProcessor processor(output_root, verbose, extract);
  std::unique_ptr<ZipExtractor> extractor(ZipExtractor::Create(zipfile,
                                                               &processor));
  if (extractor.get() == NULL) {
    fprintf(stderr, "Unable to open zip file %s: %s.\n", zipfile,
            strerror(errno));
    return -1;
  }

  if (extractor->ProcessAll() < 0) {
    fprintf(stderr, "%s.\n", extractor->GetError());
    return -1;
  }
  return 0;
}

// add a file to the zip
int add_file(std::unique_ptr<ZipBuilder> const &builder, char *file,
             bool flatten, bool verbose, bool compress) {
  struct stat statst;
  if (stat(file, &statst) < 0) {
    fprintf(stderr, "Cannot stat file %s: %s.\n", file, strerror(errno));
    return -1;
  }
  bool isdir = (statst.st_mode & S_IFDIR) != 0;

  if (flatten && isdir) {
    return 0;
  }

  // Compute the path, flattening it if requested
  char path[PATH_MAX];
  size_t len = strlen(file);
  if (len > PATH_MAX) {
    fprintf(stderr, "Path too long: %s.\n", file);
    return -1;
  }
  if (flatten) {
    basename(file, path, PATH_MAX);
  } else {
    strncpy(path, file, PATH_MAX);
    path[PATH_MAX - 1] = 0;
    if (isdir && len < PATH_MAX - 1) {
      // Add the trailing slash for folders
      path[len] = '/';
      path[len + 1] = 0;
    }
  }

  if (verbose) {
    mode_t perm = statst.st_mode & 0777;
    printf("%c %o %s\n", isdir ? 'd' : 'f', perm, path);
  }

  u1 *buffer = builder->NewFile(path, mode_to_zipattr(statst.st_mode));
  if (isdir || statst.st_size == 0) {
    builder->FinishFile(0);
  } else {
    // read the input file
    int fd = open(file, O_RDONLY);
    if (fd < 0) {
      fprintf(stderr, "Can't open file %s for reading: %s.\n", file,
              strerror(errno));
      return -1;
    }
    if (copy_file_to_buffer(fd, statst.st_size, buffer) < 0) {
      fprintf(stderr, "Can't read file %s: %s.\n", file, strerror(errno));
      close(fd);
      return -1;
    }
    close(fd);
    builder->FinishFile(statst.st_size, compress, true);
  }
  return 0;
}

// Read a list of files separated by newlines. The resulting array can be
// freed using the free method.
char **read_filelist(char *filename) {
  struct stat statst;
  int fd = open(filename, O_RDONLY);
  if (fd < 0) {
    fprintf(stderr, "Can't open file %s for reading: %s.\n", filename,
            strerror(errno));
    return NULL;
  }
  if (fstat(fd, &statst) < 0) {
    fprintf(stderr, "Cannot stat file %s: %s.\n", filename, strerror(errno));
    return NULL;
  }

  char *data = static_cast<char *>(malloc(statst.st_size));
  if (copy_file_to_buffer(fd, statst.st_size, data) < 0) {
    fprintf(stderr, "Can't read file %s: %s.\n", filename, strerror(errno));
    close(fd);
    return NULL;
  }
  close(fd);

  int nb_entries = 1;
  for (int i = 0; i < statst.st_size; i++) {
    if (data[i] == '\n') {
      nb_entries++;
    }
  }

  size_t sizeof_array = sizeof(char *) * (nb_entries + 1);
  void *result = malloc(sizeof_array + statst.st_size);
  // copy the content
  char **filelist = static_cast<char **>(result);
  char *content = static_cast<char *>(result) + sizeof_array;
  memcpy(content, data, statst.st_size);
  free(data);
  // Create the corresponding array
  int j = 1;
  filelist[0] = content;
  for (int i = 0; i < statst.st_size; i++) {
    if (content[i] == '\n') {
      content[i] = 0;
      if (i + 1 < statst.st_size) {
        filelist[j] = content + i + 1;
        j++;
      }
    }
  }
  filelist[j] = NULL;
  return filelist;
}

// Execute the create operation
int create(char *zipfile, char **files, bool flatten, bool verbose,
           bool compress) {
  u8 size = ZipBuilder::EstimateSize(files);
  if (size == 0) {
    return -1;
  }
  std::unique_ptr<ZipBuilder> builder(ZipBuilder::Create(zipfile, size));
  if (builder.get() == NULL) {
    fprintf(stderr, "Unable to create zip file %s: %s.\n",
            zipfile, strerror(errno));
    return -1;
  }
  for (int i = 0; files[i] != NULL; i++) {
    if (add_file(builder, files[i], flatten, verbose, compress) < 0) {
      return -1;
    }
  }
  if (builder->Finish() < 0) {
    fprintf(stderr, "%s\n", builder->GetError());
    return -1;
  }
  return 0;
}

}  // namespace devtools_ijar

//
// main method
//
static void usage(char *progname) {
  fprintf(stderr, "Usage: %s [vxc[fC]] x.zip [file1...filen]\n", progname);
  fprintf(stderr, "  v verbose - list all file in x.zip\n");
  fprintf(stderr, "  x extract - extract file in x.zip in current directory\n");
  fprintf(stderr, "  c create  - add files to x.zip\n");
  fprintf(stderr, "  f flatten - flatten files to use with create operation\n");
  fprintf(stderr,
          "  C compress - compress files when using the create operation\n");
  fprintf(stderr, "x and c cannot be used in the same command-line.\n");
  exit(1);
}

int main(int argc, char **argv) {
  bool extract = false;
  bool verbose = false;
  bool create = false;
  bool compress = false;
  bool flatten = false;

  if (argc < 3) {
    usage(argv[0]);
  }

  for (int i = 0; argv[1][i] != 0; i++) {
    switch (argv[1][i]) {
    case 'x':
      extract = true;
      break;
    case 'v':
      verbose = true;
      break;
    case 'c':
      create = true;
      break;
    case 'f':
      flatten = true;
      break;
    case 'C':
      compress = true;
      break;
    default:
      usage(argv[0]);
    }
  }
  if (create) {
    if (extract) {
      usage(argv[0]);
    }
    // Create a zip
    char **filelist = argv + 3;
    if (argc == 4 && argv[3][0] == '@') {
      // We never free that list because it needs to be allocated during the
      // whole execution, the system will reclaim memory.
      filelist = devtools_ijar::read_filelist(argv[3] + 1);
      if (filelist == NULL) {
        return -1;
      }
    }
    return devtools_ijar::create(argv[2], filelist, flatten, verbose, compress);
  } else {
    if (flatten) {
      usage(argv[0]);
    }
    // Extraction / list mode
    return devtools_ijar::extract(argv[2], verbose, extract);
  }
}
