// Copyright 2016 The Bazel Authors. All rights reserved.
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

#ifndef SRC_TOOLS_SINGLEJAR_COMBINED_JAR_H_
#define SRC_TOOLS_SINGLEJAR_COMBINED_JAR_H_

#include <stdio.h>

#include <cinttypes>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// Must be included before <io.h> (on Windows) and <fcntl.h>.
#include "src/tools/singlejar/port.h"
// Need newline so clang-format won't alpha-sort with other headers.

#include "src/tools/singlejar/combiners.h"
#include "src/tools/singlejar/options.h"

/*
 * Jar file we are writing.
 */
class OutputJar {
 public:
  // Constructor.
  OutputJar();
  // Do all that needs to be done. Can be called only once.
  int Doit(Options *options);
  // Destructor.
  virtual ~OutputJar();
  // Add a combiner to handle the entries with given name. OutputJar will
  // own the instance of the combiner and will delete it on self destruction.
  void ExtraCombiner(const std::string& entry_name, Combiner *combiner);
  // Additional file handler to be redefined by a subclass.
  virtual void ExtraHandler(const std::string &input_jar_path, const CDH *entry,
                            const std::string *input_jar_aux_label);
  // Return jar path.
  const char *path() const { return options_->output_jar.c_str(); }
  // True if an entry with given name have not been added to this archive.
  bool NewEntry(const std::string& entry_name) {
    return known_members_.count(entry_name) == 0;
  }

 protected:
  // The purpose  of these two tiny utility methods is to avoid creating a
  // std::string instance (which always involves allocating an object on the
  // heap) when we just need to check that a sequence of bytes in memory has
  // given prefix or suffix.
  static bool begins_with(const char *str, size_t n, const char *head) {
    const size_t n_head = strlen(head);
    return n >= n_head && !strncmp(str, head, n_head);
  }
  static bool ends_with(const char *str, size_t n, const char *tail) {
    const size_t n_tail = strlen(tail);
    return n >= n_tail && !strncmp(str + n - n_tail, tail, n_tail);
  }

 private:
  // Open output jar.
  bool Open();
  // Add the contents of the given input jar.
  bool AddJar(int jar_path_index);
  // Returns the current output position.
  off64_t Position();
  // Write Jar entry.
  void WriteEntry(void *local_header_and_payload);
  // Write META_INF/ entry (the first entry on output).
  void WriteMetaInf();
  // Write a directory entry.
  void WriteDirEntry(const std::string &name, const uint8_t *extra_fields,
                     const uint16_t n_extra_fields);
  // Create output Central Directory Header for the given input entry and
  // append it to CEN (Central Directory) buffer.
  void AppendToDirectoryBuffer(const CDH *cdh, off64_t lh_pos,
                               uint16_t normalized_time, bool fix_timestamp);
  // Reserve space in CEN buffer.
  uint8_t *ReserveCdr(size_t chunk_size);
  // Reserve space for the Central Directory Header in CEN buffer.
  uint8_t *ReserveCdh(size_t size);
  // Close output.
  bool Close();
  // Set classpath resource with given resource name and path.
  void ClasspathResource(const std::string& resource_name,
                         const std::string& resource_path);
  // Append file starting at page boundary.
  off64_t PageAlignedAppendFile(const std::string &file_path,
                                size_t *file_size);
  void AppendPageAlignedFile(const std::string &file,
                             const std::string &offset_manifest_attr_name,
                             const std::string &size_manifest_attr_name,
                             const std::string &property_name);
  // Append data from the file specified by file_path.
  size_t AppendFile(Options *options, const char *file_path);
  // Copy 'count' bytes starting at 'offset' from the given file.
  ssize_t CopyAppendData(int in_fd, off64_t offset, size_t count);
  // Write bytes to the output file, return true on success.
  bool WriteBytes(const void *buffer, size_t count);

  Options *options_;
  struct EntryInfo {
    EntryInfo(Combiner *combiner, int index = -1)
        : combiner_(combiner), input_jar_index_(index) {}
    Combiner *combiner_;
    int input_jar_index_;  // Input jar index for the plain entry or -1.
  };

  std::unordered_map<std::string, struct EntryInfo> known_members_;
  FILE *file_;
  off64_t outpos_;
  std::unique_ptr<char[]> buffer_;
  int entries_;
  int duplicate_entries_;
  uint8_t *cen_;
  size_t cen_size_;
  size_t cen_capacity_;
  Concatenator spring_handlers_;
  Concatenator spring_schemas_;
  Concatenator protobuf_meta_handler_;
  ManifestCombiner manifest_;
  PropertyCombiner build_properties_;
  NullCombiner null_combiner_;
  std::vector<std::unique_ptr<Concatenator> > service_handlers_;
  std::vector<std::unique_ptr<Concatenator> > classpath_resources_;
  std::vector<std::unique_ptr<Combiner> > extra_combiners_;
};

#endif  //   SRC_TOOLS_SINGLEJAR_COMBINED_JAR_H_
