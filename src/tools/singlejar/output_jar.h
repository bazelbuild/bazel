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

#include <stdint.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

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
  ~OutputJar();
  // Add a combiner to handle the entries with given name. OutputJar will
  // own the instance of the combiner and will delete it on self destruction.
  void ExtraCombiner(const std::string& entry_name, Combiner *combiner);
  // Return jar path.
  const char *path() const { return options_->output_jar.c_str(); }

 private:
  // Open output jar.
  bool Open();
  // Add the contents of the given input jar.
  bool AddJar(int jar_path_index);
  // Returns the current output position.
  off_t Position();
  // Write Jar entry.
  void WriteEntry(void *local_header_and_payload);
  // Write a directory entry.
  void AddDirectory(const char *path);
  // Append given Central Directory Header to CEN (Central Directory) buffer.
  CDH *AppendToDirectoryBuffer(const CDH *cdh);
  // Reserve space in CEN buffer.
  uint8_t *ReserveCdr(size_t chunk_size);
  // Reserve space for the Central Directory Header in CEN buffer.
  uint8_t *ReserveCdh(size_t size);
  // Close output.
  bool Close();
  // Set classpath resource with given resource name and path.
  void ClasspathResource(const std::string& resource_name,
                         const std::string& resource_path);
  // Copy the bytes from the given file.
  ssize_t AppendFile(int in_fd, off_t *in_offset, size_t count);
  // Write bytes to the output file, return true on success.
  bool WriteBytes(uint8_t *buffer, size_t count);

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

  Options *options_;
  struct EntryInfo {
    EntryInfo(Combiner *combiner, int index = -1)
        : combiner_(combiner), input_jar_index_(index) {}
    Combiner *combiner_;
    int input_jar_index_;  // Input jar index for the plain entry or -1.
  };

  std::unordered_map<std::string, struct EntryInfo> known_members_;
  int fd_;
  int entries_;
  int duplicate_entries_;
  uint8_t *cen_;
  size_t cen_size_;
  size_t cen_capacity_;
  Concatenator spring_handlers_;
  Concatenator spring_schemas_;
  Concatenator protobuf_meta_handler_;
  Concatenator manifest_;
  PropertyCombiner build_properties_;
  NullCombiner null_combiner_;
  std::vector<std::unique_ptr<Concatenator> > service_handlers_;
  std::vector<std::unique_ptr<Concatenator> > classpath_resources_;
  std::vector<std::unique_ptr<Combiner> > extra_combiners_;
};

#endif  //   SRC_TOOLS_SINGLEJAR_COMBINED_JAR_H_
