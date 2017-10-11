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

#ifndef SRC_TOOLS_SINGLEJAR_COMBINERS_H_
#define SRC_TOOLS_SINGLEJAR_COMBINERS_H_ 1

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "src/tools/singlejar/transient_bytes.h"
#include "src/tools/singlejar/zip_headers.h"

// An interface for combining the files.
class Combiner {
 public:
  virtual ~Combiner();
  // Merges the contents of the given Zip entry to this instance.
  virtual bool Merge(const CDH *cdh, const LH *lh) = 0;
  // Returns a point to the buffer containing Local Header followed by the
  // payload. The caller is responsible of freeing the buffer. If `compress'
  // is not set, the payload is a copy of the bytes held by this combiner.
  // Otherwise the payload is compressed, provided that the compressed data
  // is smaller than the original.
  virtual void *OutputEntry(bool compress) = 0;
};

// An output jar entry consisting of a concatenation of the input jar
// entries. Byte sequences can be appended to it, too.
class Concatenator : public Combiner {
 public:
  Concatenator(const std::string &filename, bool insert_newlines = true)
      : filename_(filename), insert_newlines_(insert_newlines) {}

  ~Concatenator() override;

  bool Merge(const CDH *cdh, const LH *lh) override;

  void *OutputEntry(bool compress) override;

  void Append(const char *s, size_t n) {
    CreateBuffer();
    buffer_->Append(reinterpret_cast<const uint8_t *>(s), n);
  }

  void Append(const char *s) { Append(s, strlen(s)); }

  void Append(const std::string &str) { Append(str.c_str(), str.size()); }

  const std::string &filename() const { return filename_; }

 private:
  void CreateBuffer() {
    if (!buffer_.get()) {
      buffer_.reset(new TransientBytes());
    }
  }
  const std::string filename_;
  std::unique_ptr<TransientBytes> buffer_;
  std::unique_ptr<Inflater> inflater_;
  bool insert_newlines_;
};

// The combiner that does nothing. Useful to represent for instance directory
// entries: once a directory entry has been created and added to the output
// jar, the subsequent entries are ignored on input, and nothing is output.
class NullCombiner : public Combiner {
 public:
  ~NullCombiner() override;
  bool Merge(const CDH *cdh, const LH *lh) override;
  void *OutputEntry(bool compress) override;
};

// Combines the contents of the multiple input entries which are XML
// files into a single XML output entry with given top level XML tag.
class XmlCombiner : public Combiner {
 public:
  XmlCombiner(const std::string &filename, const std::string &xml_tag)
      : filename_(filename),
        start_tag_("<" + xml_tag + ">"),
        end_tag_("</" + xml_tag + ">") {}
  ~XmlCombiner() override;

  bool Merge(const CDH *cdh, const LH *lh) override;

  void *OutputEntry(bool compress) override;

  const std::string filename() const { return filename_; }

 private:
  const std::string filename_;
  const std::string start_tag_;
  const std::string end_tag_;
  std::unique_ptr<Concatenator> concatenator_;
  std::unique_ptr<Inflater> inflater_;
};

// A wrapper around Concatenator allowing to append
//   NAME=VALUE
// lines to the contents.
// NOTE that it does not allow merging existing entries.
class PropertyCombiner : public Concatenator {
 public:
  PropertyCombiner(const std::string &filename) : Concatenator(filename) {}
  ~PropertyCombiner();

  bool Merge(const CDH *cdh, const LH *lh) override;

  void AddProperty(const char *key, const char *value) {
    // TODO(asmundak): deduplicate properties.
    Append(key);
    Append("=", 1);
    Append(value);
    Append("\n", 1);
  }

  void AddProperty(const std::string &key, const std::string &value) {
    // TODO(asmundak): deduplicate properties.
    Append(key);
    Append("=", 1);
    Append(value);
    Append("\n", 1);
  }
};

// Combiner that checks META-INF/desugar_deps files (b/65645388) to ensure
// correct bytecode desugaring, specifically of default and static interface
// methods, across an entire binary.  Two checks are performed:
// 1. Make sure that any dependency assumed by the desugaring process is in
//    fact part of the binary.  This protects against ill-advised uses of
//    neverlink, where a library is only on the compile-time classpath but not
//    the runtime classpath.
// 2. To paper over incomplete classpaths during desugaring (b/65211436), check
//    that interfaces that couldn't be found don't declare or inherit default
//    methods.  Desugar emits extra metadata to avoid us having to open up and
//    parse .class files for this purpose.
class Java8DesugarDepsChecker : public Combiner {
 public:
  Java8DesugarDepsChecker(std::function<bool(const std::string &)> known_member,
                          bool verbose)
      : Java8DesugarDepsChecker(std::move(known_member), verbose, true) {}
  ~Java8DesugarDepsChecker() override {}

  bool Merge(const CDH *cdh, const LH *lh) override;

  void *OutputEntry(bool compress) override;

 private:
  Java8DesugarDepsChecker(std::function<bool(const std::string &)> known_member,
                          bool verbose, bool fail_on_error)
      : known_member_(std::move(known_member)),
        verbose_(verbose),
        fail_on_error_(fail_on_error),
        error_(false) {}
  /// Computes and caches whether the given interface has default methods.
  /// \param interface_name interface name as it would appear in bytecode, e.g.,
  ///        "java/lang/Runnable"
  bool HasDefaultMethods(const std::string &interface_name);

  const std::function<bool(const std::string &)> known_member_;
  const bool verbose_;
  const bool fail_on_error_;  // For testing

  std::unique_ptr<TransientBytes> buffer_;
  std::unique_ptr<Inflater> inflater_;
  /// Reverse mapping from needed dependencies to one of the users.
  std::map<std::string, std::string> needed_deps_;
  /// Reverse mapping from missing interfaces to one of the classes that missed
  /// them.
  std::map<std::string, std::string> missing_interfaces_;
  std::unordered_map<std::string, std::vector<std::string> >
      extended_interfaces_;
  /// Cache of interfaces known to definitely define or inherit default methods
  /// or definitely not define and not inherit default methods.  Merge()
  /// populates initial entries and HasDefaultMethods() adds to the cache as
  /// needed.
  std::unordered_map<std::string, bool> has_default_methods_;
  bool error_;

  friend class CombinersTest;
};

#endif  //  SRC_TOOLS_SINGLEJAR_COMBINERS_H_
