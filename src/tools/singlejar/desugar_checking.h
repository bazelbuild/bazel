// Copyright 2017 The Bazel Authors. All rights reserved.
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

#ifndef SRC_TOOLS_SINGLEJAR_DESUGAR_CHECKING_H_
#define SRC_TOOLS_SINGLEJAR_DESUGAR_CHECKING_H_ 1

#include <functional>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "src/tools/singlejar/combiners.h"
#include "src/tools/singlejar/transient_bytes.h"
#include "src/tools/singlejar/zip_headers.h"

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
  Java8DesugarDepsChecker(std::function<bool (const std::string&)> known_member,
                          bool verbose)
      : Java8DesugarDepsChecker(std::move(known_member), verbose, true) {}
  ~Java8DesugarDepsChecker() override {}

  bool Merge(const CDH *cdh, const LH *lh) override;

  void *OutputEntry(bool compress) override;

 private:
  Java8DesugarDepsChecker(std::function<bool (const std::string&)> known_member,
                          bool verbose, bool fail_on_error)
      : known_member_(std::move(known_member)),
        verbose_(verbose),
        fail_on_error_(fail_on_error),
        error_(false) {}
  /// Computes and caches whether the given interface has default methods.
  /// \param interface_name interface name as it would appear in bytecode, e.g.,
  ///        "java/lang/Runnable"
  bool HasDefaultMethods(const std::string &interface_name);

  const std::function<bool (const std::string&)> known_member_;
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

  friend class Java8DesugarDepsCheckerTest;
};

#endif  // SRC_TOOLS_SINGLEJAR_DESUGAR_CHECKING_H_
