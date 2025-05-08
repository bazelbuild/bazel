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

#include "src/tools/singlejar/desugar_checking.h"
#include "src/tools/singlejar/diag.h"
#include "src/main/protobuf/desugar_deps.pb.h"

bool Java8DesugarDepsChecker::Merge(const CDH *cdh, const LH *lh) {
  // Throw away anything previously read, no need to concatenate
  buffer_.reset(new TransientBytes());
  if (Z_NO_COMPRESSION == lh->compression_method()) {
    buffer_->ReadEntryContents(cdh, lh);
  } else if (Z_DEFLATED == lh->compression_method()) {
    if (!inflater_) {
      inflater_.reset(new Inflater());
    }
    buffer_->DecompressEntryContents(cdh, lh, inflater_.get());
  } else {
    diag_errx(2, "META-INF/desugar_deps is neither stored nor deflated");
  }

  // TODO(kmb): Wrap buffer_ as ZeroCopyInputStream to avoid copying out.
  // Note we only copy one file at a time, so overhead should be modest.
  uint32_t checksum;
  const size_t data_size = buffer_->data_size();
  uint8_t *buf = reinterpret_cast<uint8_t *>(malloc(data_size));
  buffer_->CopyOut(reinterpret_cast<uint8_t *>(buf), &checksum);
  buffer_.reset();  // release buffer eagerly

  bazel::tools::desugar::DesugarDepsInfo deps_info;
  google::protobuf::io::CodedInputStream content(buf, data_size);
  if (!deps_info.ParseFromCodedStream(&content)) {
    diag_errx(2, "META-INF/desugar_deps: unable to parse");
  }
  if (!content.ConsumedEntireMessage()) {
    diag_errx(2, "META-INF/desugar_deps: unexpected trailing content");
  }
  free(buf);

  for (const auto &assume_present : deps_info.assume_present()) {
    // This means we need file named <target>.class in the output.  Remember
    // the first origin of this requirement for error messages, drop others.
    needed_deps_.emplace(assume_present.target().binary_name() + ".class",
                         assume_present.origin().binary_name());
  }

  for (const auto &missing : deps_info.missing_interface()) {
    // Remember the first origin of this requirement for error messages, drop
    // subsequent ones.
    missing_interfaces_.emplace(missing.target().binary_name(),
                                missing.origin().binary_name());
  }

  for (const auto &extends : deps_info.interface_with_supertypes()) {
    // Remember interface hierarchy the first time we see this interface, drop
    // subsequent ones for consistency with how singlejar will keep the first
    // occurrence of the file defining the interface.  We'll lazily derive
    // whether missing_interfaces_ inherit default methods with this data later.
    if (extends.extended_interface_size() > 0) {
      std::vector<std::string> extended;
      extended.reserve(extends.extended_interface_size());
      for (const auto &itf : extends.extended_interface()) {
        extended.push_back(itf.binary_name());
      }
      extended_interfaces_.emplace(extends.origin().binary_name(),
                                   std::move(extended));
    }
  }

  for (const auto &companion : deps_info.interface_with_companion()) {
    // Only remember interfaces that definitely have default methods for now.
    // For all other interfaces we'll transitively check extended interfaces
    // in HasDefaultMethods.
    if (companion.num_default_methods() > 0) {
      has_default_methods_[companion.origin().binary_name()] = true;
    }
  }
  return true;
}

void *Java8DesugarDepsChecker::OutputEntry(bool compress) {
  if (verbose_) {
    fprintf(stderr, "Needed deps: %zu\n", needed_deps_.size());
    fprintf(stderr, "Interfaces to check: %zu\n", missing_interfaces_.size());
    fprintf(stderr, "Sub-interfaces: %zu\n", extended_interfaces_.size());
    fprintf(stderr, "Interfaces w/ default methods: %zu\n",
            has_default_methods_.size());
  }
  for (const auto &needed : needed_deps_) {
    if (verbose_) {
      fprintf(stderr, "Looking for %s\n", needed.first.c_str());
    }
    if (!known_member_(needed.first)) {
      if (fail_on_error_) {
        diag_errx(2,
                  "%s referenced by %s but not found.  Is the former defined"
                  " in a neverlink library?",
                  needed.first.c_str(), needed.second.c_str());
      } else {
        error_ = true;
      }
    }
  }

  for (const auto &missing : missing_interfaces_) {
    if (verbose_) {
      fprintf(stderr, "Checking %s\n", missing.first.c_str());
    }
    if (HasDefaultMethods(missing.first)) {
      if (fail_on_error_) {
        diag_errx(
            2,
            "%s needed on the classpath for desugaring %s.  Please add"
            " the missing dependency to the target containing the latter.",
            missing.first.c_str(), missing.second.c_str());
      } else {
        error_ = true;
      }
    }
  }

  // We don't want these files in the output, just check them for consistency
  return nullptr;
}

bool Java8DesugarDepsChecker::HasDefaultMethods(
    const std::string &interface_name) {
  auto cached = has_default_methods_.find(interface_name);
  if (cached != has_default_methods_.end()) {
    return cached->second;
  }

  // Prime with false in case there's a cycle.  We'll update with the true value
  // (ignoring the cycle) below.
  has_default_methods_.emplace(interface_name, false);

  for (const std::string &extended : extended_interfaces_[interface_name]) {
    if (HasDefaultMethods(extended)) {
      has_default_methods_[interface_name] = true;
      return true;
    }
  }
  has_default_methods_[interface_name] = false;
  return false;
}
