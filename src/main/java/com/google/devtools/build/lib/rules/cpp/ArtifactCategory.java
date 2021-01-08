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
// limitations under the License
package com.google.devtools.build.lib.rules.cpp;

import com.google.common.collect.ImmutableList;

/**
 * A category of artifacts that are candidate input/output to an action, for which the toolchain can
 * select a single artifact.
 */
public enum ArtifactCategory {
  STATIC_LIBRARY("lib", ".a", ".lib"),
  ALWAYSLINK_STATIC_LIBRARY("lib", ".lo", ".lo.lib"),
  DYNAMIC_LIBRARY("lib", ".so", ".dylib", ".dll"),
  EXECUTABLE("", "", ".exe"),
  INTERFACE_LIBRARY("lib", ".ifso", ".tbd", ".if.lib"),
  PIC_FILE("", ".pic"),
  INCLUDED_FILE_LIST("", ".d"),
  OBJECT_FILE("", ".o", ".obj"),
  PIC_OBJECT_FILE("", ".pic.o"),
  CPP_MODULE("", ".pcm"),
  GENERATED_ASSEMBLY("", ".s", ".asm"),
  PROCESSED_HEADER("", ".processed"),
  GENERATED_HEADER("", ".h"),
  PREPROCESSED_C_SOURCE("", ".i"),
  PREPROCESSED_CPP_SOURCE("", ".ii"),
  COVERAGE_DATA_FILE("", ".gcno"),
  // A matched-clif protobuf. Typically in binary format, but could be text depending on
  // the options passed to the clif_matcher.
  CLIF_OUTPUT_PROTO("", ".opb");

  private final String defaultPrefix;
  private final String defaultExtension;
  private final String starlarkName;
  // The extensions allowed for this artifact name pattern, Bazel should recognized them as
  // corresponding file type in CppFileTypes.java
  final ImmutableList<String> allowedExtensions;

  ArtifactCategory(
      String defaultPrefix,
      String defaultExtension,
      String... extraAllowedExtensions) {
    this.defaultPrefix = defaultPrefix;
    this.defaultExtension = defaultExtension;
    this.allowedExtensions =
        new ImmutableList.Builder<String>()
            .add(defaultExtension)
            .add(extraAllowedExtensions)
            .build();

    this.starlarkName = toString().toLowerCase();
  }

  public String getStarlarkName() {
    return starlarkName;
  }

  /** Returns the name of the category. */
  public String getCategoryName() {
    return this.toString().toLowerCase();
  }

  public String getDefaultPrefix() {
    return defaultPrefix;
  }

  public String getDefaultExtension() {
    return defaultExtension;
  }

  public ImmutableList<String> getAllowedExtensions() {
    return allowedExtensions;
  }
}
