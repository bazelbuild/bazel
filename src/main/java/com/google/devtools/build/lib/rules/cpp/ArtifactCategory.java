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

/**
 * A category of artifacts that are candidate input/output to an action, for which the toolchain can
 * select a single artifact.
 */
public enum ArtifactCategory {
  STATIC_LIBRARY("lib%{base_name}.a"),
  ALWAYSLINK_STATIC_LIBRARY("lib%{base_name}.lo"),
  DYNAMIC_LIBRARY("lib%{base_name}.so"),
  EXECUTABLE("%{base_name}"),
  INTERFACE_LIBRARY("lib%{base_name}.ifso"),
  PIC_FILE("%{output_name}.pic"),
  INCLUDED_FILE_LIST("%{output_name}.d"),
  OBJECT_FILE("%{output_name}.o"),
  PIC_OBJECT_FILE("%{output_name}.pic.o"),
  CPP_MODULE("%{output_name}.pcm"),
  GENERATED_ASSEMBLY("%{output_name}.s"),
  PROCESSED_HEADER("%{output_name}.processed"),
  GENERATED_HEADER("%{output_name}.h"),
  PREPROCESSED_C_SOURCE("%{output_name}.i"),
  PREPROCESSED_CPP_SOURCE("%{output_name}.ii"),
  COVERAGE_DATA_FILE("%{output_name}.gcno"),
  // A matched-clif protobuf. Typically in binary format, but could be text depending on
  // the options passed to the clif_matcher.
  CLIF_OUTPUT_PROTO("%{output_name}.opb");

  private final String defaultPattern;

  ArtifactCategory(String defaultPattern) {
    this.defaultPattern = defaultPattern;
  }

  /** Returns the name of the category. */
  public String getCategoryName() {
    return this.toString().toLowerCase();
  }

  public String getDefaultPattern() {
    return defaultPattern;
  }
}
