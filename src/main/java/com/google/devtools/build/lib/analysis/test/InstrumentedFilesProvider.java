// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.test;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.util.Pair;

/** A provider of instrumented file sources and instrumentation metadata. */
public interface InstrumentedFilesProvider extends TransitiveInfoProvider {
  /**
   * The transitive closure of instrumented source files.
   */
  NestedSet<Artifact> getInstrumentedFiles();

  /**
   * Returns a collection of instrumentation metadata files.
   */
  NestedSet<Artifact> getInstrumentationMetadataFiles();

  /**
   * The transitive closure of instrumented source files for which baseline coverage should be
   * generated. In general, this is a subset of the instrumented source files: it only contains
   * instrumented source files from rules that support baseline coverage.
   */
  NestedSet<Artifact> getBaselineCoverageInstrumentedFiles();

  /**
   * The output artifact of the baseline coverage action; this is only ever a single artifact, which
   * contains baseline coverage for the entire transitive closure of source files.
   */
  // TODO(ulfjack): Change this to a single Artifact. Also change how it's generated. It's better to
  // generate actions such that each action only covers the source files of a single rule, in
  // particular because baseline coverage is language-specific (it requires a parser for the
  // specific language), and we don't want to depend on all language parsers from any single rule.
  NestedSet<Artifact> getBaselineCoverageArtifacts();

  /**
   * Extra files that are needed on the inputs of test actions for coverage collection to happen,
   * for example, {@code gcov}.
   *
   * <p>They aren't mentioned in the instrumented files manifest.
   */
  NestedSet<Artifact> getCoverageSupportFiles();

  /**
   * Environment variables that need to be set for tests collecting code coverage.
   */
  NestedSet<Pair<String, String>> getCoverageEnvironment();

  /**
   * A map from the reported source file path to the actual source file path, relative to the
   * workspace directory, if the two values are different. If the reported source file is the same
   * as the actual source path it will not be included in this map.
   *
   * <p> This is useful for virtual include paths in C++, which get reported at the include location
   * and not the real source path. For example, the reported include source file can be
   * "bazel-out/k8-fastbuild/bin/include/common/_virtual_includes/strategy/strategy.h", but its
   * actual source path is "include/common/strategy.h".
   */
  ImmutableMap<String, String> getReportedToActualSources();
}
