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

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skylarkbuildapi.test.InstrumentedFilesInfoApi;
import com.google.devtools.build.lib.util.Pair;

/** An implementation class for the InstrumentedFilesProvider interface. */
@AutoCodec
public final class InstrumentedFilesInfo extends NativeInfo implements InstrumentedFilesInfoApi {
  /** Singleton provider instance for {@link InstrumentedFilesInfo}. */
  public static final InstrumentedFilesProvider STARLARK_CONSTRUCTOR =
      new InstrumentedFilesProvider();

  public static final InstrumentedFilesInfo EMPTY =
      new InstrumentedFilesInfo(
          NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
          NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
          NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
          NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
          NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
          NestedSetBuilder.<Pair<String, String>>emptySet(Order.COMPILE_ORDER),
          NestedSetBuilder.emptySet(Order.STABLE_ORDER));

  private final NestedSet<Artifact> instrumentedFiles;
  private final NestedSet<Artifact> instrumentationMetadataFiles;
  private final NestedSet<Artifact> baselineCoverageFiles;
  private final NestedSet<Artifact> baselineCoverageArtifacts;
  private final NestedSet<Artifact> coverageSupportFiles;
  private final NestedSet<Pair<String, String>> coverageEnvironment;
  private final NestedSet<Pair<String, String>> reportedToActualSources;

  public InstrumentedFilesInfo(
      NestedSet<Artifact> instrumentedFiles,
      NestedSet<Artifact> instrumentationMetadataFiles,
      NestedSet<Artifact> baselineCoverageFiles,
      NestedSet<Artifact> baselineCoverageArtifacts,
      NestedSet<Artifact> coverageSupportFiles,
      NestedSet<Pair<String, String>> coverageEnvironment,
      NestedSet<Pair<String, String>> reportedToActualSources) {
    super(STARLARK_CONSTRUCTOR);
    this.instrumentedFiles = instrumentedFiles;
    this.instrumentationMetadataFiles = instrumentationMetadataFiles;
    this.baselineCoverageFiles = baselineCoverageFiles;
    this.baselineCoverageArtifacts = baselineCoverageArtifacts;
    this.coverageSupportFiles = coverageSupportFiles;
    this.coverageEnvironment = coverageEnvironment;
    this.reportedToActualSources = reportedToActualSources;
  }

  @Override
  public InstrumentedFilesProvider getProvider() {
    return STARLARK_CONSTRUCTOR;
  }

  /** The transitive closure of instrumented source files. */
  public NestedSet<Artifact> getInstrumentedFiles() {
    return instrumentedFiles;
  }

  @Override
  public Depset getInstrumentedFilesForStarlark() {
    return Depset.of(Artifact.TYPE, getInstrumentedFiles());
  }

  /** Returns a collection of instrumentation metadata files. */
  public NestedSet<Artifact> getInstrumentationMetadataFiles() {
    return instrumentationMetadataFiles;
  }

  @Override
  public Depset getInstrumentationMetadataFilesForStarlark() {
    return Depset.of(Artifact.TYPE, getInstrumentationMetadataFiles());
  }

  /**
   * The transitive closure of instrumented source files for which baseline coverage should be
   * generated. In general, this is a subset of the instrumented source files: it only contains
   * instrumented source files from rules that support baseline coverage.
   */
  // TODO(ulfjack): Change this to a single Artifact. Also change how it's generated. It's better to
  // generate actions such that each action only covers the source files of a single rule, in
  // particular because baseline coverage is language-specific (it requires a parser for the
  // specific language), and we don't want to depend on all language parsers from any single rule.
  public NestedSet<Artifact> getBaselineCoverageInstrumentedFiles() {
    return baselineCoverageFiles;
  }

  /**
   * The output artifact of the baseline coverage action; this is only ever a single artifact, which
   * contains baseline coverage for the entire transitive closure of source files.
   */
  public NestedSet<Artifact> getBaselineCoverageArtifacts() {
    return baselineCoverageArtifacts;
  }

  /**
   * Extra files that are needed on the inputs of test actions for coverage collection to happen,
   * for example, {@code gcov}.
   *
   * <p>They aren't mentioned in the instrumented files manifest.
   */
  public NestedSet<Artifact> getCoverageSupportFiles() {
    return coverageSupportFiles;
  }

  /** Environment variables that need to be set for tests collecting code coverage. */
  public NestedSet<Pair<String, String>> getCoverageEnvironment() {
    return coverageEnvironment;
  }

  /**
   * A set of pairs of reported source file path and the actual source file path, relative to the
   * workspace directory, if the two values are different. If the reported source file is the same
   * as the actual source path it will not be included in this set.
   *
   * <p>This is useful for virtual include paths in C++, which get reported at the include location
   * and not the real source path. For example, the reported include source file can be
   * "bazel-out/k8-fastbuild/bin/include/common/_virtual_includes/strategy/strategy.h", but its
   * actual source path is "include/common/strategy.h".
   */
  public NestedSet<Pair<String, String>> getReportedToActualSources() {
    return reportedToActualSources;
  }

  /** Provider implementation for {@link InstrumentedFilesInfo}. */
  public static class InstrumentedFilesProvider extends BuiltinProvider<InstrumentedFilesInfo> {

    public InstrumentedFilesProvider() {
      super("InstrumentedFilesInfo", InstrumentedFilesInfo.class);
    }
  }
}
