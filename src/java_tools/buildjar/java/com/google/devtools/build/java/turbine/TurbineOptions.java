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

package com.google.devtools.build.java.turbine;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;

import javax.annotation.Nullable;

/** Header compilation options. */
public class TurbineOptions {

  private final String output;
  private final ImmutableList<String> classPath;
  private final ImmutableList<String> bootClassPath;
  private final ImmutableList<String> sources;
  private final ImmutableList<String> processorPath;
  private final ImmutableList<String> processors;
  private final String tempDir;
  private final ImmutableList<String> sourceJars;
  private final Optional<String> outputDeps;
  private final ImmutableMap<String, String> directJarsToTargets;
  private final ImmutableMap<String, String> indirectJarsToTargets;
  private final String targetLabel;
  private final ImmutableList<String> depsArtifacts;
  private final String strictDepsMode;
  private final String ruleKind;
  private final ImmutableList<String> javacOpts;

  private TurbineOptions(
      String output,
      ImmutableList<String> classPath,
      ImmutableList<String> bootClassPath,
      ImmutableList<String> sources,
      ImmutableList<String> processorPath,
      ImmutableList<String> processors,
      String tempDir,
      ImmutableList<String> sourceJars,
      @Nullable String outputDeps,
      ImmutableMap<String, String> directJarsToTargets,
      ImmutableMap<String, String> indirectJarsToTargets,
      String targetLabel,
      ImmutableList<String> depsArtifacts,
      String strictDepsMode,
      String ruleKind,
      ImmutableList<String> javacOpts) {
    this.output = checkNotNull(output, "output must not be null");
    this.classPath = checkNotNull(classPath, "classPath must not be null");
    this.bootClassPath = checkNotNull(bootClassPath, "bootClassPath must not be null");
    this.sources = checkNotNull(sources, "sources must not be null");
    this.processorPath = checkNotNull(processorPath, "processorPath must not be null");
    this.processors = checkNotNull(processors, "processors must not be null");
    this.tempDir = checkNotNull(tempDir, "tempDir must not be null");
    this.sourceJars = checkNotNull(sourceJars, "sourceJars must not be null");
    this.outputDeps = Optional.fromNullable(outputDeps);
    this.directJarsToTargets =
        checkNotNull(directJarsToTargets, "directJarsToTargets must not be null");
    this.indirectJarsToTargets =
        checkNotNull(indirectJarsToTargets, "indirectJarsToTargets must not be null");
    this.targetLabel = checkNotNull(targetLabel, "targetLabel must not be null");
    this.depsArtifacts = checkNotNull(depsArtifacts, "depsArtifacts must not be null");
    this.strictDepsMode = checkNotNull(strictDepsMode, "strictDepsMode must not be null");
    this.ruleKind = checkNotNull(ruleKind, "ruleKind must not be null");
    this.javacOpts = checkNotNull(javacOpts, "javacOpts must not be null");
  }

  /** Paths to the Java source files to compile. */
  public ImmutableList<String> sources() {
    return sources;
  }

  /** Paths to classpath artifacts. */
  public ImmutableList<String> classPath() {
    return classPath;
  }

  /** Paths to compilation bootclasspath artifacts. */
  public ImmutableList<String> bootClassPath() {
    return bootClassPath;
  }

  /** The output jar. */
  public String outputFile() {
    return output;
  }

  /** A temporary directory, e.g. for extracting sourcejar entries to before compilation. */
  public String tempDir() {
    return tempDir;
  }

  /** Paths to annotation processor artifacts. */
  public ImmutableList<String> processorPath() {
    return processorPath;
  }

  /** Annotation processor class names. */
  public ImmutableList<String> processors() {
    return processors;
  }

  /** Source jars for compilation. */
  public ImmutableList<String> sourceJars() {
    return sourceJars;
  }

  /** Output jdeps file. */
  public Optional<String> outputDeps() {
    return outputDeps;
  }

  /** The mapping from the path to a direct dependency to its build label. */
  public ImmutableMap<String, String> directJarsToTargets() {
    return directJarsToTargets;
  }

  /** The mapping from the path to an indirect dependency to its build label. */
  public ImmutableMap<String, String> indirectJarsToTargets() {
    return indirectJarsToTargets;
  }

  /** The label of the target being compiled. */
  public String targetLabel() {
    return targetLabel;
  }

  /** The .jdeps artifacts for direct dependencies. */
  public ImmutableList<String> depsArtifacts() {
    return depsArtifacts;
  }

  /**
   * The Strict Java Deps mode.
   *
   * <p>See {@link com.google.devtools.build.buildjar.javac.plugins.dependency.DependencyModule.StrictJavaDeps}.
   */
  public String strictDepsMode() {
    return strictDepsMode;
  }

  /** The kind of the build rule being compiled (e.g. {@code java_library}). */
  public String ruleKind() {
    return ruleKind;
  }

  /** Additional Java compiler flags. */
  public ImmutableList<String> javacOpts() {
    return javacOpts;
  }

  public static Builder builder() {
    return new Builder();
  }

  /** A {@link Builder} for {@link TurbineOptions}. */
  public static class Builder {

    private String output;
    private final ImmutableList.Builder<String> classPath = ImmutableList.builder();
    private final ImmutableList.Builder<String> sources = ImmutableList.builder();
    private final ImmutableList.Builder<String> processorPath = ImmutableList.builder();
    private final ImmutableList.Builder<String> processors = ImmutableList.builder();
    private String tempDir;
    private final ImmutableList.Builder<String> sourceJars = ImmutableList.builder();
    private final ImmutableList.Builder<String> bootClassPath = ImmutableList.builder();
    private String outputDeps;
    private final ImmutableMap.Builder<String, String> directJarsToTargets = ImmutableMap.builder();
    private final ImmutableMap.Builder<String, String> indirectJarsToTargets =
        ImmutableMap.builder();
    private String targetLabel;
    private final ImmutableList.Builder<String> depsArtifacts = ImmutableList.builder();
    private String strictDepsMode = "OFF";
    private String ruleKind;
    private final ImmutableList.Builder<String> javacOpts = ImmutableList.builder();

    public TurbineOptions build() {
      return new TurbineOptions(
          output,
          classPath.build(),
          bootClassPath.build(),
          sources.build(),
          processorPath.build(),
          processors.build(),
          tempDir,
          sourceJars.build(),
          outputDeps,
          directJarsToTargets.build(),
          indirectJarsToTargets.build(),
          targetLabel,
          depsArtifacts.build(),
          strictDepsMode,
          ruleKind,
          javacOpts.build());
    }

    public Builder setStrictJavaDeps(String strictDepsMode) {
      this.strictDepsMode = strictDepsMode;
      return this;
    }

    public Builder setOutput(String output) {
      this.output = output;
      return this;
    }

    public Builder addClassPathEntries(Iterable<String> classPath) {
      this.classPath.addAll(classPath);
      return this;
    }

    public Builder addBootClassPathEntries(Iterable<String> bootClassPath) {
      this.bootClassPath.addAll(bootClassPath);
      return this;
    }

    public Builder addSources(Iterable<String> sources) {
      this.sources.addAll(sources);
      return this;
    }

    public Builder addProcessorPathEntries(Iterable<String> processorPath) {
      this.processorPath.addAll(processorPath);
      return this;
    }

    public Builder setProcessors(Iterable<String> processors) {
      this.processors.addAll(processors);
      return this;
    }

    public Builder setTempDir(String tempDir) {
      this.tempDir = tempDir;
      return this;
    }

    public Builder setSourceJars(Iterable<String> sourceJars) {
      this.sourceJars.addAll(sourceJars);
      return this;
    }

    public Builder setOutputDeps(String outputDeps) {
      this.outputDeps = outputDeps;
      return this;
    }

    public Builder addDirectJarToTarget(String jar, String target) {
      directJarsToTargets.put(jar, target);
      return this;
    }

    public Builder addIndirectJarToTarget(String jar, String target) {
      indirectJarsToTargets.put(jar, target);
      return this;
    }

    public Builder setTargetLabel(String targetLabel) {
      this.targetLabel = targetLabel;
      return this;
    }

    public Builder addAllDepsArtifacts(Iterable<String> depsArtifacts) {
      this.depsArtifacts.addAll(depsArtifacts);
      return this;
    }

    public Builder setRuleKind(String ruleKind) {
      this.ruleKind = ruleKind;
      return this;
    }

    public Builder addAllJavacOpts(Iterable<String> javacOpts) {
      this.javacOpts.addAll(javacOpts);
      return this;
    }
  }
}
