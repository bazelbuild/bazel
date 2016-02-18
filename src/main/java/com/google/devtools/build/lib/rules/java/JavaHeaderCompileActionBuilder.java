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

package com.google.devtools.build.lib.rules.java;

import static com.google.devtools.build.lib.util.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine.CustomMultiArgv;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.StrictDepsMode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import javax.annotation.Nullable;

/**
 * Builder for Java header compilation actions, to be used if --experimental_java_header_compilation
 * is enabled.
 *
 * <p>The header compiler consumes the inputs of a java compilation, and produces an interface
 * jar that can be used as a compile-time jar by upstream targets. The header interface jar is
 * equivalent to the output of ijar, but unlike ijar the header compiler operates directly on
 * Java source files instead post-processing the class outputs of the compilation. Compiling the
 * interface jar from source moves javac off the build's critical path.
 *
 * <p>The implementation of the header compiler tool can be found under
 * {@code //src/java_tools/buildjar/java/com/google/devtools/build/java/turbine}.
 */
public class JavaHeaderCompileActionBuilder {

  static final ResourceSet RESOURCE_SET =
      ResourceSet.createWithRamCpuIo(/*memoryMb=*/ 750.0, /*cpuUsage=*/ 0.5, /*ioUsage=*/ 0.0);

  private final RuleContext ruleContext;

  private Artifact outputJar;
  @Nullable private Artifact outputDepsProto;
  private final Collection<Artifact> sourceFiles = new ArrayList<>();
  private final Collection<Artifact> sourceJars = new ArrayList<>();
  private NestedSet<Artifact> classpathEntries = NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
  private List<Artifact> bootclasspathEntries = new ArrayList<>();
  @Nullable private String ruleKind;
  @Nullable private Label targetLabel;
  private PathFragment tempDirectory;
  private BuildConfiguration.StrictDepsMode strictJavaDeps = BuildConfiguration.StrictDepsMode.OFF;
  private List<Artifact> directJars = new ArrayList<>();
  private List<Artifact> compileTimeDependencyArtifacts = new ArrayList<>();
  private ImmutableList<String> javacOpts;
  private List<Artifact> processorPath = new ArrayList<>();
  private List<String> processorNames = new ArrayList<>();

  /** Creates a builder using the configuration of the rule as the action configuration. */
  public JavaHeaderCompileActionBuilder(RuleContext ruleContext) {
    this.ruleContext = ruleContext;
  }

  /** Sets the output jdeps file. */
  public JavaHeaderCompileActionBuilder setOutputDepsProto(@Nullable Artifact outputDepsProto) {
    this.outputDepsProto = outputDepsProto;
    return this;
  }

  /** Sets the direct dependency artifacts. */
  public JavaHeaderCompileActionBuilder addDirectJars(Collection<Artifact> directJars) {
    checkNotNull(directJars, "directJars must not be null");
    this.directJars.addAll(directJars);
    return this;
  }

  /** Sets the .jdeps artifacts for direct dependencies. */
  public JavaHeaderCompileActionBuilder addCompileTimeDependencyArtifacts(
      Collection<Artifact> dependencyArtifacts) {
    checkNotNull(dependencyArtifacts, "dependencyArtifacts must not be null");
    this.compileTimeDependencyArtifacts.addAll(dependencyArtifacts);
    return this;
  }

  /** Sets Java compiler flags. */
  public JavaHeaderCompileActionBuilder setJavacOpts(ImmutableList<String> javacOpts) {
    checkNotNull(javacOpts, "javacOpts must not be null");
    this.javacOpts = javacOpts;
    return this;
  }

  /** Sets the output jar. */
  public JavaHeaderCompileActionBuilder setOutputJar(Artifact outputJar) {
    checkNotNull(outputJar, "outputJar must not be null");
    this.outputJar = outputJar;
    return this;
  }

  /** Adds a Java source file to compile. */
  public JavaHeaderCompileActionBuilder addSourceFile(Artifact sourceFile) {
    checkNotNull(sourceFile, "sourceFile must not be null");
    sourceFiles.add(sourceFile);
    return this;
  }

  /** Adds Java source files to compile. */
  public JavaHeaderCompileActionBuilder addSourceFiles(Collection<Artifact> sourceFiles) {
    checkNotNull(sourceFiles, "sourceFiles must not be null");
    this.sourceFiles.addAll(sourceFiles);
    return this;
  }

  /** Adds a jar archive of Java sources to compile. */
  public JavaHeaderCompileActionBuilder addSourceJars(Collection<Artifact> sourceJars) {
    checkNotNull(sourceJars, "sourceJars must not be null");
    this.sourceJars.addAll(sourceJars);
    return this;
  }

  /** Sets the compilation classpath entries. */
  public JavaHeaderCompileActionBuilder setClasspathEntries(NestedSet<Artifact> classpathEntries) {
    checkNotNull(classpathEntries, "classpathEntries must not be null");
    this.classpathEntries = classpathEntries;
    return this;
  }

  /** Sets the compilation bootclasspath entries. */
  public JavaHeaderCompileActionBuilder addAllBootclasspathEntries(
      Collection<Artifact> bootclasspathEntries) {
    checkNotNull(bootclasspathEntries, "bootclasspathEntries must not be null");
    this.bootclasspathEntries.addAll(bootclasspathEntries);
    return this;
  }

  /** Sets the compilation extclasspath entries. */
  public JavaHeaderCompileActionBuilder addAllExtClasspathEntries(
      Collection<Artifact> extclasspathEntries) {
    checkNotNull(extclasspathEntries, "extclasspathEntries must not be null");
    // fold extclasspath entries into the bootclasspath; that's what javac ends up doing
    this.bootclasspathEntries.addAll(extclasspathEntries);
    return this;
  }

  /** Sets the annotation processors classpath entries. */
  public JavaHeaderCompileActionBuilder addProcessorPaths(Collection<Artifact> processorPaths) {
    checkNotNull(processorPaths, "processorPaths must not be null");
    this.processorPath.addAll(processorPaths);
    return this;
  }

  /** Sets the fully-qualified class names of annotation processors to run. */
  public JavaHeaderCompileActionBuilder addProcessorNames(Collection<String> processorNames) {
    checkNotNull(processorNames, "processorNames must not be null");
    this.processorNames.addAll(processorNames);
    return this;
  }

  /** Sets the kind of the build rule being compiled (e.g. {@code java_library}). */
  public JavaHeaderCompileActionBuilder setRuleKind(@Nullable String ruleKind) {
    this.ruleKind = ruleKind;
    return this;
  }

  /** Sets the label of the target being compiled. */
  public JavaHeaderCompileActionBuilder setTargetLabel(@Nullable Label targetLabel) {
    this.targetLabel = targetLabel;
    return this;
  }

  /**
   * Sets the path to a temporary directory, e.g. for extracting sourcejar entries to before
   * compilation.
   */
  public JavaHeaderCompileActionBuilder setTempDirectory(PathFragment tempDirectory) {
    checkNotNull(tempDirectory, "tempDirectory must not be null");
    this.tempDirectory = tempDirectory;
    return this;
  }

  /** Sets the Strict Java Deps mode. */
  public JavaHeaderCompileActionBuilder setStrictJavaDeps(StrictDepsMode strictJavaDeps) {
    checkNotNull(strictJavaDeps, "strictJavaDeps must not be null");
    this.strictJavaDeps = strictJavaDeps;
    return this;
  }

  /** Builds and registers the {@link SpawnAction} for a header compilation. */
  public void build() {
    checkNotNull(outputDepsProto, "outputDepsProto must not be null");
    checkNotNull(sourceFiles, "sourceFiles must not be null");
    checkNotNull(sourceJars, "sourceJars must not be null");
    checkNotNull(classpathEntries, "classpathEntries must not be null");
    checkNotNull(bootclasspathEntries, "bootclasspathEntries must not be null");
    checkNotNull(tempDirectory, "tempDirectory must not be null");
    checkNotNull(strictJavaDeps, "strictJavaDeps must not be null");
    checkNotNull(directJars, "directJars must not be null");
    checkNotNull(compileTimeDependencyArtifacts, "compileTimeDependencyArtifacts must not be null");
    checkNotNull(javacOpts, "javacOpts must not be null");
    checkNotNull(processorPath, "processorPath must not be null");
    checkNotNull(processorNames, "processorNames must not be null");

    SpawnAction.Builder builder = new SpawnAction.Builder();

    builder.addOutput(outputJar);
    if (outputDepsProto != null) {
      builder.addOutput(outputDepsProto);
    }

    builder.useParameterFile(ParameterFileType.SHELL_QUOTED);
    builder.setCommandLine(buildCommandLine(ruleContext.getConfiguration().getHostPathSeparator()));

    builder.addTransitiveInputs(JavaCompilationHelper.getHostJavabaseInputs(ruleContext));
    builder.addInputs(classpathEntries);
    builder.addInputs(bootclasspathEntries);
    builder.addInputs(processorPath);
    builder.addInputs(sourceJars);
    builder.addInputs(sourceFiles);
    builder.addInputs(directJars);
    builder.addInputs(compileTimeDependencyArtifacts);

    Artifact langtools = ruleContext.getPrerequisiteArtifact("$java_langtools", Mode.HOST);
    builder.addTool(langtools);

    List<String> jvmArgs =
        ImmutableList.<String>builder()
            .addAll(JavaToolchainProvider.getDefaultJavacJvmOptions(ruleContext))
            .add("-Xbootclasspath/p:" + langtools.getExecPath().getPathString())
            .build();
    builder.setJarExecutable(
        ruleContext.getHostConfiguration().getFragment(Jvm.class).getJavaExecutable(),
        JavaToolchainProvider.getHeaderCompilerJar(ruleContext),
        jvmArgs);

    builder.setResources(RESOURCE_SET);

    builder.setMnemonic("Turbine");

    int count = sourceFiles.size() + sourceJars.size();
    builder.setProgressMessage(
        "Compiling Java headers " + outputJar.prettyPrint() + " (" + count + " files)");

    ruleContext.registerAction(builder.build(ruleContext));
  }

  /** Builds the header compiler command line. */
  private CommandLine buildCommandLine(String hostPathSeparator) {
    CustomCommandLine.Builder result = CustomCommandLine.builder();

    result.addExecPath("--output", outputJar);

    if (outputDepsProto != null) {
      result.addExecPath("--output_deps", outputDepsProto);
    }

    result.add("--temp_dir").addPath(tempDirectory);

    result.addJoinExecPaths("--classpath", hostPathSeparator, classpathEntries);
    result.addJoinExecPaths(
        "--bootclasspath", hostPathSeparator, bootclasspathEntries);

    if (!processorNames.isEmpty()) {
      result.add("--processors", processorNames);
    }
    if (!processorPath.isEmpty()) {
      result.addJoinExecPaths(
          "--processorpath", hostPathSeparator, processorPath);
    }

    result.addExecPaths("--sources", sourceFiles);

    if (!sourceJars.isEmpty()) {
      result.addExecPaths("--source_jars", sourceJars);
    }

    result.add("--javacopts", javacOpts);

    if (ruleKind != null) {
      result.add("--rule_kind");
      result.add(ruleKind);
    }
    if (targetLabel != null) {
      result.add("--target_label");
      if (targetLabel.getPackageIdentifier().getRepository().isDefault()) {
        result.add(targetLabel.toString());
      } else {
        // @-prefixed strings will be assumed to be params filenames and expanded,
        // so add an extra @ to escape it.
        result.add("@" + targetLabel);
      }
    }

    if (strictJavaDeps != BuildConfiguration.StrictDepsMode.OFF) {
      result.add("--strict_java_deps");
      result.add(strictJavaDeps.toString());
      result.add(
          new CustomMultiArgv() {
            @Override
            public Iterable<String> argv() {
              return JavaCompileAction.addJarsToTargets(classpathEntries, directJars);
            }
          });

      if (!compileTimeDependencyArtifacts.isEmpty()) {
        result.addExecPaths("--deps_artifacts", compileTimeDependencyArtifacts);
      }
    }

    return result.build();
  }
}
