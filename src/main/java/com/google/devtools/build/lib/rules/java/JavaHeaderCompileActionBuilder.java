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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;
import static java.nio.charset.StandardCharsets.ISO_8859_1;
import static java.util.stream.Collectors.joining;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ParamFileInfo;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.rules.java.JavaPluginInfoProvider.JavaPluginInfo;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.LazyString;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Collection;
import javax.annotation.Nullable;

/**
 * Action builder for Java header compilation, to be used if --java_header_compilation is enabled.
 *
 * <p>The header compiler consumes the inputs of a java compilation, and produces an interface jar
 * that can be used as a compile-time jar by upstream targets. The header interface jar is
 * equivalent to the output of ijar, but unlike ijar the header compiler operates directly on Java
 * source files instead post-processing the class outputs of the compilation. Compiling the
 * interface jar from source moves javac off the build's critical path.
 *
 * <p>The implementation of the header compiler tool can be found under {@code
 * //src/java_tools/buildjar/java/com/google/devtools/build/java/turbine}.
 */
public class JavaHeaderCompileActionBuilder {

  private final RuleContext ruleContext;

  private Artifact outputJar;
  @Nullable private Artifact outputDepsProto;
  private ImmutableSet<Artifact> sourceFiles = ImmutableSet.of();
  private final Collection<Artifact> sourceJars = new ArrayList<>();
  private NestedSet<Artifact> classpathEntries = NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
  private ImmutableList<Artifact> bootclasspathEntries = ImmutableList.of();
  @Nullable private Label targetLabel;
  @Nullable private String injectingRuleKind;
  private PathFragment tempDirectory;
  private BuildConfiguration.StrictDepsMode strictJavaDeps = BuildConfiguration.StrictDepsMode.OFF;
  private NestedSet<Artifact> directJars = NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
  private NestedSet<Artifact> compileTimeDependencyArtifacts =
      NestedSetBuilder.emptySet(Order.STABLE_ORDER);
  private ImmutableList<String> javacOpts;
  private JavaPluginInfo plugins = JavaPluginInfo.empty();

  private NestedSet<Artifact> additionalInputs = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
  private Artifact javacJar;
  private NestedSet<Artifact> toolsJars = NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);

  public JavaHeaderCompileActionBuilder(RuleContext ruleContext) {
    this.ruleContext = ruleContext;
  }

  /** Sets the output jdeps file. */
  public JavaHeaderCompileActionBuilder setOutputDepsProto(@Nullable Artifact outputDepsProto) {
    this.outputDepsProto = outputDepsProto;
    return this;
  }

  /** Sets the direct dependency artifacts. */
  public JavaHeaderCompileActionBuilder setDirectJars(NestedSet<Artifact> directJars) {
    checkNotNull(directJars, "directJars must not be null");
    this.directJars = directJars;
    return this;
  }

  /** Sets the .jdeps artifacts for direct dependencies. */
  public JavaHeaderCompileActionBuilder setCompileTimeDependencyArtifacts(
      NestedSet<Artifact> dependencyArtifacts) {
    checkNotNull(dependencyArtifacts, "dependencyArtifacts must not be null");
    this.compileTimeDependencyArtifacts = dependencyArtifacts;
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

  /** Adds Java source files to compile. */
  public JavaHeaderCompileActionBuilder setSourceFiles(ImmutableSet<Artifact> sourceFiles) {
    checkNotNull(sourceFiles, "sourceFiles must not be null");
    this.sourceFiles = sourceFiles;
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
  public JavaHeaderCompileActionBuilder setBootclasspathEntries(
      ImmutableList<Artifact> bootclasspathEntries) {
    checkNotNull(bootclasspathEntries, "bootclasspathEntries must not be null");
    this.bootclasspathEntries = bootclasspathEntries;
    return this;
  }

  /** Sets the annotation processors classpath entries. */
  public JavaHeaderCompileActionBuilder setPlugins(JavaPluginInfo plugins) {
    checkNotNull(plugins, "plugins must not be null");
    checkState(this.plugins.isEmpty());
    this.plugins = plugins;
    return this;
  }

  /** Sets the label of the target being compiled. */
  public JavaHeaderCompileActionBuilder setTargetLabel(@Nullable Label targetLabel) {
    this.targetLabel = targetLabel;
    return this;
  }

  /** Sets the injecting rule kind of the target being compiled. */
  public JavaHeaderCompileActionBuilder setInjectingRuleKind(@Nullable String injectingRuleKind) {
    this.injectingRuleKind = injectingRuleKind;
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
  public JavaHeaderCompileActionBuilder setStrictJavaDeps(
      BuildConfiguration.StrictDepsMode strictJavaDeps) {
    checkNotNull(strictJavaDeps, "strictJavaDeps must not be null");
    this.strictJavaDeps = strictJavaDeps;
    return this;
  }

  /** Sets the javabase inputs. */
  public JavaHeaderCompileActionBuilder setAdditionalInputs(NestedSet<Artifact> additionalInputs) {
    checkNotNull(additionalInputs, "additionalInputs must not be null");
    this.additionalInputs = additionalInputs;
    return this;
  }

  /** Sets the javac jar. */
  public JavaHeaderCompileActionBuilder setJavacJar(Artifact javacJar) {
    checkNotNull(javacJar, "javacJar must not be null");
    this.javacJar = javacJar;
    return this;
  }

  /** Sets the tools jars. */
  public JavaHeaderCompileActionBuilder setToolsJars(NestedSet<Artifact> toolsJars) {
    checkNotNull(toolsJars, "toolsJars must not be null");
    this.toolsJars = toolsJars;
    return this;
  }

  /** Builds and registers the action for a header compilation. */
  public void build(JavaToolchainProvider javaToolchain, JavaRuntimeInfo hostJavabase) {
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

    // Invariant: if strictJavaDeps is OFF, then directJars and
    // dependencyArtifacts are ignored
    if (strictJavaDeps == BuildConfiguration.StrictDepsMode.OFF) {
      directJars = NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
      compileTimeDependencyArtifacts = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }

    // The compilation uses API-generating annotation processors and has to fall back to
    // javac-turbine.
    boolean requiresAnnotationProcessing = !plugins.isEmpty();

    SpawnAction.Builder builder = new SpawnAction.Builder();

    builder.setEnvironment(JavaCompileActionBuilder.UTF8_ENVIRONMENT);

    builder.setProgressMessage(
        new ProgressMessage(
            this.outputJar, sourceFiles.size() + sourceJars.size(), plugins.processorClasses()));

    builder.addTool(javacJar);
    builder.addTransitiveTools(toolsJars);

    builder.addOutput(outputJar);
    builder.addOutput(outputDepsProto);

    builder.addTransitiveInputs(additionalInputs);
    builder.addInputs(bootclasspathEntries);
    builder.addInputs(sourceJars);
    builder.addInputs(sourceFiles);

    FilesToRunProvider headerCompiler =
        (!requiresAnnotationProcessing && javaToolchain.getHeaderCompilerDirect() != null)
            ? javaToolchain.getHeaderCompilerDirect()
            : javaToolchain.getHeaderCompiler();
    // The header compiler is either a jar file that needs to be executed using
    // `java -jar <path>`, or an executable that can be run directly.
    if (!headerCompiler.getExecutable().getExtension().equals("jar")) {
      builder.setExecutable(headerCompiler);
    } else {
      builder.addTransitiveInputs(hostJavabase.javaBaseInputsMiddleman());
      builder.setJarExecutable(
          hostJavabase.javaBinaryExecPath(),
          headerCompiler.getExecutable(),
          javaToolchain.getJvmOptions());
    }

    CustomCommandLine.Builder commandLine =
        CustomCommandLine.builder()
            .addExecPath("--output", outputJar)
            .addExecPath("--output_deps", outputDepsProto)
            .addPath("--temp_dir", tempDirectory)
            .addExecPaths("--bootclasspath", bootclasspathEntries)
            .addExecPaths("--sources", sourceFiles)
            .addExecPaths("--source_jars", sourceJars)
            .add("--injecting_rule_kind", injectingRuleKind);

    if (!javacOpts.isEmpty()) {
      commandLine.addAll("--javacopts", javacOpts);
      // terminate --javacopts with `--` to support javac flags that start with `--`
      commandLine.add("--");
    }

    if (targetLabel != null) {
      commandLine.add("--target_label");
      if (targetLabel.getPackageIdentifier().getRepository().isDefault()
          || targetLabel.getPackageIdentifier().getRepository().isMain()) {
        commandLine.addLabel(targetLabel);
      } else {
        // @-prefixed strings will be assumed to be params filenames and expanded,
        // so add an extra @ to escape it.
        commandLine.addPrefixedLabel("@", targetLabel);
      }
    }

    // The action doesn't require annotation processing, so use the non-javac-based turbine
    // implementation.
    if (!requiresAnnotationProcessing) {
      NestedSet<Artifact> classpath;
      if (!directJars.isEmpty() || classpathEntries.isEmpty()) {
        classpath = directJars;
      } else {
        classpath = classpathEntries;
      }
      builder.addTransitiveInputs(classpath);

      commandLine.addExecPaths("--classpath", classpath);
      commandLine.add("--nojavac_fallback");

      ruleContext.registerAction(
          builder
              .addCommandLine(
                  commandLine.build(), ParamFileInfo.builder(ParameterFileType.UNQUOTED).build())
              .setMnemonic("Turbine")
              .build(ruleContext));
      return;
    }

    // If we get here the action requires annotation processing, so add additional inputs and
    // flags needed for the javac-based header compiler implementatino that supports
    // annotation processing.

    builder.addTransitiveInputs(classpathEntries);
    builder.addTransitiveInputs(plugins.processorClasspath());
    builder.addTransitiveInputs(plugins.data());
    builder.addTransitiveInputs(compileTimeDependencyArtifacts);

    commandLine.addExecPaths("--classpath", classpathEntries);
    commandLine.addAll("--processors", plugins.processorClasses());
    commandLine.addExecPaths("--processorpath", plugins.processorClasspath());
    if (strictJavaDeps != BuildConfiguration.StrictDepsMode.OFF) {
      commandLine.addExecPaths("--direct_dependencies", directJars);
      if (!compileTimeDependencyArtifacts.isEmpty()) {
        commandLine.addExecPaths("--deps_artifacts", compileTimeDependencyArtifacts);
      }
    }

    ruleContext.registerAction(
        builder
            .addCommandLine(
                commandLine.build(),
                ParamFileInfo.builder(ParameterFileType.UNQUOTED).setCharset(ISO_8859_1).build())
            .setMnemonic("JavacTurbine")
            .build(ruleContext));
  }

  /** Static class to avoid keeping a reference to this builder after build() is called. */
  @AutoCodec.VisibleForSerialization
  @AutoCodec
  static class ProgressMessage extends LazyString {

    private final Artifact outputJar;
    private final int fileCount;
    private final NestedSet<String> processorClasses;

    public ProgressMessage(Artifact outputJar, int fileCount, NestedSet<String> processorClasses) {
      this.outputJar = outputJar;
      this.fileCount = fileCount;
      this.processorClasses = processorClasses;
    }

    @Override
    public String toString() {
      return String.format(
          "Compiling Java headers %s (%d files)%s",
          outputJar.prettyPrint(),
          fileCount,
          processorClasses.isEmpty()
              ? ""
              : processorClasses.toCollection().stream()
                  .map(name -> name.substring(name.lastIndexOf('.') + 1))
                  .collect(joining(", ", " and running annotation processors (", ")")));
    }
  }
}
