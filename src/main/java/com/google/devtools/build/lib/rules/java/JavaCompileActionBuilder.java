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

package com.google.devtools.build.lib.rules.java;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.extra.ExtraActionInfo;
import com.google.devtools.build.lib.actions.extra.JavaCompileInfo;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.config.CoreOptionConverters.StrictDepsMode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.java.JavaCompileAction.ProgressMessage;
import com.google.devtools.build.lib.rules.java.JavaConfiguration.JavaClasspathMode;
import com.google.devtools.build.lib.rules.java.JavaPluginInfo.JavaPluginData;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.Collections;
import java.util.Objects;
import java.util.Optional;
import java.util.stream.Stream;
import javax.annotation.Nullable;

/** Java compilation action builder. */
public final class JavaCompileActionBuilder {

  private static final String JACOCO_INSTRUMENTATION_PROCESSOR = "jacoco";

  /** Environment variable that sets the UTF-8 charset. */
  static final ImmutableMap<String, String> UTF8_ENVIRONMENT =
      ImmutableMap.of("LC_CTYPE", "en_US.UTF-8");

  static final String MNEMONIC = "Javac";

  /** Returns true if this is a Java compile action. */
  public static boolean isJavaCompileAction(ActionAnalysisMetadata action) {
    return action != null && action.getMnemonic().equals(MNEMONIC);
  }

  @ThreadCompatible
  @Immutable
  private static final class JavaCompileExtraActionInfoSupplier
      implements JavaCompileAction.ExtraActionInfoSupplier {

    private final Artifact outputJar;

    /** The list of classpath entries to specify to javac. */
    private final NestedSet<Artifact> classpathEntries;

    /** The list of bootclasspath entries to specify to javac. */
    private final NestedSet<Artifact> bootclasspathEntries;

    /** An argument to the javac >= 9 {@code --system} flag. */
    @Nullable private final Optional<PathFragment> system;

    /** The list of classpath entries to search for annotation processors. */
    private final NestedSet<Artifact> processorPath;

    /** The list of annotation processor classes to run. */
    private final NestedSet<String> processorNames;

    /** Set of additional Java source files to compile. */
    private final ImmutableList<Artifact> sourceJars;

    /** The set of explicit Java source files to compile. */
    private final ImmutableSet<Artifact> sourceFiles;

    /** The compiler options to pass to javac. */
    private final ImmutableList<String> javacOpts;

    JavaCompileExtraActionInfoSupplier(
        Artifact outputJar,
        NestedSet<Artifact> classpathEntries,
        NestedSet<Artifact> bootclasspathEntries,
        Optional<PathFragment> system,
        NestedSet<Artifact> processorPath,
        NestedSet<String> processorNames,
        ImmutableList<Artifact> sourceJars,
        ImmutableSet<Artifact> sourceFiles,
        ImmutableList<String> javacOpts) {
      this.outputJar = outputJar;
      this.classpathEntries = classpathEntries;
      this.bootclasspathEntries = bootclasspathEntries;
      this.system = system;
      this.processorPath = processorPath;
      this.processorNames = processorNames;
      this.sourceJars = sourceJars;
      this.sourceFiles = sourceFiles;
      this.javacOpts = javacOpts;
    }

    @Override
    public void extend(ExtraActionInfo.Builder builder, ImmutableList<String> arguments) {
      JavaCompileInfo.Builder info =
          JavaCompileInfo.newBuilder()
              .addAllSourceFile(Artifact.toExecPaths(sourceFiles))
              .addAllClasspath(Artifact.toExecPaths(classpathEntries.toList()))
              .addAllBootclasspath(Artifact.toExecPaths(bootclasspathEntries.toList()))
              .addAllSourcepath(Artifact.toExecPaths(sourceJars))
              .addAllJavacOpt(javacOpts)
              .addAllProcessor(processorNames.toList())
              .addAllProcessorpath(Artifact.toExecPaths(processorPath.toList()))
              .setOutputjar(outputJar.getExecPathString());
      if (system.isPresent()) {
        info.setSystem(system.get().toString());
      }
      info.addAllArgument(arguments);
      builder.setExtension(JavaCompileInfo.javaCompileInfo, info.build());
    }
  }

  private final RuleContext ruleContext;
  private final JavaToolchainProvider toolchain;
  private final String execGroup;
  private ImmutableSet<Artifact> additionalOutputs = ImmutableSet.of();
  private Artifact coverageArtifact;
  private ImmutableSet<Artifact> sourceFiles = ImmutableSet.of();
  private ImmutableList<Artifact> sourceJars = ImmutableList.of();
  private StrictDepsMode strictJavaDeps = StrictDepsMode.ERROR;
  private String fixDepsTool = "add_dep";
  private NestedSet<Artifact> directJars = NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
  private NestedSet<Artifact> compileTimeDependencyArtifacts =
      NestedSetBuilder.emptySet(Order.STABLE_ORDER);
  private ImmutableList<String> javacOpts = ImmutableList.of();
  private ImmutableMap<String, String> executionInfo = ImmutableMap.of();
  private boolean compressJar;
  private NestedSet<Artifact> classpathEntries = NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
  private BootClassPathInfo bootClassPath = BootClassPathInfo.empty();
  private ImmutableList<Artifact> sourcePathEntries = ImmutableList.of();
  private JavaToolchainTool javaBuilder;
  private NestedSet<Artifact> toolsJars = NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
  private JavaPluginData plugins = JavaPluginData.empty();
  private NestedSet<Artifact> extraData = NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
  private Label targetLabel;
  @Nullable private String injectingRuleKind;
  private ImmutableList<Artifact> additionalInputs = ImmutableList.of();
  private Artifact genSourceOutput;
  private JavaCompileOutputs<Artifact> outputs;
  private JavaClasspathMode classpathMode;
  private Artifact manifestOutput;

  public JavaCompileActionBuilder(
      RuleContext ruleContext, JavaToolchainProvider toolchain, String execGroup) {
    this.ruleContext = ruleContext;
    this.toolchain = toolchain;
    this.execGroup = execGroup;
  }

  public JavaCompileAction build() throws RuleErrorException {
    // TODO(bazel-team): all the params should be calculated before getting here, and the various
    // aggregation code below should go away.

    // Invariant: if strictJavaDeps is OFF, then directJars and
    // dependencyArtifacts are ignored
    if (strictJavaDeps == StrictDepsMode.OFF) {
      directJars = NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
      compileTimeDependencyArtifacts = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }

    // Invariant: if java_classpath is set to 'off', dependencyArtifacts are ignored
    if (!Collections.disjoint(
        plugins.processorClasses().toSet(),
        toolchain.getReducedClasspathIncompatibleProcessors())) {
      classpathMode = JavaClasspathMode.OFF;
    }
    if (classpathMode == JavaClasspathMode.OFF) {
      compileTimeDependencyArtifacts = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }

    NestedSetBuilder<Artifact> toolsBuilder = NestedSetBuilder.compileOrder();
    javaBuilder.addInputs(toolchain, toolsBuilder);
    toolsBuilder.addTransitive(toolsJars);

    NestedSetBuilder<Artifact> mandatoryInputsBuilder = NestedSetBuilder.stableOrder();
    mandatoryInputsBuilder
        .addTransitive(plugins.processorClasspath())
        .addTransitive(plugins.data())
        .addTransitive(extraData)
        .addAll(sourceJars)
        .addAll(sourceFiles)
        .addTransitive(toolchain.getJavaRuntime().javaBaseInputs())
        .addTransitive(bootClassPath.bootclasspath())
        .addAll(sourcePathEntries)
        .addAll(additionalInputs)
        .addTransitive(bootClassPath.systemInputs());
    if (coverageArtifact != null) {
      mandatoryInputsBuilder.add(coverageArtifact);
    }

    JavaCompileExtraActionInfoSupplier extraActionInfoSupplier =
        new JavaCompileExtraActionInfoSupplier(
            outputs.output(),
            classpathEntries,
            bootClassPath.bootclasspath(),
            bootClassPath.systemPath(),
            plugins.processorClasspath(),
            plugins.processorClasses(),
            sourceJars,
            sourceFiles,
            javacOpts);

    // TODO(b/123076347): outputDepsProto should never be null if SJD is enabled
    if (strictJavaDeps == StrictDepsMode.OFF || outputs.depsProto() == null) {
      classpathMode = JavaClasspathMode.OFF;
    }

    NestedSet<Artifact> tools = toolsBuilder.build();
    mandatoryInputsBuilder.addTransitive(tools);
    NestedSet<Artifact> mandatoryInputs = mandatoryInputsBuilder.build();

    CustomCommandLine executableLine = javaBuilder.getCommandLine(toolchain);

    return new JavaCompileAction(
        /* compilationType= */ JavaCompileAction.CompilationType.JAVAC,
        /* owner= */ ruleContext.getActionOwner(execGroup),
        /* tools= */ tools,
        /* progressMessage= */ new ProgressMessage(
            /* prefix= */ "Building",
            /* output= */ outputs.output(),
            /* sourceFiles= */ sourceFiles,
            /* sourceJars= */ sourceJars,
            /* plugins= */ plugins),
        /* mandatoryInputs= */ mandatoryInputs,
        /* transitiveInputs= */ classpathEntries,
        /* directJars= */ directJars,
        /* outputs= */ allOutputs(),
        /* executionInfo= */ executionInfo,
        /* extraActionInfoSupplier= */ extraActionInfoSupplier,
        /* executableLine= */ executableLine,
        /* flagLine= */ buildParamFileContents(javacOpts),
        /* configuration= */ ruleContext.getConfiguration(),
        /* dependencyArtifacts= */ compileTimeDependencyArtifacts,
        /* outputDepsProto= */ outputs.depsProto(),
        /* classpathMode= */ classpathMode);
  }

  private ImmutableSet<Artifact> allOutputs() {
    ImmutableSet.Builder<Artifact> result =
        ImmutableSet.<Artifact>builder()
            .add(outputs.output())
            .addAll(additionalOutputs);
    Stream.of(outputs.depsProto(), outputs.nativeHeader(), genSourceOutput, manifestOutput)
        .filter(Objects::nonNull)
        .forEachOrdered(result::add);
    return result.build();
  }

  private CustomCommandLine buildParamFileContents(ImmutableList<String> javacOpts)
      throws RuleErrorException {

    CustomCommandLine.Builder result = CustomCommandLine.builder();

    result.addExecPath("--output", outputs.output());
    result.addExecPath("--native_header_output", outputs.nativeHeader());
    result.addExecPath("--generated_sources_output", genSourceOutput);
    result.addExecPath("--output_manifest_proto", manifestOutput);
    if (compressJar) {
      result.add("--compress_jar");
    }
    result.addExecPath("--output_deps_proto", outputs.depsProto());
    result.addExecPaths("--bootclasspath", bootClassPath.bootclasspath());
    if (bootClassPath.systemPath().isPresent()) {
      result.addPath("--system", bootClassPath.systemPath().get());
    }
    result.addExecPaths("--sourcepath", sourcePathEntries);
    result.addExecPaths("--processorpath", plugins.processorClasspath());
    result.addAll("--processors", plugins.processorClasses());
    result.addExecPaths("--source_jars", sourceJars);
    result.addExecPaths("--sources", sourceFiles);
    if (!javacOpts.isEmpty()) {
      result.add("--javacopts").addObject(javacOpts);
      // terminate --javacopts with `--` to support javac flags that start with `--`
      result.add("--");
    }
    if (targetLabel != null) {
      result.add("--target_label");
      if (targetLabel.getRepository().isMain()) {
        result.addLabel(targetLabel);
      } else {
        // @-prefixed strings will be assumed to be filenames and expanded by
        // {@link JavaLibraryBuildRequest}, so add an extra &at; to escape it.
        result.addPrefixedLabel("@", targetLabel);
      }
    }
    result.add("--injecting_rule_kind", injectingRuleKind);
    // strict_java_deps controls whether the mapping from jars to targets is
    // written out and whether we try to minimize the compile-time classpath.
    if (strictJavaDeps != StrictDepsMode.OFF) {
      result.add("--strict_java_deps", strictJavaDeps.toString());
      result.addExecPaths("--direct_dependencies", directJars);
    }
    result.add("--experimental_fix_deps_tool", fixDepsTool);

    // Chose what artifact to pass to JavaBuilder, as input to jacoco instrumentation processor.
    if (coverageArtifact != null) {
      result.add("--post_processor");
      result.addExecPath(JACOCO_INSTRUMENTATION_PROCESSOR, coverageArtifact);
      result.addPath(ruleContext.getCoverageMetadataDirectory().getExecPath());
      result.add("-*Test");
      result.add("-*TestCase");
    }
    return result.build();
  }

  @CanIgnoreReturnValue
  public JavaCompileActionBuilder setAdditionalOutputs(ImmutableSet<Artifact> outputs) {
    this.additionalOutputs = outputs;
    return this;
  }

  @CanIgnoreReturnValue
  public JavaCompileActionBuilder setSourceFiles(ImmutableSet<Artifact> sourceFiles) {
    this.sourceFiles = sourceFiles;
    return this;
  }

  @CanIgnoreReturnValue
  public JavaCompileActionBuilder setSourceJars(ImmutableList<Artifact> sourceJars) {
    checkState(this.sourceJars.isEmpty());
    this.sourceJars = checkNotNull(sourceJars, "sourceJars must not be null");
    return this;
  }

  /** Sets the strictness of Java dependency checking, see {@link StrictDepsMode}. */
  @CanIgnoreReturnValue
  public JavaCompileActionBuilder setStrictJavaDeps(StrictDepsMode strictDeps) {
    strictJavaDeps = strictDeps;
    return this;
  }

  /** Sets the tool with which to fix dependency errors. */
  @CanIgnoreReturnValue
  public JavaCompileActionBuilder setFixDepsTool(String depsTool) {
    fixDepsTool = depsTool;
    return this;
  }

  /** Accumulates the given jar artifacts as being provided by direct dependencies. */
  @CanIgnoreReturnValue
  public JavaCompileActionBuilder setDirectJars(NestedSet<Artifact> directJars) {
    this.directJars = checkNotNull(directJars, "directJars must not be null");
    return this;
  }

  @CanIgnoreReturnValue
  public JavaCompileActionBuilder setCompileTimeDependencyArtifacts(
      NestedSet<Artifact> dependencyArtifacts) {
    checkNotNull(compileTimeDependencyArtifacts, "dependencyArtifacts must not be null");
    this.compileTimeDependencyArtifacts = dependencyArtifacts;
    return this;
  }

  @CanIgnoreReturnValue
  public JavaCompileActionBuilder setJavacOpts(ImmutableList<String> copts) {
    this.javacOpts = Preconditions.checkNotNull(copts);
    return this;
  }

  @CanIgnoreReturnValue
  public JavaCompileActionBuilder setJavacExecutionInfo(
      ImmutableMap<String, String> executionInfo) {
    this.executionInfo = executionInfo;
    return this;
  }

  @CanIgnoreReturnValue
  public JavaCompileActionBuilder setCompressJar(boolean compressJar) {
    this.compressJar = compressJar;
    return this;
  }

  @CanIgnoreReturnValue
  public JavaCompileActionBuilder setClasspathEntries(NestedSet<Artifact> classpathEntries) {
    this.classpathEntries = classpathEntries;
    return this;
  }

  @CanIgnoreReturnValue
  public JavaCompileActionBuilder setBootClassPath(BootClassPathInfo bootClassPath) {
    this.bootClassPath = bootClassPath;
    return this;
  }

  @CanIgnoreReturnValue
  public JavaCompileActionBuilder setSourcePathEntries(ImmutableList<Artifact> sourcePathEntries) {
    this.sourcePathEntries = Preconditions.checkNotNull(sourcePathEntries);
    return this;
  }

  @CanIgnoreReturnValue
  public JavaCompileActionBuilder setPlugins(JavaPluginData plugins) {
    checkNotNull(plugins, "plugins must not be null");
    checkState(this.plugins.isEmpty());
    this.plugins = plugins;
    return this;
  }

  public void setExtraData(NestedSet<Artifact> extraData) {
    checkNotNull(extraData, "extraData must not be null");
    checkState(this.extraData.isEmpty());
    this.extraData = extraData;
  }

  /** Sets the tools jars. */
  @CanIgnoreReturnValue
  public JavaCompileActionBuilder setToolsJars(NestedSet<Artifact> toolsJars) {
    checkNotNull(toolsJars, "toolsJars must not be null");
    this.toolsJars = toolsJars;
    return this;
  }

  @CanIgnoreReturnValue
  public JavaCompileActionBuilder setJavaBuilder(JavaToolchainTool javaBuilder) {
    this.javaBuilder = javaBuilder;
    return this;
  }

  @CanIgnoreReturnValue
  public JavaCompileActionBuilder setCoverageArtifact(Artifact coverageArtifact) {
    this.coverageArtifact = coverageArtifact;
    return this;
  }

  @CanIgnoreReturnValue
  public JavaCompileActionBuilder setTargetLabel(Label targetLabel) {
    this.targetLabel = targetLabel;
    return this;
  }

  @CanIgnoreReturnValue
  public JavaCompileActionBuilder setInjectingRuleKind(@Nullable String injectingRuleKind) {
    this.injectingRuleKind = injectingRuleKind;
    return this;
  }

  @CanIgnoreReturnValue
  public JavaCompileActionBuilder setAdditionalInputs(ImmutableList<Artifact> additionalInputs) {
    checkNotNull(additionalInputs, "additionalInputs must not be null");
    this.additionalInputs = additionalInputs;
    return this;
  }

  public void setGenSourceOutput(Artifact genSourceOutput) {
    this.genSourceOutput = genSourceOutput;
  }

  public void setOutputs(JavaCompileOutputs<Artifact> outputs) {
    this.outputs = outputs;
  }

  public void setClasspathMode(JavaClasspathMode classpathMode) {
    this.classpathMode = classpathMode;
  }

  public void setManifestOutput(Artifact manifestOutput) {
    this.manifestOutput = manifestOutput;
  }
}
