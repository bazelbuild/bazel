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
import static com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType.UNQUOTED;
import static com.google.devtools.build.lib.rules.java.JavaCompileActionBuilder.UTF8_ENVIRONMENT;
import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionEnvironment;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLines;
import com.google.devtools.build.lib.actions.EmptyRunfilesSupplier;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.ParamFileInfo;
import com.google.devtools.build.lib.actions.ResourceSetOrBuilder;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.config.CoreOptionConverters.StrictDepsMode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.rules.java.JavaCompileAction.ProgressMessage;
import com.google.devtools.build.lib.rules.java.JavaConfiguration.JavaClasspathMode;
import com.google.devtools.build.lib.rules.java.JavaPluginInfo.JavaPluginData;
import com.google.devtools.build.lib.util.OnDemandString;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.proto.Deps;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.stream.Stream;
import javax.annotation.Nullable;

/**
 * Action for Java header compilation, to be used if --java_header_compilation is enabled.
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
public final class JavaHeaderCompileAction extends SpawnAction {

  private final boolean insertDependencies;

  private JavaHeaderCompileAction(
      ActionOwner owner,
      NestedSet<Artifact> tools,
      NestedSet<Artifact> inputs,
      Iterable<? extends Artifact> outputs,
      ResourceSetOrBuilder resourceSetOrBuilder,
      CommandLines commandLines,
      ActionEnvironment env,
      ImmutableMap<String, String> executionInfo,
      CharSequence progressMessage,
      RunfilesSupplier runfilesSupplier,
      String mnemonic,
      boolean stripOutputPaths,
      boolean insertDependencies) {
    super(
        owner,
        tools,
        inputs,
        outputs,
        resourceSetOrBuilder,
        commandLines,
        env,
        executionInfo,
        progressMessage,
        runfilesSupplier,
        mnemonic,
        stripOutputPaths);
    this.insertDependencies = insertDependencies;
  }

  @Override
  protected void afterExecute(ActionExecutionContext context, List<SpawnResult> spawnResults) {
    SpawnResult spawnResult = Iterables.getOnlyElement(spawnResults);
    Artifact outputDepsProto = Iterables.get(getOutputs(), 1);
    try {
      Deps.Dependencies fullOutputDeps =
          JavaCompileAction.createFullOutputDeps(
              spawnResult, outputDepsProto, getInputs(), context, stripOutputPaths);
      JavaCompileActionContext javaContext = context.getContext(JavaCompileActionContext.class);
      if (insertDependencies && javaContext != null) {
        javaContext.insertDependencies(outputDepsProto, fullOutputDeps);
      }
    } catch (IOException e) {
      // Left empty. If we cannot read the .jdeps file now, we will read it later or throw an
      // appropriate error then.
    }
  }

  public static Builder newBuilder(RuleContext ruleContext) {
    return new Builder(ruleContext);
  }

  /** Builder for {@link JavaHeaderCompileAction}. */
  public static final class Builder {

    private static final ParamFileInfo PARAM_FILE_INFO =
        ParamFileInfo.builder(UNQUOTED).setCharset(ISO_8859_1).build();

    private final RuleContext ruleContext;

    private Artifact outputJar;
    // Only non-null before set.
    private Artifact outputDepsProto;
    @Nullable private Artifact manifestOutput;
    @Nullable private Artifact gensrcOutputJar;
    @Nullable private Artifact resourceOutputJar;
    private ImmutableSet<Artifact> additionalOutputs = ImmutableSet.of();
    private ImmutableSet<Artifact> sourceFiles = ImmutableSet.of();
    private ImmutableList<Artifact> sourceJars = ImmutableList.of();
    private NestedSet<Artifact> classpathEntries =
        NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
    private NestedSet<Artifact> bootclasspathEntries =
        NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
    @Nullable private Label targetLabel;
    @Nullable private String injectingRuleKind;
    private StrictDepsMode strictJavaDeps = StrictDepsMode.OFF;
    private NestedSet<Artifact> directJars = NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
    private NestedSet<Artifact> compileTimeDependencyArtifacts =
        NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    private ImmutableList<String> javacOpts = ImmutableList.of();
    private boolean addTurbineHjarJavacOpt = false;
    private JavaPluginData plugins = JavaPluginData.empty();

    private ImmutableList<Artifact> additionalInputs = ImmutableList.of();
    private NestedSet<Artifact> toolsJars = NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);

    private boolean enableHeaderCompilerDirect = true;

    private boolean enableDirectClasspath = true;

    private Builder(RuleContext ruleContext) {
      this.ruleContext = ruleContext;
    }

    /** Sets the output jdeps file. */
    @CanIgnoreReturnValue
    public Builder setOutputDepsProto(Artifact outputDepsProto) {
      this.outputDepsProto = checkNotNull(outputDepsProto);
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setManifestOutput(@Nullable Artifact manifestOutput) {
      this.manifestOutput = manifestOutput;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setGensrcOutputJar(@Nullable Artifact gensrcOutputJar) {
      this.gensrcOutputJar = gensrcOutputJar;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setResourceOutputJar(@Nullable Artifact resourceOutputJar) {
      this.resourceOutputJar = resourceOutputJar;
      return this;
    }

    /** Sets the direct dependency artifacts. */
    @CanIgnoreReturnValue
    public Builder setDirectJars(NestedSet<Artifact> directJars) {
      checkNotNull(directJars, "directJars must not be null");
      this.directJars = directJars;
      return this;
    }

    /** Sets the .jdeps artifacts for direct dependencies. */
    @CanIgnoreReturnValue
    public Builder setCompileTimeDependencyArtifacts(NestedSet<Artifact> dependencyArtifacts) {
      checkNotNull(dependencyArtifacts, "dependencyArtifacts must not be null");
      this.compileTimeDependencyArtifacts = dependencyArtifacts;
      return this;
    }

    /** Sets Java compiler flags. */
    @CanIgnoreReturnValue
    public Builder setJavacOpts(ImmutableList<String> javacOpts) {
      this.javacOpts = checkNotNull(javacOpts);
      return this;
    }

    /**
     * Adds {@code -Aexperimental_turbine_hjar} to Java compiler flags without creating an entirely
     * new list.
     */
    @CanIgnoreReturnValue
    public Builder addTurbineHjarJavacOpt() {
      this.addTurbineHjarJavacOpt = true;
      return this;
    }

    /** Sets the output jar. */
    @CanIgnoreReturnValue
    public Builder setOutputJar(Artifact outputJar) {
      checkNotNull(outputJar, "outputJar must not be null");
      this.outputJar = outputJar;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setAdditionalOutputs(ImmutableSet<Artifact> outputs) {
      checkNotNull(outputs, "outputs must not be null");
      this.additionalOutputs = outputs;
      return this;
    }

    /** Adds Java source files to compile. */
    @CanIgnoreReturnValue
    public Builder setSourceFiles(ImmutableSet<Artifact> sourceFiles) {
      checkNotNull(sourceFiles, "sourceFiles must not be null");
      this.sourceFiles = sourceFiles;
      return this;
    }

    /** Adds a jar archive of Java sources to compile. */
    @CanIgnoreReturnValue
    public Builder setSourceJars(ImmutableList<Artifact> sourceJars) {
      checkNotNull(sourceJars, "sourceJars must not be null");
      this.sourceJars = sourceJars;
      return this;
    }

    /** Sets the compilation classpath entries. */
    @CanIgnoreReturnValue
    public Builder setClasspathEntries(NestedSet<Artifact> classpathEntries) {
      checkNotNull(classpathEntries, "classpathEntries must not be null");
      this.classpathEntries = classpathEntries;
      return this;
    }

    /** Sets the compilation bootclasspath entries. */
    @CanIgnoreReturnValue
    public Builder setBootclasspathEntries(NestedSet<Artifact> bootclasspathEntries) {
      checkNotNull(bootclasspathEntries, "bootclasspathEntries must not be null");
      this.bootclasspathEntries = bootclasspathEntries;
      return this;
    }

    /** Sets the annotation processors classpath entries. */
    @CanIgnoreReturnValue
    public Builder setPlugins(JavaPluginData plugins) {
      checkNotNull(plugins, "plugins must not be null");
      checkState(this.plugins.isEmpty());
      this.plugins = plugins;
      return this;
    }

    /** Sets the label of the target being compiled. */
    @CanIgnoreReturnValue
    public Builder setTargetLabel(@Nullable Label targetLabel) {
      this.targetLabel = targetLabel;
      return this;
    }

    /** Sets the injecting rule kind of the target being compiled. */
    @CanIgnoreReturnValue
    public Builder setInjectingRuleKind(@Nullable String injectingRuleKind) {
      this.injectingRuleKind = injectingRuleKind;
      return this;
    }

    /** Sets the Strict Java Deps mode. */
    @CanIgnoreReturnValue
    public Builder setStrictJavaDeps(StrictDepsMode strictJavaDeps) {
      checkNotNull(strictJavaDeps, "strictJavaDeps must not be null");
      this.strictJavaDeps = strictJavaDeps;
      return this;
    }

    /** Sets additional inputs, e.g. for databinding support. */
    @CanIgnoreReturnValue
    public Builder setAdditionalInputs(ImmutableList<Artifact> additionalInputs) {
      checkNotNull(additionalInputs, "additionalInputs must not be null");
      this.additionalInputs = additionalInputs;
      return this;
    }

    /** Sets the tools jars. */
    @CanIgnoreReturnValue
    public Builder setToolsJars(NestedSet<Artifact> toolsJars) {
      checkNotNull(toolsJars, "toolsJars must not be null");
      this.toolsJars = toolsJars;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder enableHeaderCompilerDirect(boolean enableHeaderCompilerDirect) {
      this.enableHeaderCompilerDirect = enableHeaderCompilerDirect;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder enableDirectClasspath(boolean enableDirectClasspath) {
      this.enableDirectClasspath = enableDirectClasspath;
      return this;
    }

    /** Builds and registers the action for a header compilation. */
    public void build(JavaToolchainProvider javaToolchain) {
      checkNotNull(outputDepsProto, "outputDepsProto must not be null");
      checkNotNull(sourceFiles, "sourceFiles must not be null");
      checkNotNull(sourceJars, "sourceJars must not be null");
      checkNotNull(classpathEntries, "classpathEntries must not be null");
      checkNotNull(bootclasspathEntries, "bootclasspathEntries must not be null");
      checkNotNull(strictJavaDeps, "strictJavaDeps must not be null");
      checkNotNull(directJars, "directJars must not be null");
      checkNotNull(
          compileTimeDependencyArtifacts, "compileTimeDependencyArtifacts must not be null");

      // Invariant: if strictJavaDeps is OFF, then directJars and
      // dependencyArtifacts are ignored
      if (strictJavaDeps == StrictDepsMode.OFF) {
        directJars = NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
        compileTimeDependencyArtifacts = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
      }

      // Enable the direct classpath optimization if there are no annotation processors.
      // N.B. we only check if the processor classes are empty, we don't care if there is plugin
      // data or dependencies if there are no annotation processors to run. This differs from
      // javac where java_plugin may be used with processor_class unset to declare Error Prone
      // plugins.
      boolean useDirectClasspath = enableDirectClasspath && plugins.processorClasses().isEmpty();

      // Use the optimized 'direct' implementation if it is available, and either there are no
      // annotation processors or they are built in to the tool and listed in
      // java_toolchain.header_compiler_direct_processors.
      ImmutableSet<String> processorClasses = plugins.processorClasses().toSet();
      boolean useHeaderCompilerDirect =
          enableHeaderCompilerDirect
              && javaToolchain.getHeaderCompilerDirect() != null
              && javaToolchain.getHeaderCompilerBuiltinProcessors().containsAll(processorClasses);
      JavaConfiguration javaConfiguration =
          ruleContext.getConfiguration().getFragment(JavaConfiguration.class);
      JavaClasspathMode classpathMode = javaConfiguration.getReduceJavaClasspath();
      if (!Collections.disjoint(
          processorClasses, javaToolchain.getReducedClasspathIncompatibleProcessors())) {
        classpathMode = JavaClasspathMode.OFF;
      }

      ActionEnvironment actionEnvironment =
          ruleContext
              .getConfiguration()
              .getActionEnvironment()
              .withAdditionalFixedVariables(UTF8_ENVIRONMENT);

      OnDemandString progressMessage =
          new ProgressMessage(
              /* prefix= */ "Compiling Java headers",
              /* output= */ outputJar,
              /* sourceFiles= */ sourceFiles,
              /* sourceJars= */ sourceJars,
              /* plugins= */ plugins);

      ImmutableSet.Builder<Artifact> outputs =
          ImmutableSet.<Artifact>builder()
              .add(outputJar)
              .add(outputDepsProto)
              .addAll(additionalOutputs);
      Stream.of(gensrcOutputJar, resourceOutputJar, manifestOutput)
          .filter(Objects::nonNull)
          .forEachOrdered(outputs::add);

      NestedSetBuilder<Artifact> mandatoryInputsBuilder =
          NestedSetBuilder.<Artifact>stableOrder()
              .addAll(additionalInputs)
              .addTransitive(bootclasspathEntries)
              .addAll(sourceJars)
              .addAll(sourceFiles)
              .addTransitive(toolsJars);

      JavaToolchainTool headerCompiler =
          useHeaderCompilerDirect
              ? javaToolchain.getHeaderCompilerDirect()
              : javaToolchain.getHeaderCompiler();
      // The header compiler is either a jar file that needs to be executed using
      // `java -jar <path>`, or an executable that can be run directly.
      headerCompiler.addInputs(javaToolchain, mandatoryInputsBuilder);
      CustomCommandLine.Builder commandLine =
          CustomCommandLine.builder()
              .addExecPath("--output", outputJar)
              .addExecPath("--gensrc_output", gensrcOutputJar)
              .addExecPath("--resource_output", resourceOutputJar)
              .addExecPath("--output_manifest_proto", manifestOutput)
              .addExecPath("--output_deps", outputDepsProto)
              .addExecPaths("--bootclasspath", bootclasspathEntries)
              .addExecPaths("--sources", sourceFiles)
              .addExecPaths("--source_jars", sourceJars)
              .add("--injecting_rule_kind", injectingRuleKind);

      if (!javacOpts.isEmpty() || addTurbineHjarJavacOpt) {
        commandLine.add("--javacopts");
        if (!javacOpts.isEmpty()) {
          commandLine.addObject(javacOpts);
        }
        if (addTurbineHjarJavacOpt) {
          commandLine.add("-Aexperimental_turbine_hjar");
        }
        // terminate --javacopts with `--` to support javac flags that start with `--`
        commandLine.add("--");
      }

      if (targetLabel != null) {
        commandLine.add("--target_label");
        if (targetLabel.getRepository().isMain()) {
          commandLine.addLabel(targetLabel);
        } else {
          // @-prefixed strings will be assumed to be params filenames and expanded,
          // so add an extra @ to escape it.
          commandLine.addPrefixedLabel("@", targetLabel);
        }
      }

      ImmutableMap<String, String> executionInfo =
          TargetUtils.getExecutionInfo(ruleContext.getRule(), ruleContext.isAllowTagsPropagation());
      if (javaConfiguration.inmemoryJdepsFiles()) {
        executionInfo =
            ImmutableMap.of(
                ExecutionRequirements.REMOTE_EXECUTION_INLINE_OUTPUTS,
                outputDepsProto.getExecPathString());
      }
      if (useDirectClasspath) {
        NestedSet<Artifact> classpath;
        if (!directJars.isEmpty() || classpathEntries.isEmpty()) {
          classpath = directJars;
        } else {
          classpath = classpathEntries;
        }
        mandatoryInputsBuilder.addTransitive(classpath);

        commandLine.addExecPaths("--classpath", classpath);
        commandLine.add("--reduce_classpath_mode", "NONE");

        NestedSet<Artifact> allInputs = mandatoryInputsBuilder.build();
        boolean stripOutputPaths =
            JavaCompilationHelper.stripOutputPaths(allInputs, ruleContext.getConfiguration());
        @Nullable
        PathFragment strippedOutputBase =
            stripOutputPaths ? JavaCompilationHelper.outputBase(outputJar) : null;
        CustomCommandLine executableLine =
            headerCompiler.getCommandLine(javaToolchain, strippedOutputBase);

        if (strippedOutputBase != null) {
          commandLine.stripOutputPaths(strippedOutputBase);
        }

        ruleContext.registerAction(
            new JavaHeaderCompileAction(
                /* owner= */ ruleContext.getActionOwner(),
                /* tools= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
                /* inputs= */ allInputs,
                /* outputs= */ outputs.build(),
                /* resourceSetOrBuilder= */ AbstractAction.DEFAULT_RESOURCE_SET,
                /* commandLines= */ CommandLines.builder()
                    .addCommandLine(executableLine)
                    .addCommandLine(commandLine.build(), PARAM_FILE_INFO)
                    .build(),
                /* env= */ actionEnvironment,
                /* executionInfo= */ ruleContext
                    .getConfiguration()
                    .modifiedExecutionInfo(executionInfo, "Turbine"),
                /* progressMessage= */ progressMessage,
                /* runfilesSupplier= */ EmptyRunfilesSupplier.INSTANCE,
                /* mnemonic= */ "Turbine",
                /* stripOutputPaths= */ stripOutputPaths,
                // If classPathMode == BAZEL, also make sure to inject the dependencies to be
                // available to downstream actions. Else just do enough work to locally create the
                // full .jdeps from the .stripped .jdeps produced on the executor.
                /* insertDependencies= */ classpathMode == JavaClasspathMode.BAZEL));
        return;
      }

      // If we get here the action requires annotation processing, so add additional inputs and
      // flags needed for the javac-based header compiler implementations that supports
      // annotation processing.

      if (!useHeaderCompilerDirect) {
        mandatoryInputsBuilder.addTransitive(plugins.processorClasspath());
        mandatoryInputsBuilder.addTransitive(plugins.data());
      }

      commandLine.addAll(
          "--builtin_processors",
          Sets.intersection(
              plugins.processorClasses().toSet(),
              javaToolchain.getHeaderCompilerBuiltinProcessors()));
      commandLine.addAll("--processors", plugins.processorClasses());
      if (!useHeaderCompilerDirect) {
        commandLine.addExecPaths("--processorpath", plugins.processorClasspath());
      }
      if (strictJavaDeps != StrictDepsMode.OFF) {
        commandLine.addExecPaths("--direct_dependencies", directJars);
      }

      NestedSet<Artifact> mandatoryInputs = mandatoryInputsBuilder.build();

      boolean pathStrippingEnabled =
          JavaCompilationHelper.stripOutputPaths(
              NestedSetBuilder.<Artifact>stableOrder()
                  .addTransitive(mandatoryInputs)
                  .addTransitive(classpathEntries)
                  .addTransitive(compileTimeDependencyArtifacts)
                  .build(),
              ruleContext.getConfiguration());

      PathFragment strippedOutputBase =
          pathStrippingEnabled ? JavaCompilationHelper.outputBase(outputJar) : null;
      if (strippedOutputBase != null) {
        commandLine.stripOutputPaths(strippedOutputBase);
      }
      CustomCommandLine executableLine =
          headerCompiler.getCommandLine(javaToolchain, strippedOutputBase);

      ruleContext.registerAction(
          new JavaCompileAction(
              /* compilationType= */ JavaCompileAction.CompilationType.TURBINE,
              /* owner= */ ruleContext.getActionOwner(),
              /* tools= */ toolsJars,
              /* progressMessage= */ progressMessage,
              /* mandatoryInputs= */ mandatoryInputs,
              /* transitiveInputs= */ classpathEntries,
              /* directJars= */ directJars,
              /* outputs= */ outputs.build(),
              /* executionInfo= */ executionInfo,
              /* extraActionInfoSupplier= */ null,
              /* executableLine= */ executableLine,
              /* flagLine= */ commandLine.build(),
              /* configuration= */ ruleContext.getConfiguration(),
              /* dependencyArtifacts= */ compileTimeDependencyArtifacts,
              /* outputDepsProto= */ outputDepsProto,
              /* classpathMode= */ classpathMode));
    }
  }
}
