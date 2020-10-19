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
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLines;
import com.google.devtools.build.lib.actions.CompositeRunfilesSupplier;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.ParamFileInfo;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
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
import com.google.devtools.build.lib.rules.java.JavaPluginInfoProvider.JavaPluginInfo;
import com.google.devtools.build.lib.util.LazyString;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.view.proto.Deps;
import com.google.protobuf.ExtensionRegistry;
import java.io.IOException;
import java.io.InputStream;
import java.io.Serializable;
import java.util.Collections;
import java.util.List;
import java.util.function.Consumer;
import java.util.stream.Stream;
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

  private static final ParamFileInfo PARAM_FILE_INFO =
      ParamFileInfo.builder(UNQUOTED).setCharset(ISO_8859_1).build();

  private final RuleContext ruleContext;

  private Artifact outputJar;
  @Nullable private Artifact outputDepsProto;
  @Nullable private Artifact manifestOutput;
  @Nullable private Artifact gensrcOutputJar;
  @Nullable private Artifact resourceOutputJar;
  private ImmutableSet<Artifact> additionalOutputs = ImmutableSet.of();
  private ImmutableSet<Artifact> sourceFiles = ImmutableSet.of();
  private ImmutableList<Artifact> sourceJars = ImmutableList.of();
  private NestedSet<Artifact> classpathEntries = NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
  private NestedSet<Artifact> bootclasspathEntries =
      NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
  @Nullable private Label targetLabel;
  @Nullable private String injectingRuleKind;
  private StrictDepsMode strictJavaDeps = StrictDepsMode.OFF;
  private NestedSet<Artifact> directJars = NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
  private NestedSet<Artifact> compileTimeDependencyArtifacts =
      NestedSetBuilder.emptySet(Order.STABLE_ORDER);
  private final ImmutableList.Builder<String> javacOptsBuilder = ImmutableList.builder();
  private JavaPluginInfo plugins = JavaPluginInfo.empty();

  private NestedSet<Artifact> additionalInputs = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
  private NestedSet<Artifact> toolsJars = NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);

  private boolean enableHeaderCompilerDirect = true;

  public JavaHeaderCompileActionBuilder(RuleContext ruleContext) {
    this.ruleContext = ruleContext;
  }

  /** Sets the output jdeps file. */
  public JavaHeaderCompileActionBuilder setOutputDepsProto(@Nullable Artifact outputDepsProto) {
    this.outputDepsProto = outputDepsProto;
    return this;
  }

  public JavaHeaderCompileActionBuilder setManifestOutput(@Nullable Artifact manifestOutput) {
    this.manifestOutput = manifestOutput;
    return this;
  }

  public JavaHeaderCompileActionBuilder setGensrcOutputJar(@Nullable Artifact gensrcOutputJar) {
    this.gensrcOutputJar = gensrcOutputJar;
    return this;
  }

  public JavaHeaderCompileActionBuilder setResourceOutputJar(@Nullable Artifact resourceOutputJar) {
    this.resourceOutputJar = resourceOutputJar;
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

  /** Adds Java compiler flags. */
  public JavaHeaderCompileActionBuilder addAllJavacOpts(Iterable<String> javacOpts) {
    this.javacOptsBuilder.addAll(javacOpts);
    return this;
  }

  /** Adds a Java compiler flag. */
  public JavaHeaderCompileActionBuilder addJavacOpt(String javacOpt) {
    this.javacOptsBuilder.add(javacOpt);
    return this;
  }

  /** Sets the output jar. */
  public JavaHeaderCompileActionBuilder setOutputJar(Artifact outputJar) {
    checkNotNull(outputJar, "outputJar must not be null");
    this.outputJar = outputJar;
    return this;
  }

  public JavaHeaderCompileActionBuilder setAdditionalOutputs(ImmutableSet<Artifact> outputs) {
    checkNotNull(outputs, "outputs must not be null");
    this.additionalOutputs = outputs;
    return this;
  }

  /** Adds Java source files to compile. */
  public JavaHeaderCompileActionBuilder setSourceFiles(ImmutableSet<Artifact> sourceFiles) {
    checkNotNull(sourceFiles, "sourceFiles must not be null");
    this.sourceFiles = sourceFiles;
    return this;
  }

  /** Adds a jar archive of Java sources to compile. */
  public JavaHeaderCompileActionBuilder setSourceJars(ImmutableList<Artifact> sourceJars) {
    checkNotNull(sourceJars, "sourceJars must not be null");
    this.sourceJars = sourceJars;
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
      NestedSet<Artifact> bootclasspathEntries) {
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

  /** Sets the Strict Java Deps mode. */
  public JavaHeaderCompileActionBuilder setStrictJavaDeps(StrictDepsMode strictJavaDeps) {
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

  /** Sets the tools jars. */
  public JavaHeaderCompileActionBuilder setToolsJars(NestedSet<Artifact> toolsJars) {
    checkNotNull(toolsJars, "toolsJars must not be null");
    this.toolsJars = toolsJars;
    return this;
  }

  public JavaHeaderCompileActionBuilder enableHeaderCompilerDirect(
      boolean enableHeaderCompilerDirect) {
    this.enableHeaderCompilerDirect = enableHeaderCompilerDirect;
    return this;
  }

  /** Builds and registers the action for a header compilation. */
  public void build(JavaToolchainProvider javaToolchain) throws InterruptedException {
    checkNotNull(outputDepsProto, "outputDepsProto must not be null");
    checkNotNull(sourceFiles, "sourceFiles must not be null");
    checkNotNull(sourceJars, "sourceJars must not be null");
    checkNotNull(classpathEntries, "classpathEntries must not be null");
    checkNotNull(bootclasspathEntries, "bootclasspathEntries must not be null");
    checkNotNull(strictJavaDeps, "strictJavaDeps must not be null");
    checkNotNull(directJars, "directJars must not be null");
    checkNotNull(compileTimeDependencyArtifacts, "compileTimeDependencyArtifacts must not be null");

    ImmutableList<String> javacOpts = javacOptsBuilder.build();

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
    boolean useDirectClasspath = plugins.processorClasses().isEmpty();

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
        ruleContext.getConfiguration().getActionEnvironment().addFixedVariables(UTF8_ENVIRONMENT);

    LazyString progressMessage =
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
        .filter(x -> x != null)
        .forEachOrdered(outputs::add);

    NestedSetBuilder<Artifact> mandatoryInputs =
        NestedSetBuilder.<Artifact>stableOrder()
            .addTransitive(additionalInputs)
            .addTransitive(bootclasspathEntries)
            .addAll(sourceJars)
            .addAll(sourceFiles)
            .addTransitive(toolsJars);

    ImmutableList<RunfilesSupplier> runfilesSuppliers = ImmutableList.of();
    FilesToRunProvider headerCompiler =
        useHeaderCompilerDirect
            ? javaToolchain.getHeaderCompilerDirect()
            : javaToolchain.getHeaderCompiler();
    // The header compiler is either a jar file that needs to be executed using
    // `java -jar <path>`, or an executable that can be run directly.
    CustomCommandLine executableLine;
    if (!headerCompiler.getExecutable().getExtension().equals("jar")) {
      runfilesSuppliers = ImmutableList.of(headerCompiler.getRunfilesSupplier());
      mandatoryInputs.addTransitive(headerCompiler.getFilesToRun());
      executableLine =
          CustomCommandLine.builder().addExecPath(headerCompiler.getExecutable()).build();
    } else {
      mandatoryInputs
          .addTransitive(javaToolchain.getJavaRuntime().javaBaseInputsMiddleman())
          .add(headerCompiler.getExecutable());
      executableLine =
          CustomCommandLine.builder()
              .addPath(javaToolchain.getJavaRuntime().javaBinaryExecPathFragment())
              .add("-Xverify:none")
              .addAll(javaToolchain.getTurbineJvmOptions())
              .add("-jar")
              .addExecPath(headerCompiler.getExecutable())
              .build();
    }

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

    ImmutableMap<String, String> executionInfo =
        TargetUtils.getExecutionInfo(ruleContext.getRule(), ruleContext.isAllowTagsPropagation());
    Consumer<Pair<ActionExecutionContext, List<SpawnResult>>> resultConsumer = null;
    if (classpathMode == JavaClasspathMode.BAZEL) {
      if (javaConfiguration.inmemoryJdepsFiles()) {
        executionInfo =
            ImmutableMap.of(
                ExecutionRequirements.REMOTE_EXECUTION_INLINE_OUTPUTS,
                outputDepsProto.getExecPathString());
      }
      resultConsumer = createResultConsumer(outputDepsProto);
    }

    if (useDirectClasspath) {
      NestedSet<Artifact> classpath;
      if (!directJars.isEmpty() || classpathEntries.isEmpty()) {
        classpath = directJars;
      } else {
        classpath = classpathEntries;
      }
      mandatoryInputs.addTransitive(classpath);

      commandLine.addExecPaths("--classpath", classpath);
      commandLine.add("--reduce_classpath_mode", "NONE");

      ruleContext.registerAction(
          new SpawnAction(
              /* owner= */ ruleContext.getActionOwner(),
              /* tools= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
              /* inputs= */ mandatoryInputs.build(),
              /* outputs= */ outputs.build(),
              /* primaryOutput= */ outputJar,
              /* resourceSet= */ AbstractAction.DEFAULT_RESOURCE_SET,
              /* commandLines= */ CommandLines.builder()
                  .addCommandLine(executableLine)
                  .addCommandLine(commandLine.build(), PARAM_FILE_INFO)
                  .build(),
              /* commandLineLimits= */ ruleContext.getConfiguration().getCommandLineLimits(),
              /* isShellCommand= */ false,
              /* env= */ actionEnvironment,
              /* executionInfo= */ ruleContext
                  .getConfiguration()
                  .modifiedExecutionInfo(executionInfo, "Turbine"),
              /* progressMessage= */ progressMessage,
              /* runfilesSupplier= */ CompositeRunfilesSupplier.fromSuppliers(runfilesSuppliers),
              /* mnemonic= */ "Turbine",
              /* executeUnconditionally= */ false,
              /* extraActionInfoSupplier= */ null,
              /* resultConsumer= */ resultConsumer));
      return;
    }

    // If we get here the action requires annotation processing, so add additional inputs and
    // flags needed for the javac-based header compiler implementations that supports
    // annotation processing.

    if (!useHeaderCompilerDirect) {
      mandatoryInputs.addTransitive(plugins.processorClasspath());
      mandatoryInputs.addTransitive(plugins.data());
    }
    mandatoryInputs.addTransitive(compileTimeDependencyArtifacts);

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

    ruleContext.registerAction(
        new JavaCompileAction(
            /* compilationType= */ JavaCompileAction.CompilationType.TURBINE,
            /* owner= */ ruleContext.getActionOwner(),
            /* env= */ actionEnvironment,
            /* tools= */ toolsJars,
            /* runfilesSupplier= */ CompositeRunfilesSupplier.fromSuppliers(runfilesSuppliers),
            /* progressMessage= */ progressMessage,
            /* mandatoryInputs= */ mandatoryInputs.build(),
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

  /**
   * Creates a consumer that reads the produced .jdeps file into memory. Pulled out into a separate
   * function to avoid capturing a data member, which would keep the entire builder instance alive.
   */
  private static Consumer<Pair<ActionExecutionContext, List<SpawnResult>>> createResultConsumer(
      Artifact outputDepsProto) {
    return (Consumer<Pair<ActionExecutionContext, List<SpawnResult>>> & Serializable)
        contextAndResults -> {
          ActionExecutionContext context = contextAndResults.getFirst();
          JavaCompileActionContext javaContext = context.getContext(JavaCompileActionContext.class);
          if (javaContext == null) {
            return;
          }
          SpawnResult spawnResult = Iterables.getOnlyElement(contextAndResults.getSecond());
          try {
            InputStream inMemoryOutput = spawnResult.getInMemoryOutput(outputDepsProto);
            try (InputStream input =
                inMemoryOutput == null
                    ? context.getInputPath(outputDepsProto).getInputStream()
                    : inMemoryOutput) {
              javaContext.insertDependencies(
                  outputDepsProto,
                  Deps.Dependencies.parseFrom(input, ExtensionRegistry.getEmptyRegistry()));
            }
          } catch (IOException e) {
            // Left empty. If we cannot read the .jdeps file now, we will read it later or throw
            // an appropriate error then.
          }
        };
  }
}
