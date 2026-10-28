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

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Ascii;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionEnvironment;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.Expander;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.CoreOptionConverters.StrictDepsMode;
import com.google.devtools.build.lib.analysis.configuredtargets.AbstractConfiguredTarget;
import com.google.devtools.build.lib.analysis.configuredtargets.MergedConfiguredTarget;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.analysis.starlark.Args;
import com.google.devtools.build.lib.analysis.starlark.StarlarkActionFactory;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.Depset.TypeException;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.BuiltinRestriction;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.StarlarkInfo;
import com.google.devtools.build.lib.packages.StarlarkInfoNoSchema;
import com.google.devtools.build.lib.packages.StarlarkInfoWithSchema;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.rules.cpp.CppFileTypes;
import com.google.devtools.build.lib.rules.java.JavaConfiguration.JavaClasspathMode;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.java.JavaCommonApi;
import com.google.devtools.build.lib.util.OnDemandString;
import javax.annotation.Nullable;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkThread;

/** A module that contains Starlark utilities for Java support. */
public class JavaStarlarkCommon
    implements JavaCommonApi<
        Artifact,
        ConstraintValueInfo,
        StarlarkRuleContext,
        StarlarkActionFactory> {

  private final JavaSemantics javaSemantics;

  private static StrictDepsMode getStrictDepsMode(String strictDepsMode) {
    switch (strictDepsMode) {
      case "OFF":
        return StrictDepsMode.OFF;
      case "ERROR":
      case "DEFAULT":
        return StrictDepsMode.ERROR;
      case "WARN":
        return StrictDepsMode.WARN;
      default:
        throw new IllegalArgumentException(
            "StrictDepsMode "
                + strictDepsMode
                + " not allowed."
                + " Only OFF and ERROR values are accepted.");
    }
  }

  private void checkJavaToolchainIsDeclaredOnRule(RuleContext ruleContext)
      throws EvalException, LabelSyntaxException {
    ToolchainInfo toolchainInfo =
        ruleContext.getToolchainInfo(Label.parseCanonical(javaSemantics.getJavaToolchainType()));
    if (toolchainInfo == null) {
      String ruleLocation = ruleContext.getRule().getLocation().toString();
      String ruleClass = ruleContext.getRule().getRuleClassObject().getName();
      throw Starlark.errorf(
          "Rule '%s' in '%s' must declare '%s' toolchain in order to use java_common. See"
              + " https://github.com/bazelbuild/bazel/issues/18970.",
          ruleClass, ruleLocation, javaSemantics.getJavaToolchainType());
    }
  }

  @Override
  public void checkJavaToolchainIsDeclaredOnRuleForStarlark(
      StarlarkActionFactory actions, StarlarkThread thread)
      throws EvalException, LabelSyntaxException {
    checkPrivateAccess(thread);
    checkJavaToolchainIsDeclaredOnRule(actions.getRuleContext());
  }

  public JavaStarlarkCommon(JavaSemantics javaSemantics) {
    this.javaSemantics = javaSemantics;
  }

  @Override
  public void createHeaderCompilationAction(
      StarlarkRuleContext ctx,
      Info toolchain,
      Artifact headerJar,
      Artifact headerDepsProto,
      Info pluginInfo,
      Depset sourceFiles,
      Sequence<?> sourceJars,
      Depset compileTimeClasspath,
      Depset directJars,
      Object bootClassPathUnchecked,
      Depset compileTimeJavaDeps,
      Depset javacOpts,
      String strictDepsMode,
      Label targetLabel,
      Object injectingRuleKind,
      boolean enableDirectClasspath,
      Sequence<?> additionalInputs,
      Artifact headerCompilationJar,
      Depset headerCompilationDirectDeps)
      throws EvalException,
          TypeException,
          RuleErrorException,
          LabelSyntaxException,
          InterruptedException {
    checkJavaToolchainIsDeclaredOnRule(ctx.getRuleContext());
    JavaTargetAttributes.Builder attributesBuilder =
        new JavaTargetAttributes.Builder()
            .addSourceJars(Sequence.cast(sourceJars, Artifact.class, "source_jars"))
            .addSourceFiles(sourceFiles.toList(Artifact.class))
            .addDirectJars(directJars.getSet(Artifact.class))
            .addHeaderCompilationDirectJars(headerCompilationDirectDeps.getSet(Artifact.class))
            .setCompileTimeClassPathEntriesWithPrependedDirectJars(
                compileTimeClasspath.getSet(Artifact.class))
            .setStrictJavaDeps(getStrictDepsMode(Ascii.toUpperCase(strictDepsMode)))
            .setTargetLabel(targetLabel)
            .setInjectingRuleKind(
                injectingRuleKind == Starlark.NONE ? null : (String) injectingRuleKind)
            .addPlugin(JavaPluginInfo.wrap(pluginInfo))
            .addCompileTimeDependencyArtifacts(compileTimeJavaDeps.getSet(Artifact.class));
    if (bootClassPathUnchecked instanceof Info) {
      BootClassPathInfo bootClassPathInfo = BootClassPathInfo.wrap((Info) bootClassPathUnchecked);
      if (!bootClassPathInfo.isEmpty()) {
        attributesBuilder.setBootClassPath(bootClassPathInfo);
      }
    }
    JavaCompilationHelper compilationHelper =
        new JavaCompilationHelper(
            ctx.getRuleContext(),
            javaSemantics,
            JavaHelper.tokenizeJavaOptions(Depset.cast(javacOpts, String.class, "javac_opts")),
            attributesBuilder,
            JavaToolchainProvider.wrap(toolchain),
            Sequence.cast(additionalInputs, Artifact.class, "additional_inputs")
                .getImmutableList());
    compilationHelper.enableDirectClasspath(enableDirectClasspath);
    compilationHelper.createHeaderCompilationAction(
        headerJar, headerCompilationJar, headerDepsProto);
  }

  @Override
  public void createCompilationAction(
      StarlarkRuleContext ctx,
      Info javaToolchain,
      Artifact output,
      Artifact manifestProto,
      Info pluginInfo,
      Depset compileTimeClasspath,
      Depset directJars,
      Object bootClassPathUnchecked,
      Depset javaBuilderJvmFlags,
      Depset compileTimeJavaDeps,
      Depset javacOpts,
      String strictDepsMode,
      Label targetLabel,
      Object depsProto,
      Object genClass,
      Object genSource,
      Object nativeHeader,
      Object sourceFiles,
      Sequence<?> sourceJars,
      Sequence<?> resources,
      Object resourceJars,
      Sequence<?> classpathResources,
      Sequence<?> sourcepath,
      Object injectingRuleKind,
      boolean enableJSpecify,
      boolean enableDirectClasspath,
      Sequence<?> additionalInputs,
      Sequence<?> additionalOutputs,
      Sequence<?> extraArgs)
      throws EvalException,
          TypeException,
          RuleErrorException,
          LabelSyntaxException,
          InterruptedException {
    checkJavaToolchainIsDeclaredOnRule(ctx.getRuleContext());
    JavaCompileOutputs<Artifact> outputs =
        JavaCompileOutputs.builder()
            .output(output)
            .depsProto(depsProto == Starlark.NONE ? null : (Artifact) depsProto)
            .genClass(genClass == Starlark.NONE ? null : (Artifact) genClass)
            .genSource(genSource == Starlark.NONE ? null : (Artifact) genSource)
            .nativeHeader(nativeHeader == Starlark.NONE ? null : (Artifact) nativeHeader)
            .manifestProto(manifestProto)
            .build();
    ImmutableList.Builder<CommandLine> extraCommandLineArgs = ImmutableList.builder();
    for (Args args : Sequence.cast(extraArgs, Args.class, "extra_args")) {
      extraCommandLineArgs.add(
          args.build(ctx.getRuleContext().getAnalysisEnvironment()::getMainRepoMapping));
    }
    JavaTargetAttributes.Builder attributesBuilder =
        new JavaTargetAttributes.Builder()
            .addSourceJars(Sequence.cast(sourceJars, Artifact.class, "source_jars"))
            .addSourceFiles(Depset.noneableCast(sourceFiles, Artifact.class, "sources").toList())
            .addDirectJars(directJars.getSet(Artifact.class))
            .setCompileTimeClassPathEntriesWithPrependedDirectJars(
                compileTimeClasspath.getSet(Artifact.class))
            .addClassPathResources(
                Sequence.cast(classpathResources, Artifact.class, "classpath_resources"))
            .setStrictJavaDeps(getStrictDepsMode(Ascii.toUpperCase(strictDepsMode)))
            .setTargetLabel(targetLabel)
            .setInjectingRuleKind(
                injectingRuleKind == Starlark.NONE ? null : (String) injectingRuleKind)
            .setSourcePath(
                Sequence.cast(sourcepath, Artifact.class, "source_path").getImmutableList())
            .addPlugin(JavaPluginInfo.wrap(pluginInfo))
            .addAdditionalOutputs(
                Sequence.cast(additionalOutputs, Artifact.class, "additional_outputs"));
    if (bootClassPathUnchecked instanceof Info) {
      BootClassPathInfo bootClassPathInfo = BootClassPathInfo.wrap((Info) bootClassPathUnchecked);
      if (!bootClassPathInfo.isEmpty()) {
        attributesBuilder.setBootClassPath(bootClassPathInfo);
      }
    }
    for (Artifact resource : Sequence.cast(resources, Artifact.class, "resources")) {
      attributesBuilder.addResource(
          JavaHelper.getJavaResourcePath(javaSemantics, ctx.getRuleContext(), resource), resource);
    }
    attributesBuilder.addResourceJars(
        Depset.noneableCast(resourceJars, Artifact.class, "resource_jars"));
    attributesBuilder.addCompileTimeDependencyArtifacts(compileTimeJavaDeps.getSet(Artifact.class));
    JavaCompilationHelper compilationHelper =
        new JavaCompilationHelper(
            ctx.getRuleContext(),
            javaSemantics,
            JavaHelper.tokenizeJavaOptions(Depset.cast(javacOpts, String.class, "javac_opts")),
            attributesBuilder,
            JavaToolchainProvider.wrap(javaToolchain),
            Sequence.cast(additionalInputs, Artifact.class, "additional_inputs")
                .getImmutableList());
    compilationHelper.javaBuilderJvmFlags(
        Depset.cast(javaBuilderJvmFlags, String.class, "javabuilder_jvm_flags"));
    compilationHelper.enableJspecify(enableJSpecify);
    compilationHelper.enableDirectClasspath(enableDirectClasspath);
    compilationHelper.setExtraCommandLineArgs(extraCommandLineArgs.build());
    compilationHelper.createCompileAction(outputs);
  }

  @Override
  public String getTargetKind(Object target, StarlarkThread thread) throws EvalException {
    checkPrivateAccess(thread);
    if (target instanceof MergedConfiguredTarget mergedConfiguredTarget) {
      target = mergedConfiguredTarget.getBaseConfiguredTarget();
    }
    if (target instanceof ConfiguredTarget configuredTarget) {
      target = configuredTarget.getActual();
    }
    if (target instanceof AbstractConfiguredTarget abstractConfiguredTarget) {
      return abstractConfiguredTarget.getRuleClassString();
    }
    return "";
  }

  protected static void checkPrivateAccess(StarlarkThread thread) throws EvalException {
    BuiltinRestriction.failIfCalledOutsideDefaultAllowlist(thread);
  }

  private static ImmutableList<Artifact> getDynamicLibrariesForLinking(
      NestedSet<StarlarkInfo> libraries) {
    ImmutableList.Builder<Artifact> dynamicLibrariesForLinkingBuilder = ImmutableList.builder();
    for (StarlarkInfo libraryToLink : libraries.toList()) {
      if (libraryToLink.getValue("interface_library") instanceof Artifact artifact) {
        dynamicLibrariesForLinkingBuilder.add(artifact);
      } else if (libraryToLink.getValue("dynamic_library") instanceof Artifact artifact) {
        dynamicLibrariesForLinkingBuilder.add(artifact);
      }
    }
    return dynamicLibrariesForLinkingBuilder.build();
  }

  @Override
  public Sequence<String> collectNativeLibsDirs(Depset libraries, StarlarkThread thread)
      throws EvalException, TypeException {
    checkPrivateAccess(thread);
    ImmutableList<Artifact> nativeLibraries =
        getDynamicLibrariesForLinking(libraries.getSet(StarlarkInfo.class));
    ImmutableList<String> uniqueDirs =
        nativeLibraries.stream()
            .filter(
                nativeLibrary -> {
                  String name = nativeLibrary.getFilename();
                  if (CppFileTypes.INTERFACE_SHARED_LIBRARY.matches(name)) {
                    return false;
                  }
                  if (!(CppFileTypes.SHARED_LIBRARY.matches(name)
                      || CppFileTypes.VERSIONED_SHARED_LIBRARY.matches(name))) {
                    throw new IllegalArgumentException(
                        "not a shared library :" + nativeLibrary.prettyPrint());
                  }
                  return true;
                })
            .map(artifact -> artifact.getRunfilesPath().getParentDirectory().getPathString())
            .distinct()
            .collect(toImmutableList());
    return StarlarkList.immutableCopyOf(uniqueDirs);
  }

  @Override
  public Depset getRuntimeClasspathForArchive(
      Depset runtimeClasspath, Depset excludedArtifacts, StarlarkThread thread)
      throws EvalException, TypeException {
    checkPrivateAccess(thread);
    if (excludedArtifacts.isEmpty()) {
      return runtimeClasspath;
    } else {
      return Depset.of(
          Artifact.class,
          NestedSetBuilder.wrap(
              Order.STABLE_ORDER,
              Iterables.filter(
                  runtimeClasspath.toList(Artifact.class),
                  Predicates.not(Predicates.in(excludedArtifacts.getSet().toSet())))));
    }
  }

  @Override
  public void checkProviderInstances(
      Sequence<?> providers, String what, ProviderApi providerType, StarlarkThread thread)
      throws EvalException {
    checkPrivateAccess(thread);
    if (providerType instanceof Provider) {
      for (int i = 0; i < providers.size(); i++) {
        Object elem = providers.get(i);
        if (!isInstanceOfProvider(elem, (Provider) providerType)) {
          throw Starlark.errorf(
              "at index %d of %s, got element of type %s, want %s",
              i, what, printableType(elem), ((Provider) providerType).getPrintableName());
        }
      }
    } else {
      throw Starlark.errorf("wanted Provider, got %s", Starlark.type(providerType));
    }
  }

  @VisibleForTesting
  static String printableType(Object elem) {
    if (elem instanceof StarlarkInfoWithSchema starlarkInfoWithSchema) {
      return starlarkInfoWithSchema.getProvider().getPrintableName();
    } else if (elem instanceof NativeInfo nativeInfo) {
      return nativeInfo.getProvider().getPrintableName();
    }
    return Starlark.type(elem);
  }

  @Override
  public boolean isLegacyGoogleApiEnabled(StarlarkThread thread) throws EvalException {
    checkPrivateAccess(thread);
    return thread.getSemantics().getBool(BuildLanguageOptions.EXPERIMENTAL_GOOGLE_LEGACY_API);
  }

  @Override
  public boolean isJavaInfoMergeRuntimeModuleFlagsEnabled(StarlarkThread thread)
      throws EvalException {
    checkPrivateAccess(thread);
    return thread
        .getSemantics()
        .getBool(BuildLanguageOptions.INCOMPATIBLE_JAVA_INFO_MERGE_RUNTIME_MODULE_FLAGS);
  }

  @Override
  public boolean incompatibleDisableNonExecutableJavaBinary(StarlarkThread thread) {
    return thread
        .getSemantics()
        .getBool(BuildLanguageOptions.INCOMPATIBLE_DISABLE_NON_EXECUTABLE_JAVA_BINARY);
  }

  @Override
  public Sequence<?> expandJavaOpts(
      StarlarkRuleContext ctx, String attr, boolean tokenize, boolean execPaths)
      throws InterruptedException {
    Expander expander;
    if (execPaths) {
      expander = ctx.getRuleContext().getExpander().withExecLocations(ImmutableMap.of());
    } else {
      expander = ctx.getRuleContext().getExpander().withDataLocations();
    }
    if (tokenize) {
      return StarlarkList.immutableCopyOf(expander.tokenized(attr));
    } else {
      return StarlarkList.immutableCopyOf(expander.list(attr));
    }
  }

  @Override
  public Sequence<?> tokenizeJavacOpts(Sequence<?> opts) throws EvalException {
    return StarlarkList.immutableCopyOf(
        JavaHelper.tokenizeJavaOptions(Sequence.noneableCast(opts, String.class, "opts")));
  }

  @Override
  public void createJvmCompileAction(
      StarlarkRuleContext ctx,
      Sequence<?> outputs,
      Depset mandatoryInputs,
      Depset transitiveInputs,
      Depset directJars,
      Object executableUnchecked,
      Object toolsUnchecked,
      Sequence<?> arguments,
      String mnemonic,
      String progressMessage,
      Dict<?, ?> envUnchecked,
      Dict<?, ?> executionRequirementsUnchecked,
      Object toolchainUnchecked,
      Object outputDepsProto,
      Depset dependencyArtifacts,
      boolean insertDependencies,
      boolean enableDirectClasspath,
      boolean reduceClasspathIncompatible,
      Sequence<?> executableArgs,
      StarlarkThread thread)
      throws EvalException, RuleErrorException, InterruptedException {
    checkPrivateAccess(thread);
    RuleContext ruleContext = ctx.getRuleContext();
    BuildConfigurationValue configuration = ruleContext.getConfiguration();
    NestedSetBuilder<Artifact> toolsBuilder = NestedSetBuilder.stableOrder();
    CustomCommandLine.Builder executableLine = CustomCommandLine.builder();
    switch (executableUnchecked) {
      case Artifact executable -> {
        toolsBuilder.add(executable);
        executableLine.addExecPath(executable);
      }
      case FilesToRunProvider executable -> {
        Artifact executableArtifact = executable.getExecutable();
        if (executableArtifact == null) {
          throw Starlark.errorf("executable must be a single file");
        }
        toolsBuilder.addTransitive(executable.getFilesToRun());
        executableLine.addExecPath(executableArtifact);
      }
      case String executable -> executableLine.addDynamicString(executable);
      default ->
          throw Starlark.errorf(
              "expected file, FilesToRunProvider, or string for executable, got: %s",
              Starlark.type(executableUnchecked));
    }
    switch (toolsUnchecked) {
      case Depset depset ->
          toolsBuilder.addTransitive(Depset.cast(depset, Artifact.class, "tools"));
      case Sequence<?> seq -> {
        for (Object tool : seq) {
          switch (tool) {
            case Artifact a -> toolsBuilder.add(a);
            case FilesToRunProvider f -> toolsBuilder.addTransitive(f.getFilesToRun());
            case Depset d -> toolsBuilder.addTransitive(Depset.cast(d, Artifact.class, "tools"));
            default ->
                throw Starlark.errorf(
                    "expected File, FilesToRunProvider, or depset in tools, got: %s",
                    Starlark.type(tool));
          }
        }
      }
      default ->
          throw Starlark.errorf(
              "expected sequence or depset for tools, got: %s", Starlark.type(toolsUnchecked));
    }
    executableLine.addAll(Sequence.cast(executableArgs, String.class, "executable_args"));

    @Nullable
    Artifact outputDepsProtoArtifact =
        outputDepsProto instanceof Artifact artifact ? artifact : null;

    NestedSet<Artifact> mandatoryInputsSet =
        Depset.cast(mandatoryInputs, Artifact.class, "mandatory_inputs");
    NestedSet<Artifact> transitiveInputsSet =
        Depset.cast(transitiveInputs, Artifact.class, "transitive_inputs");
    NestedSet<Artifact> directJarsSet = Depset.cast(directJars, Artifact.class, "direct_jars");
    NestedSet<Artifact> additionalArtifactsForPathMapping =
        NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    JavaClasspathMode configClasspathMode =
        configuration.getFragment(JavaConfiguration.class).getReduceJavaClasspath();
    JavaClasspathMode classpathMode;
    if (enableDirectClasspath) {
      NestedSet<Artifact> classpath;
      if (!directJarsSet.isEmpty() || transitiveInputsSet.isEmpty()) {
        classpath = directJarsSet;
        additionalArtifactsForPathMapping = transitiveInputsSet;
      } else {
        classpath = transitiveInputsSet;
      }
      mandatoryInputsSet =
          NestedSetBuilder.fromNestedSet(mandatoryInputsSet).addTransitive(classpath).build();
      transitiveInputsSet = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
      directJarsSet = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
      classpathMode = JavaClasspathMode.OFF;
      insertDependencies =
          insertDependencies
              && (configClasspathMode == JavaClasspathMode.BAZEL
                  || configClasspathMode == JavaClasspathMode.BAZEL_NO_FALLBACK);
      for (Args args : Sequence.cast(arguments, Args.class, "arguments")) {
        if (!classpath.isEmpty()) {
          args.addAll(
              "--classpath",
              Depset.of(Artifact.class, classpath),
              /* mapEach= */ Starlark.NONE,
              /* formatEach= */ Starlark.NONE,
              /* beforeEach= */ Starlark.NONE,
              /* omitIfEmpty= */ true,
              /* uniquify= */ false,
              /* expandDirectories= */ true,
              /* terminateWith= */ Starlark.NONE,
              /* allowClosure= */ false,
              thread);
        }
        args.addArgument("--reduce_classpath_mode", "NONE", /* format= */ Starlark.NONE, thread);
        break;
      }
    } else {
      classpathMode =
          outputDepsProtoArtifact != null && !reduceClasspathIncompatible
              ? configClasspathMode
              : JavaClasspathMode.OFF;
    }

    CommandLine flagLine = CommandLine.empty();
    ImmutableList.Builder<CommandLine> extraCommandLineArgs = ImmutableList.builder();
    boolean firstArgs = true;
    for (Args args : Sequence.cast(arguments, Args.class, "arguments")) {
      CommandLine commandLine =
          args.build(ruleContext.getAnalysisEnvironment()::getMainRepoMapping);
      if (firstArgs) {
        flagLine = commandLine;
        firstArgs = false;
      } else {
        extraCommandLineArgs.add(commandLine);
      }
    }

    ImmutableMap<String, String> env =
        ImmutableMap.copyOf(Dict.noneableCast(envUnchecked, String.class, String.class, "env"));
    ActionEnvironment actionEnvironment =
        configuration.getActionEnvironment().withAdditionalFixedVariables(env);

    ImmutableMap<String, String> executionRequirements =
        TargetUtils.getFilteredExecutionInfo(
            executionRequirementsUnchecked,
            ruleContext.getRule(),
            configuration.allowUnresolvedSymlinks());

    toolchainUnchecked = ctx.maybeOverrideToolchain(toolchainUnchecked);
    String execGroup =
        StarlarkActionFactory.determineExecGroup(ruleContext, Starlark.NONE, toolchainUnchecked);

    NestedSet<Artifact> dependencyArtifactsSet =
        classpathMode != JavaClasspathMode.OFF
            ? Depset.cast(dependencyArtifacts, Artifact.class, "dependency_artifacts")
            : NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    NestedSet<Artifact> tools = toolsBuilder.build();
    ruleContext.registerAction(
        new JavaCompileAction(
            /* mnemonic= */ mnemonic,
            /* insertDependencies= */ insertDependencies,
            /* owner= */ ruleContext.getActionOwner(execGroup),
            /* tools= */ tools,
            /* progressMessage= */ new OnDemandString() {
              @Override
              public String toString() {
                return progressMessage;
              }
            },
            /* mandatoryInputs= */ NestedSetBuilder.fromNestedSet(mandatoryInputsSet)
                .addTransitive(tools)
                .build(),
            /* transitiveInputs= */ transitiveInputsSet,
            /* directJars= */ directJarsSet,
            /* outputs= */ ImmutableSet.copyOf(
                Sequence.cast(outputs, Artifact.class, "outputs").getImmutableList()),
            /* env= */ actionEnvironment,
            /* executionInfo= */ executionRequirements,
            /* executableLine= */ executableLine.build(),
            /* flagLine= */ flagLine,
            /* extraCommandLineArgs= */ extraCommandLineArgs.build(),
            /* configuration= */ configuration,
            /* dependencyArtifacts= */ dependencyArtifactsSet,
            /* additionalArtifactsForPathMapping= */ additionalArtifactsForPathMapping,
            /* outputDepsProto= */ outputDepsProtoArtifact,
            /* classpathMode= */ classpathMode));
  }

  static boolean isInstanceOfProvider(Object obj, Provider provider) {
    if (obj instanceof NativeInfo nativeInfo) {
      return nativeInfo.getProvider().getKey().equals(provider.getKey());
    } else if (obj instanceof StarlarkInfoWithSchema starlarkInfoWithSchema) {
      return starlarkInfoWithSchema.getProvider().getKey().equals(provider.getKey());
    } else if (obj instanceof StarlarkInfoNoSchema starlarkInfoNoSchema) {
      return starlarkInfoNoSchema.getProvider().getKey().equals(provider.getKey());
    }
    return false;
  }
}
