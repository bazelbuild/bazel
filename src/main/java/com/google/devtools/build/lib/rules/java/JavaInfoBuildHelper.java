// Copyright 2018 The Bazel Authors. All rights reserved.
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

import static com.google.common.collect.Iterables.concat;
import static com.google.devtools.build.lib.rules.java.JavaCompilationArgs.ClasspathType.BOTH;
import static com.google.devtools.build.lib.rules.java.JavaCompilationArgs.ClasspathType.COMPILE_ONLY;
import static com.google.devtools.build.lib.rules.java.JavaCompilationArgs.ClasspathType.RUNTIME_ONLY;
import static java.util.stream.Stream.concat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionRegistry;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.StrictDepsMode;
import com.google.devtools.build.lib.analysis.skylark.SkylarkActionFactory;
import com.google.devtools.build.lib.analysis.skylark.SkylarkRuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgs.ClasspathType;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import java.util.List;
import java.util.Objects;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;
import javax.annotation.Nullable;

/** Implements logic for creating JavaInfo from different set of input parameters. */
final class JavaInfoBuildHelper {
  private static final JavaInfoBuildHelper INSTANCE = new JavaInfoBuildHelper();

  private JavaInfoBuildHelper() {}

  public static JavaInfoBuildHelper getInstance() {
    return INSTANCE;
  }

  /**
   * Creates JavaInfo instance from outputJar.
   *
   * @param outputJar the jar that was created as a result of a compilation (e.g. javac, scalac,
   *     etc)
   * @param sourceFiles the sources that were used to create the output jar
   * @param sourceJars the source jars that were used to create the output jar
   * @param useIjar if an ijar of the output jar should be created and stored in the provider
   * @param neverlink if true only use this library for compilation and not at runtime
   * @param compileTimeDeps compile time dependencies that were used to create the output jar
   * @param runtimeDeps runtime dependencies that are needed for this library
   * @param exports libraries to make available for users of this library. <a
   *     href="https://docs.bazel.build/versions/master/be/java.html#java_library"
   *     target="_top">java_library.exports</a>
   * @param actions used to create the ijar and single jar actions
   * @param javaToolchain the toolchain to be used for retrieving the ijar tool
   * @return new created JavaInfo instance
   * @throws EvalException if some mandatory parameter are missing
   */
  @Deprecated
  JavaInfo createJavaInfoLegacy(
      Artifact outputJar,
      SkylarkList<Artifact> sourceFiles,
      SkylarkList<Artifact> sourceJars,
      Boolean useIjar,
      Boolean neverlink,
      SkylarkList<JavaInfo> compileTimeDeps,
      SkylarkList<JavaInfo> runtimeDeps,
      SkylarkList<JavaInfo> exports,
      Object actions,
      Object javaToolchain,
      Object hostJavabase,
      Location location)
      throws EvalException {
    final Artifact sourceJar;
    if (sourceFiles.isEmpty() && sourceJars.isEmpty()) {
      sourceJar = null;
    } else if (sourceFiles.isEmpty() && sourceJars.size() == 1) {
      sourceJar = sourceJars.get(0);
    } else {
      if (!(actions instanceof SkylarkActionFactory)) {
        throw new EvalException(location, "Must pass ctx.actions when packing sources.");
      }
      if (!(javaToolchain instanceof ConfiguredTarget)) {
        throw new EvalException(location, "Must pass java_toolchain when packing sources.");
      }
      if (!(hostJavabase instanceof ConfiguredTarget)) {
        throw new EvalException(location, "Must pass host_javabase when packing sources.");
      }
      sourceJar =
          packSourceFiles(
              (SkylarkActionFactory) actions,
              outputJar,
              sourceFiles,
              sourceJars,
              (ConfiguredTarget) javaToolchain,
              (ConfiguredTarget) hostJavabase);
    }
    final Artifact iJar;
    if (useIjar) {
      if (!(actions instanceof SkylarkActionFactory)) {
        throw new EvalException(
            location,
            "The value of use_ijar is True. Make sure the ctx.actions argument is valid.");
      }
      if (!(javaToolchain instanceof ConfiguredTarget)) {
        throw new EvalException(
            location,
            "The value of use_ijar is True. Make sure the java_toolchain argument is valid.");
      }
      iJar =
          buildIjar(
              (SkylarkActionFactory) actions, outputJar, null, (ConfiguredTarget) javaToolchain);
    } else {
      iJar = outputJar;
    }

    return createJavaInfo(
        outputJar, iJar, sourceJar, neverlink, compileTimeDeps, runtimeDeps, exports, location);
  }

  /**
   * Creates JavaInfo instance from outputJar.
   *
   * @param outputJar the jar that was created as a result of a compilation (e.g. javac, scalac,
   *     etc)
   * @param compileJar Jar added as a compile-time dependency to other rules. Typically produced by
   *     ijar.
   * @param sourceJar the source jar that was used to create the output jar
   * @param neverlink if true only use this library for compilation and not at runtime
   * @param compileTimeDeps compile time dependencies that were used to create the output jar
   * @param runtimeDeps runtime dependencies that are needed for this library
   * @param exports libraries to make available for users of this library. <a
   *     href="https://docs.bazel.build/versions/master/be/java.html#java_library"
   *     target="_top">java_library.exports</a>
   * @return new created JavaInfo instance
   */
  JavaInfo createJavaInfo(
      Artifact outputJar,
      Artifact compileJar,
      @Nullable Artifact sourceJar,
      Boolean neverlink,
      SkylarkList<JavaInfo> compileTimeDeps,
      SkylarkList<JavaInfo> runtimeDeps,
      SkylarkList<JavaInfo> exports,
      Location location) {
    compileJar = compileJar != null ? compileJar : outputJar;
    ImmutableList<Artifact> sourceJars =
        sourceJar != null ? ImmutableList.of(sourceJar) : ImmutableList.of();
    JavaInfo.Builder javaInfoBuilder = JavaInfo.Builder.create();
    javaInfoBuilder.setLocation(location);

    JavaCompilationArgs.Builder javaCompilationArgsBuilder = JavaCompilationArgs.builder();

    javaCompilationArgsBuilder.addFullCompileTimeJar(outputJar);

    if (!neverlink) {
      javaCompilationArgsBuilder.addRuntimeJar(outputJar);
    }
    javaCompilationArgsBuilder.addCompileTimeJar(compileJar);

    JavaRuleOutputJarsProvider javaRuleOutputJarsProvider =
        JavaRuleOutputJarsProvider.builder()
            .addOutputJar(outputJar, compileJar, sourceJars)
            .build();
    javaInfoBuilder.addProvider(JavaRuleOutputJarsProvider.class, javaRuleOutputJarsProvider);

    JavaCompilationArgs.Builder recursiveJavaCompilationArgsBuilder =
        JavaCompilationArgs.Builder.copyOf(javaCompilationArgsBuilder);

    ClasspathType type = neverlink ? COMPILE_ONLY : BOTH;

    fetchProviders(exports, JavaCompilationArgsProvider.class)
        .map(JavaCompilationArgsProvider::getJavaCompilationArgs)
        .forEach(args -> javaCompilationArgsBuilder.addTransitiveArgs(args, type));

    fetchProviders(concat(exports, compileTimeDeps), JavaCompilationArgsProvider.class)
        .map(JavaCompilationArgsProvider::getRecursiveJavaCompilationArgs)
        .forEach(args -> recursiveJavaCompilationArgsBuilder.addTransitiveArgs(args, type));

    fetchProviders(runtimeDeps, JavaCompilationArgsProvider.class)
        .map(JavaCompilationArgsProvider::getRecursiveJavaCompilationArgs)
        .forEach(args -> recursiveJavaCompilationArgsBuilder.addTransitiveArgs(args, RUNTIME_ONLY));

    javaInfoBuilder.addProvider(
        JavaCompilationArgsProvider.class,
        JavaCompilationArgsProvider.create(
            javaCompilationArgsBuilder.build(), recursiveJavaCompilationArgsBuilder.build()));

    javaInfoBuilder.addProvider(JavaExportsProvider.class, createJavaExportsProvider(exports));

    javaInfoBuilder.addProvider(
        JavaSourceJarsProvider.class,
        createJavaSourceJarsProvider(sourceJars, concat(compileTimeDeps, runtimeDeps, exports)));

    javaInfoBuilder.setRuntimeJars(ImmutableList.of(outputJar));

    return javaInfoBuilder.build();
  }

  /**
   * Creates action which creates archive with all source files inside. Takes all filer from
   * sourceFiles collection and all files from every sourceJars. Name of Artifact generated based on
   * outputJar.
   *
   * @param outputJar name of output Jar artifact.
   * @return generated artifact, or null if there's nothing to pack
   */
  @Nullable
  Artifact packSourceFiles(
      SkylarkActionFactory actions,
      Artifact outputJar,
      SkylarkList<Artifact> sourceFiles,
      SkylarkList<Artifact> sourceJars,
      ConfiguredTarget javaToolchain,
      ConfiguredTarget hostJavabase)
      throws EvalException {
    // No sources to pack, return None
    if (sourceFiles.isEmpty() && sourceJars.isEmpty()) {
      return null;
    }
    // If we only have one source jar, return it directly to avoid action creation
    if (sourceFiles.isEmpty() && sourceJars.size() == 1) {
      return sourceJars.get(0);
    }
    ActionRegistry actionRegistry = createActionRegistry(actions);
    Artifact outputSrcJar = getSourceJar(actions.getActionConstructionContext(), outputJar);
    JavaRuntimeInfo javaRuntimeInfo = JavaRuntimeInfo.from(hostJavabase, null);
    JavaToolchainProvider javaToolchainProvider = getJavaToolchainProvider(javaToolchain);
    JavaSemantics javaSemantics = javaToolchainProvider.getJavaSemantics();
    SingleJarActionBuilder.createSourceJarAction(
        actionRegistry,
        actions.getActionConstructionContext(),
        javaSemantics,
        ImmutableList.copyOf(sourceFiles),
        NestedSetBuilder.<Artifact>stableOrder().addAll(sourceJars).build(),
        outputSrcJar,
        javaToolchainProvider,
        javaRuntimeInfo);
    return outputSrcJar;
  }

  private ActionRegistry createActionRegistry(SkylarkActionFactory skylarkActionFactory) {
    return new ActionRegistry() {

      @Override
      public void registerAction(ActionAnalysisMetadata... actions) {
        skylarkActionFactory.registerAction(actions);
      }

      @Override
      public ArtifactOwner getOwner() {
        return skylarkActionFactory
            .getActionConstructionContext()
            .getAnalysisEnvironment()
            .getOwner();
      }
    };
  }

  /** Creates a {@link JavaSourceJarsProvider} from the given lists of source jars. */
  private static JavaSourceJarsProvider createJavaSourceJarsProvider(
      List<Artifact> sourceJars, NestedSet<Artifact> transitiveSourceJars) {
    NestedSet<Artifact> javaSourceJars = NestedSetBuilder.wrap(Order.STABLE_ORDER, sourceJars);
    return JavaSourceJarsProvider.create(transitiveSourceJars, javaSourceJars);
  }

  private JavaSourceJarsProvider createJavaSourceJarsProvider(
      Iterable<Artifact> sourceJars, Iterable<JavaInfo> transitiveDeps) {
    NestedSetBuilder<Artifact> transitiveSourceJars = NestedSetBuilder.stableOrder();

    transitiveSourceJars.addAll(sourceJars);

    fetchSourceJars(transitiveDeps)
        .forEach(transitiveSourceJars::addTransitive);

    return JavaSourceJarsProvider.create(transitiveSourceJars.build(), sourceJars);
  }

  private Iterable<NestedSet<Artifact>> fetchSourceJars(Iterable<JavaInfo> javaInfos) {
    Stream<NestedSet<Artifact>> sourceJars =
        fetchProviders(javaInfos, JavaSourceJarsProvider.class)
            .map(JavaSourceJarsProvider::getSourceJars)
            .map(sourceJarsList -> NestedSetBuilder.wrap(Order.STABLE_ORDER, sourceJarsList));

    Stream<NestedSet<Artifact>> transitiveSourceJars =
        fetchProviders(javaInfos, JavaSourceJarsProvider.class)
            .map(JavaSourceJarsProvider::getTransitiveSourceJars);

    return concat(sourceJars, transitiveSourceJars)::iterator;
  }

  /**
   * Returns Stream of not null Providers.
   *
   * Gets Stream from dependencies, transforms to Provider defined by providerClass param
   * and filters nulls.
   *
   * @see JavaInfo#merge(List)
   */
  private <P extends TransitiveInfoProvider>Stream<P> fetchProviders(Iterable<JavaInfo> javaInfos,
      Class<P> providerClass){
    return StreamSupport.stream(javaInfos.spliterator(), /*parallel=*/ false)
        .map(javaInfo -> javaInfo.getProvider(providerClass))
        .filter(Objects::nonNull);
  }

  private JavaExportsProvider createJavaExportsProvider(Iterable<JavaInfo> javaInfos) {
    NestedSet<Label> exportsNestedSet = fetchExports(javaInfos);

    // TODO(b/69780248): I need to add javaInfos there too. See #3769
    // The problem is JavaInfo can not be converted to Label.
    return new JavaExportsProvider(exportsNestedSet);
  }

  private NestedSet<Label> fetchExports(Iterable<JavaInfo> javaInfos){
    NestedSetBuilder<Label> builder = NestedSetBuilder.stableOrder();

    fetchProviders(javaInfos, JavaExportsProvider.class)
        .map(JavaExportsProvider::getTransitiveExports)
        .forEach(builder::addTransitive);

    return builder.build();
  }

  @Deprecated
  public JavaInfo create(
      @Nullable Object actions,
      NestedSet<Artifact> compileTimeJars,
      NestedSet<Artifact> runtimeJars,
      Boolean useIjar,
      @Nullable Object javaToolchain,
      NestedSet<Artifact> transitiveCompileTimeJars,
      NestedSet<Artifact> transitiveRuntimeJars,
      NestedSet<Artifact> sourceJars,
      Location location)
      throws EvalException {

    JavaCompilationArgs.Builder javaCompilationArgsBuilder = JavaCompilationArgs.builder();
    if (useIjar && !compileTimeJars.isEmpty()) {
      javaCompilationArgsBuilder.addFullCompileTimeJars(compileTimeJars);
      if (!(actions instanceof SkylarkActionFactory)) {
        throw new EvalException(
            location,
            "The value of use_ijar is True. Make sure the ctx.actions argument is valid.");
      }
      if (!(javaToolchain instanceof ConfiguredTarget)) {
        throw new EvalException(
            location,
            "The value of use_ijar is True. Make sure the java_toolchain argument is valid.");
      }
      for (Artifact compileJar : compileTimeJars) {
        javaCompilationArgsBuilder.addCompileTimeJar(
            buildIjar(
                (SkylarkActionFactory) actions,
                compileJar,
                null,
                (ConfiguredTarget) javaToolchain));
      }
    } else {
      javaCompilationArgsBuilder.addCompileTimeJars(compileTimeJars);
      javaCompilationArgsBuilder.addFullCompileTimeJars(compileTimeJars);
    }

    JavaCompilationArgs javaCompilationArgs =
        javaCompilationArgsBuilder.addTransitiveRuntimeJars(runtimeJars).build();

    JavaCompilationArgs.Builder recursiveJavaCompilationArgs =
        JavaCompilationArgs.builder()
            .addTransitiveArgs(javaCompilationArgs, ClasspathType.BOTH)
            .addTransitiveCompileTimeJars(transitiveCompileTimeJars)
            .addTransitiveRuntimeJars(transitiveRuntimeJars);

    JavaInfo javaInfo =
        JavaInfo.Builder.create()
            .addProvider(
                JavaCompilationArgsProvider.class,
                JavaCompilationArgsProvider.create(
                    javaCompilationArgs, recursiveJavaCompilationArgs.build()))
            .addProvider(
                JavaSourceJarsProvider.class,
                JavaSourceJarsProvider.create(
                    NestedSetBuilder.emptySet(Order.STABLE_ORDER), sourceJars))
            .setRuntimeJars(ImmutableList.copyOf(runtimeJars))
            .build();
    return javaInfo;
  }

  public JavaInfo createJavaCompileAction(
      SkylarkRuleContext skylarkRuleContext,
      SkylarkList<Artifact> sourceJars,
      SkylarkList<Artifact> sourceFiles,
      Artifact outputJar,
      SkylarkList<String> javacOpts,
      SkylarkList<JavaInfo> deps,
      SkylarkList<JavaInfo> exports,
      SkylarkList<JavaInfo> plugins,
      SkylarkList<JavaInfo> exportedPlugins,
      String strictDepsMode,
      ConfiguredTarget javaToolchain,
      ConfiguredTarget hostJavabase,
      SkylarkList<Artifact> sourcepathEntries,
      SkylarkList<Artifact> resources,
      Boolean neverlink,
      JavaSemantics javaSemantics)
      throws EvalException, InterruptedException {
    if (sourceJars.isEmpty() && sourceFiles.isEmpty() && exports.isEmpty()) {
      throw new EvalException(
          null, "source_jars, sources and exports cannot be simultaneous empty");
    }

    JavaRuntimeInfo javaRuntimeInfo =
        JavaRuntimeInfo.from(hostJavabase, skylarkRuleContext.getRuleContext());
    if (javaRuntimeInfo == null) {
      throw new EvalException(null, "'host_javabase' must point to a Java runtime");
    }

    JavaToolchainProvider toolchainProvider = getJavaToolchainProvider(javaToolchain);

    JavaLibraryHelper helper =
        new JavaLibraryHelper(skylarkRuleContext.getRuleContext())
            .setOutput(outputJar)
            .addSourceJars(sourceJars)
            .addSourceFiles(sourceFiles)
            .addResources(resources)
            .setSourcePathEntries(sourcepathEntries)
            .setJavacOpts(
                ImmutableList.<String>builder()
                    .addAll(
                        JavaCommon.computeToolchainJavacOpts(
                            skylarkRuleContext.getRuleContext(), toolchainProvider))
                    .addAll(
                        javaSemantics.getCompatibleJavacOptions(
                            skylarkRuleContext.getRuleContext(), toolchainProvider))
                    .addAll(javacOpts)
                    .build());

    List<JavaCompilationArgsProvider> depsCompilationArgsProviders =
        JavaInfo.fetchProvidersFromList(deps, JavaCompilationArgsProvider.class);
    List<JavaCompilationArgsProvider> exportsCompilationArgsProviders =
        JavaInfo.fetchProvidersFromList(exports, JavaCompilationArgsProvider.class);
    helper.addAllDeps(depsCompilationArgsProviders);
    helper.addAllExports(exportsCompilationArgsProviders);
    helper.setCompilationStrictDepsMode(getStrictDepsMode(strictDepsMode.toUpperCase()));

    helper.addAllPlugins(JavaInfo.fetchProvidersFromList(plugins, JavaPluginInfoProvider.class));
    helper.addAllPlugins(JavaInfo.fetchProvidersFromList(deps, JavaPluginInfoProvider.class));
    helper.setNeverlink(neverlink);

    JavaRuleOutputJarsProvider.Builder outputJarsBuilder = JavaRuleOutputJarsProvider.builder();

    boolean generateMergedSourceJar =
        (sourceJars.size() > 1 || !sourceFiles.isEmpty())
            || (sourceJars.isEmpty() && sourceFiles.isEmpty() && !exports.isEmpty());
    Artifact outputSourceJar =
        generateMergedSourceJar
            ? getSourceJar(skylarkRuleContext.getRuleContext(), outputJar)
            : sourceJars.get(0);

    JavaInfo.Builder javaInfoBuilder = JavaInfo.Builder.create();
    JavaCompilationArtifacts artifacts =
        helper.build(
            javaSemantics,
            toolchainProvider,
            javaRuntimeInfo,
            SkylarkList.createImmutable(ImmutableList.of()),
            outputJarsBuilder,
            /*createOutputSourceJar*/ generateMergedSourceJar,
            outputSourceJar,
            javaInfoBuilder);

    JavaCompilationArgsProvider javaCompilationArgsProvider =
        helper.buildCompilationArgsProvider(artifacts, true, neverlink);
    Runfiles runfiles =
        new Runfiles.Builder(skylarkRuleContext.getWorkspaceName())
            .addTransitiveArtifactsWrappedInStableOrder(
                javaCompilationArgsProvider.getRuntimeJars())
            .build();

    JavaPluginInfoProvider transitivePluginsProvider =
        JavaPluginInfoProvider.merge(
            concat(
                JavaInfo.getProvidersFromListOfJavaProviders(
                    JavaPluginInfoProvider.class, exportedPlugins),
                JavaInfo.getProvidersFromListOfJavaProviders(
                    JavaPluginInfoProvider.class, exports)));

    ImmutableList<Artifact> outputSourceJars = ImmutableList.of(outputSourceJar);

    NestedSetBuilder<Artifact> transitiveSourceJars =
        NestedSetBuilder.<Artifact>stableOrder().addAll(outputSourceJars);
    for (JavaSourceJarsProvider sourceJarsProvider :
        JavaInfo.getProvidersFromListOfJavaProviders(JavaSourceJarsProvider.class, deps)) {
      transitiveSourceJars.addTransitive(sourceJarsProvider.getTransitiveSourceJars());
    }

    return javaInfoBuilder
        .addProvider(JavaCompilationArgsProvider.class, javaCompilationArgsProvider)
        .addProvider(
            JavaSourceJarsProvider.class,
            createJavaSourceJarsProvider(outputSourceJars, transitiveSourceJars.build()))
        .addProvider(JavaRuleOutputJarsProvider.class, outputJarsBuilder.build())
        .addProvider(JavaRunfilesProvider.class, new JavaRunfilesProvider(runfiles))
        .addProvider(JavaPluginInfoProvider.class, transitivePluginsProvider)
        .setNeverlink(neverlink)
        .setRuntimeJars(ImmutableList.of(outputJar))
        .build();
  }

  public Artifact buildIjar(
      SkylarkActionFactory actions,
      Artifact inputJar,
      @Nullable Label targetLabel,
      ConfiguredTarget javaToolchainConfiguredTarget)
      throws EvalException {
    String ijarBasename = FileSystemUtils.removeExtension(inputJar.getFilename()) + "-ijar.jar";
    Artifact interfaceJar = actions.declareFile(ijarBasename, inputJar);
    FilesToRunProvider ijarTarget =
        getJavaToolchainProvider(javaToolchainConfiguredTarget).getIjar();
    CustomCommandLine.Builder commandLine =
        CustomCommandLine.builder().addExecPath(inputJar).addExecPath(interfaceJar);
    if (targetLabel != null) {
      commandLine.addLabel("--target_label", targetLabel);
    }
    SpawnAction.Builder actionBuilder =
        new SpawnAction.Builder()
            .addInput(inputJar)
            .addOutput(interfaceJar)
            .setExecutable(ijarTarget)
            .setProgressMessage("Extracting interface for jar %s", inputJar.getFilename())
            .addCommandLine(commandLine.build())
            .useDefaultShellEnvironment()
            .setMnemonic("JavaIjar");
    actions.registerAction(actionBuilder.build(actions.getActionConstructionContext()));
    return interfaceJar;
  }

  public Artifact stampJar(
      SkylarkActionFactory actions,
      Artifact inputJar,
      Label targetLabel,
      ConfiguredTarget javaToolchainConfiguredTarget)
      throws EvalException {
    String basename = FileSystemUtils.removeExtension(inputJar.getFilename()) + "-stamped.jar";
    Artifact outputJar = actions.declareFile(basename, inputJar);
    // ijar doubles as a stamping tool
    FilesToRunProvider ijarTarget =
        getJavaToolchainProvider(javaToolchainConfiguredTarget).getIjar();
    CustomCommandLine.Builder commandLine =
        CustomCommandLine.builder()
            .addExecPath(inputJar)
            .addExecPath(outputJar)
            .add("--nostrip_jar")
            .addLabel("--target_label", targetLabel);
    SpawnAction.Builder actionBuilder =
        new SpawnAction.Builder()
            .addInput(inputJar)
            .addOutput(outputJar)
            .setExecutable(ijarTarget)
            .setProgressMessage("Stamping target label into jar %s", inputJar.getFilename())
            .addCommandLine(commandLine.build())
            .useDefaultShellEnvironment()
            .setMnemonic("JavaIjar");
    actions.registerAction(actionBuilder.build(actions.getActionConstructionContext()));
    return outputJar;
  }

  JavaToolchainProvider getJavaToolchainProvider(ConfiguredTarget javaToolchain)
      throws EvalException {
    JavaToolchainProvider javaToolchainProvider = JavaToolchainProvider.from(javaToolchain);
    if (javaToolchainProvider == null) {
      throw new EvalException(
          null, javaToolchain.getLabel() + " does not provide JavaToolchainProvider.");
    }
    return javaToolchainProvider;
  }

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

  private static Artifact getSourceJar(ActionConstructionContext context, Artifact outputJar) {
    return JavaCompilationHelper.derivedArtifact(context, outputJar, "", "-src.jar");
  }
}
