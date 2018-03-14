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
import com.google.devtools.build.lib.analysis.actions.SpawnAction.Builder;
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
import com.google.devtools.build.lib.syntax.SkylarkType;
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
   * @param outputJar the jar that was created as a result of
   *                  a compilation (e.g. javac, scalac, etc)
   * @param sourceFiles the sources that were used to create the output jar
   * @param sourceJars the source jars that were used to create the output jar
   * @param useIjar if an ijar of the output jar should be created and stored in the provider
   * @param neverlink if true only use this library for compilation and not at runtime
   * @param compileTimeDeps compile time dependencies that were used to create the output jar
   * @param runtimeDeps runtime dependencies that are needed for this library
   * @param exports libraries to make available for users of this library.
   *                <a href="https://docs.bazel.build/versions/master/be/java.html#java_library"
   *                target="_top">java_library.exports</a>
   * @param action used to create the ijar and single jar actions
   * @param javaToolchain the toolchain to be used for retrieving the ijar tool
   * @return new created JavaInfo instance
   * @throws EvalException if some mandatory parameter are missing
   */
  public JavaInfo createJavaInfo(
      Artifact outputJar,
      SkylarkList<Artifact> sourceFiles,
      SkylarkList<Artifact> sourceJars,
      Boolean useIjar,
      Boolean neverlink,
      SkylarkList<JavaInfo> compileTimeDeps,
      SkylarkList<JavaInfo> runtimeDeps,
      SkylarkList<JavaInfo> exports,
      Object action,
      Object javaToolchain,
      Object hostJavabase,
      Location location)
      throws EvalException {

    if (sourceFiles.isEmpty() && sourceJars.isEmpty() && exports.isEmpty()) {
      throw new EvalException(
          null, "source_jars, sources and exports cannot be simultaneous empty");
    }


    ImmutableList<Artifact> outputSourceJars;

    if (sourceFiles.isEmpty() && sourceJars.size() == 1){
      outputSourceJars = ImmutableList.copyOf(sourceJars);
    } else if (!sourceFiles.isEmpty() || !sourceJars.isEmpty()){
      Artifact source = packSourceFiles(
              outputJar, sourceFiles, sourceJars, action, javaToolchain, hostJavabase, location);
      outputSourceJars = ImmutableList.of(source);
    } else {
      outputSourceJars = ImmutableList.of();
    }

    JavaInfo.Builder javaInfoBuilder = JavaInfo.Builder.create();
    javaInfoBuilder.setLocation(location);

    JavaCompilationArgs.Builder javaCompilationArgsBuilder = JavaCompilationArgs.builder();

    javaCompilationArgsBuilder.addFullCompileTimeJar(outputJar);

    if (!neverlink) {
      javaCompilationArgsBuilder.addRuntimeJar(outputJar);
    }

    Artifact iJar = outputJar;
    if (useIjar) {
      SkylarkActionFactory skylarkActionFactory = checkActionType(action, location);
      ConfiguredTarget javaToolchainConfiguredTarget =
          checkConfiguredTargetType(javaToolchain, "java_toolchain", location);
      iJar = buildIjar(outputJar, skylarkActionFactory, javaToolchainConfiguredTarget);
    }
    javaCompilationArgsBuilder.addCompileTimeJar(iJar);

    JavaRuleOutputJarsProvider javaRuleOutputJarsProvider = JavaRuleOutputJarsProvider.builder()
            .addOutputJar(outputJar, iJar, outputSourceJars).build();
    javaInfoBuilder.addProvider(JavaRuleOutputJarsProvider.class, javaRuleOutputJarsProvider);

    JavaCompilationArgs.Builder recursiveJavaCompilationArgsBuilder =
        JavaCompilationArgs.Builder.copyOf(javaCompilationArgsBuilder);

    ClasspathType type = neverlink ? COMPILE_ONLY : BOTH;

    fetchProviders(exports, JavaCompilationArgsProvider.class)
        .map(JavaCompilationArgsProvider::getJavaCompilationArgs)
        .forEach(args->javaCompilationArgsBuilder.addTransitiveArgs(args, type));

    fetchProviders(concat(exports, compileTimeDeps), JavaCompilationArgsProvider.class)
        .map(JavaCompilationArgsProvider::getRecursiveJavaCompilationArgs)
        .forEach(args->recursiveJavaCompilationArgsBuilder.addTransitiveArgs(args, type));

    fetchProviders(runtimeDeps, JavaCompilationArgsProvider.class)
        .map(JavaCompilationArgsProvider::getRecursiveJavaCompilationArgs)
        .forEach(args->recursiveJavaCompilationArgsBuilder.addTransitiveArgs(args, RUNTIME_ONLY));

    javaInfoBuilder.addProvider(JavaCompilationArgsProvider.class,
        JavaCompilationArgsProvider.create(
            javaCompilationArgsBuilder.build(), recursiveJavaCompilationArgsBuilder.build()));

    javaInfoBuilder.addProvider(JavaExportsProvider.class, createJavaExportsProvider(exports));

    javaInfoBuilder.addProvider(
        JavaSourceJarsProvider.class,
        createJavaSourceJarsProvider(
            outputSourceJars, concat(compileTimeDeps, runtimeDeps, exports)));

    return javaInfoBuilder.build();
  }

  /**
   * Creates action which creates archive with all source files inside.
   * Takes all filer from sourceFiles collection and all files from every sourceJars.
   * Name of Artifact generated based on outputJar.
   *
   * @param outputJar name of output Jar artifact.
   * @return generated artifact
   */
  private Artifact packSourceFiles(
      Artifact outputJar,
      SkylarkList<Artifact> sourceFiles,
      SkylarkList<Artifact> sourceJars,
      Object action,
      Object javaToolchain,
      Object hostJavabase,
      Location location)
      throws EvalException {

    SkylarkActionFactory skylarkActionFactory = checkActionType(action, location);
    ActionRegistry actionRegistry = createActionRegistry(skylarkActionFactory);

    Artifact outputSrcJar =
        getSourceJar(
            skylarkActionFactory.getActionConstructionContext(), outputJar);

    ConfiguredTarget hostJavabaseConfiguredTarget =
        checkConfiguredTargetType(hostJavabase, "host_javabase", location);
    JavaRuntimeInfo javaRuntimeInfo = JavaRuntimeInfo.from(hostJavabaseConfiguredTarget, null);

    ConfiguredTarget javaToolchainConfiguredTarget =
        checkConfiguredTargetType(javaToolchain, "java_toolchain", location);
    JavaToolchainProvider javaToolchainProvider =
        getJavaToolchainProvider(javaToolchainConfiguredTarget);

    JavaSemantics javaSemantics = javaToolchainProvider.getJavaSemantics();

    SingleJarActionBuilder.createSourceJarAction(
        actionRegistry,
        skylarkActionFactory.getActionConstructionContext(),
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

  public JavaInfo create(
      @Nullable Object actionsUnchecked,
      NestedSet<Artifact> compileTimeJars,
      NestedSet<Artifact> runtimeJars,
      Boolean useIjar,
      @Nullable Object javaToolchainUnchecked,
      NestedSet<Artifact> transitiveCompileTimeJars,
      NestedSet<Artifact> transitiveRuntimeJars,
      NestedSet<Artifact> sourceJars)
      throws EvalException {

    JavaCompilationArgs.Builder javaCompilationArgsBuilder = JavaCompilationArgs.builder();
    if (useIjar && !compileTimeJars.isEmpty()) {
      javaCompilationArgsBuilder.addFullCompileTimeJars(compileTimeJars);
      SkylarkActionFactory skylarkActionFactory =
          checkActionType(actionsUnchecked);
      ConfiguredTarget configuredTarget =
          checkConfiguredTargetType(javaToolchainUnchecked, "java_toolchain");
      for (Artifact compileJar : compileTimeJars) {
        javaCompilationArgsBuilder.addCompileTimeJar(
            buildIjar(compileJar, skylarkActionFactory, configuredTarget));
      }
    } else {
      javaCompilationArgsBuilder.addCompileTimeJars(compileTimeJars);
      javaCompilationArgsBuilder.addFullCompileTimeJars(compileTimeJars);
    }

    JavaCompilationArgs javaCompilationArgs =
        javaCompilationArgsBuilder.addTransitiveRuntimeJars(runtimeJars).build();

    JavaCompilationArgs.Builder recursiveJavaCompilationArgs = JavaCompilationArgs.builder();
    if (transitiveCompileTimeJars.isEmpty()) {
      recursiveJavaCompilationArgs.addTransitiveCompileTimeJars(
          javaCompilationArgs.getCompileTimeJars());
      recursiveJavaCompilationArgs.addTransitiveFullCompileTimeJars(
          javaCompilationArgs.getFullCompileTimeJars());
    } else {
      recursiveJavaCompilationArgs.addTransitiveCompileTimeJars(transitiveCompileTimeJars);
    }

    if (transitiveRuntimeJars.isEmpty()) {
      recursiveJavaCompilationArgs.addTransitiveRuntimeJars(runtimeJars);
    } else {
      recursiveJavaCompilationArgs.addTransitiveRuntimeJars(transitiveRuntimeJars);
    }

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

    JavaLibraryHelper helper =
        new JavaLibraryHelper(skylarkRuleContext.getRuleContext())
            .setOutput(outputJar)
            .addSourceJars(sourceJars)
            .addSourceFiles(sourceFiles)
            .addResources(resources)
            .setSourcePathEntries(sourcepathEntries)
            .setJavacOpts(javacOpts);

    helper.addAllDeps(deps);
    helper.addAllExports(exports);
    helper.setCompilationStrictDepsMode(getStrictDepsMode(strictDepsMode.toUpperCase()));

    helper.addAllPlugins(plugins);
    helper.addAllPlugins(deps);

    JavaRuleOutputJarsProvider.Builder outputJarsBuilder = JavaRuleOutputJarsProvider.builder();

    boolean generateMergedSourceJar =
        (sourceJars.size() > 1 || !sourceFiles.isEmpty())
            || (sourceJars.isEmpty() && sourceFiles.isEmpty() && !exports.isEmpty());
    Artifact outputSourceJar =
        generateMergedSourceJar
            ? getSourceJar(skylarkRuleContext.getRuleContext(), outputJar)
            : sourceJars.get(0);

    JavaCompilationArtifacts artifacts =
        helper.build(
            javaSemantics,
            getJavaToolchainProvider(javaToolchain),
            javaRuntimeInfo,
            SkylarkList.createImmutable(ImmutableList.of()),
            outputJarsBuilder,
            /*createOutputSourceJar*/ generateMergedSourceJar,
            outputSourceJar);

    JavaCompilationArgsProvider javaCompilationArgsProvider =
        helper.buildCompilationArgsProvider(artifacts, true, neverlink);
    Runfiles runfiles =
        new Runfiles.Builder(skylarkRuleContext.getWorkspaceName())
            .addTransitiveArtifactsWrappedInStableOrder(
                javaCompilationArgsProvider.getRecursiveJavaCompilationArgs().getRuntimeJars())
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

    return JavaInfo.Builder.create()
        .addProvider(JavaCompilationArgsProvider.class, javaCompilationArgsProvider)
        .addProvider(
            JavaSourceJarsProvider.class,
            createJavaSourceJarsProvider(outputSourceJars, transitiveSourceJars.build()))
        .addProvider(JavaRuleOutputJarsProvider.class, outputJarsBuilder.build())
        .addProvider(JavaRunfilesProvider.class, new JavaRunfilesProvider(runfiles))
        .addProvider(JavaPluginInfoProvider.class, transitivePluginsProvider)
        .setNeverlink(neverlink)
        .build();
  }

  private SkylarkActionFactory checkActionType(Object action) throws EvalException {
    return checkActionType(action, /*location=*/ null);
  }

  private SkylarkActionFactory checkActionType(Object action, Location location)
      throws EvalException {
    return SkylarkType.cast(
        action,
        SkylarkActionFactory.class,
        location,
        "The value of use_ijar is True. Make sure the ctx.actions argument is valid.");
  }

  private ConfiguredTarget checkConfiguredTargetType(Object javaToolchain, String typeName)
      throws EvalException {
    return checkConfiguredTargetType(javaToolchain, typeName, /*location=*/ null);
  }

  private ConfiguredTarget checkConfiguredTargetType(
      Object javaToolchain, String typeName, Location location) throws EvalException {
    Objects.requireNonNull(typeName);
    return SkylarkType.cast(
        javaToolchain,
        ConfiguredTarget.class,
        location,
        "The value of use_ijar is True. Make sure the " + typeName + " argument is a valid.");
  }

  private Artifact buildIjar(
      Artifact inputJar,
      SkylarkActionFactory actions,
      ConfiguredTarget javaToolchainConfiguredTarget)
      throws EvalException {
    String ijarBasename = FileSystemUtils.removeExtension(inputJar.getFilename()) + "-ijar.jar";
    Artifact interfaceJar = actions.declareFile(ijarBasename, inputJar);
    FilesToRunProvider ijarTarget =
        getJavaToolchainProvider(javaToolchainConfiguredTarget).getIjar();
    SpawnAction.Builder actionBuilder =
        new Builder()
            .addInput(inputJar)
            .addOutput(interfaceJar)
            .setExecutable(ijarTarget)
            .setProgressMessage("Extracting interface for jar %s", inputJar.getFilename())
            .addCommandLine(
                CustomCommandLine.builder().addExecPath(inputJar).addExecPath(interfaceJar).build())
            .useDefaultShellEnvironment()
            .setMnemonic("JavaIjar");
    actions.registerAction(actionBuilder.build(actions.getActionConstructionContext()));
    return interfaceJar;
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
