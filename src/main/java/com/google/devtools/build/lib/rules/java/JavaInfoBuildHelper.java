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
import static com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider.ClasspathType.BOTH;
import static com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider.ClasspathType.COMPILE_ONLY;
import static com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider.ClasspathType.RUNTIME_ONLY;
import static com.google.devtools.build.lib.rules.java.JavaInfo.streamProviders;
import static java.util.stream.Stream.concat;

import com.google.common.base.Ascii;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionRegistry;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
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
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider.ClasspathType;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import java.util.stream.Stream;
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
   * @param jdeps optional jdeps information for outputJar
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
      @Nullable Artifact jdeps,
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
      if (!isValidJavaToolchain(javaToolchain)) {
        throw new EvalException(location, "Must pass java_toolchain when packing sources.");
      }
      if (!isValidJavaRuntime(hostJavabase)) {
        throw new EvalException(location, "Must pass host_javabase when packing sources.");
      }
      sourceJar =
          packSourceFiles(
              (SkylarkActionFactory) actions,
              outputJar,
              sourceFiles,
              sourceJars,
              javaToolchain,
              hostJavabase,
              location);
    }
    final Artifact iJar;
    if (useIjar) {
      if (!(actions instanceof SkylarkActionFactory)) {
        throw new EvalException(
            location,
            "The value of use_ijar is True. Make sure the ctx.actions argument is valid.");
      }
      if (!isValidJavaToolchain(javaToolchain)) {
        throw new EvalException(
            location,
            "The value of use_ijar is True. Make sure the java_toolchain argument is valid.");
      }
      iJar =
          buildIjar(
              (SkylarkActionFactory) actions,
              outputJar,
              null,
              getJavaToolchainProvider(location, javaToolchain),
              location);
    } else {
      iJar = outputJar;
    }

    return createJavaInfo(
        outputJar,
        iJar,
        sourceJar,
        neverlink,
        compileTimeDeps,
        runtimeDeps,
        exports,
        jdeps,
        location);
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
   * @param jdeps optional jdeps information for outputJar
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
      @Nullable Artifact jdeps,
      Location location) {
    compileJar = compileJar != null ? compileJar : outputJar;
    ImmutableList<Artifact> sourceJars =
        sourceJar != null ? ImmutableList.of(sourceJar) : ImmutableList.of();
    JavaInfo.Builder javaInfoBuilder = JavaInfo.Builder.create();
    javaInfoBuilder.setLocation(location);

    JavaCompilationArgsProvider.Builder javaCompilationArgsBuilder =
        JavaCompilationArgsProvider.builder();

    if (!neverlink) {
      javaCompilationArgsBuilder.addRuntimeJar(outputJar);
    }
    javaCompilationArgsBuilder.addDirectCompileTimeJar(
        /* interfaceJar= */ compileJar, /* fullJar= */ outputJar);

    JavaRuleOutputJarsProvider javaRuleOutputJarsProvider =
        JavaRuleOutputJarsProvider.builder()
            .addOutputJar(outputJar, compileJar, null /* manifestProto */, sourceJars)
            .setJdeps(jdeps)
            .build();
    javaInfoBuilder.addProvider(JavaRuleOutputJarsProvider.class, javaRuleOutputJarsProvider);

    ClasspathType type = neverlink ? COMPILE_ONLY : BOTH;

    streamProviders(exports, JavaCompilationArgsProvider.class)
        .forEach(args -> javaCompilationArgsBuilder.addExports(args, type));
    streamProviders(compileTimeDeps, JavaCompilationArgsProvider.class)
        .forEach(args -> javaCompilationArgsBuilder.addDeps(args, type));

    streamProviders(runtimeDeps, JavaCompilationArgsProvider.class)
        .forEach(args -> javaCompilationArgsBuilder.addDeps(args, RUNTIME_ONLY));

    javaInfoBuilder.addProvider(
        JavaCompilationArgsProvider.class, javaCompilationArgsBuilder.build());

    javaInfoBuilder.addProvider(JavaExportsProvider.class, createJavaExportsProvider(exports));

    javaInfoBuilder.addProvider(JavaPluginInfoProvider.class, createJavaPluginsProvider(exports));

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
      Object javaToolchain,
      Object hostJavabase,
      Location location)
      throws EvalException {
    // No sources to pack, return None
    if (sourceFiles.isEmpty() && sourceJars.isEmpty()) {
      return null;
    }
    // If we only have one source jar, return it directly to avoid action creation
    if (sourceFiles.isEmpty() && sourceJars.size() == 1) {
      return sourceJars.get(0);
    }
    ActionRegistry actionRegistry = actions.asActionRegistry(location, actions);
    Artifact outputSrcJar = getSourceJar(actions.getActionConstructionContext(), outputJar);
    JavaRuntimeInfo javaRuntimeInfo = getJavaRuntimeProvider(location, hostJavabase, null);
    JavaToolchainProvider javaToolchainProvider = getJavaToolchainProvider(location, javaToolchain);
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

  private JavaSourceJarsProvider createJavaSourceJarsProvider(
      Iterable<Artifact> sourceJars, Iterable<JavaInfo> transitiveDeps) {
    NestedSetBuilder<Artifact> transitiveSourceJars = NestedSetBuilder.stableOrder();

    transitiveSourceJars.addAll(sourceJars);

    fetchSourceJars(transitiveDeps).forEach(transitiveSourceJars::addTransitive);

    return JavaSourceJarsProvider.create(transitiveSourceJars.build(), sourceJars);
  }

  private Stream<NestedSet<Artifact>> fetchSourceJars(Iterable<JavaInfo> javaInfos) {
    // TODO(b/123265803): This step should be only necessary if transitive source jar doesn't
    // include sourcejar at this level but they should.
    Stream<NestedSet<Artifact>> sourceJars =
        streamProviders(javaInfos, JavaSourceJarsProvider.class)
            .map(JavaSourceJarsProvider::getSourceJars)
            .map(sourceJarsList -> NestedSetBuilder.wrap(Order.STABLE_ORDER, sourceJarsList));

    Stream<NestedSet<Artifact>> transitiveSourceJars =
        streamProviders(javaInfos, JavaSourceJarsProvider.class)
            .map(JavaSourceJarsProvider::getTransitiveSourceJars);

    return concat(sourceJars, transitiveSourceJars);
  }

  private JavaExportsProvider createJavaExportsProvider(Iterable<JavaInfo> javaInfos) {
    return JavaExportsProvider.merge(
        JavaInfo.fetchProvidersFromList(javaInfos, JavaExportsProvider.class));
  }

  private JavaPluginInfoProvider createJavaPluginsProvider(Iterable<JavaInfo> javaInfos) {
    return JavaPluginInfoProvider.merge(
        JavaInfo.fetchProvidersFromList(javaInfos, JavaPluginInfoProvider.class));
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

    JavaCompilationArgsProvider.Builder javaCompilationArgsBuilder =
        JavaCompilationArgsProvider.builder();
    if (useIjar && !compileTimeJars.isEmpty()) {
      if (!(actions instanceof SkylarkActionFactory)) {
        throw new EvalException(
            location,
            "The value of use_ijar is True. Make sure the ctx.actions argument is valid.");
      }
      if (!isValidJavaToolchain(javaToolchain)) {
        throw new EvalException(
            location,
            "The value of use_ijar is True. Make sure the java_toolchain argument is valid.");
      }
      NestedSetBuilder<Artifact> builder = NestedSetBuilder.naiveLinkOrder();
      for (Artifact compileJar : compileTimeJars) {
        builder.add(
            buildIjar(
                (SkylarkActionFactory) actions,
                compileJar,
                null,
                getJavaToolchainProvider(location, javaToolchain),
                location));
      }
      javaCompilationArgsBuilder.addDirectCompileTimeJars(
          /* interfaceJars = */ builder.build(), /* fullJars= */ compileTimeJars);
    } else {
      javaCompilationArgsBuilder.addDirectCompileTimeJars(
          /* interfaceJars = */ compileTimeJars, /* fullJars= */ compileTimeJars);
    }
    javaCompilationArgsBuilder
        .addTransitiveCompileTimeJars(transitiveCompileTimeJars)
        .addRuntimeJars(runtimeJars)
        .addRuntimeJars(transitiveRuntimeJars);

    JavaInfo javaInfo =
        JavaInfo.Builder.create()
            .addProvider(JavaCompilationArgsProvider.class, javaCompilationArgsBuilder.build())
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
      Object javaToolchain,
      Object hostJavabase,
      SkylarkList<Artifact> sourcepathEntries,
      SkylarkList<Artifact> resources,
      Boolean neverlink,
      JavaSemantics javaSemantics,
      Location location,
      Environment environment)
      throws EvalException {

    if (sourceJars.isEmpty() && sourceFiles.isEmpty() && exports.isEmpty()) {
      throw new EvalException(
          location, "source_jars, sources and exports cannot be simultaneous empty");
    }

    JavaRuntimeInfo javaRuntimeInfo =
        getJavaRuntimeProvider(location, hostJavabase, skylarkRuleContext.getRuleContext());
    if (javaRuntimeInfo == null) {
      throw new EvalException(location, "'host_javabase' must point to a Java runtime");
    }

    JavaToolchainProvider toolchainProvider = getJavaToolchainProvider(location, javaToolchain);

    JavaLibraryHelper helper =
        new JavaLibraryHelper(skylarkRuleContext.getRuleContext())
            .setOutput(outputJar)
            .addSourceJars(sourceJars)
            .addSourceFiles(sourceFiles)
            .addResources(resources)
            .setSourcePathEntries(sourcepathEntries)
            .setJavacOpts(
                ImmutableList.<String>builder()
                    .addAll(toolchainProvider.getJavacOptions())
                    .addAll(
                        javaSemantics.getCompatibleJavacOptions(
                            skylarkRuleContext.getRuleContext(), toolchainProvider))
                    .addAll(
                        JavaCommon.computePerPackageJavacOpts(
                            skylarkRuleContext.getRuleContext(), toolchainProvider))
                    .addAll(javacOpts)
                    .build());

    streamProviders(deps, JavaCompilationArgsProvider.class).forEach(helper::addDep);
    streamProviders(exports, JavaCompilationArgsProvider.class).forEach(helper::addExport);
    helper.setCompilationStrictDepsMode(getStrictDepsMode(Ascii.toUpperCase(strictDepsMode)));
    helper.setPlugins(createJavaPluginsProvider(concat(plugins, deps)));
    helper.setNeverlink(neverlink);

    JavaRuleOutputJarsProvider.Builder outputJarsBuilder = JavaRuleOutputJarsProvider.builder();

    boolean createOutputSourceJar;
    Artifact outputSourceJar;
    if (environment.getSemantics().incompatibleGenerateJavaCommonSourceJar()) {
      outputSourceJar = getSourceJar(skylarkRuleContext.getRuleContext(), outputJar);
      createOutputSourceJar = true;
    } else {
      createOutputSourceJar =
          (sourceJars.size() > 1 || !sourceFiles.isEmpty())
              || (sourceJars.isEmpty() && sourceFiles.isEmpty() && !exports.isEmpty());
      outputSourceJar =
          createOutputSourceJar
              ? getSourceJar(skylarkRuleContext.getRuleContext(), outputJar)
              : sourceJars.get(0);
    }

    JavaInfo.Builder javaInfoBuilder = JavaInfo.Builder.create();
    JavaCompilationArtifacts artifacts =
        helper.build(
            javaSemantics,
            toolchainProvider,
            javaRuntimeInfo,
            SkylarkList.createImmutable(ImmutableList.of()),
            outputJarsBuilder,
            /*createOutputSourceJar=*/ createOutputSourceJar,
            outputSourceJar,
            javaInfoBuilder,
            // Include JavaGenJarsProviders from both deps and exports in the JavaGenJarsProvider
            // added to javaInfoBuilder for this target.
            NestedSetBuilder.wrap(
                Order.STABLE_ORDER,
                JavaInfo.fetchProvidersFromList(concat(deps, exports), JavaGenJarsProvider.class)));

    JavaCompilationArgsProvider javaCompilationArgsProvider =
        helper.buildCompilationArgsProvider(artifacts, true, neverlink);
    Runfiles runfiles =
        new Runfiles.Builder(skylarkRuleContext.getWorkspaceName())
            .addTransitiveArtifactsWrappedInStableOrder(
                javaCompilationArgsProvider.getRuntimeJars())
            .build();

    ImmutableList<Artifact> outputSourceJars = ImmutableList.of(outputSourceJar);

    // When sources are not provided, the subsequent output Jar will be empty. As such, the output
    // Jar is omitted from the set of Runtime Jars.
    if (!sourceJars.isEmpty() || !sourceFiles.isEmpty()) {
      javaInfoBuilder.setRuntimeJars(ImmutableList.of(outputJar));
    }

    return javaInfoBuilder
        .addProvider(JavaCompilationArgsProvider.class, javaCompilationArgsProvider)
        .addProvider(
            JavaSourceJarsProvider.class,
            createJavaSourceJarsProvider(outputSourceJars, concat(deps, exports)))
        .addProvider(JavaRuleOutputJarsProvider.class, outputJarsBuilder.build())
        .addProvider(JavaRunfilesProvider.class, new JavaRunfilesProvider(runfiles))
        .addProvider(
            JavaPluginInfoProvider.class,
            createJavaPluginsProvider(concat(exportedPlugins, exports)))
        .setNeverlink(neverlink)
        .build();
  }

  public Artifact buildIjar(
      SkylarkActionFactory actions,
      Artifact inputJar,
      @Nullable Label targetLabel,
      Object javaToolchain,
      Location location)
      throws EvalException {
    String ijarBasename = FileSystemUtils.removeExtension(inputJar.getFilename()) + "-ijar.jar";
    Artifact interfaceJar = actions.declareFile(ijarBasename, inputJar);
    FilesToRunProvider ijarTarget = getJavaToolchainProvider(location, javaToolchain).getIjar();
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
    actions.registerAction(location, actionBuilder.build(actions.getActionConstructionContext()));
    return interfaceJar;
  }

  public Artifact stampJar(
      SkylarkActionFactory actions,
      Artifact inputJar,
      Label targetLabel,
      Object javaToolchain,
      Location location)
      throws EvalException {
    String basename = FileSystemUtils.removeExtension(inputJar.getFilename()) + "-stamped.jar";
    Artifact outputJar = actions.declareFile(basename, inputJar);
    // ijar doubles as a stamping tool
    FilesToRunProvider ijarTarget = getJavaToolchainProvider(location, javaToolchain).getIjar();
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
    actions.registerAction(location, actionBuilder.build(actions.getActionConstructionContext()));
    return outputJar;
  }

  private static boolean isValidJavaToolchain(Object javaToolchain) {
    return javaToolchain instanceof ConfiguredTarget
        || javaToolchain instanceof JavaToolchainProvider;
  }

  JavaToolchainProvider getJavaToolchainProvider(Location location, Object javaToolchain)
      throws EvalException {
    if (javaToolchain instanceof ConfiguredTarget) {
      // TODO(b/122738702): remove support for passing toolchains as configured targets
      ConfiguredTarget target = (ConfiguredTarget) javaToolchain;
      JavaToolchainProvider javaToolchainProvider = JavaToolchainProvider.from(target);
      if (javaToolchainProvider == null) {
        throw new EvalException(
            location, target.getLabel() + " does not provide JavaToolchainProvider.");
      }
      return javaToolchainProvider;
    }
    if (javaToolchain instanceof JavaToolchainProvider) {
      return (JavaToolchainProvider) javaToolchain;
    }
    throw new EvalException(null, javaToolchain + " is not a JavaToolchainProvider.");
  }

  private static boolean isValidJavaRuntime(Object javaRuntime) {
    return javaRuntime instanceof ConfiguredTarget || javaRuntime instanceof JavaRuntimeInfo;
  }

  private JavaRuntimeInfo getJavaRuntimeProvider(
      Location location, Object javabase, RuleContext ruleContext) throws EvalException {
    if (javabase instanceof TransitiveInfoCollection) {
      // TODO(b/122738702): remove support for passing toolchains as configured targets
      return JavaRuntimeInfo.from((TransitiveInfoCollection) javabase, ruleContext);
    }
    if (javabase instanceof JavaRuntimeInfo) {
      return (JavaRuntimeInfo) javabase;
    }
    throw new EvalException(location, javabase + " is not a JavaRuntimeInfo.");
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
