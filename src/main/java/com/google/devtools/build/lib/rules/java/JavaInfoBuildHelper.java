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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.Iterables.concat;
import static com.google.common.collect.Streams.stream;
import static com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider.ClasspathType.BOTH;
import static com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider.ClasspathType.COMPILE_ONLY;
import static com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider.ClasspathType.RUNTIME_ONLY;
import static com.google.devtools.build.lib.rules.java.JavaInfo.streamProviders;
import static java.util.stream.Stream.concat;

import com.google.common.base.Ascii;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.ActionRegistry;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.config.CoreOptionConverters.StrictDepsMode;
import com.google.devtools.build.lib.analysis.starlark.StarlarkActionFactory;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider.ClasspathType;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider.JavaOutput;
import com.google.devtools.build.lib.shell.ShellUtils;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Stream;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.syntax.Location;

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
   * @param javaOutput the artifacts that were created as a result of a compilation (e.g. javac,
   *     scalac, etc)
   * @param neverlink if true only use this library for compilation and not at runtime
   * @param compileTimeDeps compile time dependencies that were used to create the output jar
   * @param runtimeDeps runtime dependencies that are needed for this library
   * @param exports libraries to make available for users of this library. <a
   *     href="https://docs.bazel.build/versions/master/be/java.html#java_library"
   *     target="_top">java_library.exports</a>
   * @param exportedPlugins A list of exported plugins.
   * @param nativeLibraries CC library dependencies that are needed for this library
   * @return new created JavaInfo instance
   */
  JavaInfo createJavaInfo(
      JavaOutput javaOutput,
      Boolean neverlink,
      Sequence<JavaInfo> compileTimeDeps,
      Sequence<JavaInfo> runtimeDeps,
      Sequence<JavaInfo> exports,
      Sequence<JavaPluginInfo> exportedPlugins,
      Sequence<CcInfo> nativeLibraries,
      Location location,
      boolean withExportsProvider) {
    JavaInfo.Builder javaInfoBuilder = JavaInfo.Builder.create();
    javaInfoBuilder.setLocation(location);

    JavaCompilationArgsProvider.Builder javaCompilationArgsBuilder =
        JavaCompilationArgsProvider.builder();

    if (!neverlink) {
      javaCompilationArgsBuilder.addRuntimeJar(javaOutput.getClassJar());
    }
    if (javaOutput.getCompileJar() != null) {
      javaCompilationArgsBuilder.addDirectCompileTimeJar(
          /* interfaceJar= */ javaOutput.getCompileJar(), /* fullJar= */ javaOutput.getClassJar());
    }
    if (javaOutput.getCompileJdeps() != null) {
      javaCompilationArgsBuilder.addCompileTimeJavaDependencyArtifacts(
          NestedSetBuilder.create(Order.STABLE_ORDER, javaOutput.getCompileJdeps()));
    }

    JavaRuleOutputJarsProvider javaRuleOutputJarsProvider =
        JavaRuleOutputJarsProvider.builder().addJavaOutput(javaOutput).build();
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

    if (withExportsProvider) {
      javaInfoBuilder.addProvider(JavaExportsProvider.class, createJavaExportsProvider(exports));
    }

    javaInfoBuilder.javaPluginInfo(mergeExportedJavaPluginInfo(exportedPlugins, exports));

    javaInfoBuilder.addProvider(
        JavaSourceJarsProvider.class,
        createJavaSourceJarsProvider(
            javaOutput.getSourceJars(), concat(compileTimeDeps, runtimeDeps, exports)));

    javaInfoBuilder.addProvider(
        JavaGenJarsProvider.class,
        JavaGenJarsProvider.create(
            false,
            javaOutput.getGeneratedClassJar(),
            javaOutput.getGeneratedSourceJar(),
            JavaPluginInfo.empty(),
            JavaInfo.fetchProvidersFromList(
                concat(compileTimeDeps, exports), JavaGenJarsProvider.class)));

    javaInfoBuilder.setRuntimeJars(ImmutableList.of(javaOutput.getClassJar()));

    ImmutableList<JavaCcInfoProvider> transitiveNativeLibraries =
        Streams.concat(
                streamProviders(runtimeDeps, JavaCcInfoProvider.class),
                streamProviders(exports, JavaCcInfoProvider.class),
                streamProviders(compileTimeDeps, JavaCcInfoProvider.class),
                Stream.of(new JavaCcInfoProvider(CcInfo.merge(nativeLibraries))))
            .collect(toImmutableList());
    javaInfoBuilder.addProvider(
        JavaCcInfoProvider.class, JavaCcInfoProvider.merge(transitiveNativeLibraries));

    return javaInfoBuilder.build();
  }

  /**
   * Creates action which creates archive with all source files inside. Takes all filer from
   * sourceFiles collection and all files from every sourceJars. Name of Artifact generated based on
   * outputJar.
   *
   * @param outputJar name of output Jar artifact.
   * @param outputSourceJar name of output source Jar artifact, or {@code null}. If unset, defaults
   *     to base name of the output jar with the suffix {@code -src.jar}.
   * @return generated artifact (can also be empty)
   */
  Artifact packSourceFiles(
      StarlarkActionFactory actions,
      Artifact outputJar,
      Artifact outputSourceJar,
      List<Artifact> sourceFiles,
      List<Artifact> sourceJars,
      JavaToolchainProvider javaToolchain)
      throws EvalException {
    if (outputJar == null && outputSourceJar == null) {
      throw Starlark.errorf(
          "pack_sources requires at least one of the parameters output_jar or output_source_jar");
    }
    // If we only have one source jar, return it directly to avoid action creation
    if (sourceFiles.isEmpty() && sourceJars.size() == 1 && outputSourceJar == null) {
      return sourceJars.get(0);
    }
    ActionRegistry actionRegistry = actions.asActionRegistry(actions);
    if (outputSourceJar == null) {
      outputSourceJar = getDerivedSourceJar(actions.getActionConstructionContext(), outputJar);
    }
    SingleJarActionBuilder.createSourceJarAction(
        actionRegistry,
        actions.getActionConstructionContext(),
        javaToolchain.getJavaSemantics(),
        NestedSetBuilder.<Artifact>wrap(Order.STABLE_ORDER, sourceFiles),
        NestedSetBuilder.<Artifact>wrap(Order.STABLE_ORDER, sourceJars),
        outputSourceJar,
        javaToolchain);
    return outputSourceJar;
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

    return concat(transitiveSourceJars, sourceJars);
  }

  private JavaExportsProvider createJavaExportsProvider(Iterable<JavaInfo> exports) {
    return JavaExportsProvider.merge(
        JavaInfo.fetchProvidersFromList(exports, JavaExportsProvider.class));
  }

  private JavaPluginInfo mergeExportedJavaPluginInfo(
      Iterable<JavaPluginInfo> plugins, Iterable<JavaInfo> javaInfos) {
    return JavaPluginInfo.merge(
        concat(
            plugins,
            stream(javaInfos)
                .map(JavaInfo::getJavaPluginInfo)
                .filter(Objects::nonNull)
                .collect(toImmutableList())));
  }

  public JavaInfo createJavaCompileAction(
      StarlarkRuleContext starlarkRuleContext,
      List<Artifact> sourceJars,
      List<Artifact> sourceFiles,
      Artifact outputJar,
      Artifact outputSourceJar,
      List<String> javacOpts,
      List<JavaInfo> deps,
      List<JavaInfo> runtimeDeps,
      List<JavaInfo> experimentalLocalCompileTimeDeps,
      List<JavaInfo> exports,
      List<JavaPluginInfo> plugins,
      List<JavaPluginInfo> exportedPlugins,
      List<CcInfo> nativeLibraries,
      List<Artifact> annotationProcessorAdditionalInputs,
      List<Artifact> annotationProcessorAdditionalOutputs,
      String strictDepsMode,
      JavaToolchainProvider javaToolchain,
      ImmutableList<Artifact> sourcepathEntries,
      List<Artifact> resources,
      Boolean neverlink,
      Boolean enableAnnotationProcessing,
      JavaSemantics javaSemantics,
      StarlarkThread thread)
      throws EvalException, InterruptedException {

    JavaToolchainProvider toolchainProvider = javaToolchain;

    JavaLibraryHelper helper =
        new JavaLibraryHelper(starlarkRuleContext.getRuleContext())
            .setOutput(outputJar)
            .addSourceJars(sourceJars)
            .addSourceFiles(sourceFiles)
            .addResources(resources)
            .setSourcePathEntries(sourcepathEntries)
            .addAdditionalOutputs(annotationProcessorAdditionalOutputs)
            .setJavacOpts(
                ImmutableList.<String>builder()
                    .addAll(toolchainProvider.getJavacOptions(starlarkRuleContext.getRuleContext()))
                    .addAll(
                        javaSemantics.getCompatibleJavacOptions(
                            starlarkRuleContext.getRuleContext(), toolchainProvider))
                    .addAll(
                        JavaCommon.computePerPackageJavacOpts(
                            starlarkRuleContext.getRuleContext(), toolchainProvider))
                    .addAll(tokenize(javacOpts))
                    .build());

    streamProviders(runtimeDeps, JavaCompilationArgsProvider.class).forEach(helper::addRuntimeDep);
    streamProviders(deps, JavaCompilationArgsProvider.class).forEach(helper::addDep);
    streamProviders(exports, JavaCompilationArgsProvider.class).forEach(helper::addExport);
    helper.setCompilationStrictDepsMode(getStrictDepsMode(Ascii.toUpperCase(strictDepsMode)));
    JavaPluginInfo pluginInfo = mergeExportedJavaPluginInfo(plugins, deps);
    if (!enableAnnotationProcessing) {
      pluginInfo = pluginInfo.disableAnnotationProcessing();
    }
    helper.setPlugins(pluginInfo);
    helper.setNeverlink(neverlink);

    NestedSet<Artifact> localCompileTimeDeps =
        JavaCompilationArgsProvider.merge(
                streamProviders(experimentalLocalCompileTimeDeps, JavaCompilationArgsProvider.class)
                    .collect(toImmutableList()))
            .getTransitiveCompileTimeJars();

    JavaRuleOutputJarsProvider.Builder outputJarsBuilder = JavaRuleOutputJarsProvider.builder();

    if (outputSourceJar == null) {
      outputSourceJar = getDerivedSourceJar(starlarkRuleContext.getRuleContext(), outputJar);
    }

    JavaInfo.Builder javaInfoBuilder = JavaInfo.Builder.create();
    JavaCompilationArtifacts artifacts =
        helper.build(
            javaSemantics,
            toolchainProvider,
            outputJarsBuilder,
            /*createOutputSourceJar=*/ true,
            outputSourceJar,
            javaInfoBuilder,
            // Include JavaGenJarsProviders from both deps and exports in the JavaGenJarsProvider
            // added to javaInfoBuilder for this target.
            JavaInfo.fetchProvidersFromList(concat(deps, exports), JavaGenJarsProvider.class),
            ImmutableList.copyOf(annotationProcessorAdditionalInputs),
            localCompileTimeDeps);

    JavaCompilationArgsProvider javaCompilationArgsProvider =
        helper.buildCompilationArgsProvider(artifacts, true, neverlink);

    ImmutableList<Artifact> outputSourceJars = ImmutableList.of(outputSourceJar);

    // When sources are not provided, the subsequent output Jar will be empty. As such, the output
    // Jar is omitted from the set of Runtime Jars.
    if (!sourceJars.isEmpty() || !sourceFiles.isEmpty()) {
      javaInfoBuilder.setRuntimeJars(ImmutableList.of(outputJar));
    }

    ImmutableList<JavaCcInfoProvider> transitiveNativeLibraries =
        Streams.concat(
                streamProviders(runtimeDeps, JavaCcInfoProvider.class),
                streamProviders(exports, JavaCcInfoProvider.class),
                streamProviders(deps, JavaCcInfoProvider.class),
                Stream.of(new JavaCcInfoProvider(CcInfo.merge(nativeLibraries))))
            .collect(toImmutableList());

    return javaInfoBuilder
        .addProvider(JavaCompilationArgsProvider.class, javaCompilationArgsProvider)
        .addProvider(
            JavaSourceJarsProvider.class,
            createJavaSourceJarsProvider(outputSourceJars, concat(runtimeDeps, exports, deps)))
        .addProvider(JavaRuleOutputJarsProvider.class, outputJarsBuilder.build())
        .javaPluginInfo(mergeExportedJavaPluginInfo(exportedPlugins, exports))
        .addProvider(JavaExportsProvider.class, createJavaExportsProvider(exports))
        .addProvider(JavaCcInfoProvider.class, JavaCcInfoProvider.merge(transitiveNativeLibraries))
        .addTransitiveOnlyRuntimeJarsToJavaInfo(deps)
        .addTransitiveOnlyRuntimeJarsToJavaInfo(exports)
        .addTransitiveOnlyRuntimeJarsToJavaInfo(runtimeDeps)
        .setNeverlink(neverlink)
        .build();
  }

  private static List<String> tokenize(List<String> input) throws EvalException {
    List<String> output = new ArrayList<>();
    for (String token : input) {
      try {
        ShellUtils.tokenize(output, token);
      } catch (ShellUtils.TokenizationException e) {
        throw Starlark.errorf("%s", e.getMessage());
      }
    }
    return output;
  }

  public Artifact buildIjar(
      StarlarkActionFactory actions,
      Artifact inputJar,
      @Nullable Label targetLabel,
      JavaToolchainProvider javaToolchain)
      throws EvalException {
    String ijarBasename = FileSystemUtils.removeExtension(inputJar.getFilename()) + "-ijar.jar";
    Artifact interfaceJar = actions.declareFile(ijarBasename, inputJar);
    FilesToRunProvider ijarTarget = javaToolchain.getIjar();
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
      StarlarkActionFactory actions,
      Artifact inputJar,
      Label targetLabel,
      JavaToolchainProvider javaToolchain)
      throws EvalException {
    String basename = FileSystemUtils.removeExtension(inputJar.getFilename()) + "-stamped.jar";
    Artifact outputJar = actions.declareFile(basename, inputJar);
    // ijar doubles as a stamping tool
    FilesToRunProvider ijarTarget = (javaToolchain).getIjar();
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

  private static Artifact getDerivedSourceJar(
      ActionConstructionContext context, Artifact outputJar) {
    return JavaCompilationHelper.derivedArtifact(context, outputJar, "", "-src.jar");
  }
}
