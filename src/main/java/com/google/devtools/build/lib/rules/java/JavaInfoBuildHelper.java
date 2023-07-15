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
import static com.google.devtools.build.lib.rules.java.JavaInfo.streamProviders;
import static java.util.stream.Stream.concat;

import com.google.common.base.Ascii;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.config.CoreOptionConverters.StrictDepsMode;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleContext;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.shell.ShellUtils;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Stream;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkThread;

/** Implements logic for creating JavaInfo from different set of input parameters. */
final class JavaInfoBuildHelper {
  private static final JavaInfoBuildHelper INSTANCE = new JavaInfoBuildHelper();

  private JavaInfoBuildHelper() {}

  public static JavaInfoBuildHelper getInstance() {
    return INSTANCE;
  }

  private static ImmutableList<JavaGenJarsProvider> collectJavaGenJarsProviders(
      Iterable<JavaInfo> javaInfos) throws RuleErrorException, EvalException {
    ImmutableList.Builder<JavaGenJarsProvider> builder = ImmutableList.builder();
    for (JavaInfo javaInfo : javaInfos) {
      JavaGenJarsProvider provider = javaInfo.getGenJarsProvider();
      if (provider != null && !provider.isEmpty()) {
        builder.add(provider);
      }
    }
    return builder.build();
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

  private static JavaModuleFlagsProvider createJavaModuleFlagsProvider(
      List<String> addExports, List<String> addOpens, Iterable<JavaInfo> transitiveDeps) {
    return JavaModuleFlagsProvider.create(
        addExports, addOpens, streamProviders(transitiveDeps, JavaModuleFlagsProvider.class));
  }

  private JavaPluginInfo mergeExportedJavaPluginInfo(
      Iterable<JavaPluginInfo> plugins, Iterable<JavaInfo> javaInfos) {
    return JavaPluginInfo.mergeWithoutJavaOutputs(
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
      List<JavaInfo> exports,
      List<JavaPluginInfo> plugins,
      List<JavaPluginInfo> exportedPlugins,
      List<CcInfo> nativeLibraries,
      List<Artifact> annotationProcessorAdditionalInputs,
      List<Artifact> annotationProcessorAdditionalOutputs,
      String strictDepsMode,
      JavaToolchainProvider javaToolchain,
      BootClassPathInfo bootClassPath,
      ImmutableList<Artifact> sourcepathEntries,
      List<Artifact> resources,
      List<Artifact> resourceJars,
      List<Artifact> classpathResources,
      Boolean neverlink,
      Boolean enableAnnotationProcessing,
      Boolean enableCompileJarAction,
      boolean enableJSpecify,
      boolean includeCompilationInfo,
      JavaSemantics javaSemantics,
      Object injectingRuleKind,
      List<String> addExports,
      List<String> addOpens,
      StarlarkThread thread)
      throws EvalException, InterruptedException, RuleErrorException {

    JavaToolchainProvider toolchainProvider = javaToolchain;

    JavaPluginInfo pluginInfo = mergeExportedJavaPluginInfo(plugins, deps);
    ImmutableList.Builder<String> allJavacOptsBuilder =
        ImmutableList.<String>builder()
            .addAll(toolchainProvider.getJavacOptions(starlarkRuleContext.getRuleContext()))
            .addAll(
                javaSemantics.getCompatibleJavacOptions(
                    starlarkRuleContext.getRuleContext(), toolchainProvider));
    if (pluginInfo
        .plugins()
        .processorClasses()
        .toSet()
        .contains("com.google.devtools.build.runfiles.AutoBazelRepositoryProcessor")) {
      allJavacOptsBuilder.add(
          "-Abazel.repository=" + starlarkRuleContext.getRuleContext().getRepository().getName());
    }
    allJavacOptsBuilder
        .addAll(
            JavaCommon.computePerPackageJavacOpts(
                starlarkRuleContext.getRuleContext(), toolchainProvider))
        .addAll(JavaModuleFlagsProvider.toFlags(addExports, addOpens))
        .addAll(tokenize(javacOpts));

    JavaLibraryHelper helper =
        new JavaLibraryHelper(starlarkRuleContext.getRuleContext())
            .setOutput(outputJar)
            .addSourceJars(sourceJars)
            .addSourceFiles(sourceFiles)
            .addResources(resources)
            .addResourceJars(resourceJars)
            .addClasspathResources(classpathResources)
            .setSourcePathEntries(sourcepathEntries)
            .addAdditionalOutputs(annotationProcessorAdditionalOutputs)
            .enableJspecify(enableJSpecify)
            .setJavacOpts(allJavacOptsBuilder.build());

    if (injectingRuleKind != Starlark.NONE) {
      helper.setInjectingRuleKind((String) injectingRuleKind);
    }

    streamProviders(runtimeDeps, JavaCompilationArgsProvider.class).forEach(helper::addRuntimeDep);
    streamProviders(deps, JavaCompilationArgsProvider.class).forEach(helper::addDep);
    streamProviders(exports, JavaCompilationArgsProvider.class).forEach(helper::addExport);
    helper.setCompilationStrictDepsMode(getStrictDepsMode(Ascii.toUpperCase(strictDepsMode)));
    // Optimization: skip this if there are no annotation processors, to avoid unnecessarily
    // disabling the direct classpath optimization if `enable_annotation_processor = False`
    // but there aren't any annotation processors.
    if (!enableAnnotationProcessing && !pluginInfo.plugins().processorClasses().isEmpty()) {
      pluginInfo = pluginInfo.disableAnnotationProcessing();
      helper.enableDirectClasspath(false);
    }
    helper.setPlugins(pluginInfo);
    helper.setNeverlink(neverlink);

    JavaRuleOutputJarsProvider.Builder outputJarsBuilder = JavaRuleOutputJarsProvider.builder();

    boolean createOutputSourceJar =
        !(sourceJars.size() == 1
            && sourceFiles.isEmpty()
            && sourceJars.get(0).equals(outputSourceJar));
    if (outputSourceJar == null) {
      outputSourceJar = getDerivedSourceJar(starlarkRuleContext.getRuleContext(), outputJar);
    }

    JavaInfo.Builder javaInfoBuilder = JavaInfo.Builder.create();
    JavaCompilationArtifacts artifacts =
        helper.build(
            javaSemantics,
            toolchainProvider,
            bootClassPath,
            outputJarsBuilder,
            createOutputSourceJar,
            includeCompilationInfo,
            outputSourceJar,
            enableCompileJarAction,
            javaInfoBuilder,
            // Include JavaGenJarsProviders from both deps and exports in the JavaGenJarsProvider
            // added to javaInfoBuilder for this target.
            collectJavaGenJarsProviders(concat(deps, exports)),
            ImmutableList.copyOf(annotationProcessorAdditionalInputs));

    JavaCompilationArgsProvider javaCompilationArgsProvider =
        helper.buildCompilationArgsProvider(artifacts, true, neverlink);

    ImmutableList<Artifact> outputSourceJars = ImmutableList.of(outputSourceJar);

    // When sources are not provided, the subsequent output Jar will be empty. As such, the output
    // Jar is omitted from the set of Runtime Jars.
    if (!sourceJars.isEmpty() || !sourceFiles.isEmpty() || !resources.isEmpty()) {
      javaInfoBuilder.setRuntimeJars(ImmutableList.of(outputJar));
    }

    ImmutableList<JavaCcInfoProvider> transitiveNativeLibraries =
        Streams.concat(
                streamProviders(runtimeDeps, JavaCcInfoProvider.class),
                streamProviders(exports, JavaCcInfoProvider.class),
                streamProviders(deps, JavaCcInfoProvider.class),
                Stream.of(JavaCcInfoProvider.create(CcInfo.merge(nativeLibraries))))
            .collect(toImmutableList());

    return javaInfoBuilder
        .javaCompilationArgs(javaCompilationArgsProvider)
        .javaSourceJars(
            createJavaSourceJarsProvider(outputSourceJars, concat(runtimeDeps, exports, deps)))
        .javaRuleOutputs(outputJarsBuilder.build())
        .javaPluginInfo(mergeExportedJavaPluginInfo(exportedPlugins, exports))
        .javaCcInfo(JavaCcInfoProvider.merge(transitiveNativeLibraries))
        .javaModuleFlags(
            createJavaModuleFlagsProvider(addExports, addOpens, concat(runtimeDeps, exports, deps)))
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
