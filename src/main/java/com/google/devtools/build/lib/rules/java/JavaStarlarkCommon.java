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

import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.configuredtargets.AbstractConfiguredTarget;
import com.google.devtools.build.lib.analysis.configuredtargets.MergedConfiguredTarget;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.analysis.starlark.StarlarkActionFactory;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleContext;
import com.google.devtools.build.lib.cmdline.BazelModuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.Depset.TypeException;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.StarlarkInfoNoSchema;
import com.google.devtools.build.lib.packages.StarlarkInfoWithSchema;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.rules.cpp.CppFileTypes;
import com.google.devtools.build.lib.rules.cpp.LibraryToLink;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.java.JavaCommonApi;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;

/** A module that contains Starlark utilities for Java support. */
public class JavaStarlarkCommon
    implements JavaCommonApi<
        Artifact,
        JavaInfo,
        JavaToolchainProvider,
        BootClassPathInfo.Provider,
        ConstraintValueInfo,
        StarlarkRuleContext,
        StarlarkActionFactory> {

  private static final ImmutableSet<String> PRIVATE_STARLARKIFACTION_ALLOWLIST =
      ImmutableSet.of("bazel_internal/test");
  private final JavaSemantics javaSemantics;

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
  public JavaInfo createJavaCompileAction(
      StarlarkRuleContext starlarkRuleContext,
      Sequence<?> sourceJars, // <Artifact> expected
      Sequence<?> sourceFiles, // <Artifact> expected
      Artifact outputJar,
      Object outputSourceJar,
      Sequence<?> javacOpts, // <String> expected
      Sequence<?> deps, // <JavaInfo> expected
      Sequence<?> runtimeDeps, // <JavaInfo> expected
      Sequence<?> exports, // <JavaInfo> expected
      Sequence<?> plugins, // <JavaPluginInfo> expected
      Sequence<?> exportedPlugins, // <JavaPluginInfo> expected
      Sequence<?> nativeLibraries, // <CcInfo> expected.
      Sequence<?> annotationProcessorAdditionalInputs, // <Artifact> expected
      Sequence<?> annotationProcessorAdditionalOutputs, // <Artifact> expected
      String strictDepsMode,
      JavaToolchainProvider javaToolchain,
      Object bootClassPath,
      Object hostJavabase,
      Sequence<?> sourcepathEntries, // <Artifact> expected
      Sequence<?> resources, // <Artifact> expected
      Sequence<?> resourceJars, // <Artifact> expected
      Sequence<?> classpathResources, // <Artifact> expected
      Boolean neverlink,
      Boolean enableAnnotationProcessing,
      Boolean enableCompileJarAction,
      Boolean enableJSpecify,
      boolean includeCompilationInfo,
      Object injectingRuleKind,
      Sequence<?> addExports, // <String> expected
      Sequence<?> addOpens, // <String> expected
      StarlarkThread thread)
      throws EvalException, InterruptedException, RuleErrorException, LabelSyntaxException {
    checkJavaToolchainIsDeclaredOnRule(starlarkRuleContext.getRuleContext());

    final ImmutableList<JavaPluginInfo> pluginsParam =
        JavaPluginInfo.wrapSequence(plugins, "plugins");
    final ImmutableList<JavaPluginInfo> exportedPluginsParam =
        JavaPluginInfo.wrapSequence(exportedPlugins, "exported_plugins");

    // checks for private API access
    if (!enableCompileJarAction
        || !enableJSpecify
        || !includeCompilationInfo
        || !classpathResources.isEmpty()
        || !resourceJars.isEmpty()
        || injectingRuleKind != Starlark.NONE) {
      checkPrivateAccess(thread);
    }

    if (starlarkRuleContext.getRuleContext().useAutoExecGroups()) {
      String javaToolchainType = javaSemantics.getJavaToolchainType();
      if (!starlarkRuleContext.getRuleContext().hasToolchainContext(javaToolchainType)) {
        throw Starlark.errorf(
            "Action declared for non-existent toolchain '%s'.", javaToolchainType);
      }
    }

    return JavaInfoBuildHelper.getInstance()
        .createJavaCompileAction(
            starlarkRuleContext,
            Sequence.cast(sourceJars, Artifact.class, "source_jars"),
            Sequence.cast(sourceFiles, Artifact.class, "source_files"),
            outputJar,
            outputSourceJar == Starlark.NONE ? null : (Artifact) outputSourceJar,
            Sequence.cast(javacOpts, String.class, "javac_opts"),
            JavaInfo.wrapSequence(deps, "deps"),
            JavaInfo.wrapSequence(runtimeDeps, "runtime_deps"),
            JavaInfo.wrapSequence(exports, "exports"),
            pluginsParam,
            exportedPluginsParam,
            Sequence.cast(nativeLibraries, CcInfo.class, "native_libraries"),
            Sequence.cast(
                annotationProcessorAdditionalInputs,
                Artifact.class,
                "annotation_processor_additional_inputs"),
            Sequence.cast(
                annotationProcessorAdditionalOutputs,
                Artifact.class,
                "annotation_processor_additional_outputs"),
            strictDepsMode,
            javaToolchain,
            bootClassPath == Starlark.NONE ? null : (BootClassPathInfo) bootClassPath,
            ImmutableList.copyOf(Sequence.cast(sourcepathEntries, Artifact.class, "sourcepath")),
            Sequence.cast(resources, Artifact.class, "resources"),
            Sequence.cast(resourceJars, Artifact.class, "resource_jars"),
            Sequence.cast(classpathResources, Artifact.class, "classpath_resources"),
            neverlink,
            enableAnnotationProcessing,
            enableCompileJarAction,
            enableJSpecify,
            includeCompilationInfo,
            javaSemantics,
            injectingRuleKind,
            Sequence.cast(addExports, String.class, "add_exports"),
            Sequence.cast(addOpens, String.class, "add_opens"),
            thread);
  }

  @Override
  // TODO(b/78512644): migrate callers to passing explicit javacopts or using custom toolchains, and
  // delete
  public ImmutableList<String> getDefaultJavacOpts(JavaToolchainProvider javaToolchain)
      throws EvalException {
    // We don't have a rule context if the default_javac_opts.java_toolchain parameter is set
    return ((JavaToolchainProvider) javaToolchain).getJavacOptions(/* ruleContext= */ null);
  }

  @Override
  public JavaInfo mergeJavaProviders(
      Sequence<?> providers, /* <JavaInfo> expected. */
      boolean mergeJavaOutputs,
      boolean mergeSourceJars,
      StarlarkThread thread)
      throws EvalException {
    if (!mergeJavaOutputs || !mergeSourceJars) {
      checkPrivateAccess(thread);
    }
    return JavaInfo.merge(
        JavaInfo.wrapSequence(providers, "providers"), mergeJavaOutputs, mergeSourceJars);
  }

  @Override
  public Provider getJavaToolchainProvider() {
    return JavaToolchainProvider.PROVIDER;
  }

  @Override
  public Provider getJavaRuntimeProvider() {
    return JavaRuntimeInfo.PROVIDER;
  }

  @Override
  public ProviderApi getBootClassPathInfo() {
    return BootClassPathInfo.PROVIDER;
  }

  @Override
  public String getTargetKind(Object target, boolean dereferenceAliases, StarlarkThread thread)
      throws EvalException {
    checkPrivateAccess(thread);
    if (target instanceof MergedConfiguredTarget) {
      target = ((MergedConfiguredTarget) target).getBaseConfiguredTarget();
    }
    if (dereferenceAliases && target instanceof ConfiguredTarget) {
      target = ((ConfiguredTarget) target).getActual();
    }
    if (target instanceof AbstractConfiguredTarget) {
      return ((AbstractConfiguredTarget) target).getRuleClassString();
    }
    return "";
  }

  protected static void checkPrivateAccess(StarlarkThread thread) throws EvalException {
    Label label =
        ((BazelModuleContext) Module.ofInnermostEnclosingStarlarkFunction(thread).getClientData())
            .label();
    if (!PRIVATE_STARLARKIFACTION_ALLOWLIST.contains(label.getPackageName())
        && !label.getPackageIdentifier().getRepository().getName().equals("_builtins")) {
      throw Starlark.errorf("Rule in '%s' cannot use private API", label.getPackageName());
    }
  }

  @Override
  public Sequence<Artifact> getBuildInfo(
      StarlarkRuleContext starlarkRuleContext, boolean isStampingEnabled, StarlarkThread thread)
      throws EvalException, InterruptedException {
    checkPrivateAccess(thread);
    RuleContext ruleContext = starlarkRuleContext.getRuleContext();
    return StarlarkList.immutableCopyOf(
        ruleContext
            .getAnalysisEnvironment()
            .getBuildInfo(
                isStampingEnabled, JavaBuildInfoFactory.KEY, ruleContext.getConfiguration()));
  }

  @Override
  public boolean getExperimentalJavaProtoLibraryDefaultHasServices(
      StarlarkSemantics starlarkSemantics) throws EvalException {
    return starlarkSemantics.getBool(
        BuildLanguageOptions.EXPERIMENTAL_JAVA_PROTO_LIBRARY_DEFAULT_HAS_SERVICES);
  }

  @Override
  public Sequence<String> collectNativeLibsDirs(Depset libraries, StarlarkThread thread)
      throws EvalException, TypeException {
    checkPrivateAccess(thread);
    ImmutableList<Artifact> nativeLibraries =
        LibraryToLink.getDynamicLibrariesForLinking(libraries.getSet(LibraryToLink.class));
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
            .map(artifact -> artifact.getRootRelativePath().getParentDirectory().getPathString())
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

  private static String printableType(Object elem) {
    if (elem instanceof StarlarkInfoWithSchema) {
      return ((StarlarkInfoWithSchema) elem).getProvider().getPrintableName();
    }
    return Starlark.type(elem);
  }

  @Override
  public boolean isLegacyGoogleApiEnabled(StarlarkThread thread) throws EvalException {
    checkPrivateAccess(thread);
    return thread.getSemantics().getBool(BuildLanguageOptions.EXPERIMENTAL_GOOGLE_LEGACY_API);
  }

  @Override
  public boolean isDepsetForJavaOutputSourceJarsEnabled(StarlarkThread thread)
      throws EvalException {
    checkPrivateAccess(thread);
    return thread
        .getSemantics()
        .getBool(BuildLanguageOptions.INCOMPATIBLE_DEPSET_FOR_JAVA_OUTPUT_SOURCE_JARS);
  }

  static boolean isInstanceOfProvider(Object obj, Provider provider) {
    if (obj instanceof NativeInfo) {
      return ((NativeInfo) obj).getProvider().getKey().equals(provider.getKey());
    } else if (obj instanceof StarlarkInfoWithSchema) {
      return ((StarlarkInfoWithSchema) obj).getProvider().getKey().equals(provider.getKey());
    } else if (obj instanceof StarlarkInfoNoSchema) {
      return ((StarlarkInfoNoSchema) obj).getProvider().getKey().equals(provider.getKey());
    }
    return false;
  }
}
