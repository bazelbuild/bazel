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

package com.google.devtools.build.lib.starlarkbuildapi.java;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.Depset.TypeException;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkActionFactoryApi;
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkRuleContextApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.platform.ConstraintValueInfoApi;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.StarlarkValue;

/** Utilities for Java compilation support in Starlark. */
@StarlarkBuiltin(
    name = "java_common",
    category = DocCategory.TOP_LEVEL_MODULE,
    doc = "Utilities for Java compilation support in Starlark.")
public interface JavaCommonApi<
        FileT extends FileApi,
        JavaInfoT extends JavaInfoApi<FileT, ?, ?>,
        JavaToolchainT extends JavaToolchainStarlarkApiProviderApi,
        BootClassPathT extends ProviderApi,
        ConstraintValueT extends ConstraintValueInfoApi,
        StarlarkRuleContextT extends StarlarkRuleContextApi<ConstraintValueT>,
        StarlarkActionFactoryT extends StarlarkActionFactoryApi>
    extends StarlarkValue {

  @StarlarkMethod(
      name = "create_header_compilation_action",
      documented = false,
      parameters = {
        @Param(name = "ctx"),
        @Param(name = "java_toolchain"),
        @Param(name = "compile_jar"),
        @Param(name = "compile_deps_proto"),
        @Param(name = "plugin_info"),
        @Param(name = "source_files"),
        @Param(name = "source_jars"),
        @Param(name = "compilation_classpath"),
        @Param(name = "direct_jars"),
        @Param(name = "bootclasspath"),
        @Param(name = "compile_time_java_deps"),
        @Param(name = "javac_opts"),
        @Param(name = "strict_deps_mode"),
        @Param(name = "target_label"),
        @Param(name = "injecting_rule_kind"),
        @Param(name = "enable_direct_classpath"),
        @Param(name = "additional_inputs"),
      })
  void createHeaderCompilationAction(
      StarlarkRuleContextT ctx,
      JavaToolchainT javaToolchain,
      FileT compileJar,
      FileT compileDepsProto,
      Info pluginInfo,
      Depset sourceFiles,
      Sequence<?> sourceJars,
      Depset compileTimeClasspath,
      Depset directJars,
      Object bootClassPath,
      Depset compileTimeDeps,
      Sequence<?> javacOpts,
      String strictDepsMode,
      Label targetLabel,
      Object injectingRuleKind,
      boolean enableDirectClasspath,
      Sequence<?> additionalInputs)
      throws EvalException, TypeException, RuleErrorException, LabelSyntaxException;

  @StarlarkMethod(
      name = "create_compilation_action",
      documented = false,
      parameters = {
        @Param(name = "ctx"),
        @Param(name = "java_toolchain"),
        @Param(name = "output"),
        @Param(name = "deps_proto"),
        @Param(name = "gen_class"),
        @Param(name = "gen_source"),
        @Param(name = "manifest_proto"),
        @Param(name = "native_header_jar"),
        @Param(name = "plugin_info"),
        @Param(name = "sources"),
        @Param(name = "source_jars"),
        @Param(name = "resources"),
        @Param(name = "resource_jars"),
        @Param(name = "compilation_classpath"),
        @Param(name = "classpath_resources"),
        @Param(name = "sourcepath"),
        @Param(name = "direct_jars"),
        @Param(name = "bootclasspath"),
        @Param(name = "compile_time_java_deps"),
        @Param(name = "javac_opts"),
        @Param(name = "strict_deps_mode"),
        @Param(name = "target_label"),
        @Param(name = "injecting_rule_kind"),
        @Param(name = "enable_jspecify"),
        @Param(name = "enable_direct_classpath"),
        @Param(name = "additional_inputs"),
        @Param(name = "additional_outputs"),
      })
  void createCompilationAction(
      StarlarkRuleContextT ctx,
      JavaToolchainT javaToolchain,
      FileT output,
      Object depsProto,
      Object genClass,
      Object genSource,
      FileT manifestProto,
      FileT nativeHeader,
      Info pluginInfo,
      Depset sourceFiles,
      Sequence<?> sourceJars,
      Sequence<?> resources,
      Depset resourceJars,
      Depset compileTimeClasspath,
      Sequence<?> classpathResources,
      Sequence<?> sourcepath,
      Depset directJars,
      Object bootClassPath,
      Depset compileTimeJavaDeps,
      Sequence<?> javacOpts,
      String strictDepsMode,
      Label targetLabel,
      Object injectingRuleKind,
      boolean enableJSpecify,
      boolean enableDirectClasspath,
      Sequence<?> additionalInputs,
      Sequence<?> additionalOutputs)
      throws EvalException, TypeException, RuleErrorException, LabelSyntaxException;

  @StarlarkMethod(
      name = "default_javac_opts",
      // This function is experimental for now.
      documented = false,
      parameters = {
        @Param(
            name = "java_toolchain",
            positional = false,
            named = true,
            allowedTypes = {@ParamType(type = JavaToolchainStarlarkApiProviderApi.class)},
            doc =
                "A JavaToolchainInfo to be used for retrieving the ijar "
                    + "tool. Only set when use_ijar is True."),
      })
  // TODO(b/78512644): migrate callers to passing explicit javacopts or using custom toolchains, and
  // delete
  ImmutableList<String> getDefaultJavacOpts(JavaToolchainT javaToolchain) throws EvalException;

  @StarlarkMethod(
      name = "JavaToolchainInfo",
      doc =
          "The key used to retrieve the provider that contains information about the Java "
              + "toolchain being used.",
      structField = true)
  ProviderApi getJavaToolchainProvider();

  @StarlarkMethod(
      name = "JavaRuntimeInfo",
      doc =
          "The key used to retrieve the provider that contains information about the Java "
              + "runtime being used.",
      structField = true)
  ProviderApi getJavaRuntimeProvider();

  @StarlarkMethod(
      name = "BootClassPathInfo",
      doc = "The provider used to supply bootclasspath information",
      structField = true)
  ProviderApi getBootClassPathInfo();

  /** Returns target kind. */
  @StarlarkMethod(
      name = "target_kind",
      parameters = {
        @Param(name = "target", positional = true, named = false, doc = "The target."),
        @Param(
            name = "dereference_aliases",
            positional = false,
            named = true,
            defaultValue = "False",
            documented = false),
      },
      documented = false,
      useStarlarkThread = true)
  String getTargetKind(Object target, boolean dereferenceAliases, StarlarkThread thread)
      throws EvalException;

  @StarlarkMethod(
      name = "get_build_info",
      documented = false,
      parameters = {@Param(name = "ctx"), @Param(name = "is_stamping_enabled")},
      useStarlarkThread = true)
  Sequence<FileT> getBuildInfo(
      StarlarkRuleContextT ruleContext, boolean isStampingEnabled, StarlarkThread thread)
      throws EvalException, InterruptedException;

  @StarlarkMethod(
      name = "experimental_java_proto_library_default_has_services",
      documented = false,
      useStarlarkSemantics = true,
      structField = true,
      doc = "Default value of java_proto_library.has_services")
  boolean getExperimentalJavaProtoLibraryDefaultHasServices(StarlarkSemantics starlarkSemantics)
      throws EvalException;

  @StarlarkMethod(
      name = "collect_native_deps_dirs",
      parameters = {@Param(name = "libraries")},
      useStarlarkThread = true,
      documented = false)
  Sequence<String> collectNativeLibsDirs(Depset libraries, StarlarkThread thread)
      throws EvalException, RuleErrorException, TypeException;

  @StarlarkMethod(
      name = "get_runtime_classpath_for_archive",
      parameters = {@Param(name = "jars"), @Param(name = "excluded_jars")},
      useStarlarkThread = true,
      documented = false)
  Depset getRuntimeClasspathForArchive(
      Depset runtimeClasspath, Depset excludedArtifacts, StarlarkThread thread)
      throws EvalException, TypeException;

  @StarlarkMethod(
      name = "check_provider_instances",
      documented = false,
      parameters = {
        @Param(name = "providers"),
        @Param(name = "what"),
        @Param(name = "provider_type")
      },
      useStarlarkThread = true)
  void checkProviderInstances(
      Sequence<?> providers, String what, ProviderApi providerType, StarlarkThread thread)
      throws EvalException;

  @StarlarkMethod(name = "_google_legacy_api_enabled", documented = false, useStarlarkThread = true)
  boolean isLegacyGoogleApiEnabled(StarlarkThread thread) throws EvalException;

  @StarlarkMethod(
      name = "_check_java_toolchain_is_declared_on_rule",
      documented = false,
      parameters = {@Param(name = "actions")},
      useStarlarkThread = true)
  void checkJavaToolchainIsDeclaredOnRuleForStarlark(
      StarlarkActionFactoryT actions, StarlarkThread thread)
      throws EvalException, LabelSyntaxException;

  @StarlarkMethod(
      name = "_incompatible_depset_for_java_output_source_jars",
      documented = false,
      useStarlarkThread = true)
  boolean isDepsetForJavaOutputSourceJarsEnabled(StarlarkThread thread) throws EvalException;

  @StarlarkMethod(
      name = "wrap_java_info",
      parameters = {@Param(name = "java_info")},
      documented = false,
      useStarlarkThread = true)
  JavaInfoT wrapJavaInfo(Info javaInfo, StarlarkThread thread)
      throws EvalException, RuleErrorException;

  @StarlarkMethod(
      name = "intern_javac_opts",
      parameters = {@Param(name = "javac_opts")},
      documented = false)
  Sequence<String> internJavacOpts(Object javacOpts) throws EvalException, RuleErrorException;
}
