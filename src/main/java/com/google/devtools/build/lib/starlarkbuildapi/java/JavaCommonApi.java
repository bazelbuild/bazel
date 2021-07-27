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
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkActionFactoryApi;
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkRuleContextApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.CcInfoApi;
import com.google.devtools.build.lib.starlarkbuildapi.platform.ConstraintValueInfoApi;
import javax.annotation.Nullable;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.NoneType;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.StarlarkValue;

/** Utilities for Java compilation support in Starlark. */
@StarlarkBuiltin(name = "java_common", doc = "Utilities for Java compilation support in Starlark.")
public interface JavaCommonApi<
        FileT extends FileApi,
        JavaInfoT extends JavaInfoApi<FileT, ?, ?>,
        JavaToolchainT extends JavaToolchainStarlarkApiProviderApi,
        ConstraintValueT extends ConstraintValueInfoApi,
        StarlarkRuleContextT extends StarlarkRuleContextApi<ConstraintValueT>,
        StarlarkActionFactoryT extends StarlarkActionFactoryApi>
    extends StarlarkValue {

  @StarlarkMethod(
      name = "provider",
      structField = true,
      doc =
          "Returns the Java declared provider. <br>"
              + "The same value is accessible as <code>JavaInfo</code>. <br>"
              + "Prefer using <code>JavaInfo</code> in new code.")
  ProviderApi getJavaProvider();

  @StarlarkMethod(
      name = "compile",
      doc =
          "Compiles Java source files/jars from the implementation of a Starlark rule and returns "
              + "a provider that represents the results of the compilation and can be added to "
              + "the set of providers emitted by this rule.",
      parameters = {
        @Param(name = "ctx", positional = true, named = false, doc = "The rule context."),
        @Param(
            name = "source_jars",
            positional = false,
            named = true,
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = FileApi.class)},
            defaultValue = "[]",
            doc =
                "A list of the jars to be compiled. At least one of source_jars or source_files"
                    + " should be specified."),
        @Param(
            name = "source_files",
            positional = false,
            named = true,
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = FileApi.class)},
            defaultValue = "[]",
            doc =
                "A list of the Java source files to be compiled. At least one of source_jars or "
                    + "source_files should be specified."),
        @Param(name = "output", positional = false, named = true),
        @Param(
            name = "output_source_jar",
            positional = false,
            named = true,
            allowedTypes = {
              @ParamType(type = FileApi.class),
              @ParamType(type = NoneType.class),
            },
            defaultValue = "None",
            doc = "The output source jar. Optional. Defaults to `{output_jar}-src.jar` if unset."),
        @Param(
            name = "javac_opts",
            positional = false,
            named = true,
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = String.class)},
            defaultValue = "[]",
            doc = "A list of the desired javac options. Optional."),
        @Param(
            name = "deps",
            positional = false,
            named = true,
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = JavaInfoApi.class)},
            defaultValue = "[]",
            doc = "A list of dependencies. Optional."),
        @Param(
            name = "runtime_deps",
            positional = false,
            named = true,
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = JavaInfoApi.class)},
            defaultValue = "[]",
            doc = "A list of runtime dependencies. Optional."),
        @Param(
            name = "exports",
            positional = false,
            named = true,
            allowedTypes = {
              @ParamType(type = Sequence.class, generic1 = JavaInfoApi.class),
            },
            defaultValue = "[]",
            doc = "A list of exports. Optional."),
        @Param(
            name = "plugins",
            positional = false,
            named = true,
            allowedTypes = {
              @ParamType(type = Sequence.class, generic1 = JavaPluginInfoApi.class),
              @ParamType(type = Sequence.class, generic1 = JavaInfoApi.class)
            },
            defaultValue = "[]",
            doc = "A list of plugins. Optional."),
        @Param(
            name = "exported_plugins",
            positional = false,
            named = true,
            allowedTypes = {
              @ParamType(type = Sequence.class, generic1 = JavaPluginInfoApi.class),
              @ParamType(type = Sequence.class, generic1 = JavaInfoApi.class)
            },
            defaultValue = "[]",
            doc = "A list of exported plugins. Optional."),
        @Param(
            name = "native_libraries",
            positional = false,
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = CcInfoApi.class)},
            named = true,
            defaultValue = "[]",
            doc = "CC native library dependencies that are needed for this library."),
        @Param(
            name = "annotation_processor_additional_inputs",
            positional = false,
            named = true,
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = FileApi.class)},
            defaultValue = "[]",
            doc =
                "A list of inputs that the Java compilation action will take in addition to the "
                    + "Java sources for annotation processing."),
        @Param(
            name = "annotation_processor_additional_outputs",
            positional = false,
            named = true,
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = FileApi.class)},
            defaultValue = "[]",
            doc =
                "A list of outputs that the Java compilation action will output in addition to "
                    + "the class jar from annotation processing."),
        @Param(
            name = "strict_deps",
            defaultValue = "'ERROR'",
            positional = false,
            named = true,
            doc =
                "A string that specifies how to handle strict deps. Possible values: 'OFF', "
                    + "'ERROR', 'WARN' and 'DEFAULT'. For more details see "
                    + "https://docs.bazel.build/versions/main/bazel-user-manual.html#"
                    + "flag--strict_java_deps. By default 'ERROR'."),
        @Param(
            name = "java_toolchain",
            positional = false,
            named = true,
            doc = "A JavaToolchainInfo to be used for this compilation. Mandatory."),
        @Param(
            name = "host_javabase",
            positional = false,
            named = true,
            doc =
                "Deprecated: You can drop this parameter (host_javabase is provided with "
                    + "java_toolchain)",
            defaultValue = "None",
            disableWithFlag = BuildLanguageOptions.INCOMPATIBLE_JAVA_COMMON_PARAMETERS,
            valueWhenDisabled = "None"),
        @Param(
            name = "sourcepath",
            positional = false,
            named = true,
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = FileApi.class)},
            defaultValue = "[]"),
        @Param(
            name = "resources",
            positional = false,
            named = true,
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = FileApi.class)},
            defaultValue = "[]"),
        @Param(name = "neverlink", positional = false, named = true, defaultValue = "False"),
        @Param(
            name = "enable_annotation_processing",
            positional = false,
            named = true,
            defaultValue = "True",
            doc =
                "Disables annotation processing in this compilation, causing any annotation"
                    + " processors provided in plugins or in exported_plugins of deps to be"
                    + " ignored.")
      },
      useStarlarkThread = true)
  JavaInfoT createJavaCompileAction(
      StarlarkRuleContextT starlarkRuleContext,
      Sequence<?> sourceJars, // <FileT> expected.
      Sequence<?> sourceFiles, // <FileT> expected.
      FileT outputJar,
      Object outputSourceJar,
      Sequence<?> javacOpts, // <String> expected.
      Sequence<?> deps, // <JavaInfoT> expected.
      Sequence<?> runtimeDeps, // <JavaInfoT> expected.
      Sequence<?> exports, // <JavaInfoT> expected.
      Sequence<?> plugins, // <JavaInfoT> expected.
      Sequence<?> exportedPlugins, // <JavaInfoT> expected.
      Sequence<?> nativeLibraries, // <CcInfoT> expected.
      Sequence<?> annotationProcessorAdditionalInputs, // <FileT> expected.
      Sequence<?> annotationProcessorAdditionalOutputs, // <FileT> expected.
      String strictDepsMode,
      JavaToolchainT javaToolchain,
      Object hostJavabase,
      Sequence<?> sourcepathEntries, // <FileT> expected.
      Sequence<?> resources, // <FileT> expected.
      Boolean neverlink,
      Boolean enableAnnotationProcessing,
      StarlarkThread thread)
      throws EvalException, InterruptedException;

  @StarlarkMethod(
      name = "run_ijar",
      doc =
          "Runs ijar on a jar, stripping it of its method bodies. This helps reduce rebuilding "
              + "of dependent jars during any recompiles consisting only of simple changes to "
              + "method implementations. The return value is typically passed to "
              + "<code><a class=\"anchor\" href=\"JavaInfo.html\">"
              + "JavaInfo</a>#compile_jar</code>.",
      parameters = {
        @Param(name = "actions", named = true, doc = "ctx.actions"),
        @Param(name = "jar", positional = false, named = true, doc = "The jar to run ijar on."),
        @Param(
            name = "target_label",
            positional = false,
            named = true,
            allowedTypes = {
              @ParamType(type = Label.class),
              @ParamType(type = NoneType.class),
            },
            defaultValue = "None",
            doc =
                "A target label to stamp the jar with. Used for <code>add_dep</code> support. "
                    + "Typically, you would pass <code>ctx.label</code> to stamp the jar "
                    + "with the current rule's label."),
        @Param(
            name = "java_toolchain",
            positional = false,
            named = true,
            doc = "A JavaToolchainInfo to used to find the ijar tool."),
      })
  FileApi runIjar(
      StarlarkActionFactoryT actions, FileT jar, Object targetLabel, JavaToolchainT javaToolchain)
      throws EvalException;

  @StarlarkMethod(
      name = "stamp_jar",
      doc =
          "Stamps a jar with a target label for <code>add_dep</code> support. "
              + "The return value is typically passed to "
              + "<code><a class=\"anchor\" href=\"JavaInfo.html\">"
              + "JavaInfo</a>#compile_jar</code>. "
              + "Prefer to use "
              + "<code><a class=\"anchor\" href=\"java_common.html#run_ijar\">run_ijar</a></code> "
              + "when possible.",
      parameters = {
        @Param(
            name = "actions",
            named = true,
            doc = "ctx.actions"),
        @Param(
            name = "jar",
            positional = false,
            named = true,
            doc = "The jar to run stamp_jar on."),
        @Param(
            name = "target_label",
            positional = false,
            named = true,
            doc =
                "A target label to stamp the jar with. Used for <code>add_dep</code> support. "
                    + "Typically, you would pass <code>ctx.label</code> to stamp the jar "
                    + "with the current rule's label."),
        @Param(
            name = "java_toolchain",
            positional = false,
            named = true,
            doc = "A JavaToolchainInfo to used to find the stamp_jar tool."),
      })
  FileApi stampJar(
      StarlarkActionFactoryT actions, FileT jar, Label targetLabel, JavaToolchainT javaToolchain)
      throws EvalException;

  @StarlarkMethod(
      name = "pack_sources",
      doc =
          "Packs sources and source jars into a single source jar file. "
              + "The return value is typically passed to"
              + "<p><code><a class=\"anchor\" href=\"JavaInfo.html\">"
              + "JavaInfo</a>#source_jar</code></p>."
              + "At least one of parameters output_jar or output_source_jar is required.",
      parameters = {
        @Param(name = "actions", named = true, doc = "ctx.actions"),
        @Param(
            name = "output_jar",
            positional = false,
            named = true,
            allowedTypes = {
              @ParamType(type = FileApi.class),
              @ParamType(type = NoneType.class),
            },
            defaultValue = "None",
            doc =
                "Deprecated: The output jar of the rule. Used to name the resulting source jar. "
                    + "The parameter sets output_source_jar parameter to `{output_jar}-src.jar`."
                    + "Use output_source_jar parameter directly instead.",
            disableWithFlag = BuildLanguageOptions.INCOMPATIBLE_JAVA_COMMON_PARAMETERS,
            valueWhenDisabled = "None"),
        @Param(
            name = "output_source_jar",
            positional = false,
            named = true,
            allowedTypes = {
              @ParamType(type = FileApi.class),
              @ParamType(type = NoneType.class),
            },
            defaultValue = "None",
            doc = "The output source jar."),
        @Param(
            name = "sources",
            positional = false,
            named = true,
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = FileApi.class)},
            defaultValue = "[]",
            doc = "A list of Java source files to be packed into the source jar."),
        @Param(
            name = "source_jars",
            positional = false,
            named = true,
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = FileApi.class)},
            defaultValue = "[]",
            doc = "A list of source jars to be packed into the source jar."),
        @Param(
            name = "java_toolchain",
            positional = false,
            named = true,
            doc = "A JavaToolchainInfo to used to find the ijar tool."),
        @Param(
            name = "host_javabase",
            positional = false,
            named = true,
            doc =
                "Deprecated: You can drop this parameter (host_javabase is provided with "
                    + "java_toolchain)",
            defaultValue = "None",
            disableWithFlag = BuildLanguageOptions.INCOMPATIBLE_JAVA_COMMON_PARAMETERS,
            valueWhenDisabled = "None"),
      },
      allowReturnNones = true)
  @Nullable
  FileApi packSources(
      StarlarkActionFactoryT actions,
      Object outputJar,
      Object outputSourceJar,
      Sequence<?> sourceFiles, // <FileT> expected.
      Sequence<?> sourceJars, // <FileT> expected.
      JavaToolchainT javaToolchain,
      Object hostJavabase)
      throws EvalException;

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
      name = "merge",
      doc = "Merges the given providers into a single JavaInfo.",
      parameters = {
        @Param(
            name = "providers",
            positional = true,
            named = false,
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = JavaInfoApi.class)},
            doc = "The list of providers to merge."),
      },
      useStarlarkThread = true)
  JavaInfoT mergeJavaProviders(
      Sequence<?> providers /* <JavaInfoT> expected. */, StarlarkThread thread)
      throws EvalException;

  @StarlarkMethod(
      name = "make_non_strict",
      doc =
          "Returns a new Java provider whose direct-jars part is the union of both the direct and"
              + " indirect jars of the given Java provider.",
      parameters = {
        @Param(name = "java_info", positional = true, named = false, doc = "The java info."),
      })
  JavaInfoT makeNonStrict(JavaInfoT javaInfo);

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
      name = "is_java_toolchain_resolution_enabled_do_not_use",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true, doc = "The rule context."),
      },
      doc = "Returns true if --incompatible_use_toolchain_resolution_for_java_rules is enabled.")
  boolean isJavaToolchainResolutionEnabled(StarlarkRuleContextT ruleContext) throws EvalException;

  @StarlarkMethod(
      name = "MessageBundleInfo",
      doc = "The provider used to supply message bundles for translation",
      structField = true,
      enableOnlyWithFlag = BuildLanguageOptions.EXPERIMENTAL_GOOGLE_LEGACY_API)
  ProviderApi getMessageBundleInfo();

  @StarlarkMethod(
      name = "add_constraints",
      doc = "Returns a copy of the given JavaInfo with the given constraints added.",
      parameters = {
        @Param(
            name = "java_info",
            positional = true,
            named = false,
            doc = "The JavaInfo to enhance."),
        @Param(
            name = "constraints",
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = String.class)},
            named = true,
            positional = false,
            defaultValue = "[]",
            doc = "Constraints to add")
      },
      enableOnlyWithFlag = BuildLanguageOptions.EXPERIMENTAL_GOOGLE_LEGACY_API)
  JavaInfoT addConstraints(JavaInfoT javaInfo, Sequence<?> constraints /* <String> expected. */)
      throws EvalException;

  @StarlarkMethod(
      name = "get_constraints",
      doc = "Returns a set of constraints added.",
      parameters = {
        @Param(
            name = "java_info",
            positional = true,
            named = false,
            doc = "The JavaInfo to get constraints from."),
      },
      enableOnlyWithFlag = BuildLanguageOptions.EXPERIMENTAL_GOOGLE_LEGACY_API)
  Sequence<String> getConstraints(JavaInfoT javaInfo);

  @StarlarkMethod(
      name = "set_annotation_processing",
      doc = "Returns a copy of the given JavaInfo with the given annotation_processing info.",
      parameters = {
        @Param(
            name = "java_info",
            positional = true,
            named = false,
            doc = "The JavaInfo to enhance."),
        @Param(
            name = "enabled",
            named = true,
            positional = false,
            defaultValue = "False",
            doc = "Returns true if the rule uses annotation processing."),
        @Param(
            name = "processor_classnames",
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = String.class)},
            named = true,
            positional = false,
            defaultValue = "[]",
            doc = "Class names of annotation processors applied to this rule."),
        @Param(
            name = "processor_classpath",
            allowedTypes = {
              @ParamType(type = Depset.class),
              @ParamType(type = NoneType.class),
            },
            named = true,
            positional = false,
            defaultValue = "None",
            doc = "Class names of annotation processors applied to this rule."),
        @Param(
            name = "class_jar",
            allowedTypes = {
              @ParamType(type = FileApi.class),
              @ParamType(type = NoneType.class),
            },
            named = true,
            positional = false,
            defaultValue = "None",
            doc = "Jar file that is the result of annotation processing for this rule, or None."),
        @Param(
            name = "source_jar",
            allowedTypes = {
              @ParamType(type = FileApi.class),
              @ParamType(type = NoneType.class),
            },
            named = true,
            positional = false,
            defaultValue = "None",
            doc = "Source archive resulting from annotation processing of this rule, or None."),
      },
      enableOnlyWithFlag = BuildLanguageOptions.EXPERIMENTAL_GOOGLE_LEGACY_API)
  JavaInfoT setAnnotationProcessing(
      JavaInfoT javaInfo,
      boolean enabled,
      Sequence<?> processorClassnames /* <String> expected. */,
      Object processorClasspath,
      Object classJar,
      Object sourceJar)
      throws EvalException;

  @StarlarkMethod(
      name = "java_toolchain_label",
      doc = "Returns the toolchain's label.",
      parameters = {
        @Param(name = "java_toolchain", positional = true, named = false, doc = "The toolchain."),
      },
      enableOnlyWithFlag = BuildLanguageOptions.EXPERIMENTAL_GOOGLE_LEGACY_API)
  Label getJavaToolchainLabel(JavaToolchainStarlarkApiProviderApi toolchain) throws EvalException;

  @StarlarkMethod(
      name = "BootClassPathInfo",
      doc = "The provider used to supply bootclasspath information",
      structField = true)
  ProviderApi getBootClassPathInfo();
}
