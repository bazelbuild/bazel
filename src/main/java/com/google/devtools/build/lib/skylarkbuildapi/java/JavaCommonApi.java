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

package com.google.devtools.build.lib.skylarkbuildapi.java;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkActionFactoryApi;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkRuleContextApi;
import com.google.devtools.build.lib.skylarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.platform.ConstraintValueInfoApi;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.ParamType;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.Depset;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.StarlarkSemantics.FlagIdentifier;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.lib.syntax.StarlarkValue;

/** Utilities for Java compilation support in Skylark. */
@SkylarkModule(name = "java_common", doc = "Utilities for Java compilation support in Starlark.")
public interface JavaCommonApi<
        FileT extends FileApi,
        JavaInfoT extends JavaInfoApi<FileT>,
        JavaToolchainT extends JavaToolchainSkylarkApiProviderApi,
        JavaRuntimeT extends JavaRuntimeInfoApi,
        ConstraintValueT extends ConstraintValueInfoApi,
        SkylarkRuleContextT extends SkylarkRuleContextApi<ConstraintValueT>,
        SkylarkActionFactoryT extends SkylarkActionFactoryApi>
    extends StarlarkValue {

  @SkylarkCallable(
      name = "provider",
      structField = true,
      doc =
          "Returns the Java declared provider. <br>"
              + "The same value is accessible as <code>JavaInfo</code>. <br>"
              + "Prefer using <code>JavaInfo</code> in new code.")
  ProviderApi getJavaProvider();

  @SkylarkCallable(
      name = "compile",
      doc =
          "Compiles Java source files/jars from the implementation of a Starlark rule and returns "
              + "a provider that represents the results of the compilation and can be added to "
              + "the set of providers emitted by this rule.",
      parameters = {
        @Param(
            name = "ctx",
            positional = true,
            named = false,
            type = SkylarkRuleContextApi.class,
            doc = "The rule context."),
        @Param(
            name = "source_jars",
            positional = false,
            named = true,
            type = Sequence.class,
            generic1 = FileApi.class,
            defaultValue = "[]",
            doc =
                "A list of the jars to be compiled. At least one of source_jars or source_files"
                    + " should be specified."),
        @Param(
            name = "source_files",
            positional = false,
            named = true,
            type = Sequence.class,
            generic1 = FileApi.class,
            defaultValue = "[]",
            doc =
                "A list of the Java source files to be compiled. At least one of source_jars or "
                    + "source_files should be specified."),
        @Param(name = "output", positional = false, named = true, type = FileApi.class),
        @Param(
            name = "output_source_jar",
            positional = false,
            named = true,
            type = FileApi.class,
            noneable = true,
            defaultValue = "None",
            doc = "The output source jar. Optional. Defaults to `{output_jar}-src.jar` if unset."),
        @Param(
            name = "javac_opts",
            positional = false,
            named = true,
            type = Sequence.class,
            generic1 = String.class,
            defaultValue = "[]",
            doc = "A list of the desired javac options. Optional."),
        @Param(
            name = "deps",
            positional = false,
            named = true,
            type = Sequence.class,
            generic1 = JavaInfoApi.class,
            defaultValue = "[]",
            doc = "A list of dependencies. Optional."),
        @Param(
            name = "exports",
            positional = false,
            named = true,
            type = Sequence.class,
            generic1 = JavaInfoApi.class,
            defaultValue = "[]",
            doc = "A list of exports. Optional."),
        @Param(
            name = "plugins",
            positional = false,
            named = true,
            type = Sequence.class,
            generic1 = JavaInfoApi.class,
            defaultValue = "[]",
            doc = "A list of plugins. Optional."),
        @Param(
            name = "exported_plugins",
            positional = false,
            named = true,
            type = Sequence.class,
            generic1 = JavaInfoApi.class,
            defaultValue = "[]",
            doc = "A list of exported plugins. Optional."),
        @Param(
            name = "annotation_processor_additional_inputs",
            positional = false,
            named = true,
            type = Sequence.class,
            generic1 = FileApi.class,
            defaultValue = "[]",
            doc =
                "A list of inputs that the Java compilation action will take in addition to the "
                    + "Java sources for annotation processing."),
        @Param(
            name = "annotation_processor_additional_outputs",
            positional = false,
            named = true,
            type = Sequence.class,
            generic1 = FileApi.class,
            defaultValue = "[]",
            doc =
                "A list of outputs that the Java compilation action will output in addition to "
                    + "the class jar from annotation processing."),
        @Param(
            name = "strict_deps",
            defaultValue = "'ERROR'",
            positional = false,
            named = true,
            type = String.class,
            doc =
                "A string that specifies how to handle strict deps. Possible values: 'OFF', "
                    + "'ERROR', 'WARN' and 'DEFAULT'. For more details see "
                    + "https://docs.bazel.build/versions/master/bazel-user-manual.html#"
                    + "flag--strict_java_deps. By default 'ERROR'."),
        @Param(
            name = "java_toolchain",
            positional = false,
            named = true,
            type = Object.class,
            allowedTypes = {@ParamType(type = JavaToolchainSkylarkApiProviderApi.class)},
            doc = "A JavaToolchainInfo to be used for this compilation. Mandatory."),
        @Param(
            name = "host_javabase",
            positional = false,
            named = true,
            type = Object.class,
            allowedTypes = {@ParamType(type = JavaRuntimeInfoApi.class)},
            doc = "A JavaRuntimeInfo to be used for this compilation. Mandatory."),
        @Param(
            name = "sourcepath",
            positional = false,
            named = true,
            type = Sequence.class,
            generic1 = FileApi.class,
            defaultValue = "[]"),
        @Param(
            name = "resources",
            positional = false,
            named = true,
            type = Sequence.class,
            generic1 = FileApi.class,
            defaultValue = "[]"),
        @Param(
            name = "neverlink",
            positional = false,
            named = true,
            type = Boolean.class,
            defaultValue = "False")
      },
      useLocation = true,
      useStarlarkThread = true)
  JavaInfoT createJavaCompileAction(
      SkylarkRuleContextT skylarkRuleContext,
      Sequence<?> sourceJars, // <FileT> expected.
      Sequence<?> sourceFiles, // <FileT> expected.
      FileT outputJar,
      Object outputSourceJar,
      Sequence<?> javacOpts, // <String> expected.
      Sequence<?> deps, // <JavaInfoT> expected.
      Sequence<?> exports, // <JavaInfoT> expected.
      Sequence<?> plugins, // <JavaInfoT> expected.
      Sequence<?> exportedPlugins, // <JavaInfoT> expected.
      Sequence<?> annotationProcessorAdditionalInputs, // <FileT> expected.
      Sequence<?> annotationProcessorAdditionalOutputs, // <FileT> expected.
      String strictDepsMode,
      JavaToolchainT javaToolchain,
      JavaRuntimeT hostJavabase,
      Sequence<?> sourcepathEntries, // <FileT> expected.
      Sequence<?> resources, // <FileT> expected.
      Boolean neverlink,
      Location loc,
      StarlarkThread thread)
      throws EvalException, InterruptedException;

  @SkylarkCallable(
      name = "run_ijar",
      doc =
          "Runs ijar on a jar, stripping it of its method bodies. This helps reduce rebuilding "
              + "of dependent jars during any recompiles consisting only of simple changes to "
              + "method implementations. The return value is typically passed to "
              + "<code><a class=\"anchor\" href=\"JavaInfo.html\">"
              + "JavaInfo</a>#compile_jar</code>.",
      parameters = {
        @Param(
            name = "actions",
            named = true,
            type = SkylarkActionFactoryApi.class,
            doc = "ctx.actions"),
        @Param(
            name = "jar",
            positional = false,
            named = true,
            type = FileApi.class,
            doc = "The jar to run ijar on."),
        @Param(
            name = "target_label",
            positional = false,
            named = true,
            type = Label.class,
            noneable = true,
            defaultValue = "None",
            doc =
                "A target label to stamp the jar with. Used for <code>add_dep</code> support. "
                    + "Typically, you would pass <code>ctx.label</code> to stamp the jar "
                    + "with the current rule's label."),
        @Param(
            name = "java_toolchain",
            positional = false,
            named = true,
            type = Object.class,
            allowedTypes = {@ParamType(type = JavaToolchainSkylarkApiProviderApi.class)},
            doc = "A JavaToolchainInfo to used to find the ijar tool."),
      },
      useLocation = true)
  FileApi runIjar(
      SkylarkActionFactoryT actions,
      FileT jar,
      Object targetLabel,
      JavaToolchainT javaToolchain,
      Location location)
      throws EvalException;

  @SkylarkCallable(
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
            type = SkylarkActionFactoryApi.class,
            doc = "ctx.actions"),
        @Param(
            name = "jar",
            positional = false,
            named = true,
            type = FileApi.class,
            doc = "The jar to run stamp_jar on."),
        @Param(
            name = "target_label",
            positional = false,
            named = true,
            type = Label.class,
            doc =
                "A target label to stamp the jar with. Used for <code>add_dep</code> support. "
                    + "Typically, you would pass <code>ctx.label</code> to stamp the jar "
                    + "with the current rule's label."),
        @Param(
            name = "java_toolchain",
            positional = false,
            named = true,
            type = Object.class,
            allowedTypes = {@ParamType(type = JavaToolchainSkylarkApiProviderApi.class)},
            doc = "A JavaToolchainInfo to used to find the stamp_jar tool."),
      },
      useLocation = true)
  FileApi stampJar(
      SkylarkActionFactoryT actions,
      FileT jar,
      Label targetLabel,
      JavaToolchainT javaToolchain,
      Location location)
      throws EvalException;

  @SkylarkCallable(
      name = "pack_sources",
      doc =
          "Packs sources and source jars into a single source jar file. "
              + "The return value is typically passed to"
              + "<p><code><a class=\"anchor\" href=\"JavaInfo.html\">"
              + "JavaInfo</a>#source_jar</code></p>.",
      parameters = {
        @Param(
            name = "actions",
            named = true,
            type = SkylarkActionFactoryApi.class,
            doc = "ctx.actions"),
        @Param(
            name = "output_jar",
            positional = false,
            named = true,
            type = FileApi.class,
            doc = "The output jar of the rule. Used to name the resulting source jar."),
        @Param(
            name = "sources",
            positional = false,
            named = true,
            type = Sequence.class,
            generic1 = FileApi.class,
            defaultValue = "[]",
            doc = "A list of Java source files to be packed into the source jar."),
        @Param(
            name = "source_jars",
            positional = false,
            named = true,
            type = Sequence.class,
            generic1 = FileApi.class,
            defaultValue = "[]",
            doc = "A list of source jars to be packed into the source jar."),
        @Param(
            name = "java_toolchain",
            positional = false,
            named = true,
            type = Object.class,
            allowedTypes = {@ParamType(type = JavaToolchainSkylarkApiProviderApi.class)},
            doc = "A JavaToolchainInfo to used to find the ijar tool."),
        @Param(
            name = "host_javabase",
            positional = false,
            named = true,
            type = Object.class,
            allowedTypes = {@ParamType(type = JavaRuntimeInfoApi.class)},
            doc = "A JavaRuntimeInfo to be used for packing sources."),
      },
      allowReturnNones = true,
      useLocation = true)
  FileApi packSources(
      SkylarkActionFactoryT actions,
      FileT outputJar,
      Sequence<?> sourceFiles, // <FileT> expected.
      Sequence<?> sourceJars, // <FileT> expected.
      JavaToolchainT javaToolchain,
      JavaRuntimeT hostJavabase,
      Location location)
      throws EvalException;

  @SkylarkCallable(
      name = "default_javac_opts",
      // This function is experimental for now.
      documented = false,
      parameters = {
        @Param(
            name = "java_toolchain",
            positional = false,
            named = true,
            type = Object.class,
            allowedTypes = {@ParamType(type = JavaToolchainSkylarkApiProviderApi.class)},
            doc =
                "A JavaToolchainInfo to be used for retrieving the ijar "
                    + "tool. Only set when use_ijar is True."),
      },
      useLocation = true)
  // TODO(b/78512644): migrate callers to passing explicit javacopts or using custom toolchains, and
  // delete
  ImmutableList<String> getDefaultJavacOpts(JavaToolchainT javaToolchain, Location loc)
      throws EvalException;

  @SkylarkCallable(
      name = "merge",
      doc = "Merges the given providers into a single JavaInfo.",
      parameters = {
        @Param(
            name = "providers",
            positional = true,
            named = false,
            type = Sequence.class,
            generic1 = JavaInfoApi.class,
            doc = "The list of providers to merge."),
      })
  JavaInfoT mergeJavaProviders(Sequence<?> providers /* <JavaInfoT> expected. */)
      throws EvalException;

  @SkylarkCallable(
      name = "make_non_strict",
      doc =
          "Returns a new Java provider whose direct-jars part is the union of both the direct and"
              + " indirect jars of the given Java provider.",
      parameters = {
        @Param(
            name = "java_info",
            positional = true,
            named = false,
            type = JavaInfoApi.class,
            doc = "The java info."),
      })
  JavaInfoT makeNonStrict(JavaInfoT javaInfo);

  @SkylarkCallable(
      name = "JavaToolchainInfo",
      doc =
          "The key used to retrieve the provider that contains information about the Java "
              + "toolchain being used.",
      structField = true)
  ProviderApi getJavaToolchainProvider();

  @SkylarkCallable(
      name = "JavaRuntimeInfo",
      doc =
          "The key used to retrieve the provider that contains information about the Java "
              + "runtime being used.",
      structField = true)
  ProviderApi getJavaRuntimeProvider();

  @SkylarkCallable(
      name = "is_java_toolchain_resolution_enabled_do_not_use",
      documented = false,
      parameters = {
        @Param(
            name = "ctx",
            positional = false,
            named = true,
            type = SkylarkRuleContextApi.class,
            doc = "The rule context."),
      },
      doc = "Returns true if --incompatible_use_toolchain_resolution_for_java_rules is enabled.")
  boolean isJavaToolchainResolutionEnabled(SkylarkRuleContextT ruleContext) throws EvalException;

  @SkylarkCallable(
      name = "MessageBundleInfo",
      doc = "The provider used to supply message bundles for translation",
      structField = true,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_GOOGLE_LEGACY_API)
  ProviderApi getMessageBundleInfo();

  @SkylarkCallable(
      name = "add_constraints",
      doc = "Returns a copy of the given JavaInfo with the given constraints added.",
      parameters = {
        @Param(
            name = "java_info",
            positional = true,
            named = false,
            type = JavaInfoApi.class,
            doc = "The JavaInfo to enhance."),
        @Param(
            name = "constraints",
            type = Sequence.class,
            generic1 = String.class,
            named = true,
            positional = false,
            defaultValue = "[]",
            doc = "Constraints to add")
      },
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_GOOGLE_LEGACY_API)
  JavaInfoT addConstraints(JavaInfoT javaInfo, Sequence<?> constraints /* <String> expected. */)
      throws EvalException;

  @SkylarkCallable(
      name = "experimental_disable_annotation_processing",
      doc =
          "Returns a copy of the given JavaInfo with any provided annotation processors disabled."
              + " Annotation processor classpaths are preserved in case they contain Error Prone"
              + " plugins, but processor names and data are excluded. For example, it can be"
              + " used to process the inputs to java_common.compile's deps and plugins parameters.",
      parameters = {
        @Param(
            name = "java_info",
            positional = true,
            named = false,
            type = JavaInfoApi.class,
            doc = "The JavaInfo to process.")
      },
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_GOOGLE_LEGACY_API)
  JavaInfoT removeAnnotationProcessors(JavaInfoT javaInfo);

  @SkylarkCallable(
      name = "compile_time_jdeps",
      doc = "Returns a depset of the given JavaInfo's compile-time jdeps files.",
      parameters = {
        @Param(
            name = "java_info",
            positional = true,
            named = false,
            type = JavaInfoApi.class,
            doc = "The JavaInfo to query."),
      },
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_GOOGLE_LEGACY_API)
  Depset /*<FileT>*/ getCompileTimeJavaDependencyArtifacts(JavaInfoT javaInfo);

  @SkylarkCallable(
      name = "add_compile_time_jdeps",
      doc = "Returns a copy of the given JavaInfo with the given compile-time jdeps files added.",
      parameters = {
        @Param(
            name = "java_info",
            positional = true,
            named = false,
            type = JavaInfoApi.class,
            doc = "The JavaInfo to clone."),
        @Param(
            name = "compile_time_jdeps",
            type = Sequence.class,
            generic1 = FileApi.class,
            named = true,
            positional = false,
            defaultValue = "[]",
            doc = "Compile-time jdeps files to add.")
      },
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_GOOGLE_LEGACY_API)
  JavaInfoT addCompileTimeJavaDependencyArtifacts(
      JavaInfoT javaInfo, Sequence<?> compileTimeJavaDependencyArtifacts /* <FileT> expected. */)
      throws EvalException;

  @SkylarkCallable(
      name = "java_toolchain_label",
      doc = "Returns the toolchain's label.",
      parameters = {
        @Param(
            name = "java_toolchain",
            positional = true,
            named = false,
            type = JavaToolchainSkylarkApiProviderApi.class,
            doc = "The toolchain."),
      },
      useLocation = true,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_GOOGLE_LEGACY_API)
  Label getJavaToolchainLabel(JavaToolchainSkylarkApiProviderApi toolchain, Location location)
      throws EvalException;
}
