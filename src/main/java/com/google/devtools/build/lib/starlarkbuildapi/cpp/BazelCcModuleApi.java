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

package com.google.devtools.build.lib.starlarkbuildapi.cpp;

import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkActionFactoryApi;
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkRuleContextApi;
import com.google.devtools.build.lib.starlarkbuildapi.platform.ConstraintValueInfoApi;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.NoneType;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkInt;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.Tuple;

/** Utilites related to C++ support. */
@StarlarkBuiltin(
    name = "cc_common",
    doc = "Utilities for C++ compilation, linking, and command line generation.")
public interface BazelCcModuleApi<
        StarlarkActionFactoryT extends StarlarkActionFactoryApi,
        FileT extends FileApi,
        ConstraintValueT extends ConstraintValueInfoApi,
        StarlarkRuleContextT extends StarlarkRuleContextApi<ConstraintValueT>,
        CcToolchainProviderT extends CcToolchainProviderApi<FeatureConfigurationT>,
        FeatureConfigurationT extends FeatureConfigurationApi,
        CompilationContextT extends CcCompilationContextApi<FileT>,
        CompilationOutputsT extends CcCompilationOutputsApi<FileT>,
        LinkingOutputsT extends CcLinkingOutputsApi<FileT>,
        LinkerInputT extends LinkerInputApi<LibraryToLinkT, FileT>,
        LibraryToLinkT extends LibraryToLinkApi<FileT>,
        LinkingContextT extends CcLinkingContextApi<FileT>,
        CcToolchainVariablesT extends CcToolchainVariablesApi,
        CcToolchainConfigInfoT extends CcToolchainConfigInfoApi,
        DebugContextT extends CcDebugInfoContextApi,
        CppModuleMapT extends CppModuleMapApi<FileT>>
    extends CcModuleApi<
        StarlarkActionFactoryT,
        FileT,
        CcToolchainProviderT,
        FeatureConfigurationT,
        CompilationContextT,
        LinkerInputT,
        LinkingContextT,
        LibraryToLinkT,
        CcToolchainVariablesT,
        ConstraintValueT,
        StarlarkRuleContextT,
        CcToolchainConfigInfoT,
        CompilationOutputsT,
        DebugContextT,
        CppModuleMapT> {

  @StarlarkMethod(
      name = "compile",
      doc =
          "Should be used for C++ compilation. Returns tuple of "
              + "(<code>CompilationContext</code>, <code>CcCompilationOutputs</code>).",
      useStarlarkThread = true,
      parameters = {
        @Param(
            name = "actions",
            positional = false,
            named = true,
            doc = "<code>actions</code> object."),
        @Param(
            name = "feature_configuration",
            doc = "<code>feature_configuration</code> to be queried.",
            positional = false,
            named = true),
        @Param(
            name = "cc_toolchain",
            doc = "<code>CcToolchainInfo</code> provider to be used.",
            positional = false,
            named = true),
        @Param(
            name = "srcs",
            doc = "The list of source files to be compiled.",
            positional = false,
            named = true,
            defaultValue = "[]"),
        @Param(
            name = "public_hdrs",
            doc =
                "List of headers needed for compilation of srcs and may be included by dependent "
                    + "rules transitively.",
            positional = false,
            named = true,
            defaultValue = "[]"),
        @Param(
            name = "private_hdrs",
            doc =
                "List of headers needed for compilation of srcs and NOT to be included by"
                    + " dependent rules.",
            positional = false,
            named = true,
            defaultValue = "[]"),
        @Param(
            name = "textual_hdrs",
            positional = false,
            named = true,
            documented = false,
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = FileApi.class)},
            defaultValue = "unbound"),
        @Param(
            name = "additional_exported_hdrs",
            positional = false,
            named = true,
            documented = false,
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = String.class)},
            defaultValue = "unbound"),
        @Param(
            name = "includes",
            doc =
                "Search paths for header files referenced both by angle bracket and quotes. "
                    + "Usually passed with -I. Propagated to dependents transitively.",
            positional = false,
            named = true,
            defaultValue = "[]"),
        @Param(
            name = "quote_includes",
            doc =
                "Search paths for header files referenced by quotes, "
                    + "e.g. #include \"foo/bar/header.h\". They can be either relative to the exec "
                    + "root or absolute. Usually passed with -iquote. Propagated to dependents "
                    + "transitively.",
            positional = false,
            named = true,
            defaultValue = "[]"),
        @Param(
            name = "system_includes",
            doc =
                "Search paths for header files referenced by angle brackets, e.g. #include"
                    + " &lt;foo/bar/header.h&gt;. They can be either relative to the exec root or"
                    + " absolute. Usually passed with -isystem. Propagated to dependents "
                    + "transitively.",
            positional = false,
            named = true,
            defaultValue = "[]"),
        @Param(
            name = "framework_includes",
            doc =
                "Search paths for header files from Apple frameworks. They can be either relative "
                    + "to the exec root or absolute. Usually passed with -F. Propagated to "
                    + "dependents transitively.",
            positional = false,
            named = true,
            defaultValue = "[]"),
        @Param(
            name = "defines",
            doc =
                "Set of defines needed to compile this target. Each define is a string. Propagated"
                    + " to dependents transitively.",
            positional = false,
            named = true,
            defaultValue = "[]"),
        @Param(
            name = "local_defines",
            doc =
                "Set of defines needed to compile this target. Each define is a string. Not"
                    + " propagated to dependents transitively.",
            positional = false,
            named = true,
            defaultValue = "[]"),
        @Param(
            name = "include_prefix",
            doc =
                "The prefix to add to the paths of the headers of this rule. When set, the "
                    + "headers in the hdrs attribute of this rule are accessible at is the "
                    + "value of this attribute prepended to their repository-relative path. "
                    + "The prefix in the strip_include_prefix attribute is removed before this "
                    + "prefix is added.",
            positional = false,
            named = true,
            defaultValue = "''"),
        @Param(
            name = "strip_include_prefix",
            doc =
                "The prefix to strip from the paths of the headers of this rule. When set, the"
                    + " headers in the hdrs attribute of this rule are accessible at their path"
                    + " with this prefix cut off. If it's a relative path, it's taken as a"
                    + " package-relative one. If it's an absolute one, it's understood as a"
                    + " repository-relative path. The prefix in the include_prefix attribute is"
                    + " added after this prefix is stripped.",
            positional = false,
            named = true,
            defaultValue = "''"),
        @Param(
            name = "user_compile_flags",
            doc = "Additional list of compilation options.",
            positional = false,
            named = true,
            defaultValue = "[]"),
        @Param(
            name = "compilation_contexts",
            doc = "Headers from dependencies used for compilation.",
            positional = false,
            named = true,
            defaultValue = "[]"),
        @Param(
            name = "name",
            doc =
                "This is used for naming the output artifacts of actions created by this "
                    + "method.",
            positional = false,
            named = true),
        @Param(
            name = "disallow_pic_outputs",
            doc = "Whether PIC outputs should be created.",
            positional = false,
            named = true,
            defaultValue = "False"),
        @Param(
            name = "disallow_nopic_outputs",
            doc = "Whether NOPIC outputs should be created.",
            positional = false,
            named = true,
            defaultValue = "False"),
        @Param(
            name = "additional_inputs",
            doc = "List of additional files needed for compilation of srcs",
            positional = false,
            named = true,
            defaultValue = "[]"),
        @Param(
            name = "module_map",
            positional = false,
            documented = false,
            defaultValue = "unbound",
            allowedTypes = {
              @ParamType(type = CppModuleMapApi.class),
              @ParamType(type = NoneType.class)
            },
            named = true),
        @Param(
            name = "additional_module_maps",
            positional = false,
            documented = false,
            defaultValue = "unbound",
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = CppModuleMapApi.class)},
            named = true),
        @Param(
            name = "propagate_module_map_to_compile_action",
            positional = false,
            named = true,
            documented = false,
            allowedTypes = {@ParamType(type = Boolean.class)},
            defaultValue = "unbound"),
        @Param(
            name = "do_not_generate_module_map",
            positional = false,
            named = true,
            documented = false,
            allowedTypes = {@ParamType(type = Boolean.class)},
            defaultValue = "unbound"),
        @Param(
            name = "code_coverage_enabled",
            positional = false,
            named = true,
            documented = false,
            allowedTypes = {@ParamType(type = Boolean.class)},
            defaultValue = "unbound"),
        @Param(
            name = "hdrs_checking_mode",
            positional = false,
            named = true,
            documented = false,
            allowedTypes = {@ParamType(type = String.class)},
            defaultValue = "unbound"),
      })
  Tuple compile(
      StarlarkActionFactoryT starlarkActionFactoryApi,
      FeatureConfigurationT starlarkFeatureConfiguration,
      CcToolchainProviderT starlarkCcToolchainProvider,
      Sequence<?> sources, // <FileT> expected
      Sequence<?> publicHeaders, // <FileT> expected
      Sequence<?> privateHeaders, // <FileT> expected
      Object textualHeaders,
      Object additionalExportedHeaders,
      Sequence<?> includes, // <String> expected
      Sequence<?> quoteIncludes, // <String> expected
      Sequence<?> systemIncludes, // <String> expected
      Sequence<?> frameworkIncludes, // <String> expected
      Sequence<?> defines, // <String> expected
      Sequence<?> localDefines, // <String> expected
      String includePrefix,
      String stripIncludePrefix,
      Sequence<?> userCompileFlags, // <String> expected
      Sequence<?> ccCompilationContexts, // <CompilationContextT> expected
      String name,
      boolean disallowPicOutputs,
      boolean disallowNopicOutputs,
      Sequence<?> additionalInputs, // <FileT> expected
      Object moduleMap,
      Object additionalModuleMaps,
      Object propagateModuleMapToCompileAction,
      Object doNotGenerateModuleMap,
      Object codeCoverageEnabled,
      Object hdrsCheckingMode,
      StarlarkThread thread)
      throws EvalException, InterruptedException;

  @StarlarkMethod(
      name = "link",
      doc = "Should be used for C++ transitive linking.",
      useStarlarkThread = true,
      parameters = {
        @Param(
            name = "actions",
            positional = false,
            named = true,
            doc = "<code>actions</code> object."),
        @Param(
            name = "feature_configuration",
            doc = "<code>feature_configuration</code> to be queried.",
            positional = false,
            named = true),
        @Param(
            name = "cc_toolchain",
            doc = "<code>CcToolchainInfo</code> provider to be used.",
            positional = false,
            named = true),
        @Param(
            name = "compilation_outputs",
            doc = "Compilation outputs containing object files to link.",
            positional = false,
            named = true,
            defaultValue = "None",
            allowedTypes = {
              @ParamType(type = CcCompilationOutputsApi.class),
              @ParamType(type = NoneType.class)
            }),
        @Param(
            name = "user_link_flags",
            doc = "Additional list of linker options.",
            positional = false,
            named = true,
            defaultValue = "[]"),
        @Param(
            name = "linking_contexts",
            doc =
                "Linking contexts from dependencies to be linked into the linking context "
                    + "generated by this rule.",
            positional = false,
            named = true,
            defaultValue = "[]"),
        @Param(
            name = "name",
            doc =
                "This is used for naming the output artifacts of actions created by this "
                    + "method.",
            positional = false,
            named = true),
        @Param(
            name = "language",
            doc = "Only C++ supported for now. Do not use this parameter.",
            positional = false,
            named = true,
            defaultValue = "'c++'"),
        @Param(
            name = "output_type",
            doc = "Can be either 'executable' or 'dynamic_library'.",
            positional = false,
            named = true,
            defaultValue = "'executable'"),
        @Param(
            name = "link_deps_statically",
            doc = " True to link dependencies statically, False dynamically.",
            positional = false,
            named = true,
            defaultValue = "True"),
        @Param(
            name = "stamp",
            doc =
                "Whether to include build information in the linked executable, if output_type is "
                    + "'executable'. If 1, build information is always included. If 0 (the "
                    + "default build information is always excluded. If -1, uses the default "
                    + "behavior, which may be overridden by the --[no]stamp flag. This should be "
                    + "unset (or set to 0) when generating the executable output for test rules.",
            positional = false,
            named = true,
            defaultValue = "0"),
        @Param(
            name = "additional_inputs",
            doc = "For additional inputs to the linking action, e.g.: linking scripts.",
            positional = false,
            named = true,
            defaultValue = "[]"),
        @Param(
            name = "grep_includes",
            positional = false,
            named = true,
            defaultValue = "None",
            allowedTypes = {
              @ParamType(type = FileApi.class),
              @ParamType(type = NoneType.class),
            }),
        @Param(
            name = "link_artifact_name_suffix",
            positional = false,
            named = true,
            documented = false,
            allowedTypes = {@ParamType(type = String.class)},
            defaultValue = "unbound"),
        @Param(
            name = "never_link",
            positional = false,
            named = true,
            documented = false,
            allowedTypes = {@ParamType(type = Boolean.class)},
            defaultValue = "unbound"),
        @Param(
            name = "test_only_target",
            positional = false,
            named = true,
            documented = false,
            allowedTypes = {@ParamType(type = Boolean.class)},
            defaultValue = "unbound"),
      })
  LinkingOutputsT link(
      StarlarkActionFactoryT starlarkActionFactoryApi,
      FeatureConfigurationT starlarkFeatureConfiguration,
      CcToolchainProviderT starlarkCcToolchainProvider,
      Object compilationOutputs,
      Sequence<?> userLinkFlags, // <String> expected
      Sequence<?> linkingContexts, // <LinkingContextT> expected
      String name,
      String language,
      String outputType,
      boolean linkDepsStatically,
      StarlarkInt stamp,
      Sequence<?> additionalInputs, // <FileT> expected
      Object grepIncludes,
      Object linkArtifactNameSuffix,
      Object neverLink,
      Object testOnlyTarget,
      StarlarkThread thread)
      throws InterruptedException, EvalException;

  @StarlarkMethod(
      name = "create_compilation_outputs",
      doc = "Create compilation outputs object.",
      parameters = {
        @Param(
            name = "objects",
            doc = "List of object files.",
            positional = false,
            named = true,
            defaultValue = "None",
            allowedTypes = {
              @ParamType(type = Depset.class),
              @ParamType(type = NoneType.class),
            }),
        @Param(
            name = "pic_objects",
            doc = "List of pic object files.",
            positional = false,
            named = true,
            defaultValue = "None",
            allowedTypes = {
              @ParamType(type = Depset.class),
              @ParamType(type = NoneType.class),
            }),
      })
  CompilationOutputsT createCompilationOutputsFromStarlark(
      Object objectsObject, Object picObjectsObject) throws EvalException;

  @StarlarkMethod(
      name = "merge_compilation_outputs",
      doc = "Merge compilation outputs.",
      parameters = {
        @Param(name = "compilation_outputs", positional = false, named = true, defaultValue = "[]"),
      })
  CompilationOutputsT mergeCcCompilationOutputsFromStarlark(
      Sequence<?> compilationOutputs) // <CompilationOutputsT> expected
      throws EvalException;
}
