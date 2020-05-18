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

package com.google.devtools.build.lib.skylarkbuildapi.cpp;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.StarlarkActionFactoryApi;
import com.google.devtools.build.lib.skylarkbuildapi.StarlarkRuleContextApi;
import com.google.devtools.build.lib.skylarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.platform.ConstraintValueInfoApi;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.ParamType;
import com.google.devtools.build.lib.skylarkinterface.StarlarkBuiltin;
import com.google.devtools.build.lib.skylarkinterface.StarlarkMethod;
import com.google.devtools.build.lib.syntax.Dict;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.NoneType;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.StarlarkSemantics.FlagIdentifier;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.lib.syntax.StarlarkValue;
import com.google.devtools.build.lib.syntax.Tuple;

/** Utilites related to C++ support. */
@StarlarkBuiltin(
    name = "cc_common",
    doc = "Utilities for C++ compilation, linking, and command line generation.")
public interface CcModuleApi<
        SkylarkActionFactoryT extends StarlarkActionFactoryApi,
        FileT extends FileApi,
        CcToolchainProviderT extends CcToolchainProviderApi<?>,
        FeatureConfigurationT extends FeatureConfigurationApi,
        CompilationContextT extends CcCompilationContextApi<FileT>,
        LinkerInputT extends LinkerInputApi<LibraryToLinkT, FileT>,
        LinkingContextT extends CcLinkingContextApi<?>,
        LibraryToLinkT extends LibraryToLinkApi<FileT>,
        CcToolchainVariablesT extends CcToolchainVariablesApi,
        ConstraintValueT extends ConstraintValueInfoApi,
        SkylarkRuleContextT extends StarlarkRuleContextApi<ConstraintValueT>,
        CcToolchainConfigInfoT extends CcToolchainConfigInfoApi,
        CompilationOutputsT extends CcCompilationOutputsApi<FileT>>
    extends StarlarkValue {

  @StarlarkMethod(
      name = "CcToolchainInfo",
      doc =
          "The key used to retrieve the provider that contains information about the C++ "
              + "toolchain being used",
      structField = true)
  ProviderApi getCcToolchainProvider();

  @Deprecated
  @StarlarkMethod(
      name = "do_not_use_tools_cpp_compiler_present",
      doc =
          "Do not use this field, its only puprose is to help with migration from "
              + "config_setting.values{'compiler') to "
              + "config_settings.flag_values{'@bazel_tools//tools/cpp:compiler'}",
      structField = true)
  default void compilerFlagExists() {}

  @StarlarkMethod(
      name = "configure_features",
      doc = "Creates a feature_configuration instance. Requires the cpp configuration fragment.",
      parameters = {
        @Param(
            name = "ctx",
            positional = false,
            named = true,
            noneable = true,
            defaultValue = "None",
            type = StarlarkRuleContextApi.class,
            doc = "The rule context."),
        @Param(
            name = "cc_toolchain",
            doc = "cc_toolchain for which we configure features.",
            positional = false,
            named = true,
            type = CcToolchainProviderApi.class),
        @Param(
            name = "requested_features",
            doc = "List of features to be enabled.",
            positional = false,
            named = true,
            defaultValue = "[]",
            type = Sequence.class),
        @Param(
            name = "unsupported_features",
            doc = "List of features that are unsupported by the current rule.",
            positional = false,
            named = true,
            defaultValue = "[]",
            type = Sequence.class),
      })
  FeatureConfigurationT configureFeatures(
      Object ruleContextOrNone,
      CcToolchainProviderT toolchain,
      Sequence<?> requestedFeatures, // <String> expected
      Sequence<?> unsupportedFeatures) // <String> expected
      throws EvalException;

  @StarlarkMethod(
      name = "get_tool_for_action",
      doc = "Returns tool path for given action.",
      parameters = {
        @Param(
            name = "feature_configuration",
            doc = "Feature configuration to be queried.",
            positional = false,
            named = true,
            type = FeatureConfigurationApi.class),
        @Param(
            name = "action_name",
            doc =
                "Name of the action. Has to be one of the names in "
                    + "@bazel_tools//tools/build_defs/cc:action_names.bzl "
                    + "(https://github.com/bazelbuild/bazel/blob/master/tools/build_defs/cc/"
                    + "action_names.bzl)",
            named = true,
            positional = false),
      })
  String getToolForAction(FeatureConfigurationT featureConfiguration, String actionName);

  @StarlarkMethod(
      name = "get_execution_requirements",
      doc = "Returns execution requirements for given action.",
      parameters = {
        @Param(
            name = "feature_configuration",
            doc = "Feature configuration to be queried.",
            positional = false,
            named = true,
            type = FeatureConfigurationApi.class),
        @Param(
            name = "action_name",
            doc =
                "Name of the action. Has to be one of the names in "
                    + "@bazel_tools//tools/build_defs/cc:action_names.bzl "
                    + "(https://github.com/bazelbuild/bazel/blob/master/tools/build_defs/cc/"
                    + "action_names.bzl)",
            named = true,
            positional = false),
      })
  Sequence<String> getExecutionRequirements(
      FeatureConfigurationT featureConfiguration, String actionName);

  @StarlarkMethod(
      name = "is_enabled",
      doc = "Returns True if given feature is enabled in the feature configuration.",
      parameters = {
        @Param(
            name = "feature_configuration",
            doc = "Feature configuration to be queried.",
            positional = false,
            named = true,
            type = FeatureConfigurationApi.class),
        @Param(
            name = "feature_name",
            doc = "Name of the feature.",
            named = true,
            positional = false),
      })
  boolean isEnabled(FeatureConfigurationT featureConfiguration, String featureName);

  @StarlarkMethod(
      name = "action_is_enabled",
      doc = "Returns True if given action_config is enabled in the feature configuration.",
      parameters = {
        @Param(
            name = "feature_configuration",
            doc = "Feature configuration to be queried.",
            positional = false,
            named = true,
            type = FeatureConfigurationApi.class),
        @Param(
            name = "action_name",
            doc = "Name of the action_config.",
            named = true,
            positional = false),
      })
  boolean actionIsEnabled(FeatureConfigurationT featureConfiguration, String actionName);

  @StarlarkMethod(
      name = "get_memory_inefficient_command_line",
      doc =
          "Returns flattened command line flags for given action, using given variables for "
              + "expansion. Flattens nested sets and ideally should not be used, or at least "
              + "should not outlive analysis. Work on memory efficient function returning Args is "
              + "ongoing.",
      parameters = {
        @Param(
            name = "feature_configuration",
            doc = "Feature configuration to be queried.",
            positional = false,
            named = true,
            type = FeatureConfigurationApi.class),
        @Param(
            name = "action_name",
            doc =
                "Name of the action. Has to be one of the names in "
                    + "@bazel_tools//tools/build_defs/cc:action_names.bzl "
                    + "(https://github.com/bazelbuild/bazel/blob/master/tools/build_defs/cc/"
                    + "action_names.bzl)",
            named = true,
            positional = false),
        @Param(
            name = "variables",
            doc = "Build variables to be used for template expansions.",
            named = true,
            positional = false,
            type = CcToolchainVariablesApi.class),
      })
  Sequence<String> getCommandLine(
      FeatureConfigurationT featureConfiguration,
      String actionName,
      CcToolchainVariablesT variables)
      throws EvalException;

  @StarlarkMethod(
      name = "get_environment_variables",
      doc = "Returns environment variables to be set for given action.",
      parameters = {
        @Param(
            name = "feature_configuration",
            doc = "Feature configuration to be queried.",
            positional = false,
            named = true,
            type = FeatureConfigurationApi.class),
        @Param(
            name = "action_name",
            doc =
                "Name of the action. Has to be one of the names in "
                    + "@bazel_tools//tools/build_defs/cc:action_names.bzl "
                    + "(https://github.com/bazelbuild/bazel/blob/master/tools/build_defs/cc/"
                    + "action_names.bzl)",
            named = true,
            positional = false),
        @Param(
            name = "variables",
            doc = "Build variables to be used for template expansion.",
            positional = false,
            named = true,
            type = CcToolchainVariablesApi.class),
      })
  Dict<String, String> getEnvironmentVariable(
      FeatureConfigurationT featureConfiguration,
      String actionName,
      CcToolchainVariablesT variables)
      throws EvalException;

  @StarlarkMethod(
      name = "create_compile_variables",
      doc = "Returns variables used for compilation actions.",
      parameters = {
        @Param(
            name = "cc_toolchain",
            doc = "cc_toolchain for which we are creating build variables.",
            positional = false,
            named = true,
            type = CcToolchainProviderApi.class),
        @Param(
            name = "feature_configuration",
            doc = "Feature configuration to be queried.",
            positional = false,
            named = true,
            type = FeatureConfigurationApi.class),
        @Param(
            name = "source_file",
            doc =
                "Optional source file for the compilation. Please prefer passing source_file here "
                    + "over appending it to the end of the command line generated from "
                    + "cc_common.get_memory_inefficient_command_line, as then it's in the power of "
                    + "the toolchain author to properly specify and position compiler flags.",
            named = true,
            positional = false,
            defaultValue = "None",
            noneable = true),
        @Param(
            name = "output_file",
            doc =
                "Optional output file of the compilation. Please prefer passing output_file here "
                    + "over appending it to the end of the command line generated from "
                    + "cc_common.get_memory_inefficient_command_line, as then it's in the power of "
                    + "the toolchain author to properly specify and position compiler flags.",
            named = true,
            positional = false,
            defaultValue = "None",
            noneable = true),
        @Param(
            name = "user_compile_flags",
            doc = "List of additional compilation flags (copts).",
            positional = false,
            named = true,
            defaultValue = "None",
            noneable = true,
            allowedTypes = {
              @ParamType(type = NoneType.class),
              @ParamType(type = Sequence.class),
            }),
        @Param(
            name = "include_directories",
            doc = "Depset of include directories.",
            positional = false,
            named = true,
            defaultValue = "None",
            noneable = true,
            allowedTypes = {@ParamType(type = NoneType.class), @ParamType(type = Depset.class)}),
        @Param(
            name = "quote_include_directories",
            doc = "Depset of quote include directories.",
            positional = false,
            named = true,
            defaultValue = "None",
            noneable = true,
            allowedTypes = {@ParamType(type = NoneType.class), @ParamType(type = Depset.class)}),
        @Param(
            name = "system_include_directories",
            doc = "Depset of system include directories.",
            positional = false,
            named = true,
            defaultValue = "None",
            noneable = true,
            allowedTypes = {@ParamType(type = NoneType.class), @ParamType(type = Depset.class)}),
        @Param(
            name = "framework_include_directories",
            doc = "Depset of framework include directories.",
            positional = false,
            named = true,
            defaultValue = "None",
            noneable = true,
            allowedTypes = {@ParamType(type = NoneType.class), @ParamType(type = Depset.class)}),
        @Param(
            name = "preprocessor_defines",
            doc = "Depset of preprocessor defines.",
            positional = false,
            named = true,
            defaultValue = "None",
            noneable = true,
            allowedTypes = {@ParamType(type = NoneType.class), @ParamType(type = Depset.class)}),
        @Param(
            name = "use_pic",
            doc = "When true the compilation will generate position independent code.",
            positional = false,
            named = true,
            defaultValue = "False"),
        // TODO(b/65151735): Remove once we migrate crosstools to features
        @Param(
            name = "add_legacy_cxx_options",
            doc = "Unused.",
            named = true,
            positional = false,
            defaultValue = "False")
      })
  CcToolchainVariablesT getCompileBuildVariables(
      CcToolchainProviderT ccToolchainProvider,
      FeatureConfigurationT featureConfiguration,
      Object sourceFile,
      Object outputFile,
      Object userCompileFlags,
      Object includeDirs,
      Object quoteIncludeDirs,
      Object systemIncludeDirs,
      Object frameworkIncludeDirs,
      Object defines,
      boolean usePic,
      boolean addLegacyCxxOptions)
      throws EvalException;

  @StarlarkMethod(
      name = "create_link_variables",
      doc = "Returns link variables used for linking actions.",
      parameters = {
        @Param(
            name = "cc_toolchain",
            doc = "cc_toolchain for which we are creating build variables.",
            positional = false,
            named = true,
            type = CcToolchainProviderApi.class),
        @Param(
            name = "feature_configuration",
            doc = "Feature configuration to be queried.",
            positional = false,
            named = true,
            type = FeatureConfigurationApi.class),
        @Param(
            name = "library_search_directories",
            doc = "Depset of directories where linker will look for libraries at link time.",
            positional = false,
            named = true,
            defaultValue = "None",
            noneable = true,
            allowedTypes = {@ParamType(type = NoneType.class), @ParamType(type = Depset.class)}),
        @Param(
            name = "runtime_library_search_directories",
            doc = "Depset of directories where loader will look for libraries at runtime.",
            positional = false,
            named = true,
            defaultValue = "None",
            noneable = true,
            allowedTypes = {@ParamType(type = NoneType.class), @ParamType(type = Depset.class)}),
        @Param(
            name = "user_link_flags",
            doc = "List of additional link flags (linkopts).",
            positional = false,
            named = true,
            defaultValue = "None",
            noneable = true,
            allowedTypes = {@ParamType(type = NoneType.class), @ParamType(type = Sequence.class)}),
        @Param(
            name = "output_file",
            doc = "Optional output file path.",
            named = true,
            positional = false,
            defaultValue = "None",
            noneable = true),
        @Param(
            name = "param_file",
            doc = "Optional param file path.",
            named = true,
            positional = false,
            defaultValue = "None",
            noneable = true),
        @Param(
            name = "def_file",
            doc = "Optional .def file path.",
            named = true,
            positional = false,
            defaultValue = "None",
            noneable = true),
        // TODO(b/65151735): Remove once we migrate crosstools to features
        @Param(
            name = "is_using_linker",
            doc =
                "True when using linker, False when archiver. Caller is responsible for keeping "
                    + "this in sync with action name used (is_using_linker = True for linking "
                    + "executable or dynamic library, is_using_linker = False for archiving static "
                    + "library).",
            named = true,
            positional = false,
            defaultValue = "True"),
        // TODO(b/65151735): Remove once we migrate crosstools to features
        @Param(
            name = "is_linking_dynamic_library",
            doc =
                "True when creating dynamic library, False when executable or static library. "
                    + "Caller is responsible for keeping this in sync with action name used. "
                    + ""
                    + "This field will be removed once b/65151735 is fixed.",
            named = true,
            positional = false,
            defaultValue = "False"),
        @Param(
            name = "must_keep_debug",
            doc =
                "When set to True, bazel will expose 'strip_debug_symbols' variable, which is "
                    + "usually used to use the linker to strip debug symbols from the output file.",
            named = true,
            positional = false,
            defaultValue = "True"),
        @Param(
            name = "use_test_only_flags",
            doc = "When set to true, 'is_cc_test' variable will be set.",
            named = true,
            positional = false,
            defaultValue = "False"),
        // TODO(b/65151735): Remove once we migrate crosstools to features
        @Param(
            name = "is_static_linking_mode",
            doc = "Unused.",
            named = true,
            positional = false,
            defaultValue = "True"),
      })
  CcToolchainVariablesT getLinkBuildVariables(
      CcToolchainProviderT ccToolchainProvider,
      FeatureConfigurationT featureConfiguration,
      Object librarySearchDirectories,
      Object runtimeLibrarySearchDirectories,
      Object userLinkFlags,
      Object outputFile,
      Object paramFile,
      Object defFile,
      boolean isUsingLinkerNotArchiver,
      boolean isCreatingSharedLibrary,
      boolean mustKeepDebug,
      boolean useTestOnlyFlags,
      boolean isStaticLinkingMode)
      throws EvalException;

  @StarlarkMethod(name = "empty_variables", documented = false)
  CcToolchainVariablesT getVariables();

  @StarlarkMethod(
      name = "create_library_to_link",
      doc = "Creates <code>LibraryToLink</code>",
      useStarlarkThread = true,
      parameters = {
        @Param(
            name = "actions",
            type = StarlarkActionFactoryApi.class,
            positional = false,
            named = true,
            doc = "<code>actions</code> object."),
        @Param(
            name = "feature_configuration",
            doc = "<code>feature_configuration</code> to be queried.",
            positional = false,
            named = true,
            type = FeatureConfigurationApi.class),
        @Param(
            name = "cc_toolchain",
            doc = "<code>CcToolchainInfo</code> provider to be used.",
            positional = false,
            named = true,
            type = CcToolchainProviderApi.class),
        @Param(
            name = "static_library",
            doc = "<code>File</code> of static library to be linked.",
            positional = false,
            named = true,
            noneable = true,
            defaultValue = "None",
            type = FileApi.class),
        @Param(
            name = "pic_static_library",
            doc = "<code>File</code> of pic static library to be linked.",
            positional = false,
            named = true,
            noneable = true,
            defaultValue = "None",
            type = FileApi.class),
        @Param(
            name = "dynamic_library",
            doc =
                "<code>File</code> of dynamic library to be linked. Always used for runtime "
                    + "and used for linking if <code>interface_library</code> is not passed.",
            positional = false,
            named = true,
            noneable = true,
            defaultValue = "None",
            type = FileApi.class),
        @Param(
            name = "interface_library",
            doc = "<code>File</code> of interface library to be linked.",
            positional = false,
            named = true,
            noneable = true,
            defaultValue = "None",
            type = FileApi.class),
        @Param(
            name = "alwayslink",
            doc = "Whether to link the static library/objects in the --whole_archive block.",
            positional = false,
            named = true,
            defaultValue = "False"),
        @Param(
            name = "dynamic_library_symlink_path",
            doc =
                "Override the default path of the dynamic library link in the solib directory. "
                    + "Empty string to use the default.",
            positional = false,
            named = true,
            type = String.class,
            defaultValue = "''"),
        @Param(
            name = "interface_library_symlink_path",
            doc =
                "Override the default path of the interface library link in the solib directory. "
                    + "Empty string to use the default.",
            positional = false,
            named = true,
            type = String.class,
            defaultValue = "''"),
      })
  LibraryToLinkT createLibraryLinkerInput(
      Object actions,
      Object featureConfiguration,
      Object ccToolchainProvider,
      Object staticLibrary,
      Object picStaticLibrary,
      Object dynamicLibrary,
      Object interfaceLibrary,
      boolean alwayslink,
      String dynamicLibraryPath,
      String interfaceLibraryPath,
      StarlarkThread thread)
      throws EvalException, InterruptedException;

  @StarlarkMethod(
      name = "create_linker_input",
      doc = "Creates a <code>LinkingContext</code>.",
      useStarlarkThread = true,
      parameters = {
        @Param(
            name = "owner",
            doc = "List of <code>LibraryToLink</code>.",
            positional = false,
            named = true,
            type = Label.class),
        @Param(
            name = "libraries",
            doc = "List of <code>LibraryToLink</code>.",
            positional = false,
            named = true,
            noneable = true,
            defaultValue = "None",
            allowedTypes = {@ParamType(type = NoneType.class), @ParamType(type = Depset.class)}),
        @Param(
            name = "user_link_flags",
            doc = "List of user link flags passed as strings.",
            positional = false,
            named = true,
            noneable = true,
            defaultValue = "None",
            allowedTypes = {@ParamType(type = NoneType.class), @ParamType(type = Depset.class)}),
        @Param(
            name = "additional_inputs",
            doc = "For additional inputs to the linking action, e.g.: linking scripts.",
            positional = false,
            named = true,
            noneable = true,
            defaultValue = "None",
            allowedTypes = {@ParamType(type = NoneType.class), @ParamType(type = Depset.class)}),
      })
  LinkerInputT createLinkerInput(
      Label owner,
      Object librariesToLinkObject,
      Object userLinkFlagsObject,
      Object nonCodeInputs,
      StarlarkThread thread)
      throws EvalException, InterruptedException;

  @StarlarkMethod(
      name = "check_experimental_cc_shared_library",
      doc = "DO NOT USE. This is to guard use of cc_shared_library.",
      useStarlarkThread = true,
      documented = false)
  void checkExperimentalCcSharedLibrary(StarlarkThread thread) throws EvalException;

  @StarlarkMethod(
      name = "create_linking_context",
      doc = "Creates a <code>LinkingContext</code>.",
      useStarlarkThread = true,
      parameters = {
        @Param(
            name = "linker_inputs",
            doc = "Depset of <code>LinkerInput</code>.",
            positional = false,
            named = true,
            noneable = true,
            defaultValue = "None",
            allowedTypes = {@ParamType(type = NoneType.class), @ParamType(type = Depset.class)}),
        @Param(
            name = "libraries_to_link",
            doc = "List of <code>LibraryToLink</code>.",
            positional = false,
            named = true,
            disableWithFlag = FlagIdentifier.INCOMPATIBLE_REQUIRE_LINKER_INPUT_CC_API,
            noneable = true,
            defaultValue = "None",
            valueWhenDisabled = "None",
            allowedTypes = {@ParamType(type = NoneType.class), @ParamType(type = Sequence.class)}),
        @Param(
            name = "user_link_flags",
            doc = "List of user link flags passed as strings.",
            positional = false,
            named = true,
            disableWithFlag = FlagIdentifier.INCOMPATIBLE_REQUIRE_LINKER_INPUT_CC_API,
            noneable = true,
            defaultValue = "None",
            valueWhenDisabled = "None",
            allowedTypes = {@ParamType(type = NoneType.class), @ParamType(type = Sequence.class)}),
        @Param(
            name = "additional_inputs",
            doc = "For additional inputs to the linking action, e.g.: linking scripts.",
            positional = false,
            named = true,
            disableWithFlag = FlagIdentifier.INCOMPATIBLE_REQUIRE_LINKER_INPUT_CC_API,
            noneable = true,
            defaultValue = "None",
            valueWhenDisabled = "None",
            allowedTypes = {@ParamType(type = NoneType.class), @ParamType(type = Sequence.class)}),
      })
  LinkingContextT createCcLinkingInfo(
      Object linkerInputs,
      Object librariesToLinkObject,
      Object userLinkFlagsObject,
      Object nonCodeInputs, // <FileT> expected
      StarlarkThread thread)
      throws EvalException, InterruptedException;

  @StarlarkMethod(
      name = "merge_cc_infos",
      doc = "Merges a list of <code>CcInfo</code>s into one.",
      parameters = {
        @Param(
            name = "cc_infos",
            doc = "List of <code>CcInfo</code>s to be merged.",
            positional = false,
            named = true,
            defaultValue = "[]",
            type = Sequence.class)
      })
  CcInfoApi<FileT> mergeCcInfos(Sequence<?> ccInfos) // <CcInfoApi> expected
      throws EvalException;

  @StarlarkMethod(
      name = "create_compilation_context",
      doc = "Creates a <code>CompilationContext</code>.",
      parameters = {
        @Param(
            name = "headers",
            doc = "Set of headers needed to compile this target",
            positional = false,
            named = true,
            defaultValue = "unbound",
            type = Object.class),
        @Param(
            name = "system_includes",
            doc =
                "Set of search paths for header files referenced by angle brackets, i.e. "
                    + "#include &lt;foo/bar/header.h&gt;. They can be either relative to the exec "
                    + "root or absolute. Usually passed with -isystem",
            positional = false,
            named = true,
            defaultValue = "unbound",
            type = Object.class),
        @Param(
            name = "includes",
            doc =
                "Set of search paths for header files referenced both by angle bracket and quotes."
                    + "Usually passed with -I",
            positional = false,
            named = true,
            defaultValue = "unbound",
            type = Object.class),
        @Param(
            name = "quote_includes",
            doc =
                "Set of search paths for header files referenced by quotes, i.e. "
                    + "#include \"foo/bar/header.h\". They can be either relative to the exec "
                    + "root or absolute. Usually passed with -iquote",
            positional = false,
            named = true,
            defaultValue = "unbound",
            type = Object.class),
        @Param(
            name = "framework_includes",
            doc = "Set of framework search paths for header files (Apple platform only)",
            positional = false,
            named = true,
            defaultValue = "unbound",
            type = Object.class),
        @Param(
            name = "defines",
            doc =
                "Set of defines needed to compile this target. Each define is a string. Propagated"
                    + " transitively to dependents.",
            positional = false,
            named = true,
            defaultValue = "unbound",
            type = Object.class),
        @Param(
            name = "local_defines",
            doc =
                "Set of defines needed to compile this target. Each define is a string. Not"
                    + " propagated transitively to dependents.",
            positional = false,
            named = true,
            defaultValue = "unbound",
            type = Object.class),
      })
  CompilationContextT createCcCompilationContext(
      Object headers,
      Object systemIncludes,
      Object includes,
      Object quoteIncludes,
      Object frameworkIncludes,
      Object defines,
      Object localDefines)
      throws EvalException;

  // TODO(b/65151735): Remove when cc_flags is entirely set from features.
  // This should only be called from the cc_flags_supplier rule.
  @StarlarkMethod(
      name = "legacy_cc_flags_make_variable_do_not_use",
      documented = false,
      parameters = {
        @Param(
            name = "cc_toolchain",
            doc = "C++ toolchain provider to be used.",
            positional = false,
            named = true,
            type = CcToolchainProviderApi.class)
      })
  String legacyCcFlagsMakeVariable(CcToolchainProviderT ccToolchain);

  @StarlarkMethod(
      name = "is_cc_toolchain_resolution_enabled_do_not_use",
      documented = false,
      parameters = {
        @Param(
            name = "ctx",
            positional = false,
            named = true,
            type = StarlarkRuleContextApi.class,
            doc = "The rule context."),
      },
      doc = "Returns true if the --incompatible_enable_cc_toolchain_resolution flag is enabled.")
  boolean isCcToolchainResolutionEnabled(SkylarkRuleContextT ruleContext);

  @StarlarkMethod(
      name = "create_cc_toolchain_config_info",
      doc = "Creates a <code>CcToolchainConfigInfo</code> provider",
      parameters = {
        @Param(
            name = "ctx",
            positional = false,
            named = true,
            type = StarlarkRuleContextApi.class,
            doc = "The rule context."),
        @Param(
            name = "features",
            positional = false,
            named = true,
            defaultValue = "[]",
            type = Sequence.class,
            doc =
                "A list of <a href=\"https://github.com/bazelbuild/bazel/blob/master/tools/cpp/"
                    + "cc_toolchain_config_lib.bzl#L336\">features</a>."),
        @Param(
            name = "action_configs",
            positional = false,
            named = true,
            defaultValue = "[]",
            type = Sequence.class,
            doc =
                "A list of <a href=\"https://github.com/bazelbuild/bazel/blob/master/tools/cpp/"
                    + "cc_toolchain_config_lib.bzl#L461\">action_configs</a>."),
        @Param(
            name = "artifact_name_patterns",
            positional = false,
            named = true,
            defaultValue = "[]",
            type = Sequence.class,
            doc =
                "A list of <a href=\"https://github.com/bazelbuild/bazel/blob/master/tools/cpp/"
                    + "cc_toolchain_config_lib.bzl#L516\">artifact_name_patterns</a>."),
        @Param(
            name = "cxx_builtin_include_directories",
            positional = false,
            named = true,
            defaultValue = "[]",
            type = Sequence.class,
            doc =
                "<p>Built-in include directories for C++ compilation. These should be the exact "
                    + "paths used by the compiler, and are generally relative to the exec root.</p>"
                    + "<p>The paths used by the compiler can be determined by 'gcc -E -xc++ - -v'."
                    + "</p><p>We currently use the C++ paths also for C compilation, which is safe "
                    + "as long as there are no name clashes between C++ and C header files.</p>"
                    + "<p>Relative paths are resolved relative to the configuration file directory."
                    + "</p><p>If the compiler has --sysroot support, then these paths should use "
                    + "%sysroot% rather than the include path, and specify the sysroot attribute "
                    + "in order to give blaze the information necessary to make the correct "
                    + "replacements.</p>"),
        @Param(
            name = "toolchain_identifier",
            positional = false,
            type = String.class,
            named = true,
            doc =
                "<p>The unique identifier of the toolchain within the crosstool release. It must "
                    + "be possible to use this as a directory name in a path.</p>"
                    + "<p>It has to match the following regex: [a-zA-Z_][\\.\\- \\w]*</p>"),
        @Param(
            name = "host_system_name",
            positional = false,
            type = String.class,
            named = true,
            doc = "The system name which is required by the toolchain to run."),
        @Param(
            name = "target_system_name",
            positional = false,
            type = String.class,
            named = true,
            doc = "The GNU System Name."),
        @Param(
            name = "target_cpu",
            positional = false,
            type = String.class,
            named = true,
            doc = "The target architecture string."),
        @Param(
            name = "target_libc",
            positional = false,
            type = String.class,
            named = true,
            doc = "The libc version string (e.g. \"glibc-2.2.2\")."),
        @Param(
            name = "compiler",
            positional = false,
            type = String.class,
            named = true,
            doc = "The compiler version string (e.g. \"gcc-4.1.1\")."),
        @Param(
            name = "abi_version",
            positional = false,
            type = String.class,
            named = true,
            doc = "The abi in use, which is a gcc version. E.g.: \"gcc-3.4\""),
        @Param(
            name = "abi_libc_version",
            positional = false,
            type = String.class,
            named = true,
            doc = "The glibc version used by the abi we're using."),
        @Param(
            name = "tool_paths",
            positional = false,
            named = true,
            defaultValue = "[]",
            type = Sequence.class,
            doc =
                "A list of <a href=\"https://github.com/bazelbuild/bazel/blob/master/tools/cpp/"
                    + "cc_toolchain_config_lib.bzl#L400\">tool_paths</a>."),
        @Param(
            name = "make_variables",
            positional = false,
            named = true,
            defaultValue = "[]",
            type = Sequence.class,
            doc =
                "A list of <a href=\"https://github.com/bazelbuild/bazel/blob/master/tools/cpp/"
                    + "cc_toolchain_config_lib.bzl#L86\">make_variables</a>."),
        @Param(
            name = "builtin_sysroot",
            positional = false,
            noneable = true,
            defaultValue = "None",
            allowedTypes = {@ParamType(type = String.class), @ParamType(type = NoneType.class)},
            named = true,
            doc =
                "The built-in sysroot. If this attribute is not present, Bazel does not "
                    + "allow using a different sysroot, i.e. through the --grte_top option."),
        @Param(
            name = "cc_target_os",
            positional = false,
            noneable = true,
            defaultValue = "None",
            allowedTypes = {@ParamType(type = String.class), @ParamType(type = NoneType.class)},
            named = true,
            doc = "Internal purpose only, do not use."),
      })
  CcToolchainConfigInfoT ccToolchainConfigInfoFromStarlark(
      SkylarkRuleContextT skylarkRuleContext,
      Sequence<?> features, // <StructApi> expected
      Sequence<?> actionConfigs, // <StructApi> expected
      Sequence<?> artifactNamePatterns, // <StructApi> expected
      Sequence<?> cxxBuiltInIncludeDirectories, // <String> expected
      String toolchainIdentifier,
      String hostSystemName,
      String targetSystemName,
      String targetCpu,
      String targetLibc,
      String compiler,
      String abiVersion,
      String abiLibcVersion,
      Sequence<?> toolPaths, // <StructApi> expected
      Sequence<?> makeVariables, // <StructApi> expected
      Object builtinSysroot,
      Object ccTargetOs)
      throws EvalException;

  @StarlarkMethod(
      name = "create_linking_context_from_compilation_outputs",
      doc =
          "Should be used for creating library rules that can propagate information downstream in"
              + " order to be linked later by a top level rule that does transitive linking to"
              + " create an executable or dynamic library. Returns tuple of "
              + "(<code>CcLinkingContext</code>, <code>CcLinkingOutputs</code>).",
      useStarlarkThread = true,
      parameters = {
        @Param(
            name = "actions",
            type = StarlarkActionFactoryApi.class,
            positional = false,
            named = true,
            doc = "<code>actions</code> object."),
        @Param(
            name = "feature_configuration",
            doc = "<code>feature_configuration</code> to be queried.",
            positional = false,
            named = true,
            type = FeatureConfigurationApi.class),
        @Param(
            name = "cc_toolchain",
            doc = "<code>CcToolchainInfo</code> provider to be used.",
            positional = false,
            named = true,
            type = CcToolchainProviderApi.class),
        @Param(
            name = "compilation_outputs",
            doc = "Compilation outputs containing object files to link.",
            positional = false,
            named = true,
            type = CcCompilationOutputsApi.class),
        @Param(
            name = "user_link_flags",
            doc = "Additional list of linking options.",
            positional = false,
            named = true,
            defaultValue = "[]",
            type = Sequence.class),
        @Param(
            name = "linking_contexts",
            doc =
                "Libraries from dependencies. These libraries will be linked into the output "
                    + "artifact of the link() call, be it a binary or a library.",
            positional = false,
            named = true,
            defaultValue = "[]",
            type = Sequence.class),
        @Param(
            name = "name",
            doc =
                "This is used for naming the output artifacts of actions created by this "
                    + "method.",
            positional = false,
            named = true,
            type = String.class),
        @Param(
            name = "language",
            doc = "Only C++ supported for now. Do not use this parameter.",
            positional = false,
            named = true,
            defaultValue = "'c++'",
            type = String.class),
        @Param(
            name = "alwayslink",
            doc = "Whether this library should always be linked.",
            positional = false,
            named = true,
            defaultValue = "False",
            type = Boolean.class),
        @Param(
            name = "additional_inputs",
            doc = "For additional inputs to the linking action, e.g.: linking scripts.",
            positional = false,
            named = true,
            defaultValue = "[]",
            type = Sequence.class),
        @Param(
            name = "disallow_static_libraries",
            doc = "Whether static libraries should be created.",
            positional = false,
            named = true,
            defaultValue = "False",
            type = Boolean.class),
        @Param(
            name = "disallow_dynamic_library",
            doc = "Whether a dynamic library should be created.",
            positional = false,
            named = true,
            defaultValue = "False",
            type = Boolean.class),
        @Param(
            name = "grep_includes",
            positional = false,
            named = true,
            noneable = true,
            defaultValue = "None",
            allowedTypes = {@ParamType(type = FileApi.class), @ParamType(type = NoneType.class)}),
      })
  Tuple<Object> createLinkingContextFromCompilationOutputs(
      SkylarkActionFactoryT skylarkActionFactoryApi,
      FeatureConfigurationT skylarkFeatureConfiguration,
      CcToolchainProviderT skylarkCcToolchainProvider,
      CompilationOutputsT compilationOutputs,
      Sequence<?> userLinkFlags, // <String> expected
      Sequence<?> linkingContexts, // <LinkingContextT> expected
      String name,
      String language,
      boolean alwayslink,
      Sequence<?> additionalInputs, // <FileT> expected
      boolean disallowStaticLibraries,
      boolean disallowDynamicLibraries,
      Object grepIncludes,
      StarlarkThread thread)
      throws InterruptedException, EvalException;
}
