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

import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkRuleContextApi;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.ParamType;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Runtime.NoneType;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;

/** Utilites related to C++ support. */
@SkylarkModule(
    name = "cc_common",
    doc = "Utilities for C++ compilation, linking, and command line generation.")
// TODO(b/111365281): Add experimental field once it's available.
public interface BazelCcModuleApi<
        FileT extends FileApi,
        SkylarkRuleContextT extends SkylarkRuleContextApi,
        CcToolchainProviderT extends CcToolchainProviderApi,
        FeatureConfigurationT extends FeatureConfigurationApi,
        CompilationInfoT extends CompilationInfoApi,
        CcCompilationContextT extends CcCompilationContextApi,
        CcCompilationOutputsT extends CcCompilationOutputsApi,
        LinkingInfoT extends LinkingInfoApi,
        LinkingContextT extends CcLinkingContextApi,
        LibraryToLinkWrapperT extends LibraryToLinkWrapperApi,
        CcToolchainVariablesT extends CcToolchainVariablesApi,
        CcToolchainConfigInfoT extends CcToolchainConfigInfoApi>
    extends CcModuleApi<
        CcToolchainProviderT,
        FeatureConfigurationT,
        CcCompilationContextT,
        LinkingContextT,
        LibraryToLinkWrapperT,
        CcToolchainVariablesT,
        SkylarkRuleContextT,
        CcToolchainConfigInfoT> {

  @SkylarkCallable(
      name = "compile",
      documented = false,
      useLocation = true,
      parameters = {
        @Param(
            name = "ctx",
            positional = false,
            named = true,
            type = SkylarkRuleContextApi.class,
            doc = "The rule context."),
        @Param(
            name = "feature_configuration",
            doc = "Feature configuration to be queried.",
            positional = false,
            named = true,
            type = FeatureConfigurationApi.class),
        @Param(
            name = "cc_toolchain",
            doc = "C++ toolchain provider to be used.",
            positional = false,
            named = true,
            type = CcToolchainProviderApi.class),
        @Param(
            name = "srcs",
            doc = "The list of source files to be compiled, see cc_library.srcs",
            positional = false,
            named = true,
            defaultValue = "[]",
            type = SkylarkList.class),
        @Param(
            name = "hdrs",
            doc = "The list of public headers to be provided to dependents, see cc_library.hdrs",
            positional = false,
            named = true,
            defaultValue = "[]",
            type = SkylarkList.class),
        @Param(
            name = "includes",
            doc = "Include directories",
            positional = false,
            named = true,
            noneable = true,
            defaultValue = "[]",
            allowedTypes = {
              @ParamType(type = SkylarkNestedSet.class),
              @ParamType(type = SkylarkList.class)
            }),
        @Param(
            name = "copts",
            doc = "Additional list of compiler options.",
            positional = false,
            named = true,
            noneable = true,
            defaultValue = "None",
            allowedTypes = {
              @ParamType(type = SkylarkNestedSet.class),
              @ParamType(type = NoneType.class)
            }),
        @Param(
            name = "compilation_contexts",
            doc = "compilation_context instances affecting compilation, e.g. from dependencies",
            positional = false,
            named = true,
            defaultValue = "[]",
            type = SkylarkList.class)
      })
  CompilationInfoT compile(
      SkylarkRuleContextT skylarkRuleContext,
      FeatureConfigurationT skylarkFeatureConfiguration,
      CcToolchainProviderT skylarkCcToolchainProvider,
      SkylarkList<FileT> sources,
      SkylarkList<FileT> headers,
      Object skylarkIncludes,
      Object skylarkCopts,
      SkylarkList<CcCompilationContextT> ccCompilationContexts,
      Location location)
      throws EvalException, InterruptedException;

  @SkylarkCallable(
      name = "link",
      documented = false,
      parameters = {
        @Param(
            name = "ctx",
            positional = false,
            named = true,
            type = SkylarkRuleContextApi.class,
            doc = "The rule context."),
        @Param(
            name = "feature_configuration",
            doc = "Feature configuration to be queried.",
            positional = false,
            named = true,
            type = FeatureConfigurationApi.class),
        @Param(
            name = "cc_toolchain",
            doc = "C++ toolchain provider to be used.",
            positional = false,
            named = true,
            type = CcToolchainProviderApi.class),
        @Param(
            name = "cc_compilation_outputs",
            doc = "List of object files to be linked.",
            positional = false,
            named = true,
            defaultValue = "[]",
            type = CcCompilationOutputsApi.class),
        @Param(
            name = "linkopts",
            doc = "Additional list of linker options.",
            positional = false,
            named = true,
            defaultValue = "[]",
            noneable = true,
            allowedTypes = {
              @ParamType(type = SkylarkList.class),
              @ParamType(type = SkylarkNestedSet.class)
            }),
        @Param(
            name = "dynamic_library",
            doc = "Dynamic library file.",
            positional = false,
            named = true,
            defaultValue = "None",
            noneable = true,
            allowedTypes = {@ParamType(type = NoneType.class), @ParamType(type = FileApi.class)}),
        @Param(
            name = "linking_contexts",
            doc = "linking_context instances affecting linking, e.g. from dependencies",
            positional = false,
            named = true,
            noneable = true,
            defaultValue = "[]",
            type = SkylarkList.class),
        @Param(
            name = "neverlink",
            doc = "True if this should never be linked against other libraries.",
            positional = false,
            named = true,
            defaultValue = "False"),
      })
  LinkingInfoT link(
      SkylarkRuleContextT skylarkRuleContext,
      FeatureConfigurationT skylarkFeatureConfiguration,
      CcToolchainProviderT skylarkCcToolchainProvider,
      CcCompilationOutputsT ccCompilationOutputs,
      Object skylarkLinkopts,
      Object dynamicLibrary,
      SkylarkList<LinkingContextT> skylarkCcLinkingContexts,
      boolean neverLink)
      throws InterruptedException, EvalException;
}
