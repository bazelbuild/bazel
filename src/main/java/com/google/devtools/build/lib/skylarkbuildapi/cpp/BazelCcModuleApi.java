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
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkActionFactoryApi;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkRuleContextApi;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.ParamType;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.StarlarkContext;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Runtime.NoneType;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkList.Tuple;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;

/** Utilites related to C++ support. */
@SkylarkModule(
    name = "cc_common",
    doc = "Utilities for C++ compilation, linking, and command line generation.")
// TODO(b/111365281): Add experimental field once it's available.
public interface BazelCcModuleApi<
    FileT extends FileApi,
    SkylarkRuleContextT extends SkylarkRuleContextApi,
    SkylarkActionFactoryT extends SkylarkActionFactoryApi,
    CcToolchainProviderT extends CcToolchainProviderApi,
    FeatureConfigurationT extends FeatureConfigurationApi,
    CcCompilationContextT extends CcCompilationContextApi,
    CcCompilationOutputsT extends CcCompilationOutputsApi,
    LinkingContextT extends CcLinkingContextApi,
    LibraryToLinkT extends LibraryToLinkApi,
    CcToolchainVariablesT extends CcToolchainVariablesApi,
    CcToolchainConfigInfoT extends CcToolchainConfigInfoApi>
    extends CcModuleApi<
    FileT,
    CcToolchainProviderT,
    FeatureConfigurationT,
    CcCompilationContextT,
    LinkingContextT,
    LibraryToLinkT,
    CcToolchainVariablesT,
    SkylarkRuleContextT,
    CcToolchainConfigInfoT> {

  @SkylarkCallable(
      name = "compile",
      documented = false,
      useLocation = true,
      parameters = {
          @Param(
              name = "actions",
              type = SkylarkActionFactoryApi.class,
              positional = false,
              named = true,
              doc = "<code>actions</code> object."),
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
              name = "public_hdrs",
              doc = "The list of public headers to be provided to dependents, see cc_library.hdrs",
              positional = false,
              named = true,
              defaultValue = "[]",
              type = SkylarkList.class),
          @Param(
              name = "private_hdrs",
              doc =
                  "List of headers needed for compilation of srcs and NOT to be included by"
                      + " dependent rules.",
              positional = false,
              named = true,
              defaultValue = "[]",
              type = SkylarkList.class),
          @Param(
              name = "includes",
              doc =
                  "Search paths for header files referenced both by angle bracket and quotes. "
                      + "Usually passed with -I. Propagated to dependents transitively.",
              positional = false,
              named = true,
              noneable = true,
              defaultValue = "[]",
              type = SkylarkList.class),
          @Param(
              name = "user_compile_flags",
              doc = "Additional list of compiler options.",
              positional = false,
              named = true,
              defaultValue = "[]",
              type = SkylarkList.class),
          @Param(
              name = "compilation_contexts",
              doc = "compilation_context instances affecting compilation, e.g. from dependencies",
              positional = false,
              named = true,
              defaultValue = "[]",
              type = SkylarkList.class),
          @Param(
              name = "name",
              doc =
                  "This is used for naming the output artifacts of actions created by this "
                      + "method.",
              positional = false,
              named = true,
              type = String.class),
      })
  Tuple<Object> compile(
      SkylarkActionFactoryT skylarkActionFactory,
      FeatureConfigurationT skylarkFeatureConfiguration,
      CcToolchainProviderT skylarkCcToolchainProvider,
      SkylarkList<FileT> sources,
      SkylarkList<FileT> publicHeaders,
      SkylarkList<FileT> privateHeaders,
      SkylarkList<String> skylarkIncludes,
      SkylarkList<String> skylarkUserCompileFlags,
      SkylarkList<CcCompilationContextT> ccCompilationContexts,
      String name,
      Location location)
      throws EvalException, InterruptedException;

  @SkylarkCallable(
      name = "create_linking_context_from_compilation_outputs",
      documented = false,
      useLocation = true,
      useContext = true,
      parameters = {
          @Param(
              name = "actions",
              type = SkylarkActionFactoryApi.class,
              positional = false,
              named = true,
              doc = "<code>actions</code> object."),
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
              noneable = true,
              type = SkylarkList.class),
          @Param(
              name = "linking_contexts",
              doc = "linking_context instances affecting linking, e.g. from dependencies",
              positional = false,
              named = true,
              noneable = true,
              defaultValue = "[]",
              type = SkylarkList.class),
          @Param(
              name = "name",
              doc =
                  "This is used for naming the output artifacts of actions created by this "
                      + "method.",
              positional = false,
              named = true,
              type = String.class),
      })
  Tuple<Object> createLinkingContextFromCompilationOutputs(
      SkylarkActionFactoryT skylarkActionFactory,
      FeatureConfigurationT skylarkFeatureConfiguration,
      CcToolchainProviderT skylarkCcToolchainProvider,
      CcCompilationOutputsT ccCompilationOutputs,
      SkylarkList<String> userLinkFlags,
      SkylarkList<LinkingContextT> skylarkCcLinkingContexts,
      String name,
      Location location,
      StarlarkContext starlarkContext)
      throws InterruptedException, EvalException;
}
