// Copyright 2019 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.RunfilesApi;
import com.google.devtools.build.lib.skylarkbuildapi.StarlarkRuleContextApi;
import com.google.devtools.build.lib.skylarkbuildapi.core.TransitiveInfoCollectionApi;
import com.google.devtools.build.lib.skylarkbuildapi.go.GoConfigurationApi;
import com.google.devtools.build.lib.skylarkbuildapi.go.GoContextInfoApi;
import com.google.devtools.build.lib.skylarkbuildapi.go.GoPackageInfoApi;
import com.google.devtools.build.lib.skylarkbuildapi.platform.ConstraintValueInfoApi;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.NoneType;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.Tuple;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkDocumentationCategory;
import net.starlark.java.annot.StarlarkMethod;

/**
 * Helper class for the C++ functionality needed from Starlark specifically to implement go_wrap_cc.
 * TODO(b/113797843): Remove class once all the bits and pieces specific to Go can be implemented in
 * Starlark.
 */
@StarlarkBuiltin(
    name = "go_wrap_cc_helper_do_not_use",
    documented = false,
    doc = "",
    category = StarlarkDocumentationCategory.TOP_LEVEL_TYPE)
public interface GoWrapCcHelperApi<
        FileT extends FileApi,
        ConstraintValueT extends ConstraintValueInfoApi,
        SkylarkRuleContextT extends StarlarkRuleContextApi<ConstraintValueT>,
        CcInfoT extends CcInfoApi<FileT>,
        FeatureConfigurationT extends FeatureConfigurationApi,
        CcToolchainProviderT extends CcToolchainProviderApi<FeatureConfigurationT>,
        CcLinkingContextT extends CcLinkingContextApi<FileT>,
        GoConfigurationT extends GoConfigurationApi,
        GoContextInfoT extends GoContextInfoApi,
        TransitiveInfoCollectionT extends TransitiveInfoCollectionApi,
        CompilationInfoT extends CompilationInfoApi<FileT>,
        CcCompilationContextT extends CcCompilationContextApi<FileT>,
        WrapCcIncludeProviderT extends WrapCcIncludeProviderApi>
    extends WrapCcHelperApi<
        FeatureConfigurationT,
        ConstraintValueT,
        SkylarkRuleContextT,
        CcToolchainProviderT,
        CompilationInfoT,
        FileT,
        CcCompilationContextT,
        WrapCcIncludeProviderT> {

  @StarlarkMethod(
      name = "get_go_runfiles",
      doc = "",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true, type = StarlarkRuleContextApi.class),
      })
  // TODO(b/113797843): Not written in Starlark because of GoRunfilesProvider.
  public RunfilesApi starlarkGetGoRunfiles(SkylarkRuleContextT skylarkRuleContext)
      throws EvalException, InterruptedException;

  @StarlarkMethod(
      name = "get_arch_int_size",
      doc = "",
      documented = false,
      parameters = {
        @Param(name = "go", positional = false, named = true, type = GoConfigurationApi.class),
      })
  // TODO(b/113797843): Not written in Starlark because of GoCompilationHelper.
  public int getArchIntSize(GoConfigurationT goConfig);

  @StarlarkMethod(
      name = "collect_transitive_go_context_gopkg",
      doc = "",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true, type = StarlarkRuleContextApi.class),
        @Param(name = "export", positional = false, named = true, type = FileApi.class),
        @Param(name = "pkg", positional = false, named = true, type = FileApi.class),
        @Param(name = "gopkg", positional = false, named = true, type = FileApi.class),
        @Param(
            name = "wrap_context",
            positional = false,
            named = true,
            defaultValue = "None",
            noneable = true,
            allowedTypes = {
              @ParamType(type = NoneType.class),
              @ParamType(type = GoContextInfoApi.class)
            }),
        @Param(name = "cc_info", positional = false, named = true, type = CcInfoApi.class),
      })
  public GoContextInfoT starlarkCollectTransitiveGoContextGopkg(
      SkylarkRuleContextT skylarkRuleContext,
      FileT export,
      FileT pkg,
      FileT gopkg,
      Object skylarkWrapContext,
      CcInfoT ccInfo);

  @StarlarkMethod(
      name = "go_wrap_cc_info",
      doc = "",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true, type = StarlarkRuleContextApi.class),
        @Param(name = "cc_info", positional = false, named = true, type = CcInfoApi.class),
      })
  // TODO(b/113797843): GoWrapCcInfo is not written in Starlark because several native rules use it.
  public GoWrapCcInfoApi<FileT> getGoWrapCcInfo(
      SkylarkRuleContextT skylarkRuleContext, CcInfoT ccInfo)
      throws EvalException, InterruptedException;

  @StarlarkMethod(
      name = "go_cc_link_params_provider",
      doc = "",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true, type = StarlarkRuleContextApi.class),
        @Param(
            name = "linking_context",
            positional = false,
            named = true,
            type = CcLinkingContextApi.class),
      })
  public GoCcLinkParamsInfoApi getGoCcLinkParamsProvider(
      SkylarkRuleContextT ruleContext, CcLinkingContextT ccLinkingContext)
      throws EvalException, InterruptedException;

  @StarlarkMethod(
      name = "create_go_compile_actions",
      doc = "",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true, type = StarlarkRuleContextApi.class),
        @Param(
            name = "cc_toolchain",
            positional = false,
            named = true,
            type = CcToolchainProviderApi.class),
        @Param(name = "srcs", positional = false, named = true, type = Sequence.class),
        @Param(name = "deps", positional = false, named = true, type = Sequence.class),
      })
  public Tuple<FileT> createGoCompileActions(
      SkylarkRuleContextT skylarkRuleContext,
      CcToolchainProviderT ccToolchainProvider,
      Sequence<?> srcs, // <FileT> expected
      Sequence<?> deps /* <TransitiveInfoCollectionT> expected */)
      throws EvalException;

  @StarlarkMethod(
      name = "create_go_compile_actions_gopkg",
      doc = "",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true, type = StarlarkRuleContextApi.class),
        @Param(
            name = "cc_toolchain",
            positional = false,
            named = true,
            type = CcToolchainProviderApi.class),
        @Param(name = "srcs", positional = false, named = true, type = Sequence.class),
        @Param(name = "deps", positional = false, named = true, type = Sequence.class),
      })
  public Tuple<FileT> createGoCompileActionsGopkg(
      SkylarkRuleContextT skylarkRuleContext,
      CcToolchainProviderT ccToolchainProvider,
      Sequence<?> srcs, // <FileT> expected
      Sequence<?> deps /* <TransitiveInfoCollectionT> expected */)
      throws EvalException;

  @StarlarkMethod(
      name = "create_transitive_gopackage_info",
      doc = "",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true, type = StarlarkRuleContextApi.class),
        @Param(name = "gopkg", positional = false, named = true, type = FileApi.class),
        @Param(name = "export", positional = false, named = true, type = FileApi.class),
        @Param(name = "swig_out_go", positional = false, named = true, type = FileApi.class),
      })
  public GoPackageInfoApi createTransitiveGopackageInfo(
      SkylarkRuleContextT skylarkRuleContext, FileT skylarkGopkg, FileT export, FileT swigOutGo);

  @StarlarkMethod(
      name = "get_gopackage_files",
      doc = "",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true, type = StarlarkRuleContextApi.class),
        @Param(name = "gopkg", positional = false, named = true, type = FileApi.class),
      })
  public Depset /*<FileT>*/ getGopackageFilesForStarlark(
      SkylarkRuleContextT skylarkRuleContext, FileT skylarkGopkg);
}
