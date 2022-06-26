// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.cpp;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.analysis.starlark.StarlarkActionFactory;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute.ComputedDefault;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.cpp.CcBinary.CcLauncherInfo;
import com.google.devtools.build.lib.rules.cpp.CcLinkingContext.Linkstamp;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.NativeComputedDefaultApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.vfs.PathFragment;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkValue;

/** Utility methods for rules in Starlark Builtins */
@StarlarkBuiltin(name = "cc_internal", category = DocCategory.BUILTIN, documented = false)
public class CcStarlarkInternal implements StarlarkValue {

  public static final String NAME = "cc_internal";

  @StarlarkMethod(
      name = "is_package_headers_checking_mode_set",
      documented = false,
      parameters = {@Param(name = "ctx", positional = false, named = true)})
  public boolean isPackageHeadersCheckingModeSetForStarlark(
      StarlarkRuleContext starlarkRuleContext) {
    return starlarkRuleContext.getRuleContext().getRule().getPackage().isDefaultHdrsCheckSet();
  }

  @StarlarkMethod(
      name = "package_headers_checking_mode",
      documented = false,
      parameters = {@Param(name = "ctx", positional = false, named = true)})
  public String getPackageHeadersCheckingModeForStarlark(StarlarkRuleContext starlarkRuleContext) {
    return starlarkRuleContext.getRuleContext().getRule().getPackage().getDefaultHdrsCheck();
  }

  @StarlarkMethod(
      name = "is_package_headers_checking_mode_set_for_aspect",
      documented = false,
      parameters = {@Param(name = "ctx", positional = false, named = true)})
  public boolean isPackageHeadersCheckingModeSetForStarlarkAspect(
      StarlarkRuleContext starlarkRuleContext) {
    return starlarkRuleContext.getRuleContext().getTarget().getPackage().isDefaultHdrsCheckSet();
  }

  @StarlarkMethod(
      name = "package_headers_checking_mode_for_aspect",
      documented = false,
      parameters = {@Param(name = "ctx", positional = false, named = true)})
  public String getPackageHeadersCheckingModeForStarlarkAspect(
      StarlarkRuleContext starlarkRuleContext) {
    return starlarkRuleContext.getRuleContext().getTarget().getPackage().getDefaultHdrsCheck();
  }

  @StarlarkMethod(
      name = "create_common",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true),
      })
  public CcCommon createCommon(StarlarkRuleContext starlarkRuleContext) throws EvalException {
    try {
      return new CcCommon(starlarkRuleContext.getRuleContext());
    } catch (RuleErrorException e) {
      throw new EvalException(e);
    }
  }

  @StarlarkMethod(
      name = "create_cc_provider",
      documented = false,
      parameters = {
        @Param(name = "cc_info", positional = false, named = true),
      })
  public CcStarlarkApiInfo createCcProvider(CcInfo ccInfo) {
    return new CcStarlarkApiInfo(ccInfo);
  }

  @StarlarkMethod(
      name = "get_build_info",
      documented = false,
      parameters = {@Param(name = "ctx")})
  public Sequence<Artifact> getBuildInfo(StarlarkRuleContext ruleContext)
      throws EvalException, InterruptedException {
    return StarlarkList.immutableCopyOf(
        ruleContext.getRuleContext().getBuildInfo(CppBuildInfo.KEY));
  }

  @StarlarkMethod(name = "launcher_provider", documented = false, structField = true)
  public ProviderApi getCcLauncherInfoProvider() throws EvalException {
    return CcLauncherInfo.PROVIDER;
  }

  @StarlarkMethod(
      name = "create_linkstamp",
      documented = false,
      parameters = {
        @Param(name = "actions", positional = false, named = true),
        @Param(name = "linkstamp", positional = false, named = true),
        @Param(name = "compilation_context", positional = false, named = true),
      })
  public Linkstamp createLinkstamp(
      StarlarkActionFactory starlarkActionFactoryApi,
      Artifact linkstamp,
      CcCompilationContext ccCompilationContext)
      throws EvalException {
    try {
      return new Linkstamp( // throws InterruptedException
          linkstamp,
          ccCompilationContext.getDeclaredIncludeSrcs(),
          starlarkActionFactoryApi.getActionConstructionContext().getActionKeyContext());
    } catch (CommandLineExpansionException | InterruptedException ex) {
      throw new EvalException(ex);
    }
  }

  static class DefaultCoptsBuiltinComputedDefault extends ComputedDefault
      implements NativeComputedDefaultApi {
    @Override
    public Object getDefault(AttributeMap rule) {
      return rule.getPackageDefaultCopts();
    }

    @Override
    public boolean resolvableWithRawAttributes() {
      return true;
    }
  }

  @StarlarkMethod(name = "default_copts_computed_default", documented = false)
  public ComputedDefault getDefaultCoptsComputedDefault() {
    return new DefaultCoptsBuiltinComputedDefault();
  }

  static class DefaultHdrsCheckBuiltinComputedDefault extends ComputedDefault
      implements NativeComputedDefaultApi {
    @Override
    public Object getDefault(AttributeMap rule) {
      return rule.isPackageDefaultHdrsCheckSet() ? rule.getPackageDefaultHdrsCheck() : "";
    }

    @Override
    public boolean resolvableWithRawAttributes() {
      return true;
    }
  }

  @StarlarkMethod(name = "default_hdrs_check_computed_default", documented = false)
  public ComputedDefault getDefaultHdrsCheckComputedDefault() {
    return new DefaultHdrsCheckBuiltinComputedDefault();
  }

  // TODO(b/207761932): perhaps move this to another internal module
  @StarlarkMethod(
      name = "declare_shareable_artifact",
      parameters = {
        @Param(name = "ctx"),
        @Param(name = "path"),
      },
      documented = false)
  public FileApi createShareableArtifact(StarlarkRuleContext ruleContext, String path)
      throws EvalException {
    return ruleContext
        .getRuleContext()
        .getShareableArtifact(PathFragment.create(path), ruleContext.getBinDirectory());
  }

  static class DefParserComputedDefault extends ComputedDefault
      implements NativeComputedDefaultApi {
    @Override
    public Object getDefault(AttributeMap rule) {
      // Every cc_rule depends implicitly on the def_parser tool.
      // The only exceptions are the rules for building def_parser itself.
      // To avoid cycles in the dependency graph, return null for rules under
      // @bazel_tools//third_party/def_parser and @bazel_tools//tools/cpp
      String label = rule.getLabel().toString();
      return label.startsWith("@bazel_tools//third_party/def_parser")
              // @bazel_tools//tools/cpp:malloc and @bazel_tools//tools/cpp:stl
              // are implicit dependencies of all cc rules,
              // thus a dependency of the def_parser.
              || label.startsWith("@bazel_tools//tools/cpp")
          ? null
          : Label.parseAbsoluteUnchecked("@bazel_tools//tools/def_parser:def_parser");
    }

    @Override
    public boolean resolvableWithRawAttributes() {
      return true;
    }
  }

  @StarlarkMethod(name = "def_parser_computed_default", documented = false)
  public ComputedDefault getDefParserComputedDefault() {
    return new DefParserComputedDefault();
  }

  /**
   * TODO(bazel-team): This can be re-written directly to Starlark but it will cause a memory
   * regression due to the way StarlarkComputedDefault is stored for each rule.
   */
  static class StlComputedDefault extends ComputedDefault implements NativeComputedDefaultApi {
    @Override
    public Object getDefault(AttributeMap rule) {
      return rule.getOrDefault("tags", Type.STRING_LIST, ImmutableList.of()).contains("__CC_STL__")
          ? null
          : Label.parseAbsoluteUnchecked("@//third_party/stl");
    }

    @Override
    public boolean resolvableWithRawAttributes() {
      return true;
    }
  }

  @StarlarkMethod(name = "stl_computed_default", documented = false)
  public ComputedDefault getStlComputedDefault() {
    return new StlComputedDefault();
  }

  @StarlarkMethod(
      name = "create_cc_launcher_info",
      doc = "Create a CcLauncherInfo instance.",
      parameters = {
        @Param(
            name = "cc_info",
            positional = false,
            named = true,
            doc = "CcInfo instance.",
            allowedTypes = {@ParamType(type = CcInfo.class)}),
        @Param(
            name = "compilation_outputs",
            positional = false,
            named = true,
            doc = "CcCompilationOutputs instance.",
            allowedTypes = {@ParamType(type = CcCompilationOutputs.class)})
      })
  public CcLauncherInfo createCcLauncherInfo(
      CcInfo ccInfo, CcCompilationOutputs compilationOutputs) {
    return new CcLauncherInfo(ccInfo, compilationOutputs);
  }
}
