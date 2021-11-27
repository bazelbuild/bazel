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

import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.StaticallyLinkedMarkerProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.configuredtargets.PackageGroupConfiguredTarget;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleContext;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.cpp.CcBinary.CcLauncherInfo;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import net.starlark.java.annot.Param;
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
      name = "get_linked_artifact",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true),
        @Param(name = "cc_toolchain", positional = false, named = true),
        @Param(name = "config", positional = false, named = true),
        @Param(name = "is_dynamic_link_type", positional = false, named = true),
      })
  public Artifact getLinkedArtifactForStarlark(
      StarlarkRuleContext starlarkRuleContext,
      CcToolchainProvider ccToolchain,
      BuildConfigurationValue config,
      Boolean isDynamicLinkType)
      throws EvalException {
    Link.LinkTargetType linkType =
        isDynamicLinkType ? Link.LinkTargetType.DYNAMIC_LIBRARY : Link.LinkTargetType.EXECUTABLE;
    try {
      return CppHelper.getLinkedArtifact(
          starlarkRuleContext.getRuleContext(), ccToolchain, config, linkType);
    } catch (RuleErrorException e) {
      throw new EvalException(e);
    }
  }

  @StarlarkMethod(
      name = "collect_compilation_prerequisites",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true),
        @Param(name = "compilation_context", positional = false, named = true),
      })
  public Depset collectCompilationPrerequisites(
      StarlarkRuleContext starlarkRuleContext, CcCompilationContext compilationContext) {
    return Depset.of(
        Artifact.TYPE,
        CcCommon.collectCompilationPrerequisites(
            starlarkRuleContext.getRuleContext(), compilationContext));
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
      name = "statically_linked_marker_provider",
      documented = false,
      parameters = {
        @Param(name = "is_linked_statically", positional = false, named = true),
      })
  public StaticallyLinkedMarkerProvider staticallyLinkedMarkerProvider(boolean isLinkedStatically) {
    return new StaticallyLinkedMarkerProvider(isLinkedStatically);
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
      name = "collect_native_cc_libraries",
      documented = false,
      parameters = {
        @Param(name = "deps", positional = false, named = true),
        @Param(name = "libraries_to_link", positional = false, named = true),
      })
  public CcNativeLibraryInfo collectNativeCcLibraries(Sequence<?> deps, Sequence<?> librariesToLink)
      throws EvalException {
    return CppHelper.collectNativeCcLibraries(
        Sequence.cast(deps, TransitiveInfoCollection.class, "deps"),
        Sequence.cast(librariesToLink, LibraryToLink.class, "libraries_to_link"));
  }

  @StarlarkMethod(
      name = "PackageGroupInfo",
      documented = false,
      structField = true,
      parameters = {})
  public Provider getPackageGroupInfo() {
    return PackageGroupConfiguredTarget.PROVIDER;
  }

  @StarlarkMethod(
      name = "strip",
      documented = false,
      parameters = {
        @Param(name = "ctx", named = true, positional = false),
        @Param(name = "toolchain", named = true, positional = false),
        @Param(name = "input", named = true, positional = false),
        @Param(name = "output", named = true, positional = false),
        @Param(name = "feature_configuration", named = true, positional = false),
      })
  public void createStripAction(
      StarlarkRuleContext ctx,
      CcToolchainProvider toolchain,
      Artifact input,
      Artifact output,
      FeatureConfigurationForStarlark featureConfig)
      throws EvalException, RuleErrorException {
    CppHelper.createStripAction(
        ctx.getRuleContext(),
        toolchain,
        ctx.getRuleContext().getFragment(CppConfiguration.class),
        input,
        output,
        featureConfig.getFeatureConfiguration());
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
}
