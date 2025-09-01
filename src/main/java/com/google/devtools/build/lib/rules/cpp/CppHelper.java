// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.vfs.PathFragment;
import javax.annotation.Nullable;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;

/**
 * Helper class for functionality shared by cpp related rules.
 *
 * <p>This class can be used only after the loading phase.
 */
public class CppHelper {

  static final PathFragment OBJS = PathFragment.create("_objs");

  private CppHelper() {
    // prevents construction
  }

  /** Returns C++ toolchain, using toolchain resolution */
  public static CcToolchainProvider getToolchain(RuleContext ruleContext)
      throws RuleErrorException {
    ToolchainInfo toolchainInfo =
        ruleContext.getToolchainInfo(Label.parseCanonicalUnchecked("//tools/cpp:toolchain_type"));
    if (toolchainInfo == null) {
      toolchainInfo =
          ruleContext.getToolchainInfo(
              Label.parseCanonicalUnchecked("@bazel_tools//tools/cpp:toolchain_type"));
    }
    if (toolchainInfo == null) {
      throw ruleContext.throwWithRuleError(
          "Unable to find a CC toolchain using toolchain resolution. Did you properly set"
              + " --platforms?");
    }
    try {
      return CcToolchainProvider.PROVIDER.wrap((Info) toolchainInfo.getValue("cc"));
    } catch (EvalException e) {
      // There is not actually any reason for toolchainInfo.getValue to throw an exception.
      throw ruleContext.throwWithRuleError(
          "Unexpected eval exception from toolchainInfo.getValue('cc')");
    }
  }

  /** Returns the directory where object files are created. */
  public static PathFragment getObjDirectory(Label ruleLabel, boolean siblingRepositoryLayout) {
    return AnalysisUtils.getUniqueDirectory(ruleLabel, OBJS, siblingRepositoryLayout);
  }

  // LINT.IfChange
  /** Returns whether binaries must be compiled with position independent code. */
  public static boolean usePicForBinaries(
      CppConfiguration cppConfiguration, FeatureConfiguration featureConfiguration) {
    return cppConfiguration.forcePic()
        || (CcToolchainProvider.usePicForDynamicLibraries(cppConfiguration, featureConfiguration)
            && (cppConfiguration.getCompilationMode() != CompilationMode.OPT
                || featureConfiguration.isEnabled(CppRuleClasses.PREFER_PIC_FOR_OPT_BINARIES)));
  }

  // LINT.ThenChange(//src/main/starlark/builtins_bzl/common/cc/cc_helper_internal.bzl)

  /** Returns the FDO build subtype. */
  @Nullable
  public static String getFdoBuildStamp(
      CppConfiguration cppConfiguration,
      FdoContext fdoContext,
      FeatureConfiguration featureConfiguration)
      throws EvalException {
    FdoContext.BranchFdoProfile branchFdoProfile = fdoContext.getBranchFdoProfile();
    if (branchFdoProfile != null) {

      if (branchFdoProfile.isAutoFdo()) {
        return featureConfiguration.isEnabled(CppRuleClasses.AUTOFDO) ? "AFDO" : null;
      }
      if (branchFdoProfile.isAutoXBinaryFdo()) {
        return featureConfiguration.isEnabled(CppRuleClasses.XBINARYFDO) ? "XFDO" : null;
      }
    }
    if (fdoContext.getBranchFdoProfile() != null
        && (fdoContext.getBranchFdoProfile().isLlvmCSFdo()
            || cppConfiguration.getCSFdoInstrument() != null)) {
      return "CSFDO";
    }
    if (fdoContext.getBranchFdoProfile() != null || cppConfiguration.getFdoInstrument() != null) {
      return "FDO";
    }
    return null;
  }

  static Artifact getCompileOutputArtifact(
      ActionConstructionContext actionConstructionContext,
      Label label,
      String outputName,
      BuildConfigurationValue config) {
    PathFragment objectDir = getObjDirectory(label, config.isSiblingRepositoryLayout());
    return actionConstructionContext.getDerivedArtifact(
        objectDir.getRelative(outputName), config.getBinDirectory(label.getRepository()));
  }

  static Dict<?, ?> asDict(Object o) {
    return o == Starlark.UNBOUND ? Dict.empty() : (Dict<?, ?>) o;
  }
}
