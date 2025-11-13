// Copyright 2017 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.actions.PathMapper;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.starlark.StarlarkActionFactory;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleContext;
import com.google.devtools.build.lib.cmdline.BazelModuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.BuiltinRestriction;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.CcModuleApi;
import javax.annotation.Nullable;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkThread;

/**
 * A module that contains Starlark utilities for C++ support.
 *
 * <p>The Bazel team is planning to rewrite all native rules in Starlark. Many of these rules use
 * C++ functionality that is not presently exposed to the public Starlark C++ API. To speed up the
 * transition to Starlark, we are exposing functionality "as is" but preventing its use externally
 * until we are comfortable with the API which would need to be supported long term.
 *
 * <p>We are not opposed to gradually adding to and improving the public C++ API but nothing should
 * merged without following proper design processes and discussions.
 */
public abstract class CcModule
    implements CcModuleApi<
        StarlarkActionFactory,
        Artifact,
        FeatureConfigurationForStarlark,
        CcToolchainVariables,
        ConstraintValueInfo,
        StarlarkRuleContext> {

  @Override
  public Provider getCcToolchainProvider() {
    // TODO: b/433485282 this will need to change for Bazel once we update rules_cc containing
    // cl/791606702
    return CcToolchainProvider.BUILTINS_PROVIDER;
  }

  @Override
  public String getToolForAction(
      FeatureConfigurationForStarlark featureConfiguration,
      String actionName,
      StarlarkThread thread)
      throws EvalException {
    isCalledFromStarlarkCcCommon(thread);
    try {
      return featureConfiguration.getFeatureConfiguration().getToolPathForAction(actionName);
    } catch (IllegalArgumentException illegalArgumentException) {
      throw new EvalException(illegalArgumentException);
    }
  }

  // TODO(blaze-team): duplicate with the getExecutionRequirements below
  @Override
  public Sequence<String> getToolRequirementForAction(
      FeatureConfigurationForStarlark featureConfiguration,
      String actionName,
      StarlarkThread thread)
      throws EvalException {
    isCalledFromStarlarkCcCommon(thread);
    return StarlarkList.immutableCopyOf(
        featureConfiguration.getFeatureConfiguration().getToolRequirementsForAction(actionName));
  }

  @Override
  public Sequence<String> getExecutionRequirements(
      FeatureConfigurationForStarlark featureConfiguration,
      String actionName,
      StarlarkThread thread)
      throws EvalException {
    isCalledFromStarlarkCcCommon(thread);
    return StarlarkList.immutableCopyOf(
        featureConfiguration.getFeatureConfiguration().getToolRequirementsForAction(actionName));
  }

  @Override
  public boolean actionIsEnabled(
      FeatureConfigurationForStarlark featureConfiguration,
      String actionName,
      StarlarkThread thread)
      throws EvalException {
    isCalledFromStarlarkCcCommon(thread);
    return featureConfiguration.getFeatureConfiguration().actionIsConfigured(actionName);
  }

  @Override
  public Sequence<String> getCommandLine(
      FeatureConfigurationForStarlark featureConfiguration,
      String actionName,
      CcToolchainVariables variables,
      StarlarkThread thread)
      throws EvalException {
    isCalledFromStarlarkCcCommon(thread);
    return StarlarkList.immutableCopyOf(
        featureConfiguration.getFeatureConfiguration().getCommandLine(actionName, variables));
  }

  @Override
  public Dict<String, String> getEnvironmentVariable(
      FeatureConfigurationForStarlark featureConfiguration,
      String actionName,
      CcToolchainVariables variables,
      StarlarkThread thread)
      throws EvalException {
    isCalledFromStarlarkCcCommon(thread);
    return Dict.immutableCopyOf(
        featureConfiguration
            .getFeatureConfiguration()
            .getEnvironmentVariables(actionName, variables, PathMapper.NOOP));
  }

  @Override
  public CcToolchainVariables getVariables(StarlarkThread thread) throws EvalException {
    isCalledFromStarlarkCcCommon(thread);
    return CcToolchainVariables.empty();
  }

  /**
   * Converts an object that can be the NoneType to the actual object if it is not or returns the
   * default value if none.
   *
   * <p>This operation is wildly unsound. It performs no dymamic checks (casts), it simply lies
   * about the type.
   */
  @SuppressWarnings("unchecked")
  protected static <T> T convertFromNoneable(Object obj, @Nullable T defaultValue) {
    if (Starlark.UNBOUND == obj || Starlark.isNullOrNone(obj)) {
      return defaultValue;
    }
    return (T) obj; // totally unsafe
  }

  @Override
  public boolean checkExperimentalCcSharedLibrary(StarlarkThread thread) throws EvalException {
    isCalledFromStarlarkCcCommon(thread);
    return thread.getSemantics().getBool(BuildLanguageOptions.EXPERIMENTAL_CC_SHARED_LIBRARY);
  }

  @Override
  public boolean getIncompatibleDisableObjcLibraryTransition(StarlarkThread thread)
      throws EvalException {
    isCalledFromStarlarkCcCommon(thread);
    return thread
        .getSemantics()
        .getBool(BuildLanguageOptions.INCOMPATIBLE_DISABLE_OBJC_LIBRARY_TRANSITION);
  }

  @Override
  public boolean addGoExecGroupsToBinaryRules(StarlarkThread thread) {
    return thread.getSemantics().getBool(BuildLanguageOptions.ADD_GO_EXEC_GROUPS_TO_BINARY_RULES);
  }

  // TODO(b/65151735): Remove when cc_flags is entirely from features.
  @Override
  public String legacyCcFlagsMakeVariable(Info ccToolchainInfo, StarlarkThread thread)
      throws EvalException {
    isCalledFromStarlarkCcCommon(thread);
    CcToolchainProvider ccToolchain = CcToolchainProvider.wrapOrThrowEvalException(ccToolchainInfo);
    return ccToolchain.getLegacyCcFlagsMakeVariable();
  }

  @Nullable
  static <T> T nullIfNone(Object object, Class<T> type) {
    return object != Starlark.NONE ? type.cast(object) : null;
  }

  public static void checkPrivateStarlarkificationAllowlist(StarlarkThread thread)
      throws EvalException {
    BuiltinRestriction.failIfCalledOutsideDefaultAllowlist(thread);
  }

  public static boolean isStarlarkCcCommonCalledFromBuiltins(StarlarkThread thread) {
    Label label =
        ((BazelModuleContext)
                Module.ofInnermostEnclosingStarlarkFunction(thread, 1).getClientData())
            .label();
    return label.getPackageIdentifier().getRepository().getName().equals("_builtins");
  }

  protected static void isCalledFromStarlarkCcCommon(StarlarkThread thread) throws EvalException {
    Label label = BazelModuleContext.ofInnermostBzlOrThrow(thread).label();
    // Allow direct access to cc_common.bzl and to C++ linking code that can't use cc_common.bzl
    // directly without creating a cycle.
    if (!label.getCanonicalForm().endsWith("_builtins//:common/cc/cc_common.bzl")
        && !label.getCanonicalForm().endsWith("_builtins//:common/cc/cc_common_bazel.bzl")
        && !label.getCanonicalForm().contains("_builtins//:common/cc/compile")
        && !label.getCanonicalForm().contains("_builtins//:common/cc/link")
        && !label.getCanonicalForm().contains("_builtins//:common/cc/toolchain_config")) {
      throw Starlark.errorf(
          "cc_common_internal can only be used by cc_common.bzl in builtins, "
              + "please use cc_common instead.");
    }
  }
}
