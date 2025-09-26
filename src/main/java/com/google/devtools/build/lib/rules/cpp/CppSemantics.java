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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.cpp.CcCommon.Language;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkValue;

/** Pluggable C++ compilation semantics. */
public interface CppSemantics extends StarlarkValue {

  /** Returns cpp toolchain type. */
  Label getCppToolchainType();

  /** What language to treat the headers. */
  Language language();

  /**
   * Called before a C++ compile action is built.
   *
   * <p>Gives the semantics implementation the opportunity to change compile actions at the last
   * minute.
   */
  void finalizeCompileActionBuilder(
      BuildConfigurationValue configuration,
      FeatureConfiguration featureConfiguration,
      CppCompileActionBuilder actionBuilder)
      throws EvalException;

  /**
   * Returns if include scanning is allowed.
   *
   * <p>If false, {@link CppCompileActionBuilder#setShouldScanIncludes(boolean)} has no effect.
   */
  boolean allowIncludeScanning();

  /** Returns true iff this build should perform .d input pruning. */
  @StarlarkMethod(
      name = "needs_dotd_input_pruning",
      documented = false,
      parameters = {@Param(name = "config")})
  boolean needsDotdInputPruning(BuildConfigurationValue configuration);

  /** Returns true iff this build requires include validation. */
  boolean needsIncludeValidation();

  /** No-op in Bazel */
  @StarlarkMethod(
      name = "validate_layering_check_features",
      documented = false,
      parameters = {
        @Param(name = "ctx", named = true),
        @Param(name = "cc_toolchain", named = true),
        @Param(name = "unsupported_features", named = true),
      })
  default void validateLayeringCheckFeaturesForStarlark(
      StarlarkRuleContext ruleContext, Info ccToolchainInfo, Sequence<?> unsupportedFeatures)
      throws EvalException {
    try {
      validateLayeringCheckFeatures(
          ruleContext.getRuleContext(),
          ruleContext.getAspectDescriptor(),
          CcToolchainProvider.wrap(ccToolchainInfo),
          ImmutableSet.copyOf(
              Sequence.cast(unsupportedFeatures, String.class, "unsupported_features")));
    } catch (RuleErrorException e) {
      throw new EvalException(e);
    }
  }

  void validateLayeringCheckFeatures(
      RuleContext ruleContext,
      AspectDescriptor aspectDescriptor,
      CcToolchainProvider ccToolchain,
      ImmutableSet<String> unsupportedFeatures)
      throws EvalException;
}
