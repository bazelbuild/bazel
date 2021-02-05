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

package com.google.devtools.build.lib.bazel.rules.cpp;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RuleErrorConsumer;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.rules.cpp.AspectLegalCppSemantics;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.cpp.CppActionNames;
import com.google.devtools.build.lib.rules.cpp.CppCompileActionBuilder;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.HeadersCheckingMode;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;

/** C++ compilation semantics. */
public class BazelCppSemantics implements AspectLegalCppSemantics {
  @AutoCodec public static final BazelCppSemantics INSTANCE = new BazelCppSemantics();

  // TODO(#10338): We need to check for both providers. With and without the @rules_cc repo name.
  //  The reason for that is that when we are in a target inside @rules_cc, the provider won't have
  // the repo name set.
  public static final Provider.Key CC_SHARED_INFO_PROVIDER_RULES_CC =
      new StarlarkProvider.Key(
          Label.parseAbsoluteUnchecked("@rules_cc//examples:experimental_cc_shared_library.bzl"),
          "CcSharedLibraryInfo");

  public static final Provider.Key CC_SHARED_INFO_PROVIDER =
      new StarlarkProvider.Key(
          Label.parseAbsoluteUnchecked("//examples:experimental_cc_shared_library.bzl"),
          "CcSharedLibraryInfo");

  private BazelCppSemantics() {
  }

  @Override
  public void finalizeCompileActionBuilder(
      BuildConfiguration configuration,
      FeatureConfiguration featureConfiguration,
      CppCompileActionBuilder actionBuilder,
      RuleErrorConsumer ruleErrorConsumer) {
    CcToolchainProvider toolchain = actionBuilder.getToolchain();
    actionBuilder
        .addTransitiveMandatoryInputs(
            configuration.getFragment(CppConfiguration.class).useSpecificToolFiles()
                    && !actionBuilder.getSourceFile().isTreeArtifact()
                ? (actionBuilder.getActionName().equals(CppActionNames.ASSEMBLE)
                    ? toolchain.getAsFiles()
                    : toolchain.getCompilerFiles())
                : toolchain.getAllFiles())
        .setShouldScanIncludes(false);
  }

  @Override
  public HeadersCheckingMode determineHeadersCheckingMode(RuleContext ruleContext) {
    return HeadersCheckingMode.STRICT;
  }

  @Override
  public HeadersCheckingMode determineStarlarkHeadersCheckingMode(
      RuleContext ruleContext, CppConfiguration cppConfig, CcToolchainProvider toolchain) {
    if (cppConfig.strictHeaderCheckingFromStarlark()) {
      return HeadersCheckingMode.STRICT;
    }
    return HeadersCheckingMode.LOOSE;
  }

  @Override
  public boolean allowIncludeScanning() {
    return false;
  }

  @Override
  public boolean needsDotdInputPruning(BuildConfiguration configuration) {
    return true;
  }

  @Override
  public void validateAttributes(RuleContext ruleContext) {
  }

  @Override
  public boolean needsIncludeValidation() {
    return true;
  }

  @Override
  public StructImpl getCcSharedLibraryInfo(TransitiveInfoCollection dep) {
    StructImpl ccSharedLibraryInfo = (StructImpl) dep.get(CC_SHARED_INFO_PROVIDER);
    if (ccSharedLibraryInfo != null) {
      return ccSharedLibraryInfo;
    }
    ccSharedLibraryInfo = (StructImpl) dep.get(CC_SHARED_INFO_PROVIDER_RULES_CC);
    if (ccSharedLibraryInfo != null) {
      return ccSharedLibraryInfo;
    }
    return null;
  }

  @Override
  public void validateLayeringCheckFeatures(
      RuleContext ruleContext,
      AspectDescriptor aspectDescriptor,
      CcToolchainProvider ccToolchain,
      ImmutableSet<String> unsupportedFeatures) {}
}
