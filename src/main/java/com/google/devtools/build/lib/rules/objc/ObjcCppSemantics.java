// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RuleErrorConsumer;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.cpp.CppCompileActionBuilder;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.HeadersCheckingMode;
import com.google.devtools.build.lib.rules.cpp.CppSemantics;
import com.google.devtools.build.lib.rules.cpp.HeaderDiscovery.DotdPruningMode;

/**
 * CppSemantics for objc builds.
 */
public class ObjcCppSemantics implements CppSemantics {
  public static final ObjcCppSemantics MODULES = new ObjcCppSemantics(true);
  public static final ObjcCppSemantics NO_MODULES = new ObjcCppSemantics(false);

  private final boolean enableModules;

  /**
   * Creates an instance of ObjcCppSemantics
   *
   * @param enableModules whether modules are enabled
   */
  public ObjcCppSemantics(boolean enableModules) {
    this.enableModules = enableModules;
  }

  @Override
  public void finalizeCompileActionBuilder(
      BuildConfiguration configuration,
      FeatureConfiguration featureConfiguration,
      CppCompileActionBuilder actionBuilder,
      RuleErrorConsumer ruleErrorConsumer) {
    actionBuilder
        // Without include scanning, we need the entire crosstool filegroup, including header files,
        // as opposed to just the "compile" filegroup.  Even with include scanning, we need the
        // system framework headers since they are not being scanned right now.
        // TODO(waltl): do better with include scanning.
        .addTransitiveMandatoryInputs(actionBuilder.getToolchain().getAllFilesMiddleman())
        .setShouldScanIncludes(
            configuration.getFragment(ObjcConfiguration.class).shouldScanIncludes());
  }

  @Override
  public HeadersCheckingMode determineHeadersCheckingMode(RuleContext ruleContext) {
    return HeadersCheckingMode.STRICT;
  }

  @Override
  public HeadersCheckingMode determineStarlarkHeadersCheckingMode(
      RuleContext context, CppConfiguration cppConfig, CcToolchainProvider toolchain) {
    if (cppConfig.strictHeaderCheckingFromStarlark()) {
      return HeadersCheckingMode.STRICT;
    }
    return HeadersCheckingMode.LOOSE;
  }

  @Override
  public boolean allowIncludeScanning() {
    return true;
  }

  @Override
  public boolean needsDotdInputPruning(BuildConfiguration configuration) {
    return configuration.getFragment(ObjcConfiguration.class).getDotdPruningPlan()
        == DotdPruningMode.USE;
  }

  @Override
  public void validateAttributes(RuleContext ruleContext) {
  }

  @Override
  public boolean needsIncludeValidation() {
    // We disable include validation when modules are enabled, because Apple uses absolute paths in
    // its module maps, which include validation does not recognize.  Modules should only be used
    // rarely and in third party code anyways.
    return !enableModules;
  }

  /** cc_shared_library is not supported with Objective-C */
  @Override
  public StructImpl getCcSharedLibraryInfo(TransitiveInfoCollection dep) {
    return null;
  }

  @Override
  public void validateLayeringCheckFeatures(
      RuleContext ruleContext,
      AspectDescriptor aspectDescriptor,
      CcToolchainProvider ccToolchain,
      ImmutableSet<String> unsupportedFeatures) {}
}
