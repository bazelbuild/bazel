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

package com.google.devtools.build.lib.rules.cpp;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RuleErrorConsumer;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.rules.cpp.CcCommon.Language;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.HeadersCheckingMode;

/**
 * CppSemantics for objc builds.
 */
public class ObjcCppSemantics implements CppSemantics {
  public static final ObjcCppSemantics INSTANCE = new ObjcCppSemantics();

  private ObjcCppSemantics() {}

  @Override
  public Language language() {
    return Language.OBJC;
  }

  @Override
  public void finalizeCompileActionBuilder(
      BuildConfigurationValue configuration,
      FeatureConfiguration featureConfiguration,
      CppCompileActionBuilder actionBuilder,
      RuleErrorConsumer ruleErrorConsumer) {
    Artifact sourceFile = actionBuilder.getSourceFile();
    CcToolchainProvider toolchain = actionBuilder.getToolchain();

    // Two things have to be true to perform #include scanning:
    //  1. The toolchain configuration has to generally enable it.
    //     This is the default unless the --noexperimental_objc_include_scanning flag was specified.
    //  2. The file must not be a not-for-preprocessing assembler source file.
    //     Assembler without C preprocessing can use the '.include' pseudo-op which is not
    //     understood by the include scanner, so we'll disable scanning, and instead require
    //     the declared sources to state (possibly overapproximate) the dependencies.
    //     Assembler with preprocessing can also use '.include', but supporting both kinds
    //     of inclusion for that use-case is ridiculous.
    boolean shouldScanIncludes =
        configuration.getFragment(CppConfiguration.class).objcShouldScanIncludes()
            && !sourceFile.isFileType(CppFileTypes.ASSEMBLER)
            && !sourceFile.isFileType(CppFileTypes.CPP_MODULE);
    if (shouldScanIncludes && actionBuilder.getGrepIncludes() == null) {
      ruleErrorConsumer.ruleError(
          "When include scanning is turned on, the rule must pass grep_includes");
    }

    actionBuilder.setShouldScanIncludes(shouldScanIncludes);

    // If include scanning is enabled, we can use the filegroup without header files - they are
    // found by include scanning.  We still need the system framework headers since they are not
    // being scanned right now, but those are placed in the "compile" file group.
    actionBuilder.addTransitiveMandatoryInputs(
        actionBuilder.getShouldScanIncludes()
            ? actionBuilder.getToolchain().getCompilerFilesWithoutIncludes()
            : configuration.getFragment(CppConfiguration.class).useSpecificToolFiles()
                    && !actionBuilder.getSourceFile().isTreeArtifact()
                ? (actionBuilder.getActionName().equals(CppActionNames.ASSEMBLE)
                    ? toolchain.getAsFiles()
                    : toolchain.getCompilerFiles())
                : toolchain.getAllFiles());
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
  public boolean needsDotdInputPruning(BuildConfigurationValue configuration) {
    return configuration.getFragment(CppConfiguration.class).objcShouldGenerateDotdFiles();
  }

  @Override
  public void validateAttributes(RuleContext ruleContext) {
  }

  @Override
  public boolean needsIncludeValidation() {
    return true;
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

  @Override
  public boolean createEmptyArchive() {
    return false;
  }

  @Override
  public boolean shouldUseInterfaceDepsBehavior(RuleContext ruleContext) {
    return false;
  }
}
