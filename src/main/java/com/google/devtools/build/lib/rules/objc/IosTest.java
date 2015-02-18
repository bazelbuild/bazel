// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.objc.ApplicationSupport.LinkedBinary;
import com.google.devtools.build.lib.rules.objc.ObjcActionsBuilder.ExtraLinkArgs;
import com.google.devtools.build.lib.rules.objc.ObjcActionsBuilder.ExtraLinkInputs;

import java.util.ArrayList;
import java.util.List;

/**
 * Contains information needed to create a {@link RuleConfiguredTarget} and invoke test runners
 * for some instantiation of this rule.
 */
// TODO(bazel-team): Extract a TestSupport class that takes on most of the logic in this class.
public abstract class IosTest implements RuleConfiguredTargetFactory {
  private static final ImmutableList<SdkFramework> AUTOMATIC_SDK_FRAMEWORKS_FOR_XCTEST =
      ImmutableList.of(new SdkFramework("XCTest"));

  public static final String TARGET_DEVICE = "target_device";
  public static final String IS_XCTEST = "xctest";
  public static final String XCTEST_APP = "xctest_app";

  @VisibleForTesting
  public static final String REQUIRES_SOURCE_ERROR =
      "ios_test requires at least one source file in srcs or non_arc_srcs";

  /**
   * Creates a target, including registering actions, just as {@link #create(RuleContext)} does.
   * The difference between {@link #create(RuleContext)} and this method is that this method does
   * only what is needed to support tests on the environment besides generate the Xcodeproj file
   * and build the app and test {@code .ipa}s. The {@link #create(RuleContext)} method delegates
   * to this method.
   */
  protected abstract ConfiguredTarget create(RuleContext ruleContext, ObjcCommon common,
      XcodeProvider xcodeProvider, NestedSet<Artifact> filesToBuild) throws InterruptedException;

  @Override
  public final ConfiguredTarget create(RuleContext ruleContext) throws InterruptedException {
    ObjcCommon common = common(ruleContext);
    OptionsProvider optionsProvider = optionsProvider(ruleContext);

    if (!common.getCompilationArtifacts().get().getArchive().isPresent()) {
      ruleContext.ruleError(REQUIRES_SOURCE_ERROR);
    }

    XcodeProvider.Builder xcodeProviderBuilder = new XcodeProvider.Builder();
    NestedSetBuilder<Artifact> filesToBuild = NestedSetBuilder.<Artifact>stableOrder()
        .addTransitive(common.getStoryboards().getOutputZips())
        .addAll(Xcdatamodel.outputZips(common.getDatamodels()));

    XcodeProductType productType;
    ExtraLinkArgs extraLinkArgs;
    ExtraLinkInputs extraLinkInputs;
    if (!isXcTest(ruleContext)) {
      productType = XcodeProductType.APPLICATION;
      extraLinkArgs = new ExtraLinkArgs();
      extraLinkInputs = new ExtraLinkInputs();
    } else {
      productType = XcodeProductType.UNIT_TEST;
      XcodeProvider appIpaXcodeProvider =
          ruleContext.getPrerequisite(XCTEST_APP, Mode.TARGET, XcodeProvider.class);
      xcodeProviderBuilder
          .setTestHost(appIpaXcodeProvider)
          .setProductType(productType);

      Artifact bundleLoader = xcTestAppProvider(ruleContext).getBundleLoader();

      // -bundle causes this binary to be linked as a bundle and not require an entry point
      // (i.e. main())
      // -bundle_loader causes the code in this test to have access to the symbols in the test rig,
      // or more specifically, the flag causes ld to consider the given binary when checking for
      // missing symbols.
      extraLinkArgs = new ExtraLinkArgs(
          "-bundle",
          "-bundle_loader", bundleLoader.getExecPathString());

      extraLinkInputs = new ExtraLinkInputs(bundleLoader);
    }

    if (ruleContext.getConfiguration().isCodeCoverageEnabled()) {
      extraLinkArgs = extraLinkArgs.appendedWith(CompilationSupport.LINKER_COVERAGE_FLAGS);
    }

    new CompilationSupport(ruleContext)
        .registerLinkActions(
            common.getObjcProvider(), extraLinkArgs, extraLinkInputs)
        .registerJ2ObjcCompileAndArchiveActions(optionsProvider, common.getObjcProvider())
        .registerCompileAndArchiveActions(common, optionsProvider)
        .addXcodeSettings(xcodeProviderBuilder, common, optionsProvider)
        .validateAttributes();

    new ApplicationSupport(
        ruleContext, common.getObjcProvider(), optionsProvider, LinkedBinary.LOCAL_AND_DEPENDENCIES)
        .registerActions()
        .addXcodeSettings(xcodeProviderBuilder)
        .addFilesToBuild(filesToBuild)
        .validateAttributes();

    new ResourceSupport(ruleContext)
        .registerActions(common.getStoryboards())
        .validateAttributes()
        .addXcodeSettings(xcodeProviderBuilder);

    new XcodeSupport(ruleContext)
        .addXcodeSettings(xcodeProviderBuilder, common.getObjcProvider(), productType)
        .addDependencies(xcodeProviderBuilder)
        .addFilesToBuild(filesToBuild)
        .registerActions(xcodeProviderBuilder.build());

    return create(ruleContext, common, xcodeProviderBuilder.build(), filesToBuild.build());
  }

  protected static boolean isXcTest(RuleContext ruleContext) {
    return ruleContext.attributes().get(IS_XCTEST, Type.BOOLEAN);
  }

  private OptionsProvider optionsProvider(RuleContext ruleContext) {
    return new OptionsProvider.Builder()
        .addCopts(ruleContext.getTokenizedStringListAttr("copts"))
        .addInfoplists(ruleContext.getPrerequisiteArtifacts("infoplist", Mode.TARGET).list())
        .addTransitive(Optional.fromNullable(
            ruleContext.getPrerequisite("options", Mode.TARGET, OptionsProvider.class)))
        .build();
  }

  /** Returns the {@link XcTestAppProvider} of the {@code xctest_app} attribute. */
  private static XcTestAppProvider xcTestAppProvider(RuleContext ruleContext) {
    return ruleContext.getPrerequisite(XCTEST_APP, Mode.TARGET, XcTestAppProvider.class);
  }

  private ObjcCommon common(RuleContext ruleContext) {
    ImmutableList<SdkFramework> extraSdkFrameworks = isXcTest(ruleContext)
        ? AUTOMATIC_SDK_FRAMEWORKS_FOR_XCTEST : ImmutableList.<SdkFramework>of();
    List<ObjcProvider> extraDepObjcProviders = new ArrayList<>();
    if (isXcTest(ruleContext)) {
      extraDepObjcProviders.add(xcTestAppProvider(ruleContext).getObjcProvider());
    }

    return ObjcLibrary.common(ruleContext, extraSdkFrameworks, /*alwayslink=*/false,
        new ObjcLibrary.ExtraImportLibraries(), new ObjcLibrary.Defines(), extraDepObjcProviders);
  }
}
