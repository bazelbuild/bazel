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

package com.google.devtools.build.lib.rules.objc;

import static com.google.devtools.build.lib.rules.objc.ObjcProvider.STORYBOARD;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.XCDATAMODEL;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.objc.CompilationSupport.ExtraLinkArgs;
import com.google.devtools.build.lib.rules.objc.ReleaseBundlingSupport.LinkedBinary;
import com.google.devtools.build.lib.rules.test.ExecutionInfoProvider;
import com.google.devtools.build.lib.rules.test.InstrumentedFilesProvider;
import com.google.devtools.build.lib.syntax.Type;

import java.util.ArrayList;
import java.util.List;

/**
 * Implementation for {@code ios_test} rule in Bazel.
 */
public final class IosTest implements RuleConfiguredTargetFactory {
  
  private static final ImmutableList<SdkFramework> AUTOMATIC_SDK_FRAMEWORKS_FOR_XCTEST =
      ImmutableList.of(new SdkFramework("XCTest"));

  public static final String TARGET_DEVICE = "target_device";
  public static final String IS_XCTEST = "xctest";
  public static final String XCTEST_APP = "xctest_app";
  public static final String MEMLEAKS_DEP = "$memleaks_dep";
  public static final String MEMLEAKS_PLUGIN = "$memleaks_plugin";

  @VisibleForTesting
  public static final String REQUIRES_SOURCE_ERROR =
      "ios_test requires at least one source file in srcs or non_arc_srcs";
  @VisibleForTesting
  public static final String NO_MULTI_CPUS_ERROR =
      "ios_test cannot be built for multiple CPUs at the same time";

  /**
   * {@inheritDoc}
   * 
   * Creates a target, including registering actions, just as {@link #create(RuleContext)} does.
   * The difference between {@link #create(RuleContext)} and this method is that this method does
   * only what is needed to support tests on the environment besides generate the Xcodeproj file
   * and build the app and test {@code .ipa}s. The {@link #create(RuleContext)} method delegates
   * to this method.
   */  
  @Override
  public final ConfiguredTarget create(RuleContext ruleContext) throws InterruptedException {
    ObjcCommon common = common(ruleContext);
  
    if (!common.getCompilationArtifacts().get().getArchive().isPresent()) {
      ruleContext.ruleError(REQUIRES_SOURCE_ERROR);
    }
  
    if (!ruleContext.getFragment(AppleConfiguration.class).getIosMultiCpus().isEmpty()) {
      ruleContext.ruleError(NO_MULTI_CPUS_ERROR);
    }
  
    XcodeProvider.Builder xcodeProviderBuilder = new XcodeProvider.Builder();
    NestedSetBuilder<Artifact> filesToBuild = NestedSetBuilder.stableOrder();
    addResourceFilesToBuild(ruleContext, common.getObjcProvider(), filesToBuild);
  
    XcodeProductType productType = getProductType(ruleContext);
    ExtraLinkArgs extraLinkArgs;
    Iterable<Artifact> extraLinkInputs;
    String bundleFormat;
    if (!isXcTest(ruleContext)) {
      extraLinkArgs = new ExtraLinkArgs();
      extraLinkInputs = ImmutableList.of();
      bundleFormat = ReleaseBundlingSupport.APP_BUNDLE_DIR_FORMAT;
    } else {
      XcodeProvider appIpaXcodeProvider =
          ruleContext.getPrerequisite(XCTEST_APP, Mode.TARGET, XcodeProvider.class);
      xcodeProviderBuilder
          .setTestHost(appIpaXcodeProvider)
          .setProductType(productType);
  
      XcTestAppProvider testApp = xcTestAppProvider(ruleContext);
      Artifact bundleLoader = testApp.getBundleLoader();
  
      // -bundle causes this binary to be linked as a bundle and not require an entry point
      // (i.e. main())
      // -bundle_loader causes the code in this test to have access to the symbols in the test rig,
      // or more specifically, the flag causes ld to consider the given binary when checking for
      // missing symbols.
      extraLinkArgs = new ExtraLinkArgs(
          "-bundle",
          "-bundle_loader", bundleLoader.getExecPathString());
  
      extraLinkInputs = ImmutableList.of(bundleLoader);
      bundleFormat = ReleaseBundlingSupport.XCTEST_BUNDLE_DIR_FORMAT;

      filesToBuild.add(testApp.getIpa());
    }
  
    new CompilationSupport(ruleContext)
        .registerLinkActions(common.getObjcProvider(), extraLinkArgs, extraLinkInputs)
        .registerCompileAndArchiveActions(common)
        .addXcodeSettings(xcodeProviderBuilder, common)
        .validateAttributes();
    
    ObjcConfiguration objcConfiguration = ObjcRuleClasses.objcConfiguration(ruleContext);
    new ReleaseBundlingSupport(
            ruleContext,
            common.getObjcProvider(),
            LinkedBinary.LOCAL_AND_DEPENDENCIES,
            bundleFormat,
            objcConfiguration.getMinimumOs())
        .registerActions()
        .addXcodeSettings(xcodeProviderBuilder)
        .addFilesToBuild(filesToBuild)
        .validateResources()
        .validateAttributes();

    new ResourceSupport(ruleContext).validateAttributes().addXcodeSettings(xcodeProviderBuilder);

    new XcodeSupport(ruleContext)
        .addXcodeSettings(xcodeProviderBuilder, common.getObjcProvider(), productType)
        .addDependencies(xcodeProviderBuilder, new Attribute("bundles", Mode.TARGET))
        .addDependencies(xcodeProviderBuilder, new Attribute("deps", Mode.TARGET))
        .addNonPropagatedDependencies(
            xcodeProviderBuilder, new Attribute("non_propagated_deps", Mode.TARGET))
        .addFilesToBuild(filesToBuild)
        .registerActions(xcodeProviderBuilder.build());

    XcodeProvider xcodeProvider = xcodeProviderBuilder.build();
    NestedSet<Artifact> filesToBuildSet = filesToBuild.build();

    Runfiles.Builder runfilesBuilder =
        new Runfiles.Builder(ruleContext.getWorkspaceName())
            .addRunfiles(ruleContext, RunfilesProvider.DEFAULT_RUNFILES);
    NestedSetBuilder<Artifact> filesToBuildBuilder =
        NestedSetBuilder.<Artifact>stableOrder().addTransitive(filesToBuildSet);

    TestSupport testSupport =
        new TestSupport(ruleContext)
            .registerTestRunnerActions()
            .addRunfiles(runfilesBuilder)
            .addFilesToBuild(filesToBuildBuilder);

    Artifact executable = testSupport.generatedTestScript();

    Runfiles runfiles = runfilesBuilder.build();
    RunfilesSupport runfilesSupport =
        RunfilesSupport.withExecutable(ruleContext, runfiles, executable);

    return new RuleConfiguredTargetBuilder(ruleContext)
        .setFilesToBuild(filesToBuildBuilder.build())
        .add(XcodeProvider.class, xcodeProvider)
        .add(RunfilesProvider.class, RunfilesProvider.simple(runfiles))
        .add(
            ExecutionInfoProvider.class,
            new ExecutionInfoProvider(ImmutableMap.of(ExecutionRequirements.REQUIRES_DARWIN, "")))
        .addProvider(
            InstrumentedFilesProvider.class,
            new CompilationSupport(ruleContext).getInstrumentedFilesProvider(common))
        .addProviders(testSupport.getExtraProviders())
        .setRunfilesSupport(runfilesSupport, executable)
        .build();
  }

  private XcodeProductType getProductType(RuleContext ruleContext) {
    if (isXcTest(ruleContext)) {
      return XcodeProductType.UNIT_TEST;
    } else {
      return XcodeProductType.APPLICATION;
    }
  }
    
  private void addResourceFilesToBuild(
      RuleContext ruleContext, ObjcProvider objcProvider, NestedSetBuilder<Artifact> filesToBuild) {
    IntermediateArtifacts intermediateArtifacts =
        ObjcRuleClasses.intermediateArtifacts(ruleContext);
  
    Iterable<Xcdatamodel> xcdatamodels =
        Xcdatamodels.xcdatamodels(intermediateArtifacts, objcProvider.get(XCDATAMODEL));
    filesToBuild.addAll(Xcdatamodel.outputZips(xcdatamodels));
  
    for (Artifact storyboard : objcProvider.get(STORYBOARD)) {
      filesToBuild.add(intermediateArtifacts.compiledStoryboardZip(storyboard));
    }
  }

  private ObjcCommon common(RuleContext ruleContext) {
    ImmutableList<SdkFramework> extraSdkFrameworks = isXcTest(ruleContext)
        ? AUTOMATIC_SDK_FRAMEWORKS_FOR_XCTEST : ImmutableList.<SdkFramework>of();
    List<ObjcProvider> extraDepObjcProviders = new ArrayList<>();
    if (isXcTest(ruleContext)) {
      extraDepObjcProviders.add(xcTestAppProvider(ruleContext).getObjcProvider());
    }
  
    // Add the memleaks library if the --ios_memleaks flag is true.  The library pauses the test
    // after all tests have been executed so that leaks can be run.
    ObjcConfiguration config = ruleContext.getFragment(ObjcConfiguration.class);
    if (config.runMemleaks()) {
      extraDepObjcProviders
          .add(ruleContext.getPrerequisite(MEMLEAKS_DEP, Mode.TARGET, ObjcProvider.class));
    }
    return ObjcLibrary.common(ruleContext, extraSdkFrameworks, /*alwayslink=*/false,
        extraDepObjcProviders);
  }
  
  protected static boolean isXcTest(RuleContext ruleContext) {
    return ruleContext.attributes().get(IS_XCTEST, Type.BOOLEAN);
  }

  /** Returns the {@link XcTestAppProvider} of the {@code xctest_app} attribute. */
  protected static XcTestAppProvider xcTestAppProvider(RuleContext ruleContext) {
    return ruleContext.getPrerequisite(XCTEST_APP, Mode.TARGET, XcTestAppProvider.class);
  }

}
