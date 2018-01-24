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
import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.test.ExecutionInfo;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.ApplePlatform.PlatformType;
import com.google.devtools.build.lib.rules.apple.XcodeConfig;
import com.google.devtools.build.lib.rules.objc.CompilationSupport.ExtraLinkArgs;
import com.google.devtools.build.lib.rules.objc.ObjcCommon.ResourceAttributes;
import com.google.devtools.build.lib.rules.objc.ReleaseBundlingSupport.LinkedBinary;
import com.google.devtools.build.lib.rules.proto.ProtoSourcesProvider;

/** Implementation for {@code ios_test} rule in Bazel. */
public final class IosTest implements RuleConfiguredTargetFactory {

  private static final ImmutableList<SdkFramework> AUTOMATIC_SDK_FRAMEWORKS_FOR_XCTEST =
      ImmutableList.of(new SdkFramework("XCTest"));

  // Attributes for IosTest rules.
  // Documentation on usage is in {@link IosTestRule@}.
  static final String OBJC_GCOV_ATTR = "$objc_gcov";
  static final String DEVICE_ARG_ATTR = "ios_device_arg";
  static final String IS_XCTEST_ATTR = "xctest";
  static final String MEMLEAKS_DEP_ATTR = "$memleaks_dep";
  static final String MEMLEAKS_PLUGIN_ATTR = "$memleaks_plugin";
  static final String PLUGINS_ATTR = "plugins";
  static final String TARGET_DEVICE = "target_device";
  static final String TEST_RUNNER_ATTR = "$test_runner";
  static final String TEST_TARGET_DEVICE_ATTR = "ios_test_target_device";
  static final String TEST_TEMPLATE_ATTR = "$test_template";
  static final String XCTEST_APP_ATTR = "xctest_app";
  static final String MCOV_TOOL_ATTR = ":mcov";

  @VisibleForTesting
  public static final String REQUIRES_SOURCE_ERROR =
      "ios_test requires at least one source file in srcs or non_arc_srcs";

  @VisibleForTesting
  public static final String NO_MULTI_CPUS_ERROR =
      "ios_test cannot be built for multiple CPUs at the same time";

  /**
   * {@inheritDoc}
   *
   * <p>Creates a target, including registering actions, just as {@link #create(RuleContext)} does.
   * The difference between {@link #create(RuleContext)} and this method is that this method does
   * only what is needed to support tests on the environment besides build the app and test {@code
   * .ipa}s. The {@link #create(RuleContext)} method delegates to this method.
   */
  @Override
  public final ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
    ruleContext.ruleWarning(
        "This rule is deprecated. Please use the new Apple build rules "
            + "(https://github.com/bazelbuild/rules_apple) to build Apple targets.");

    Iterable<ObjcProtoProvider> objcProtoProviders =
        ruleContext.getPrerequisites("deps", Mode.TARGET, ObjcProtoProvider.SKYLARK_CONSTRUCTOR);

    ProtobufSupport protoSupport =
        new ProtobufSupport(
                ruleContext,
                ruleContext.getConfiguration(),
                ImmutableList.<ProtoSourcesProvider>of(),
                objcProtoProviders,
                ProtobufSupport.getTransitivePortableProtoFilters(objcProtoProviders))
            .registerGenerationActions()
            .registerCompilationActions();
    Optional<ObjcProvider> protosObjcProvider = protoSupport.getObjcProvider();

    ObjcCommon common = common(ruleContext, protosObjcProvider);

    if (!common.getCompilationArtifacts().get().getArchive().isPresent()) {
      ruleContext.ruleError(REQUIRES_SOURCE_ERROR);
    }

    if (!ruleContext.getFragment(AppleConfiguration.class).getIosMultiCpus().isEmpty()) {
      ruleContext.ruleError(NO_MULTI_CPUS_ERROR);
    }

    NestedSetBuilder<Artifact> filesToBuild = NestedSetBuilder.stableOrder();
    addResourceFilesToBuild(ruleContext, common.getObjcProvider(), filesToBuild);

    ExtraLinkArgs extraLinkArgs;
    Iterable<Artifact> extraLinkInputs;
    String bundleFormat;
    XcTestAppProvider testApp = xcTestAppProvider(ruleContext);
    Artifact bundleLoader = testApp.getBundleLoader();

    // -bundle causes this binary to be linked as a bundle and not require an entry point
    // (i.e. main())
    // -bundle_loader causes the code in this test to have access to the symbols in the test rig,
    // or more specifically, the flag causes ld to consider the given binary when checking for
    // missing symbols.
    // -rpath @loader_path/Frameworks allows test bundles to load dylibs from the app's
    // Frameworks directory.
    extraLinkArgs =
        new ExtraLinkArgs(
            "-bundle",
            "-bundle_loader",
            bundleLoader.getExecPathString(),
            "-Xlinker",
            "-rpath",
            "-Xlinker",
            "@loader_path/Frameworks");

    extraLinkInputs = ImmutableList.of(bundleLoader);
    bundleFormat = ReleaseBundlingSupport.XCTEST_BUNDLE_DIR_FORMAT;

    filesToBuild.add(testApp.getIpa());

    J2ObjcMappingFileProvider j2ObjcMappingFileProvider =
        J2ObjcMappingFileProvider.union(
            ruleContext.getPrerequisites("deps", Mode.TARGET, J2ObjcMappingFileProvider.class));
    J2ObjcEntryClassProvider j2ObjcEntryClassProvider =
        new J2ObjcEntryClassProvider.Builder()
            .addTransitive(
                ruleContext.getPrerequisites("deps", Mode.TARGET, J2ObjcEntryClassProvider.class))
            .build();

    CompilationSupport compilationSupport =
        new CompilationSupport.Builder().setRuleContext(ruleContext).setIsTestRule().build();

    compilationSupport
        .registerLinkActions(
            common.getObjcProvider(),
            j2ObjcMappingFileProvider,
            j2ObjcEntryClassProvider,
            extraLinkArgs,
            extraLinkInputs,
            DsymOutputType.TEST)
        .registerCompileAndArchiveActions(common)
        .registerFullyLinkAction(
            common.getObjcProvider(),
            ruleContext.getImplicitOutputArtifact(CompilationSupport.FULLY_LINKED_LIB))
        .validateAttributes();

    AppleConfiguration appleConfiguration = ruleContext.getFragment(AppleConfiguration.class);
    new ReleaseBundlingSupport(
            ruleContext,
            common.getObjcProvider(),
            LinkedBinary.LOCAL_AND_DEPENDENCIES,
            bundleFormat,
            XcodeConfig.getMinimumOsForPlatformType(ruleContext, PlatformType.IOS),
            appleConfiguration.getMultiArchPlatform(PlatformType.IOS))
        .registerActions(DsymOutputType.TEST)
        .addFilesToBuild(filesToBuild, Optional.of(DsymOutputType.TEST))
        .validateResources()
        .validateAttributes();

    new ResourceSupport(ruleContext).validateAttributes();

    NestedSet<Artifact> filesToBuildSet = filesToBuild.build();

    Runfiles.Builder runfilesBuilder =
        new Runfiles.Builder(
                ruleContext.getWorkspaceName(),
                ruleContext.getConfiguration().legacyExternalRunfiles())
            .addRunfiles(ruleContext, RunfilesProvider.DEFAULT_RUNFILES);

    NestedSetBuilder<Artifact> filesToBuildBuilder =
        NestedSetBuilder.<Artifact>stableOrder().addTransitive(filesToBuildSet);

    InstrumentedFilesProvider instrumentedFilesProvider =
        new CompilationSupport.Builder()
            .setRuleContext(ruleContext)
            .build()
            .getInstrumentedFilesProvider(common);

    TestSupport testSupport =
        new TestSupport(ruleContext)
            .registerTestRunnerActions()
            .addRunfiles(runfilesBuilder, instrumentedFilesProvider)
            .addFilesToBuild(filesToBuildBuilder);

    Artifact executable = testSupport.generatedTestScript();

    Runfiles runfiles = runfilesBuilder.build();
    RunfilesSupport runfilesSupport =
        RunfilesSupport.withExecutable(ruleContext, runfiles, executable);

    ImmutableMap.Builder<String, String> execInfoMapBuilder = new ImmutableMap.Builder<>();
    execInfoMapBuilder.put(ExecutionRequirements.REQUIRES_DARWIN, "");
    if (ruleContext.getFragment(ObjcConfiguration.class).runMemleaks()) {
      execInfoMapBuilder.put("nosandbox", "");
    }

    return new RuleConfiguredTargetBuilder(ruleContext)
        .setFilesToBuild(filesToBuildBuilder.build())
        .addProvider(RunfilesProvider.simple(runfiles))
        .addNativeDeclaredProvider(new ExecutionInfo(execInfoMapBuilder.build()))
        .addNativeDeclaredProviders(testSupport.getExtraProviders())
        .addProvider(InstrumentedFilesProvider.class, instrumentedFilesProvider)
        .setRunfilesSupport(runfilesSupport, executable)
        .build();
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

  /** Constructs an {@link ObjcCommon} instance based on the attributes. */
  private ObjcCommon common(RuleContext ruleContext, Optional<ObjcProvider> protosObjcProvider) {
    CompilationArtifacts compilationArtifacts =
        CompilationSupport.compilationArtifacts(ruleContext);

    ObjcCommon.Builder builder =
        new ObjcCommon.Builder(ruleContext)
            .setCompilationAttributes(
                CompilationAttributes.Builder.fromRuleContext(ruleContext).build())
            .setCompilationArtifacts(compilationArtifacts)
            .setResourceAttributes(new ResourceAttributes(ruleContext))
            .addDefines(ruleContext.getExpander().withDataLocations().tokenized("defines"))
            .addDeps(ruleContext.getPrerequisites("deps", Mode.TARGET))
            .addRuntimeDeps(ruleContext.getPrerequisites("runtime_deps", Mode.TARGET))
            .addDeps(ruleContext.getPrerequisites("bundles", Mode.TARGET))
            .addDepObjcProviders(protosObjcProvider.asSet())
            .addNonPropagatedDepObjcProviders(
                ruleContext.getPrerequisites(
                    "non_propagated_deps", Mode.TARGET, ObjcProvider.SKYLARK_CONSTRUCTOR))
            .setIntermediateArtifacts(ObjcRuleClasses.intermediateArtifacts(ruleContext))
            .setHasModuleMap();

    builder
        .addExtraSdkFrameworks(AUTOMATIC_SDK_FRAMEWORKS_FOR_XCTEST)
        .addDepObjcProviders(ImmutableList.of(xcTestAppProvider(ruleContext).getObjcProvider()));

    // Add the memleaks library if the --ios_memleaks flag is true.  The library pauses the test
    // after all tests have been executed so that leaks can be run.
    ObjcConfiguration config = ruleContext.getFragment(ObjcConfiguration.class);
    if (config.runMemleaks()) {
      builder.addDepObjcProviders(
          ruleContext.getPrerequisites(
              MEMLEAKS_DEP_ATTR, Mode.TARGET, ObjcProvider.SKYLARK_CONSTRUCTOR));
    }

    return builder.build();
  }

  /** Returns the {@link XcTestAppProvider} of the {@code xctest_app} attribute. */
  protected static XcTestAppProvider xcTestAppProvider(RuleContext ruleContext) {
    return ruleContext.getPrerequisite(
        XCTEST_APP_ATTR, Mode.TARGET, XcTestAppProvider.SKYLARK_CONSTRUCTOR);
  }
}
