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

package com.google.devtools.build.lib.rules.objc;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.baseArtifactNames;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.getFirstArtifactEndingWith;
import static com.google.devtools.build.lib.rules.objc.LegacyCompilationSupport.AUTOMATIC_SDK_FRAMEWORKS;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.HEADER;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.INCLUDE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.MODULE_MAP;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.STORYBOARD;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.BundlingRule.FAMILIES_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.BundlingRule.INFOPLIST_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.CLANG;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.CLANG_PLUSPLUS;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.DSYMUTIL;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.LIPO;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.ReleaseBundlingRule.APP_ICON_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.ReleaseBundlingRule.BUNDLE_ID_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.ReleaseBundlingRule.ENTITLEMENTS_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.ReleaseBundlingRule.LAUNCH_IMAGE_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.ReleaseBundlingRule.PROVISIONING_PROFILE_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.SRCS_TYPE;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.STRIP;
import static com.google.devtools.build.lib.rules.objc.ReleaseBundlingSupport.NO_ASSET_CATALOG_ERROR_FORMAT;
import static org.junit.Assert.fail;

import com.dd.plist.NSDictionary;
import com.dd.plist.PropertyListParser;
import com.google.common.base.Joiner;
import com.google.common.base.Optional;
import com.google.common.base.Predicate;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.common.collect.Iterables;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Multiset;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandAction;
import com.google.devtools.build.lib.actions.ExecutionInfoSpecifier;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.FailAction;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.OutputGroupProvider;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.analysis.actions.BinaryFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine.Builder;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine.VectorArg;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.actions.ParameterFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction.Substitution;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.analysis.util.ScratchAttributeWriter;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.Attribute.SplitTransition;
import com.google.devtools.build.lib.packages.util.MockJ2ObjcSupport;
import com.google.devtools.build.lib.packages.util.MockObjcSupport;
import com.google.devtools.build.lib.packages.util.MockProtoSupport;
import com.google.devtools.build.lib.rules.apple.AppleCommandLineOptions;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration.ConfigurationDistinguisher;
import com.google.devtools.build.lib.rules.apple.ApplePlatform;
import com.google.devtools.build.lib.rules.apple.ApplePlatform.PlatformType;
import com.google.devtools.build.lib.rules.apple.AppleToolchain;
import com.google.devtools.build.lib.rules.apple.DottedVersion;
import com.google.devtools.build.lib.rules.apple.XcodeVersionProperties;
import com.google.devtools.build.lib.rules.cpp.CppCompileAction;
import com.google.devtools.build.lib.rules.cpp.CppLinkAction;
import com.google.devtools.build.lib.rules.objc.CompilationSupport.ExtraLinkArgs;
import com.google.devtools.build.lib.rules.objc.ObjcCommandLineOptions.ObjcCrosstoolMode;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos.BundleFile;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos.Control;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos.MergeZip;
import com.google.devtools.build.xcode.plmerge.proto.PlMergeProtos;
import java.io.InputStream;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import org.junit.Before;

/**
 * Superclass for all Obj-C rule tests.
 *
 * <p>TODO(matvore): split this up into more helper classes, especially the check... methods, which
 * are many and not shared by all objc_ rules.
 * <p>TODO(matvore): find a more concise way to repeat common tests (in particular, those which
 * simply call a check... method) across several rule types.
 */
public abstract class ObjcRuleTestCase extends BuildViewTestCase {
  protected static final String MOCK_ACTOOLWRAPPER_PATH =
      toolsRepoExecPath("tools/objc/actoolwrapper");
  protected static final String MOCK_IBTOOLWRAPPER_PATH =
      toolsRepoExecPath("tools/objc/ibtoolwrapper");
  protected static final String MOCK_BUNDLEMERGE_PATH = toolsRepoExecPath("tools/objc/bundlemerge");
  protected static final String MOCK_MOMCWRAPPER_PATH = toolsRepoExecPath("tools/objc/momcwrapper");
  protected static final String MOCK_SWIFTSTDLIBTOOLWRAPPER_PATH =
      toolsRepoExecPath("tools/objc/swiftstdlibtoolwrapper");
  protected static final String MOCK_LIBTOOL_PATH = toolsRepoExecPath("tools/objc/libtool");
  protected static final String MOCK_XCRUNWRAPPER_PATH =
      toolsRepoExecPath("tools/objc/xcrunwrapper");
  protected static final ImmutableList<String> FASTBUILD_COPTS =
      ImmutableList.of("-O0", "-DDEBUG=1");

  protected static final DottedVersion DEFAULT_IOS_SDK_VERSION =
      DottedVersion.fromString(AppleCommandLineOptions.DEFAULT_IOS_SDK_VERSION);

  private String artifactPrefix;

  /**
   * Returns the configuration obtained by applying the apple crosstool configuration transtion to
   * this {@code BuildViewTestCase}'s target configuration.
   */
  protected BuildConfiguration getAppleCrosstoolConfiguration() throws InterruptedException {
    return getConfiguration(targetConfig, AppleCrosstoolTransition.APPLE_CROSSTOOL_TRANSITION);
  }

  /** Specification of code coverage behavior. */
  public enum CodeCoverageMode {
    // No code coverage information.
    NONE,
    // Code coverage in gcov format.
    GCOV,
    // Code coverage in llvm-covmap format.
    LLVMCOV;
  }

  /**
   * Returns the bin dir for artifacts built for a given Apple architecture and minimum OS
   * version (as set by a configuration transition) and configuration distinguisher but the global
   * default for {@code --cpu}.
   *
   * @param arch the given Apple architecture which artifacts are built under this configuration.
   *     Note this will likely be different than the value of {@code --cpu}.
   * @param configurationDistinguisher the configuration distinguisher used to describe the
   *     a configuration transition
   * @param minOsVersion the minimum os version for which to compile artifacts in the
   *     configuration
   */
  protected String configurationBin(
      String arch, ConfigurationDistinguisher configurationDistinguisher,
      DottedVersion minOsVersion) {
    return configurationDir(arch, configurationDistinguisher, minOsVersion) + "bin/";
  }

   /**
   * Returns the genfiles dir for artifacts built for a given Apple architecture and minimum OS
   * version (as set by a configuration transition) and configuration distinguisher but the global
   * default for {@code --cpu}.
   *
   * @param arch the given Apple architecture which artifacts are built under this configuration.
   *     Note this will likely be different than the value of {@code --cpu}.
   * @param configurationDistinguisher the configuration distinguisher used to describe the
   *     a configuration transition
   * @param minOsVersion the minimum os version for which to compile artifacts in the
   *     configuration
   */
  protected String configurationGenfiles(
      String arch, ConfigurationDistinguisher configurationDistinguisher,
      DottedVersion minOsVersion) {
    return configurationDir(arch, configurationDistinguisher, minOsVersion)
        + getTargetConfiguration().getGenfilesDirectory(RepositoryName.MAIN)
            .getExecPath().getBaseName();

  }

  private String configurationDir(
      String arch, ConfigurationDistinguisher configurationDistinguisher,
      DottedVersion minOsVersion) {
    switch (configurationDistinguisher) {
      case UNKNOWN:
        return String.format("%s-out/ios_%s-fastbuild/", TestConstants.PRODUCT_NAME, arch);
      case IOS_EXTENSION: // Intentional fall-through.
      case IOS_APPLICATION:
      case APPLEBIN_IOS:
        return String.format(
            "%1$s-out/ios-%2$s-min%4$s-%3$s-ios_%2$s-fastbuild/",
            TestConstants.PRODUCT_NAME,
            arch,
            configurationDistinguisher.toString().toLowerCase(Locale.US),
            minOsVersion);
      case APPLEBIN_WATCHOS:
        return String.format(
            "%1$s-out/watchos-%2$s-min%4$s-%3$s-watchos_%2$s-fastbuild/",
            TestConstants.PRODUCT_NAME,
            arch,
            configurationDistinguisher.toString().toLowerCase(Locale.US),
            minOsVersion);
      default:
        throw new AssertionError();
    }
  }

  /**
   * Returns the bin dir for artifacts built for a given Apple architecture (as set by a
   * configuration transition) and configuration distinguisher but the global default for
   * {@code --cpu} and the platform default for minimum OS.
   *
   * @param arch the given Apple architecture which artifacts are built under this configuration.
   *     Note this will likely be different than the value of {@code --cpu}
   * @param configurationDistinguisher the configuration distinguisher used to describe the
   *     a configuration transition
   */
  protected String configurationBin(
      String arch, ConfigurationDistinguisher configurationDistinguisher) {
    return configurationBin(arch, configurationDistinguisher,
        defaultMinimumOs(configurationDistinguisher));
  }

  /**
   * Returns the bin dir for artifacts with the given iOS architecture as set through {@code --cpu}
   * and configuration distinguisher, assuming {@code --ios_multi_cpus} isn't set.
   */
  protected static String iosConfigurationCcDepsBin(
      String arch, ConfigurationDistinguisher configurationDistinguisher) {
    switch (configurationDistinguisher) {
      case IOS_EXTENSION:
      case APPLEBIN_IOS:
        return String.format(
            "%s-out/%s-ios_%s-fastbuild/bin/",
            TestConstants.PRODUCT_NAME,
            configurationDistinguisher.toString().toLowerCase(Locale.US),
            arch);
      case UNKNOWN: // Intentional fall-through.
      case IOS_APPLICATION:
        return String.format("%s-out/ios_%s-fastbuild/bin/", TestConstants.PRODUCT_NAME, arch);
      default:
        throw new AssertionError();
    }
  }

  /**
   * Returns the default minimum os version that dependencies under a given configuration
   * distinguisher (and thus a given platform type) will be compiled for.
   */
  protected static DottedVersion defaultMinimumOs(
      ConfigurationDistinguisher configurationDistinguisher) {
    switch (configurationDistinguisher) {
      case UNKNOWN:
      case IOS_EXTENSION:
        return IosExtension.EXTENSION_MINIMUM_OS_VERSION;
      case IOS_APPLICATION:
      case APPLEBIN_IOS:
        return DEFAULT_IOS_SDK_VERSION;
      case APPLEBIN_WATCHOS:
        return DottedVersion.fromString(XcodeVersionProperties.DEFAULT_WATCHOS_SDK_VERSION);
      default:
        throw new AssertionError();
    }
  }

  /**
   * Returns the genfiles dir for iOS builds in the root architecture.
   */
  protected static String rootConfigurationGenfiles() {
    return TestConstants.PRODUCT_NAME + "-out/gcc-4.4.0-glibc-2.3.6-grte-k8-fastbuild/genfiles/";
  }

  protected String execPathEndingWith(Iterable<Artifact> artifacts, String suffix) {
    return getFirstArtifactEndingWith(artifacts, suffix).getExecPathString();
  }

  @Before
  public final void initializeMockToolsConfig() throws Exception {
    MockObjcSupport.setup(mockToolsConfig);
    MockProtoSupport.setup(mockToolsConfig);

    // Set flags required by objc builds.
    useConfiguration();
  }

  protected static String frameworkDir(ConfiguredTarget target) {
    AppleConfiguration configuration =
        target.getConfiguration().getFragment(AppleConfiguration.class);
    return frameworkDir(configuration.getSingleArchPlatform());
  }

  protected static String frameworkDir(ApplePlatform platform) {
    return AppleToolchain.platformDir(
        platform.getNameInPlist()) + AppleToolchain.DEVELOPER_FRAMEWORK_PATH;
  }

  /**
   * Creates an {@code objc_library} target writer for the label indicated by the given String.
   */
  protected ScratchAttributeWriter createLibraryTargetWriter(String labelString) {
    return ScratchAttributeWriter.fromLabelString(this, "objc_library", labelString);
  }

  /**
   * Creates an {@code objc_binary} target writer for the label indicated by the given String.
   */
  protected ScratchAttributeWriter createBinaryTargetWriter(String labelString) {
    return ScratchAttributeWriter.fromLabelString(this, "objc_binary", labelString);
  }

  private static String compilationModeFlag(CompilationMode mode) {
    switch (mode) {
      case DBG:
        return "dbg";
      case OPT:
        return "opt";
      case FASTBUILD:
        return "fastbuild";
    }
    throw new AssertionError();
  }

  private static List<String> compilationModeCopts(CompilationMode mode) {
    switch (mode) {
      case DBG:
        return ImmutableList.<String>builder()
            .addAll(ObjcConfiguration.DBG_COPTS)
            .addAll(ObjcConfiguration.GLIBCXX_DBG_COPTS)
            .build();
      case OPT:
        return ObjcConfiguration.OPT_COPTS;
      case FASTBUILD:
        return FASTBUILD_COPTS;
    }
    throw new AssertionError();
  }

  protected static String listAttribute(String name, Iterable<String> values) {
    StringBuilder result = new StringBuilder();
    for (String value : values) {
      if (result.length() == 0) {
        result.append(name).append(" = [");
      }
      result.append(String.format("'%s',", value));
    }
    if (result.length() != 0) {
      result.append("],");
    }
    return result.toString();
  }

  /** Returns the treatment of the crosstool for this test case. */
  protected ObjcCrosstoolMode getObjcCrosstoolMode() {
    return ObjcCrosstoolMode.ALL;
  }

  @Override
  protected void useConfiguration(String... args) throws Exception {
    // By default, objc tests assume the case of --experimental_objc_crosstool=all.  The "Legacy"
    // subclasses explicitly override to test --experimental_objc_crosstool=off.
    useConfiguration(getObjcCrosstoolMode(), args);
  }

  protected void useConfiguration(ObjcCrosstoolMode objcCrosstoolMode, String... args)
      throws Exception {
    ImmutableList.Builder<String> extraArgsBuilder = ImmutableList.builder();
    extraArgsBuilder.addAll(TestConstants.OSX_CROSSTOOL_FLAGS);

    switch(objcCrosstoolMode) {
      case ALL:
        extraArgsBuilder.add("--experimental_objc_crosstool=all");
        break;
      case LIBRARY:
        extraArgsBuilder.add("--experimental_objc_crosstool=library");
        break;
      case OFF:
        extraArgsBuilder.add("--experimental_objc_crosstool=off");
        break;
    }

    extraArgsBuilder
        .add("--xcode_version_config=" + MockObjcSupport.XCODE_VERSION_CONFIG)
        .add("--apple_crosstool_top=" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL)
        .add("--crosstool_top=" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL);

    ImmutableList<String> extraArgs = extraArgsBuilder.build();
    args = Arrays.copyOf(args, args.length + extraArgs.size());
    for (int i = 0; i < extraArgs.size(); i++) {
      args[(args.length - extraArgs.size()) + i] = extraArgs.get(i);
    }

    super.useConfiguration(args);
  }

  /**
   * @param extraAttributes individual strings which contain a whole attribute to be added to the
   *     generated target, e.g. "deps = ['foo']"
   */
  protected ConfiguredTarget addBinaryBasedTarget(
      String ruleType,
      String packageName,
      String targetName,
      List<String> srcs,
      List<String> deps,
      String... extraAttributes)
      throws Exception {
    for (String source : srcs) {
      scratch.file(String.format("%s/%s", packageName, source));
    }
    scratch.file(String.format("%s/BUILD", packageName),
        ruleType + "(name = '" + targetName + "',",
        listAttribute("srcs", srcs),
        listAttribute("deps", deps),
        Joiner.on(",\n").join(extraAttributes),
        ")");
    return getConfiguredTarget(String.format("//%s:%s", packageName, targetName));
  }

  /**
   * @param extraAttributes individual strings which contain a whole attribute to be added to the
   *     generated target, e.g. "deps = ['foo']"
   */
  protected ConfiguredTarget addSimpleIosTest(
      String packageName,
      String targetName,
      List<String> srcs,
      List<String> deps,
      String... extraAttributes)
      throws Exception {
    return addBinaryBasedTarget(
        "ios_test", packageName, targetName, srcs, deps, extraAttributes);
  }

  /**
   * Returns the arguments to pass to clang for specifying module map artifact location and
   * module name.
   *
   * @param packagePath the path to the package this target is in
   * @param targetName the name of the target
   */
  protected List<String> moduleMapArtifactArguments(String packagePath, String targetName) {
    Artifact moduleMapArtifact =
        getGenfilesArtifact(
            targetName + ".modulemaps/module.modulemap", packagePath + ":" + targetName);
    String moduleName = packagePath.replace("//", "").replace("/", "_") + "_" + targetName;

    return ImmutableList.of("-iquote",
        moduleMapArtifact.getExecPath().getParentDirectory().toString(),
        "-fmodule-name=" + moduleName);
  }

  /**
   * Returns all child configurations resulting from a given split transition on a given
   * configuration.
   */
  protected List<BuildConfiguration> getSplitConfigurations(BuildConfiguration configuration,
      SplitTransition<BuildOptions> splitTransition) throws InterruptedException {
    ImmutableList.Builder<BuildConfiguration> splitConfigs = ImmutableList.builder();

    for (BuildOptions splitOptions : splitTransition.split(configuration.getOptions())) {
      splitConfigs.add(getSkyframeExecutor().getConfigurationForTesting(
          reporter, configuration.fragmentClasses(), splitOptions));
    }

    return splitConfigs.build();
  }

  protected void checkLinkActionCorrect(RuleType ruleType, ExtraLinkArgs extraLinkArgs)
      throws Exception {
    useConfiguration("--cpu=ios_i386");

    createLibraryTargetWriter("//lib1:lib1")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "hdr.h")
        .write();
    createLibraryTargetWriter("//lib2:lib2")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "hdr.h")
        .write();

    scratch.file("x/a.m");
    ruleType.scratchTarget(scratch,
        "srcs", "['a.m']", "deps", "['//lib1:lib1', '//lib2:lib2']");
    CommandAction action = linkAction("//x:x");
    assertRequiresDarwin(action);
    assertThat(Artifact.toRootRelativePaths(action.getInputs()))
        .containsAllOf("x/libx.a", "lib1/liblib1.a", "lib2/liblib2.a", "x/x-linker.objlist");
    assertThat(Artifact.toRootRelativePaths(action.getOutputs()))
        .containsExactly("x/x_bin");

    verifyLinkAction(Iterables.getOnlyElement(action.getOutputs()),
        getBinArtifact("x-linker.objlist", "//x:x"), "i386",
        ImmutableList.of("libx.a", "liblib1.a", "liblib2.a"), ImmutableList.<PathFragment>of(),
        extraLinkArgs);
  }

  // Regression test for b/29094356.
  protected void checkLinkActionDuplicateInputs(RuleType ruleType, ExtraLinkArgs extraLinkArgs)
      throws Exception {
    useConfiguration("--cpu=ios_i386");

    scratch.file("lib/BUILD",
        "cc_library(",
        "    name = 'cclib',",
        "    srcs = ['dep.c'],",
        "    deps = ['//lib2:lib2'],",
        ")");

    createLibraryTargetWriter("//lib1:lib1")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "hdr.h")
        .write();
    createLibraryTargetWriter("//lib2:lib2")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "hdr.h")
        .write();

    ruleType.scratchTarget(
        scratch, "srcs", "['a.m']", "deps", "['//lib:cclib', '//lib1:lib1', '//lib2:lib2']");
    CommandAction action = linkAction("//x:x");
    assertRequiresDarwin(action);

    verifyObjlist(action, "x-linker.objlist",
        execPathEndingWith(action.getInputs(), "x/libx.a"),
        execPathEndingWith(action.getInputs(), "lib2/liblib2.a"),
        execPathEndingWith(action.getInputs(), "lib1/liblib1.a"),
        execPathEndingWith(action.getInputs(), "lib/libcclib.a"));
  }

  /**
   * Verifies a {@code -filelist} file's contents.
   *
   * @param originalAction the action which uses the filelist artifact
   * @param objlistName the path suffix of the filelist artifact
   * @param inputArchives path suffixes of the expected contents of the filelist
   */
  protected void verifyObjlist(Action originalAction, String objlistName, String... inputArchives)
      throws Exception {
    Artifact filelistArtifact =
        getFirstArtifactEndingWith(originalAction.getInputs(), objlistName);

    ParameterFileWriteAction fileWriteAction =
        (ParameterFileWriteAction) getGeneratingAction(filelistArtifact);
    ImmutableList.Builder<String> execPaths = ImmutableList.builder();
    for (String inputArchive : inputArchives) {
      execPaths.add(execPathEndingWith(originalAction.getInputs(), inputArchive));
    }

    assertThat(fileWriteAction.getContents()).containsExactlyElementsIn(execPaths.build());
  }

  /**
   * Verifies a link action is registered correctly.
   *
   * @param binArtifact the output artifact which a link action should be registered to generate
   * @param filelistArtifact the input filelist artifact
   * @param arch the architecture (for example, "i386") which the binary is to be created for
   * @param inputArchives the suffixes (basenames or relative paths with basenames) of the input
   *     archive files for the link action
   * @param importedFrameworks custom framework path fragments
   * @param extraLinkArgs extra link arguments expected on the link action
   */
  protected void verifyLinkAction(
      Artifact binArtifact,
      Artifact filelistArtifact,
      String arch,
      List<String> inputArchives,
      List<PathFragment> importedFrameworks,
      ExtraLinkArgs extraLinkArgs)
      throws Exception {
    final CommandAction binAction = (CommandAction) getGeneratingAction(binArtifact);

    for (String inputArchive : inputArchives) {
      // Verify each input archive is present in the action inputs.
      getFirstArtifactEndingWith(binAction.getInputs(), inputArchive);
    }
    ImmutableList.Builder<String> frameworkPathFragmentParents = ImmutableList.builder();
    ImmutableList.Builder<String> frameworkPathBaseNames = ImmutableList.builder();
    for (PathFragment importedFramework : importedFrameworks) {
      frameworkPathFragmentParents.add(importedFramework.getParentDirectory().toString());
      frameworkPathBaseNames.add(importedFramework.getBaseName());
    }

    ImmutableList<String> expectedCommandLineFragments =
        ImmutableList.<String>builder()
            .add("-mios-simulator-version-min=" + DEFAULT_IOS_SDK_VERSION)
            .add("-arch " + arch)
            .add("-isysroot " + AppleToolchain.sdkDir())
            .add(AppleToolchain.sdkDir() + AppleToolchain.DEVELOPER_FRAMEWORK_PATH)
            .add(frameworkDir(ApplePlatform.forTarget(PlatformType.IOS, arch)))
            .addAll(frameworkPathFragmentParents.build())
            .add("-Xlinker -objc_abi_version -Xlinker 2")
            .add("-Xlinker -rpath -Xlinker @executable_path/Frameworks")
            .add("-fobjc-link-runtime")
            .add("-ObjC")
            .addAll(
                Interspersing.beforeEach(
                    "-framework", SdkFramework.names(AUTOMATIC_SDK_FRAMEWORKS)))
            .addAll(Interspersing.beforeEach("-framework", frameworkPathBaseNames.build()))
            .add("-filelist")
            .add(filelistArtifact.getExecPathString())
            .add("-o")
            .addAll(Artifact.toExecPaths(binAction.getOutputs()))
            .addAll(extraLinkArgs)
            .build();

    String linkArgs = Joiner.on(" ").join(binAction.getArguments());
    for (String expectedCommandLineFragment : expectedCommandLineFragments) {
      assertThat(linkArgs).contains(expectedCommandLineFragment);
    }
  }

  protected void checkLinkActionWithTransitiveCppDependency(
      RuleType ruleType, ExtraLinkArgs extraLinkArgs) throws Exception {

    createLibraryTargetWriter("//lib1:lib1").setAndCreateFiles("srcs", "a.mm").write();
    createLibraryTargetWriter("//lib2:lib2")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setList("deps", "//lib1")
        .write();

    scratch.file("x/c.m");
    ruleType.scratchTarget(scratch, "srcs", "['c.m']", "deps", "['//lib2:lib2']");

    CommandAction action = linkAction("//x:x");
    assertThat(action.getArguments().get(2))
        .startsWith(
            MOCK_XCRUNWRAPPER_PATH + " " + CLANG_PLUSPLUS + " -stdlib=libc++ -std=gnu++11");
  }

  protected Map<String, String> mobileProvisionProfiles(BundleMergeProtos.Control control) {
    Map<String, String> profiles = new HashMap<>();
    for (BundleFile bundleFile : control.getBundleFileList()) {
      if (bundleFile.getBundlePath()
          .endsWith(ReleaseBundlingSupport.PROVISIONING_PROFILE_BUNDLE_FILE)) {
        assertWithMessage("Should not have multiple entries for same source file")
            .that(profiles.put(bundleFile.getSourceFile(), bundleFile.getBundlePath()))
            .isNull();
      }
    }
    return profiles;
  }

  protected void checkFilesToRun(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch);
    ConfiguredTarget target = getConfiguredTarget("//x:x");
    FilesToRunProvider filesToRun = target.getProvider(FilesToRunProvider.class);
    assertThat(filesToRun.getExecutable().getRootRelativePathString())
        .isEqualTo("x/x_runner.sh");
    RunfilesSupport runfilesSupport = filesToRun.getRunfilesSupport();
    assertThat(Artifact.toRootRelativePaths(runfilesSupport.getRunfiles().getArtifacts()))
        .containsExactly(
            "x/x.ipa",
            "x/x_runner.sh",
            "tools/objc/StdRedirect.dylib");
  }

  protected void assertAppleSdkVersionEnv(Map<String, String> env) throws Exception {
    assertAppleSdkVersionEnv(env, DEFAULT_IOS_SDK_VERSION);
  }

  protected void assertAppleSdkVersionEnv(Map<String, String> env, DottedVersion versionNumber)
      throws Exception {
    assertThat(env).containsEntry("APPLE_SDK_VERSION_OVERRIDE", versionNumber.toString());
  }

  protected void assertAppleSdkPlatformEnv(
      Map<String, String> env, String platformName) throws Exception {
    assertThat(env).containsEntry("APPLE_SDK_PLATFORM", platformName);
  }

  protected void assertAppleSdkVersionEnv(CommandAction action) throws Exception {
    assertAppleSdkVersionEnv(action, DEFAULT_IOS_SDK_VERSION.toString());
  }

  protected void assertAppleSdkVersionEnv(CommandAction action, String versionString)
      throws Exception {
    assertThat(action.getEnvironment())
        .containsEntry("APPLE_SDK_VERSION_OVERRIDE", versionString);
  }

  protected void assertAppleSdkPlatformEnv(CommandAction action, String platformName)
      throws Exception {
    assertThat(action.getEnvironment()).containsEntry("APPLE_SDK_PLATFORM", platformName);
  }

  protected void assertXcodeVersionEnv(CommandAction action, String versionNumber)
      throws Exception {
    assertThat(action.getEnvironment()).containsEntry("XCODE_VERSION_OVERRIDE", versionNumber);
  }

  protected void checkNoRunfilesSupportForDevice(RuleType ruleType) throws Exception {
    useConfiguration("--cpu=ios_armv7");
    ruleType.scratchTarget(scratch);
    ConfiguredTarget target = getConfiguredTarget("//x:x");
    FilesToRunProvider filesToRun = target.getProvider(FilesToRunProvider.class);
    assertThat(filesToRun.getRunfilesSupport()).isNull();
  }

  protected void checkGenerateRunnerScriptAction(RuleType ruleType) throws Exception {
    useConfiguration(
        "--cpu=ios_i386", "--ios_simulator_device=iPhone X", "--ios_simulator_version=3");

    ruleType.scratchTarget(scratch);
    ConfiguredTarget target = getConfiguredTarget("//x:x");
    Artifact runnerScript = getBinArtifact("x_runner.sh", target);
    TemplateExpansionAction action = (TemplateExpansionAction) getGeneratingAction(runnerScript);
    assertThat(ActionsTestUtil.baseArtifactNames(action.getInputs()))
        .containsExactly("ios_runner.sh.mac_template");
    assertThat(Artifact.toRootRelativePaths(action.getOutputs()))
        .containsExactly("x/x_runner.sh");
    Map<String, String> substitutions =
        action
            .getSubstitutions()
            .stream()
            .collect(ImmutableMap.toImmutableMap(sub -> sub.getKey(), s -> s.getValue()));
    ;
    assertThat(substitutions.get("%ipa_file%")).isEqualTo("x/x.ipa");
    assertThat(substitutions.get("%sim_device%")).isEqualTo("'iPhone X'");
    assertThat(substitutions.get("%sdk_version%")).isEqualTo("3");
    assertThat(substitutions.get("%app_name%")).isEqualTo("x");
    assertThat(substitutions.get("%std_redirect_dylib_path%"))
        .endsWith("tools/objc/StdRedirect.dylib");
  }

  protected void checkGenerateRunnerScriptAction_escaped(RuleType ruleType) throws Exception {
    useConfiguration("--cpu=ios_i386", "--ios_simulator_device=iPhone X'");

    ruleType.scratchTarget(scratch);
    ConfiguredTarget target = getConfiguredTarget("//x:x");
    Artifact runnerScript = getBinArtifact("x_runner.sh", target);
    TemplateExpansionAction action = (TemplateExpansionAction) getGeneratingAction(runnerScript);
    assertThat(action.getSubstitutions())
        .contains(Substitution.of("%sim_device%", "'iPhone X'\\'''"));
  }

  protected void checkDeviceSigningAction(RuleType ruleType) throws Exception {
    useConfiguration("--cpu=ios_armv7");

    scratch.file("x/entitlements.entitlements");
    ruleType.scratchTarget(scratch, ENTITLEMENTS_ATTR, "'entitlements.entitlements'");
    SpawnAction action = (SpawnAction) ipaGeneratingAction();
    assertRequiresDarwin(action);
    assertThat(ActionsTestUtil.baseArtifactNames(action.getInputs()))
        .containsExactly("x.entitlements", "foo.mobileprovision", "x.unprocessed.ipa");
    assertThat(Artifact.toRootRelativePaths(action.getOutputs())).containsExactly("x/x.ipa");
  }

  protected void checkSigningWithCertName(RuleType ruleType) throws Exception {
    useConfiguration("--cpu=ios_armv7", "--ios_signing_cert_name=Foo Bar");

    scratch.file("x/entitlements.entitlements");
    ruleType.scratchTarget(scratch, ENTITLEMENTS_ATTR, "'entitlements.entitlements'");
    SpawnAction action = (SpawnAction) ipaGeneratingAction();
    assertThat(ActionsTestUtil.baseArtifactNames(action.getInputs()))
        .containsExactly("x.entitlements", "foo.mobileprovision", "x.unprocessed.ipa");
    assertThat(Joiner.on(' ').join(action.getArguments())).contains("--sign \"Foo Bar\"");

    assertThat(Artifact.toRootRelativePaths(action.getOutputs())).containsExactly("x/x.ipa");
  }

  protected void checkPostProcessingAction(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch, "ipa_post_processor", "'tool.sh'");

    SpawnAction action = (SpawnAction) ipaGeneratingAction();
    assertRequiresDarwin(action);
    assertThat(ActionsTestUtil.baseArtifactNames(action.getInputs()))
        .containsExactly("tool.sh", "x.unprocessed.ipa");

    assertThat(Joiner.on(' ').join(action.getArguments())).contains("x/tool.sh ${t}");

    assertThat(Artifact.toRootRelativePaths(action.getOutputs())).containsExactly("x/x.ipa");
  }

  protected void checkSigningAndPostProcessing(RuleType ruleType) throws Exception {
    useConfiguration("--cpu=ios_armv7");
    ruleType.scratchTarget(
        scratch, "ipa_post_processor", "'tool.sh'", ENTITLEMENTS_ATTR,
        "'entitlements.entitlements'");

    SpawnAction action = (SpawnAction) ipaGeneratingAction();
    assertRequiresDarwin(action);
    assertThat(ActionsTestUtil.baseArtifactNames(action.getInputs()))
        .containsExactly("tool.sh", "x.entitlements", "foo.mobileprovision", "x.unprocessed.ipa");

    assertThat(normalizeBashArgs(action.getArguments()))
        .containsAllOf("x/tool.sh", "--sign")
        .inOrder();
  }

  protected void checkNoEntitlementsDefined(RuleType ruleType) throws Exception {
    useConfiguration("--cpu=ios_armv7", "--nodevice_debug_entitlements");

    ruleType.scratchTarget(scratch);

    SpawnAction ipaAction = (SpawnAction) ipaGeneratingAction();
    Artifact entitlements = getFirstArtifactEndingWith(ipaAction.getInputs(), ".entitlements");
    SpawnAction substitutionAction = (SpawnAction) getGeneratingAction(entitlements);
    assertThat(Joiner.on(' ').join(substitutionAction.getArguments())).contains("sed");

    Artifact prefix =
        getFirstArtifactEndingWith(substitutionAction.getInputs(), ".team_prefix_file");
    SpawnAction prefixAction = (SpawnAction) getGeneratingAction(prefix);
    assertThat(baseArtifactNames(prefixAction.getInputs())).containsExactly("foo.mobileprovision");
    assertThat(Joiner.on(' ').join(prefixAction.getArguments()))
        .contains("Print ApplicationIdentifierPrefix:0");

    Artifact extractedEntitlements =
        getFirstArtifactEndingWith(substitutionAction.getInputs(), ".entitlements_with_variables");
    SpawnAction extractionAction = (SpawnAction) getGeneratingAction(extractedEntitlements);
    assertThat(baseArtifactNames(extractionAction.getInputs()))
        .containsExactly("foo.mobileprovision");
    assertThat(Joiner.on(' ').join(extractionAction.getArguments())).contains("Print Entitlements");
  }

  protected void checkEntitlementsDefined(RuleType ruleType) throws Exception {
    useConfiguration("--cpu=ios_armv7", "--nodevice_debug_entitlements");

    ruleType.scratchTarget(scratch, ENTITLEMENTS_ATTR, "'bar.entitlements'");

    SpawnAction ipaAction = (SpawnAction) ipaGeneratingAction();
    Artifact entitlements = getFirstArtifactEndingWith(ipaAction.getInputs(), ".entitlements");
    SpawnAction substitutionAction = (SpawnAction) getGeneratingAction(entitlements);

    Artifact prefix =
        getFirstArtifactEndingWith(substitutionAction.getInputs(), ".team_prefix_file");
    SpawnAction prefixAction = (SpawnAction) getGeneratingAction(prefix);
    assertThat(prefixAction).isNotNull();

    assertThat(Artifact.toExecPaths(substitutionAction.getInputs())).contains("x/bar.entitlements");
    assertThat(
            getFirstArtifactEndingWith(
                substitutionAction.getInputs(), ".entitlements_with_variables"))
        .isNull();
  }

  protected void checkExtraEntitlements(RuleType ruleType) throws Exception {
    useConfiguration("--cpu=ios_armv7", "--extra_entitlements=//foo:extra.entitlements");

    ruleType.scratchTarget(scratch);

    scratch.file("foo/extra.entitlements");
    scratch.file("foo/BUILD", "exports_files(['extra.entitlements'])");

    SpawnAction ipaAction = (SpawnAction) ipaGeneratingAction();
    Artifact entitlements = getFirstArtifactEndingWith(ipaAction.getInputs(), ".entitlements");
    SpawnAction mergeAction = (SpawnAction) getGeneratingAction(entitlements);

    assertThat(Artifact.toExecPaths(mergeAction.getInputs())).contains("foo/extra.entitlements");

    Artifact mergeControl =
        getFirstArtifactEndingWith(mergeAction.getInputs(), ".merge-entitlements-control");
    BinaryFileWriteAction mergeControleAction =
        (BinaryFileWriteAction) getGeneratingAction(mergeControl);

    PlMergeProtos.Control mergeControlProto;
    try (InputStream in = mergeControleAction.getSource().openStream()) {
      mergeControlProto = PlMergeProtos.Control.parseFrom(in);
    }

    assertThat(mergeControlProto.getSourceFileList()).contains("foo/extra.entitlements");
  }

  protected void checkFastbuildDebugEntitlements(RuleType ruleType) throws Exception {
    useConfiguration("--cpu=ios_armv7");
    assertDebugEntitlements(ruleType);
  }

  protected void checkDebugEntitlements(RuleType ruleType) throws Exception {
    useConfiguration("--cpu=ios_armv7", "--compilation_mode=dbg");
    assertDebugEntitlements(ruleType);
  }

  protected void checkOptNoDebugEntitlements(RuleType ruleType) throws Exception {
    useConfiguration("--cpu=ios_armv7", "--compilation_mode=opt");
    assertNoDebugEntitlements(ruleType);
  }

  protected void checkExplicitNoDebugEntitlements(RuleType ruleType) throws Exception {
    useConfiguration("--cpu=ios_armv7", "--nodevice_debug_entitlements");
    assertNoDebugEntitlements(ruleType);
  }

  private void assertDebugEntitlements(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch);

    SpawnAction ipaAction = (SpawnAction) ipaGeneratingAction();
    Artifact entitlements = getFirstArtifactEndingWith(ipaAction.getInputs(), ".entitlements");
    SpawnAction mergeAction = (SpawnAction) getGeneratingAction(entitlements);

    assertThat(Artifact.toExecPaths(mergeAction.getInputs()))
        .contains(toolsRepoExecPath("tools/objc/device_debug_entitlements.plist"));

    Artifact mergeControl =
        getFirstArtifactEndingWith(mergeAction.getInputs(), ".merge-entitlements-control");
    BinaryFileWriteAction mergeControleAction =
        (BinaryFileWriteAction) getGeneratingAction(mergeControl);

    PlMergeProtos.Control mergeControlProto;
    try (InputStream in = mergeControleAction.getSource().openStream()) {
      mergeControlProto = PlMergeProtos.Control.parseFrom(in);
    }

    assertThat(mergeControlProto.getSourceFileList())
        .contains(toolsRepoExecPath("tools/objc/device_debug_entitlements.plist"));
  }

  private void assertNoDebugEntitlements(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch);

    SpawnAction ipaAction = (SpawnAction) ipaGeneratingAction();
    Artifact entitlements = getFirstArtifactEndingWith(ipaAction.getInputs(), ".entitlements");
    SpawnAction entitlementsAction = (SpawnAction) getGeneratingAction(entitlements);

    assertThat(Artifact.toExecPaths(entitlementsAction.getInputs()))
        .doesNotContain(toolsRepoExecPath("tools/objc/device_debug_entitlements.plist"));
  }

  protected void checkCompilesSources(RuleType ruleType) throws Exception {
    createLibraryTargetWriter("//lib1:lib1")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "hdr.h")
        .write();
    createLibraryTargetWriter("//lib2:lib2")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "hdr.h")
        .setList("deps", "//lib1:lib1")
        .write();
    scratch.file("x/a.m");
    scratch.file("x/b.m");
    scratch.file("x/a.h");
    scratch.file("x/private.h");
    ruleType.scratchTarget(scratch, "srcs", "['a.m', 'b.m', 'private.h']", "hdrs", "['a.h']",
        "deps", "['//lib2:lib2']");
    CommandAction compileA = compileAction("//x:x", "a.o");

    assertThat(Artifact.toRootRelativePaths(compileA.getPossibleInputsForTesting()))
        .containsAllOf("x/a.m", "x/a.h", "x/private.h", "lib1/hdr.h", "lib2/hdr.h");
    assertThat(Artifact.toRootRelativePaths(compileA.getOutputs()))
        .containsExactly("x/_objs/x/x/a.o", "x/_objs/x/x/a.d");
  }

  protected void checkCompilesSourcesWithModuleMapsEnabled(RuleType ruleType) throws Exception {
    useConfiguration("--experimental_objc_enable_module_maps");
    createLibraryTargetWriter("//lib1:lib1")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "hdr.h")
        .write();
    createLibraryTargetWriter("//lib2:lib2")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "hdr.h")
        .setList("deps", "//lib1:lib1")
        .write();

    ruleType.scratchTarget(
        scratch, "srcs", "['a.m', 'b.m']", "hdrs", "['a.h']", "deps", "['//lib2:lib2']");
    CommandAction compileA = compileAction("//x:x", "a.o");

    assertThat(Artifact.toRootRelativePaths(compileA.getInputs()))
        .containsAllOf(
            "lib1/lib1.modulemaps/module.modulemap",
            "lib2/lib2.modulemaps/module.modulemap",
            "x/x.modulemaps/module.modulemap");
  }

  protected void checkCompileWithDotMFileInHeaders(RuleType ruleType) throws Exception {
    scratch.file("bin/a.m");
    scratch.file("bin/b.m");
    scratch.file("bin/h.m");
    createLibraryTargetWriter("//lib1:lib1")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "hdr.h")
        .write();
    createLibraryTargetWriter("//lib2:lib2")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "hdr.h")
        .write();
    scratch.file("x/a.m");
    ruleType.scratchTarget(scratch,
        "srcs", "['a.m', 'b.m']",
        "hdrs", "['h.m']",
        "deps", "['//lib1:lib1', '//lib2:lib2']");
    Action linkAction = linkAction("//x:x");
    Artifact libBin = getFirstArtifactEndingWith(linkAction.getInputs(), "libx.a");
    Action linkBinAFile = getGeneratingAction(libBin);
    Artifact aObjFile = getFirstArtifactEndingWith(linkBinAFile.getInputs(), "a.o");
    CommandAction compileA = (CommandAction) getGeneratingAction(aObjFile);

    assertThat(compileA.getArguments()).contains("x/a.m");
    assertThat(compileA.getArguments()).doesNotContain("x/h.m");
    assertThat(getFirstArtifactEndingWith(linkBinAFile.getInputs(), "h.o")).isNull();
  }

  protected void checkCompileWithTextualHeaders(RuleType ruleType) throws Exception {
    createLibraryTargetWriter("//lib1:lib1")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "hdr.h")
        .write();
    createLibraryTargetWriter("//lib2:lib2")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "hdr.h")
        .write();
    ruleType.scratchTarget(scratch,
        "srcs", "['a.m']",
        "textual_hdrs", "['t.h']",
        "deps", "['//lib1:lib1', '//lib2:lib2']");
    CommandAction compileA = compileAction("//x:x", "a.o");

    assertThat(Artifact.toRootRelativePaths(compileA.getPossibleInputsForTesting()))
        .containsAllOf("x/a.m", "x/t.h", "lib1/hdr.h", "lib2/hdr.h");
  }

  protected void checkLinksFrameworksOfSelfAndTransitiveDependencies(RuleType ruleType)
      throws Exception {
    createLibraryTargetWriter("//base_lib:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setList("sdk_frameworks", "foo")
        .write();
    scratch.file("x/a.m");
    ruleType.scratchTarget(scratch,
        "srcs", "['a.m']",
        "deps", "['//base_lib:lib']",
        "sdk_frameworks", "['bar']");

    assertThat(Joiner.on(" ").join(linkAction("//x:x").getArguments()))
        .contains("-framework foo -framework bar");
  }

  protected void checkLinksWeakFrameworksOfSelfAndTransitiveDependencies(RuleType ruleType)
      throws Exception {
    createLibraryTargetWriter("//base_lib:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setList("weak_sdk_frameworks", "foo")
        .write();

    ruleType.scratchTarget(
        scratch, "srcs", "['a.m']", "deps", "['//base_lib:lib']", "weak_sdk_frameworks", "['bar']");

    assertThat(Joiner.on(" ").join(linkAction("//x:x").getArguments()))
        .contains("-weak_framework foo -weak_framework bar");
  }

  protected void checkLinkWithFrameworkImportsIncludesFlagsAndInputArtifacts(RuleType ruleType)
      throws Exception {
    ConfiguredTarget lib = addLibWithDepOnFrameworkImport();
    scratch.file("x/a.m");
    ruleType.scratchTarget(scratch,
        "srcs", "['a.m']",
        "deps", "['" + lib.getLabel() + "']");

    CommandAction linkAction = linkAction("//x:x");
    String linkActionArgs = Joiner.on(" ").join(linkAction.getArguments());
    assertThat(linkActionArgs).contains("-framework fx1 -framework fx2");
    assertThat(linkActionArgs).contains("-F fx");
    assertThat(linkAction.getInputs()).containsAllOf(
        getSourceArtifact("fx/fx1.framework/a"),
        getSourceArtifact("fx/fx1.framework/b"),
        getSourceArtifact("fx/fx2.framework/c"),
        getSourceArtifact("fx/fx2.framework/d"));
  }

  protected void checkLinkIncludeOrderStaticLibsFirst(RuleType ruleType) throws Exception {
    scratch.file("fx/fx1.framework");
    scratch.file("fx/BUILD", "objc_framework(name = 'fx')");
    scratch.file("x/a.m");
    ruleType.scratchTarget(
        scratch, "srcs", "['a.m']", "sdk_frameworks", "['fx']", "sdk_dylibs", "['libdy1']");

    CommandAction linkAction = linkAction("//x:x");
    String linkActionArgs = Joiner.on(" ").join(linkAction.getArguments());

    assertThat(linkActionArgs.indexOf(".a")).isLessThan(linkActionArgs.indexOf("-F"));
    assertThat(linkActionArgs.indexOf(".a")).isLessThan(linkActionArgs.indexOf("-l"));
  }

  protected ObjcProvider providerForTarget(String label) throws Exception {
    return getConfiguredTarget(label).get(ObjcProvider.SKYLARK_CONSTRUCTOR);
  }

  protected CommandAction archiveAction(String label) throws Exception {
    ConfiguredTarget target = getConfiguredTarget(label);
    return (CommandAction)
        getGeneratingAction(getBinArtifact("lib" + target.getLabel().getName() + ".a", target));
  }

  protected Iterable<Artifact> inputsEndingWith(Action action, final String suffix) {
    return Iterables.filter(action.getInputs(), new Predicate<Artifact>() {
      @Override
      public boolean apply(Artifact artifact) {
        return artifact.getExecPathString().endsWith(suffix);
      }
    });
  }

  /**
   * Asserts that the given action can specify execution requirements, and requires execution on
   * darwin.
   */
  protected void assertRequiresDarwin(ExecutionInfoSpecifier action) {
    assertThat(action.getExecutionInfo()).containsKey(ExecutionRequirements.REQUIRES_DARWIN);
  }

  /**
   * Asserts that the given action can specify execution requirements, but does not require
   * execution on darwin.
   */
  protected void assertNotRequiresDarwin(Action action) {
    ExecutionInfoSpecifier executionInfoSpecifier = (ExecutionInfoSpecifier) action;
    assertThat(executionInfoSpecifier.getExecutionInfo())
        .doesNotContainKey(ExecutionRequirements.REQUIRES_DARWIN);
  }

  protected ConfiguredTarget addBinWithTransitiveDepOnFrameworkImport() throws Exception {
    ConfiguredTarget lib = addLibWithDepOnFrameworkImport();
    return createBinaryTargetWriter("//bin:bin")
        .setAndCreateFiles("srcs", "a.m")
        .setList("deps", lib.getLabel().toString())
        .write();

  }

  protected ConfiguredTarget addLibWithDepOnFrameworkImport() throws Exception {
    scratch.file("fx/fx1.framework/a");
    scratch.file("fx/fx1.framework/b");
    scratch.file("fx/fx2.framework/c");
    scratch.file("fx/fx2.framework/d");
    scratch.file("fx/BUILD",
        "objc_framework(",
        "    name = 'fx',",
        "    framework_imports = glob(['fx1.framework/*', 'fx2.framework/*']),",
        "    sdk_frameworks = ['CoreLocation'],",
        "    weak_sdk_frameworks = ['MediaAccessibility'],",
        "    sdk_dylibs = ['libdy1'],",
        ")");
    return createLibraryTargetWriter("//lib:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setList("deps", "//fx:fx")
        .write();
  }

  protected CommandAction compileAction(String ownerLabel, String objFileName) throws Exception {
    Action archiveAction = archiveAction(ownerLabel);
    return (CommandAction)
        getGeneratingAction(
            getFirstArtifactEndingWith(archiveAction.getInputs(), "/" + objFileName));
  }

  /**
   * Verifies simply that some rule type creates the {@link CompilationArtifacts} object
   * successfully; in particular, makes sure it is not ignoring attributes. If the scope of
   * {@link CompilationArtifacts} expands, make sure this method tests it properly.
   *
   * <p>This test only makes sure the attributes are not being ignored - it does not test any
   * other functionality in depth, which is covered by other unit tests.
   */
  protected void checkPopulatesCompilationArtifacts(RuleType ruleType) throws Exception {
    scratch.file("x/a.m");
    scratch.file("x/b.m");
    scratch.file("x/c.pch");
    ruleType.scratchTarget(scratch,
        "srcs", "['a.m']",
        "non_arc_srcs", "['b.m']",
        "pch", "'c.pch'");
    ImmutableList<String> includeFlags = ImmutableList.of("-include", "x/c.pch");
    assertContainsSublist(compileAction("//x:x", "a.o").getArguments(), includeFlags);
    assertContainsSublist(compileAction("//x:x", "b.o").getArguments(), includeFlags);
  }

  protected void checkProvidesHdrsAndIncludes(RuleType ruleType) throws Exception {
    scratch.file("x/a.h");
    ruleType.scratchTarget(scratch,
        "hdrs", "['a.h']",
        "includes", "['incdir']");
    ObjcProvider provider =
        getConfiguredTarget("//x:x", getAppleCrosstoolConfiguration())
            .get(ObjcProvider.SKYLARK_CONSTRUCTOR);
    assertThat(provider.get(HEADER)).containsExactly(getSourceArtifact("x/a.h"));
    assertThat(provider.get(INCLUDE))
        .containsExactly(
            PathFragment.create("x/incdir"),
            getAppleCrosstoolConfiguration().getGenfilesFragment().getRelative("x/incdir"));
  }

  protected void checkCompilesWithHdrs(RuleType ruleType) throws Exception {
    scratch.file("x/a.m");
    scratch.file("x/a.h");
    ruleType.scratchTarget(scratch,
        "srcs", "['a.m']",
        "hdrs", "['a.h']");
    assertThat(compileAction("//x:x", "a.o").getPossibleInputsForTesting())
        .contains(getSourceArtifact("x/a.h"));
  }

  protected void checkArchivesPrecompiledObjectFiles(RuleType ruleType) throws Exception {
    scratch.file("x/a.m");
    scratch.file("x/b.o");
    ruleType.scratchTarget(scratch, "srcs", "['a.m', 'b.o']");
    assertThat(Artifact.toRootRelativePaths(archiveAction("//x:x").getInputs())).contains("x/b.o");
  }

  protected void checkPopulatesBundling(RuleType ruleType) throws Exception {
    scratch.file("x/a.m");
    scratch.file("x/info.plist");
    scratch.file("x/assets.xcassets/1");
    ruleType.scratchTarget(scratch,
        "srcs", "['a.m']",
        INFOPLIST_ATTR, "'info.plist'",
        "asset_catalogs", "['assets.xcassets/1']");
    String targetName = "//x:x";
    ConfiguredTarget target = getConfiguredTarget(targetName);
    PlMergeProtos.Control control = plMergeControl(targetName);

    assertThat(control.getSourceFileList())
        .contains(getSourceArtifact("x/info.plist").getExecPathString());

    assertThat(linkAction("//x:x").getInputs())
        .contains(getBinArtifact("libx.a", target));

    Artifact actoolzipOutput = getBinArtifact("x.actool.zip", target);
    assertThat(getGeneratingAction(actoolzipOutput).getInputs())
        .contains(getSourceArtifact("x/assets.xcassets/1"));
  }

  /**
   * Checks that a target at {@code //x:x}, which is already created, registered a correct merge
   * bundle action based on certain arbitrary and default values which include nested bundles.
   */
  private void checkMergeBundleActionsWithNestedBundle(
      String bundleDir, BuildConfiguration bundleConfiguration) throws Exception {
    BundleFile fooResource = BundleFile.newBuilder()
        .setSourceFile("bndl/foo.data")
        .setBundlePath("foo.data")
        .setExternalFileAttribute(BundleableFile.DEFAULT_EXTERNAL_FILE_ATTRIBUTE)
        .build();

    BundleMergeProtos.Control topControl = bundleMergeControl("//x:x");
    BundleMergeProtos.Control nestedControl =
        Iterables.getOnlyElement(topControl.getNestedBundleList());
    assertThat(topControl.getBundleRoot()).isEqualTo(bundleDir);
    assertThat(nestedControl.getBundleRoot()).isEqualTo("bndl.bundle");
    assertThat(topControl.getBundleFileList()).doesNotContain(fooResource);
    assertThat(nestedControl.getBundleFileList()).contains(fooResource);

    ConfiguredTarget bndlTarget = getConfiguredTarget("//bndl:bndl", bundleConfiguration);
    Artifact bundlePlist = getBinArtifact("bndl-MergedInfo.plist", bndlTarget);
    assertThat(nestedControl.getBundleInfoPlistFile()).isEqualTo(bundlePlist.getExecPathString());

    Artifact actoolzipOutput = getBinArtifact("bndl.actool.zip", bndlTarget);
    assertThat(nestedControl.getMergeZipList())
        .containsExactly(MergeZip.newBuilder()
            .setEntryNamePrefix(bundleDir + "/bndl.bundle/")
            .setSourcePath(actoolzipOutput.getExecPathString())
            .build());

    assertThat(bundleMergeAction("//x:x").getInputs())
        .containsAllOf(getSourceArtifact("bndl/foo.data"), bundlePlist, actoolzipOutput);
  }

  protected SpawnAction bundleMergeAction(String target) throws Exception {
    Label targetLabel = Label.parseAbsolute(target);
    ConfiguredTarget binary = getConfiguredTarget(target);
    return (SpawnAction)
        getGeneratingAction(getBinArtifact(targetLabel.getName() + artifactName(".unprocessed.ipa"),
            binary));
  }

  protected void checkMergeBundleActionsWithNestedBundle(RuleType ruleType) throws Exception {
    scratch.file("bndl/BUILD",
        "objc_bundle_library(",
        "    name = 'bndl',",
        "    resources = ['foo.data'],",
        "    infoplist = 'bndl-Info.plist',",
        "    asset_catalogs = ['bar.xcassets/1'],",
        ")");
    ruleType.scratchTarget(scratch,
        "bundles", "['//bndl:bndl']");
    checkMergeBundleActionsWithNestedBundle(
        getBundlePathInsideIpa(ruleType), targetConfig);
  }

  // This checks that the proto bundling and grouping behavior works as expected. Grouping is based
  // on the proto_library targets, given that each proto_library is complete in its closure (all
  // the required deps are captured inside a proto_library).
  //
  // This particular tests sets up 3 proto groups, defined as [A, B], [B, C], [A, C, D]. The proto
  // grouping support detects that, for example, since A doesn't appear in all groups with B or C,
  // then it doesn't need any dependencies other than itself to be built. The same applies for B and
  // C, The same cannot be said about D, which only appears with A and C, so we have to assume that
  // D depends on A and C.
  //
  // These dependencies describe what the inputs will be to each of the generation/compilation
  // actions. Denoting {[in] -> [out]} as an action with "in" being the required inputs, and "out"
  // being the expected outputs, given the layout of the groups for this test, the actions should
  // be:
  //
  // {[A]       -> [A]}
  // {[B]       -> [B]}
  // {[C]       -> [C]}
  // {[A, C, D] -> [D]}
  //
  // This test ensures that, for example, to generate DataA.pbobjc.{h,m}, only data_a.proto should
  // be provided as an input, while the inputs to generate DataD.pbobjc.{h,m} should be
  // data_a.proto, data_c.proto and data_d.proto. The same applies for the compilation actions,
  // where the inputs are interpreted as .pbobjc.h files, and the output is a .pbobjc.o file.
  protected void checkProtoBundlingAndLinking(RuleType ruleType) throws Exception {
    scratch.file(
        "protos/BUILD",
        "proto_library(",
        "    name = 'protos_1',",
        "    srcs = ['data_a.proto', 'data_b.proto'],",
        ")",
        "proto_library(",
        "    name = 'protos_2',",
        "    srcs = ['data_b.proto', 'data_c.proto'],",
        ")",
        "proto_library(",
        "    name = 'protos_3',",
        "    srcs = ['data_c.proto', 'data_a.proto', 'data_d.proto'],",
        ")",
        "objc_proto_library(",
        "    name = 'objc_protos_a',",
        "    portable_proto_filters = ['filter_a.pbascii'],",
        "    deps = [':protos_1'],",
        ")",
        "objc_proto_library(",
        "    name = 'objc_protos_b',",
        "    portable_proto_filters = ['filter_b.pbascii'],",
        "    deps = [':protos_2', ':protos_3'],",
        ")");
    scratch.file(
        "libs/BUILD",
        "objc_library(",
        "    name = 'objc_lib',",
        "    srcs = ['a.m'],",
        "    deps = ['//protos:objc_protos_a', '//protos:objc_protos_b']",
        ")");

    ruleType.scratchTarget(
        scratch,
        "srcs", "['main.m']",
        "deps", "['//libs:objc_lib']",
        "defines", "['SHOULDNOTBEINPROTOS']",
        "copts", "['-ISHOULDNOTBEINPROTOS']");

    BuildConfiguration childConfig =
        Iterables.getOnlyElement(
            getSplitConfigurations(
                targetConfig,
                new MultiArchSplitTransitionProvider.AppleBinaryTransition(
                    PlatformType.IOS, Optional.<DottedVersion>absent())));

    ConfiguredTarget topTarget = getConfiguredTarget("//x:x", childConfig);

    assertObjcProtoProviderArtifactsArePropagated(topTarget);
    assertBundledGenerationActionsAreDifferent(topTarget);
    assertOnlyRequiredInputsArePresentForBundledGeneration(topTarget);
    assertOnlyRequiredInputsArePresentForBundledCompilation(topTarget);
    assertCoptsAndDefinesForBundlingTarget(topTarget);
    assertBundledGroupsGetCreatedAndLinked(topTarget);
    if (getObjcCrosstoolMode() == ObjcCrosstoolMode.ALL) {
      assertBundledCompilationUsesCrosstool(topTarget);
    }
  }

  protected ImmutableList<Artifact> getAllObjectFilesLinkedInBin(Artifact bin) {
    ImmutableList.Builder<Artifact> objects = ImmutableList.builder();
    CommandAction binAction = (CommandAction) getGeneratingAction(bin);
    for (Artifact binActionArtifact : binAction.getInputs()) {
      if (binActionArtifact.getRootRelativePath().getPathString().endsWith(".a")) {
        CommandAction linkAction = (CommandAction) getGeneratingAction(binActionArtifact);
        for (Artifact linkActionArtifact : linkAction.getInputs()) {
          if (linkActionArtifact.getRootRelativePath().getPathString().endsWith(".o")) {
            objects.add(linkActionArtifact);
          }
        }
      }
    }
    return objects.build();
  }

  private void assertObjcProtoProviderArtifactsArePropagated(ConfiguredTarget topTarget)
      throws Exception {
    ConfiguredTarget libTarget =
        view.getPrerequisiteConfiguredTargetForTesting(
            reporter, topTarget, Label.parseAbsoluteUnchecked("//libs:objc_lib"), masterConfig);

    ObjcProtoProvider protoProvider = libTarget.getProvider(ObjcProtoProvider.class);
    assertThat(protoProvider).isNotNull();
    assertThat(protoProvider.getProtoGroups().toSet()).hasSize(3);
    assertThat(
            Artifact.toExecPaths(
                ImmutableSet.copyOf(Iterables.concat(protoProvider.getProtoGroups()))))
        .containsExactly(
            "protos/data_a.proto",
            "protos/data_b.proto",
            "protos/data_c.proto",
            "protos/data_d.proto");
    assertThat(Artifact.toExecPaths(protoProvider.getPortableProtoFilters()))
        .containsExactly("protos/filter_a.pbascii", "protos/filter_b.pbascii");
  }

  private void assertBundledGenerationActionsAreDifferent(ConfiguredTarget topTarget) {
    Artifact protoHeaderA = getBinArtifact("_generated_protos/x/protos/DataA.pbobjc.h", topTarget);
    Artifact protoHeaderB = getBinArtifact("_generated_protos/x/protos/DataB.pbobjc.h", topTarget);
    Artifact protoHeaderC = getBinArtifact("_generated_protos/x/protos/DataC.pbobjc.h", topTarget);
    Artifact protoHeaderD = getBinArtifact("_generated_protos/x/protos/DataD.pbobjc.h", topTarget);
    CommandAction protoActionA = (CommandAction) getGeneratingAction(protoHeaderA);
    CommandAction protoActionB = (CommandAction) getGeneratingAction(protoHeaderB);
    CommandAction protoActionC = (CommandAction) getGeneratingAction(protoHeaderC);
    CommandAction protoActionD = (CommandAction) getGeneratingAction(protoHeaderD);
    assertThat(protoActionA).isNotNull();
    assertThat(protoActionB).isNotNull();
    assertThat(protoActionC).isNotNull();
    assertThat(protoActionD).isNotNull();
    assertThat(protoActionA).isNotEqualTo(protoActionB);
    assertThat(protoActionB).isNotEqualTo(protoActionC);
    assertThat(protoActionC).isNotEqualTo(protoActionD);
  }

  private void assertOnlyRequiredInputsArePresentForBundledGeneration(ConfiguredTarget topTarget)
      throws Exception {
    ConfiguredTarget libTarget =
        view.getPrerequisiteConfiguredTargetForTesting(
            reporter, topTarget, Label.parseAbsoluteUnchecked("//libs:objc_lib"), masterConfig);
    ObjcProtoProvider protoProvider = libTarget.getProvider(ObjcProtoProvider.class);

    Artifact protoHeaderA = getBinArtifact("_generated_protos/x/protos/DataA.pbobjc.h", topTarget);
    Artifact protoHeaderB = getBinArtifact("_generated_protos/x/protos/DataB.pbobjc.h", topTarget);
    Artifact protoHeaderC = getBinArtifact("_generated_protos/x/protos/DataC.pbobjc.h", topTarget);
    Artifact protoHeaderD = getBinArtifact("_generated_protos/x/protos/DataD.pbobjc.h", topTarget);

    CommandAction protoActionA = (CommandAction) getGeneratingAction(protoHeaderA);
    CommandAction protoActionB = (CommandAction) getGeneratingAction(protoHeaderB);
    CommandAction protoActionC = (CommandAction) getGeneratingAction(protoHeaderC);
    CommandAction protoActionD = (CommandAction) getGeneratingAction(protoHeaderD);

    assertThat(protoActionA.getInputs()).containsAllIn(protoProvider.getPortableProtoFilters());
    assertThat(protoActionB.getInputs()).containsAllIn(protoProvider.getPortableProtoFilters());
    assertThat(protoActionC.getInputs()).containsAllIn(protoProvider.getPortableProtoFilters());
    assertThat(protoActionD.getInputs()).containsAllIn(protoProvider.getPortableProtoFilters());

    assertThat(Artifact.toExecPaths(protoActionA.getInputs())).contains("protos/data_a.proto");
    assertThat(Artifact.toExecPaths(protoActionA.getInputs()))
        .containsNoneOf("protos/data_b.proto", "protos/data_c.proto", "protos/data_d.proto");

    assertThat(Artifact.toExecPaths(protoActionB.getInputs())).contains("protos/data_b.proto");
    assertThat(Artifact.toExecPaths(protoActionB.getInputs()))
        .containsNoneOf("protos/data_a.proto", "protos/data_c.proto", "protos/data_d.proto");

    assertThat(Artifact.toExecPaths(protoActionC.getInputs())).contains("protos/data_c.proto");
    assertThat(Artifact.toExecPaths(protoActionC.getInputs()))
        .containsNoneOf("protos/data_a.proto", "protos/data_b.proto", "protos/data_d.proto");

    assertThat(Artifact.toExecPaths(protoActionD.getInputs())).contains("protos/data_d.proto");
    assertThat(Artifact.toExecPaths(protoActionD.getInputs()))
        .containsAllOf("protos/data_a.proto", "protos/data_c.proto");
    assertThat(Artifact.toExecPaths(protoActionD.getInputs()))
        .doesNotContain("protos/data_b.proto");
  }

  /**
   * Ensures that all middleman artifacts in the action input are expanded so that the real inputs
   * are also included.
   */
  protected Iterable<Artifact> getExpandedActionInputs(Action action) {
    List<Artifact> containedArtifacts = new ArrayList<>();
    for (Artifact input : action.getInputs()) {
      if (input.isMiddlemanArtifact()) {
        Action middlemanAction = getGeneratingAction(input);
        Iterables.addAll(containedArtifacts, getExpandedActionInputs(middlemanAction));
      }
      containedArtifacts.add(input);
    }
    return containedArtifacts;
  }

  private void assertOnlyRequiredInputsArePresentForBundledCompilation(ConfiguredTarget topTarget) {
    Artifact protoHeaderA = getBinArtifact("_generated_protos/x/protos/DataA.pbobjc.h", topTarget);
    Artifact protoHeaderB = getBinArtifact("_generated_protos/x/protos/DataB.pbobjc.h", topTarget);
    Artifact protoHeaderC = getBinArtifact("_generated_protos/x/protos/DataC.pbobjc.h", topTarget);
    Artifact protoHeaderD = getBinArtifact("_generated_protos/x/protos/DataD.pbobjc.h", topTarget);

    Artifact protoObjectA =
        getBinArtifact("_objs/x/x/_generated_protos/x/protos/DataA.pbobjc.o", topTarget);
    Artifact protoObjectB =
        getBinArtifact("_objs/x/x/_generated_protos/x/protos/DataB.pbobjc.o", topTarget);
    Artifact protoObjectC =
        getBinArtifact("_objs/x/x/_generated_protos/x/protos/DataC.pbobjc.o", topTarget);
    Artifact protoObjectD =
        getBinArtifact("_objs/x/x/_generated_protos/x/protos/DataD.pbobjc.o", topTarget);

    CommandAction protoObjectActionA = (CommandAction) getGeneratingAction(protoObjectA);
    CommandAction protoObjectActionB = (CommandAction) getGeneratingAction(protoObjectB);
    CommandAction protoObjectActionC = (CommandAction) getGeneratingAction(protoObjectC);
    CommandAction protoObjectActionD = (CommandAction) getGeneratingAction(protoObjectD);

    assertThat(protoObjectActionA).isNotNull();
    assertThat(protoObjectActionB).isNotNull();
    assertThat(protoObjectActionC).isNotNull();
    assertThat(protoObjectActionD).isNotNull();

    assertThat(getExpandedActionInputs(protoObjectActionA))
        .containsNoneOf(protoHeaderB, protoHeaderC, protoHeaderD);
    assertThat(getExpandedActionInputs(protoObjectActionB))
        .containsNoneOf(protoHeaderA, protoHeaderC, protoHeaderD);
    assertThat(getExpandedActionInputs(protoObjectActionC))
        .containsNoneOf(protoHeaderA, protoHeaderB, protoHeaderD);
    assertThat(getExpandedActionInputs(protoObjectActionD))
        .containsAllOf(protoHeaderA, protoHeaderC, protoHeaderD);
    assertThat(getExpandedActionInputs(protoObjectActionD))
        .doesNotContain(protoHeaderB);
  }

  private void assertCoptsAndDefinesForBundlingTarget(ConfiguredTarget topTarget) throws Exception {
    Artifact protoObject =
        getBinArtifact("_objs/x/x/_generated_protos/x/protos/DataA.pbobjc.o", topTarget);
    CommandAction protoObjectAction = (CommandAction) getGeneratingAction(protoObject);
    assertThat(protoObjectAction).isNotNull();
    assertThat(protoObjectAction.getArguments())
        .containsNoneOf("-DSHOULDNOTBEINPROTOS", "-ISHOULDNOTBEINPROTOS");

    Artifact binLib = getBinArtifact("libx.a", topTarget);
    CommandAction binLibAction = (CommandAction) getGeneratingAction(binLib);
    assertThat(binLibAction).isNotNull();

    Artifact binSrcObject = getFirstArtifactEndingWith(binLibAction.getInputs(), "main.o");
    CommandAction binSrcObjectAction = (CommandAction) getGeneratingAction(binSrcObject);
    assertThat(binSrcObjectAction).isNotNull();
    assertThat(binSrcObjectAction.getArguments())
        .containsAllOf("-DSHOULDNOTBEINPROTOS", "-ISHOULDNOTBEINPROTOS");
  }

  private void assertBundledGroupsGetCreatedAndLinked(ConfiguredTarget topTarget) {
    Artifact protosGroup0Lib = getBinArtifact("libx_BundledProtos_0.a", topTarget);
    Artifact protosGroup1Lib = getBinArtifact("libx_BundledProtos_1.a", topTarget);
    Artifact protosGroup2Lib = getBinArtifact("libx_BundledProtos_2.a", topTarget);
    Artifact protosGroup3Lib = getBinArtifact("libx_BundledProtos_3.a", topTarget);

    CommandAction protosLib0Action = (CommandAction) getGeneratingAction(protosGroup0Lib);
    CommandAction protosLib1Action = (CommandAction) getGeneratingAction(protosGroup1Lib);
    CommandAction protosLib2Action = (CommandAction) getGeneratingAction(protosGroup2Lib);
    CommandAction protosLib3Action = (CommandAction) getGeneratingAction(protosGroup3Lib);
    assertThat(protosLib0Action).isNotNull();
    assertThat(protosLib1Action).isNotNull();
    assertThat(protosLib2Action).isNotNull();
    assertThat(protosLib3Action).isNotNull();

    Artifact bin = getBinArtifact("x_bin", topTarget);
    CommandAction binAction = (CommandAction) getGeneratingAction(bin);
    assertThat(binAction.getInputs())
        .containsAllOf(protosGroup0Lib, protosGroup1Lib, protosGroup2Lib, protosGroup3Lib);
  }

  private void assertBundledCompilationUsesCrosstool(ConfiguredTarget topTarget) {
    Artifact protoObjectA =
        getBinArtifact("_objs/x/x/_generated_protos/x/protos/DataA.pbobjc.o", topTarget);
    Artifact protoObjectB =
        getBinArtifact("_objs/x/x/_generated_protos/x/protos/DataB.pbobjc.o", topTarget);
    Artifact protoObjectC =
        getBinArtifact("_objs/x/x/_generated_protos/x/protos/DataC.pbobjc.o", topTarget);
    Artifact protoObjectD =
        getBinArtifact("_objs/x/x/_generated_protos/x/protos/DataD.pbobjc.o", topTarget);

    assertThat(getGeneratingAction(protoObjectA)).isInstanceOf(CppCompileAction.class);
    assertThat(getGeneratingAction(protoObjectB)).isInstanceOf(CppCompileAction.class);
    assertThat(getGeneratingAction(protoObjectC)).isInstanceOf(CppCompileAction.class);
    assertThat(getGeneratingAction(protoObjectD)).isInstanceOf(CppCompileAction.class);
  }

  protected void checkProtoBundlingDoesNotHappen(RuleType ruleType) throws Exception {
    scratch.file(
        "protos/BUILD",
        "proto_library(",
        "    name = 'protos',",
        "    srcs = ['data_a.proto'],",
        ")",
        "objc_proto_library(",
        "    name = 'objc_protos',",
        "    portable_proto_filters = ['filter_b.pbascii'],",
        "    deps = [':protos'],",
        ")");
    scratch.file(
        "libs/BUILD",
        "objc_library(",
        "    name = 'objc_lib',",
        "    srcs = ['a.m'],",
        "    deps = ['//protos:objc_protos']",
        ")");

    ruleType.scratchTarget(
        scratch,
        "srcs", "['main.m']",
        "deps", "['//libs:objc_lib']");

    ConfiguredTarget topTarget = getConfiguredTarget("//x:x");
    Artifact protoHeader = getBinArtifact("_generated_protos/x/protos/DataA.pbobjc.h", topTarget);
    CommandAction protoAction = (CommandAction) getGeneratingAction(protoHeader);
    assertThat(protoAction).isNull();
  }

  protected void checkProtoBundlingWithTargetsWithNoDeps(RuleType ruleType) throws Exception {
    scratch.file(
        "protos/BUILD",
        "proto_library(",
        "    name = 'protos_a',",
        "    srcs = ['data_a.proto'],",
        ")",
        "objc_proto_library(",
        "    name = 'objc_protos_a',",
        "    portable_proto_filters = ['filter_a.pbascii'],",
        "    deps = [':protos_a'],",
        ")");
    scratch.file(
        "libs/BUILD",
        "objc_library(",
        "    name = 'objc_lib',",
        "    srcs = ['a.m'],",
        "    deps = ['//protos:objc_protos_a', ':no_deps_target'],",
        ")",
        "objc_framework(",
        "    name = 'no_deps_target',",
        "    framework_imports = ['x.framework'],",
        ")");

    ruleType.scratchTarget(scratch, "deps", "['//libs:objc_lib']");

    ConfiguredTarget topTarget = getConfiguredTarget("//x:x");

    ConfiguredTarget libTarget =
        view.getPrerequisiteConfiguredTargetForTesting(
            reporter, topTarget, Label.parseAbsoluteUnchecked("//libs:objc_lib"), masterConfig);

    ObjcProtoProvider protoProvider = libTarget.getProvider(ObjcProtoProvider.class);
    assertThat(protoProvider).isNotNull();
  }

  protected void checkFrameworkDepLinkFlags(RuleType ruleType,
      ExtraLinkArgs extraLinkArgs) throws Exception {
    scratch.file(
        "libs/BUILD",
        "objc_library(",
        "    name = 'objc_lib',",
        "    srcs = ['a.m'],",
        "    deps = [':my_framework'],",
        ")",
        "objc_framework(",
        "    name = 'my_framework',",
        "    framework_imports = ['buzzbuzz.framework'],",
        ")");

    ruleType.scratchTarget(scratch, "deps", "['//libs:objc_lib']");

    CommandAction linkAction = linkAction("//x:x");
    Artifact binArtifact = getFirstArtifactEndingWith(linkAction.getOutputs(), "x_bin");
    Artifact objList = getFirstArtifactEndingWith(linkAction.getInputs(), "x-linker.objlist");

    verifyLinkAction(
        binArtifact,
        objList,
        "x86_64",
        ImmutableList.of("x/libx.a", "libobjc_lib.a"),
        ImmutableList.of(PathFragment.create("libs/buzzbuzz")),
        extraLinkArgs);
  }

  protected void checkBundleLoaderIsCorrectlyPassedToTheLinker(RuleType ruleType) throws Exception {
    scratch.file("bin/BUILD",
        "apple_binary(",
        "    name = 'bin',",
        "    srcs = ['a.m'],",
        "    platform_type = 'ios',",
        ")");

    ruleType.scratchTarget(scratch, "binary_type", "'loadable_bundle'", "bundle_loader",
        "'//bin:bin'");
    ConfiguredTarget binTarget = getConfiguredTarget("//bin:bin");

    CommandAction linkAction = linkAction("//x:x");
    assertThat(Joiner.on(" ").join(linkAction.getArguments()))
        .contains("-bundle_loader " + getBinArtifact("bin_lipobin", binTarget).getExecPath());
    assertThat(Joiner.on(" ").join(linkAction.getArguments()))
        .contains("-Xlinker -rpath -Xlinker @loader_path/Frameworks");
  }

  /**
   * @param bundleConfiguration the configuration in which the bundle is expected to be executed
   */
  protected void checkMergeBundleActionsWithNestedBundle(BinaryRuleTypePair ruleTypePair,
      BuildConfiguration bundleConfiguration) throws Exception {
    scratch.file("bndl/BUILD",
        "objc_bundle_library(",
        "    name = 'bndl',",
        "    resources = ['foo.data'],",
        "    infoplist = 'bndl-Info.plist',",
        "    asset_catalogs = ['bar.xcassets/1'],",
        ")");
    ruleTypePair.scratchTargets(scratch,
        "bundles", "['//bndl:bndl']");
    checkMergeBundleActionsWithNestedBundle(ruleTypePair.getBundleDir(), bundleConfiguration);
  }

  protected Action lipoLibAction(String libLabel) throws Exception {
    return actionProducingArtifact(libLabel, "_lipo.a");
  }

  protected Action lipoBinAction(String binLabel) throws Exception {
    return actionProducingArtifact(binLabel, "_lipobin");
  }

  protected CommandAction linkAction(String binLabel) throws Exception {
    CommandAction linkAction = (CommandAction) actionProducingArtifact(binLabel, "_bin");
    if (linkAction == null) {
      // For multi-architecture rules, the link action is not in the target configuration, but
      // across a configuration transition.
      Action lipoAction = lipoBinAction(binLabel);
      if (lipoAction != null) {
        Artifact binArtifact = getFirstArtifactEndingWith(lipoAction.getInputs(), "_bin");
        linkAction = (CommandAction) getGeneratingAction(binArtifact);
      }
    }
    return linkAction;
  }

  protected CommandAction linkLibAction(String libLabel) throws Exception {
    CommandAction linkAction = (CommandAction) actionProducingArtifact(libLabel, "-fl.a");

    if (linkAction == null) {
      // For multi-architecture rules, the link action is not in the target configuration, but
      // across a configuration transition.
      Action lipoAction = lipoLibAction(libLabel);
      if (lipoAction != null) {
        Artifact binArtifact = getFirstArtifactEndingWith(lipoAction.getInputs(), "-fl.a");
        linkAction = (CommandAction) getGeneratingAction(binArtifact);
      }
    }
    return linkAction;
  }

  protected Action actionProducingArtifact(String targetLabel,
      String artifactSuffix) throws Exception {
    ConfiguredTarget libraryTarget = getConfiguredTarget(targetLabel);
    Label parsedLabel = Label.parseAbsolute(targetLabel);
    Artifact linkedLibrary = getBinArtifact(
        parsedLabel.getName() + artifactSuffix,
        libraryTarget);
    return getGeneratingAction(linkedLibrary);
  }

  protected void addTargetWithAssetCatalogs(RuleType ruleType) throws Exception {
    scratch.file("x/foo.xcassets/foo");
    scratch.file("x/bar.xcassets/bar");
    ruleType.scratchTarget(scratch,
        "asset_catalogs", "['foo.xcassets/foo', 'bar.xcassets/bar']");
  }

  /**
   * Checks that a target at {@code //x:x}, which is already created, registered a correct actool
   * action based on the given targetDevice and platform, setting certain arbitrary and default
   * values.
   */
  protected void checkActoolActionCorrectness(DottedVersion minimumOsVersion, String targetDevice,
      String platform) throws Exception {
    Artifact actoolZipOut = getBinArtifact("x" + artifactName(".actool.zip"),
        getConfiguredTarget("//x:x"));
    Artifact actoolPartialInfoplist =
        getBinArtifact("x" + artifactName(".actool-PartialInfo.plist"), "//x:x");
    SpawnAction actoolZipAction = (SpawnAction) getGeneratingAction(actoolZipOut);
    assertThat(actoolZipAction.getArguments())
        .containsExactly(
            MOCK_ACTOOLWRAPPER_PATH,
            actoolZipOut.getExecPathString(),
            "--platform", platform,
            "--output-partial-info-plist", actoolPartialInfoplist.getExecPathString(),
            "--minimum-deployment-target", minimumOsVersion.toString(),
            "--target-device", targetDevice,
            "x/foo.xcassets", "x/bar.xcassets")
        .inOrder();
    assertRequiresDarwin(actoolZipAction);

    assertThat(Artifact.toExecPaths(actoolZipAction.getInputs()))
        .containsExactly(
            "x/foo.xcassets/foo",
            "x/bar.xcassets/bar",
            MOCK_ACTOOLWRAPPER_PATH);
    assertThat(Artifact.toExecPaths(actoolZipAction.getOutputs()))
        .containsExactly(
            actoolZipOut.getExecPathString(),
            actoolPartialInfoplist.getExecPathString());
  }

  /**
   * Checks that a target at {@code //x:x}, which is already created, registered a correct actool
   * action based on certain arbitrary and default values for iphone simulator.
   */
  protected void checkActoolActionCorrectness(DottedVersion minimumOsVersion) throws Exception {
    checkActoolActionCorrectness(minimumOsVersion, "iphone", "iphonesimulator");
  }

  protected void checkAssetCatalogAttributeError(RuleType ruleType, String attribute)
      throws Exception {
    checkAssetCatalogAttributeError(ruleType, attribute, INFOPLIST_ATTR, "'pl.plist'");
  }

  protected void checkAssetCatalogAttributeError(RuleType ruleType, String attribute,
      String infoplistAttribute, String infoPlists) throws Exception {
    scratch.file("x/pl.plist");
    checkError("x", "x", String.format(NO_ASSET_CATALOG_ERROR_FORMAT, "3.1415926"),
        ruleType.target(scratch, "x", "x",
            infoplistAttribute, infoPlists,
            attribute, "'3.1415926'"));
  }

  protected SpawnAction actoolZipActionForIpa(String target) throws Exception {
    Artifact binActoolZipOut =
        getFirstArtifactEndingWith(bundleMergeAction(target).getInputs(), ".actool.zip");
    return (SpawnAction) getGeneratingAction(binActoolZipOut);
  }

  /**
   * Checks that a target at {@code //x:x}, which is already created, registered an actool with
   * correct arguments based on certain arbitrary and default values.
   */
  private void checkActoolZipInvocationCorrectness(DottedVersion minimumOsVersion)
      throws Exception {
    SpawnAction actoolZipAction = actoolZipActionForIpa("//x:x");
    assertThat(actoolZipAction.getArguments())
        .containsExactly(
            MOCK_ACTOOLWRAPPER_PATH,
            execPathEndingWith(actoolZipAction.getOutputs(), "x.actool.zip"),
            "--platform", "iphonesimulator",
            "--output-partial-info-plist",
                execPathEndingWith(actoolZipAction.getOutputs(), "actool-PartialInfo.plist"),
            "--minimum-deployment-target", minimumOsVersion.toString(),
            "--target-device", "iphone",
            "lib/ac.xcassets",
            "--app-icon", "foo",
            "--launch-image", "bar")
        .inOrder();
  }

  protected void checkSpecifyAppIconAndLaunchImageUsingXcassetsOfDependency(RuleType ruleType)
      throws Exception {
    createLibraryTargetWriter("//lib:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setList("asset_catalogs", "ac.xcassets/foo")
        .write();
    ruleType.scratchTarget(scratch,
        "srcs", "['src.m']",
        "deps", "['//lib:lib']",
        APP_ICON_ATTR, "'foo'",
        LAUNCH_IMAGE_ATTR, "'bar'");
    checkActoolZipInvocationCorrectness(DEFAULT_IOS_SDK_VERSION);
  }

  protected void checkSpecifyAppIconAndLaunchImageUsingXcassetsOfDependency(
      BinaryRuleTypePair ruleTypePair, DottedVersion minimumOsVersion) throws Exception {
    createLibraryTargetWriter("//lib:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setList("asset_catalogs", "ac.xcassets/foo")
        .write();
    ruleTypePair.scratchTargets(scratch,
        "deps", "['//lib:lib']",
        APP_ICON_ATTR, "'foo'",
        LAUNCH_IMAGE_ATTR, "'bar'");
    checkActoolZipInvocationCorrectness(minimumOsVersion);
  }

  /**
   * Verifies that targeted device family information is passed to actoolzip for the given targeted
   * families.
   *
   * @param packageName where to place the rule during testing - this should be different every time
   *     the method is invoked
   * @param buildFileContents contents of the BUILD file for the {@code packageName} package
   * @param targetDevices the values to {@code --target-device} expected in the actoolzip invocation
   */
  private void checkPassesFamiliesToActool(String packageName, String buildFileContents,
      String... targetDevices) throws Exception {
    scratch.file(String.format("%s/BUILD", packageName), buildFileContents);
    ConfiguredTarget target = getConfiguredTarget(String.format("//%s:x", packageName));
    Artifact actoolZipOut = getBinArtifact("x.actool.zip", target);
    SpawnAction actoolZipAction = (SpawnAction) getGeneratingAction(actoolZipOut);

    List<String> arguments = actoolZipAction.getArguments();
    for (String targetDevice : targetDevices) {
      assertContainsSublist(arguments, ImmutableList.of("--target-device", targetDevice));
    }

    assertWithMessage("Incorrect number of --target-device flags in arguments [" + arguments + "]")
        .that(Collections.frequency(arguments, "--target-device"))
        .isEqualTo(targetDevices.length);
  }

  private void checkPassesFamiliesToActool(RuleType ruleType, String packageName,
      String familiesAttribute, String... actoolFlags) throws Exception {
    String buildFileContents = ruleType.target(scratch, packageName, "x",
        FAMILIES_ATTR, familiesAttribute,
        "asset_catalogs", "['foo.xcassets/1']");
    checkPassesFamiliesToActool(packageName, buildFileContents, actoolFlags);
  }

  protected void checkPassesFamiliesToActool(RuleType ruleType) throws Exception {
    checkPassesFamiliesToActool(ruleType, "iphone", "['iphone']", "iphone");
    checkPassesFamiliesToActool(ruleType, "ipad", "['ipad']", "ipad");
    checkPassesFamiliesToActool(ruleType, "both", "['iphone', 'ipad']", "ipad", "iphone");
    checkPassesFamiliesToActool(ruleType, "both_reverse", "['ipad', 'iphone']", "ipad", "iphone");
  }

  private void checkPassesFamiliesToActool(BinaryRuleTypePair ruleTypePair, String packageName,
      String familiesAttribute, String families, String... actoolFlags) throws Exception {
    String buildFileContents = ruleTypePair.targets(scratch, packageName,
        "asset_catalogs", "['foo.xcassets/1']",
        familiesAttribute, families);
    checkPassesFamiliesToActool(packageName, buildFileContents, actoolFlags);
  }

  protected void checkPassesFamiliesToActool(BinaryRuleTypePair ruleTypePair) throws Exception {
    checkPassesFamiliesToActool(ruleTypePair, FAMILIES_ATTR);
  }

  protected void checkPassesFamiliesToActool(BinaryRuleTypePair ruleTypePair,
      String familiesAttribute) throws Exception {
    checkPassesFamiliesToActool(ruleTypePair, "iphone", familiesAttribute, "['iphone']", "iphone");
    checkPassesFamiliesToActool(ruleTypePair, "ipad", familiesAttribute, "['ipad']", "ipad");
    checkPassesFamiliesToActool(ruleTypePair, "both", familiesAttribute, "['iphone', 'ipad']",
        "ipad", "iphone");
    checkPassesFamiliesToActool(ruleTypePair, "both_reverse", familiesAttribute,
        "['ipad', 'iphone']", "ipad", "iphone");
  }

  /**
   * Verifies that targeted device family information is passed to ibtool for the given targeted
   * families.
   *
   * @param packageName where to place the rule during testing - this should be different every time
   *     the method is invoked
   * @param buildFileContents contents of the BUILD file for the {@code packageName} package
   * @param targetDevices the values to {@code --target-device} expected in the ibtool invocation
   */
  private void checkPassesFamiliesToIbtool(String packageName, String buildFileContents,
      String... targetDevices) throws Exception {
    scratch.file(String.format("%s/BUILD", packageName), buildFileContents);
    ConfiguredTarget target = getConfiguredTarget(String.format("//%s:x", packageName));

    Artifact storyboardZipOut = getBinArtifact("x/foo.storyboard.zip", target);
    SpawnAction storyboardZipAction = (SpawnAction) getGeneratingAction(storyboardZipOut);

    List<String> arguments = storyboardZipAction.getArguments();
    for (String targetDevice : targetDevices) {
      assertContainsSublist(arguments, ImmutableList.of("--target-device", targetDevice));
    }

    assertWithMessage("Incorrect number of --target-device flags in arguments [" + arguments + "]")
        .that(Collections.frequency(arguments, "--target-device"))
        .isEqualTo(targetDevices.length);
  }

  private void checkPassesFamiliesToIbtool(RuleType ruleType, String packageName,
      String families, String... targetDevices) throws Exception {
    String buildFileContents = ruleType.target(scratch, packageName, "x",
        FAMILIES_ATTR, families,
        "storyboards", "['foo.storyboard']");
    checkPassesFamiliesToIbtool(packageName, buildFileContents, targetDevices);
  }

  protected void checkPassesFamiliesToIbtool(RuleType ruleType) throws Exception {
    checkPassesFamiliesToIbtool(ruleType, "iphone", "['iphone']", "iphone");
    checkPassesFamiliesToIbtool(ruleType, "ipad", "['ipad']", "ipad");
    checkPassesFamiliesToIbtool(ruleType, "both", "['iphone', 'ipad']",
        "ipad", "iphone");
    checkPassesFamiliesToIbtool(ruleType, "both_reverse", "['ipad', 'iphone']",
       "ipad", "iphone");
  }

  private void checkPassesFamiliesToIbtool(BinaryRuleTypePair ruleTypePair, String packageName,
      String familyAttribute, String families, String... targetDevices) throws Exception {
    String buildFileContents = ruleTypePair.targets(scratch, packageName,
        familyAttribute, families,
        "storyboards", "['foo.storyboard']");
    checkPassesFamiliesToIbtool(packageName, buildFileContents, targetDevices);
  }

  protected void checkPassesFamiliesToIbtool(BinaryRuleTypePair ruleTypePair) throws Exception {
    checkPassesFamiliesToIbtool(ruleTypePair, FAMILIES_ATTR);
  }

  protected void checkPassesFamiliesToIbtool(BinaryRuleTypePair ruleTypePair,
      String familyAttribute) throws Exception {
    checkPassesFamiliesToIbtool(ruleTypePair, "iphone", familyAttribute, "['iphone']",
        "iphone");
    checkPassesFamiliesToIbtool(ruleTypePair, "ipad", familyAttribute, "['ipad']",
        "ipad");
    checkPassesFamiliesToIbtool(ruleTypePair, "both", familyAttribute,
        "['iphone', 'ipad']", "ipad", "iphone");
    checkPassesFamiliesToIbtool(ruleTypePair, "both_reverse", familyAttribute,
        "['ipad', 'iphone']", "ipad", "iphone");
  }

  private void checkReportsErrorsForInvalidFamiliesAttribute(
      RuleType ruleType, String packageName, String familyAttribute, String families)
          throws Exception {
    checkError(packageName, "x", ReleaseBundling.INVALID_FAMILIES_ERROR,
        ruleType.target(scratch, packageName, "x", familyAttribute, families));
  }

  protected void checkReportsErrorsForInvalidFamiliesAttribute(RuleType ruleType)
      throws Exception {
    checkReportsErrorsForInvalidFamiliesAttribute(ruleType, FAMILIES_ATTR);
  }

  protected void checkReportsErrorsForInvalidFamiliesAttribute(RuleType ruleType,
      String familyAttribute) throws Exception {
    checkReportsErrorsForInvalidFamiliesAttribute(ruleType, "a", familyAttribute, "['foo']");
    checkReportsErrorsForInvalidFamiliesAttribute(ruleType, "b", familyAttribute, "[]");
    checkReportsErrorsForInvalidFamiliesAttribute(ruleType, "c", familyAttribute,
        "['iphone', 'ipad', 'iphone']");
    checkReportsErrorsForInvalidFamiliesAttribute(ruleType, "d", familyAttribute,
        "['iphone', 'bar']");
  }

  /**
   * @param extraAttributes individual strings which contain a whole attribute to be added to the
   *     generated target, e.g. "deps = ['foo']"
   */
  protected void addBinAndLibWithResources(
      String attributeName,
      String libFile,
      String binFile,
      String binaryType,
      String... extraAttributes)
      throws Exception {
    scratch.file("lib/" + libFile);

    createLibraryTargetWriter("//lib:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .set(attributeName, "['" + libFile + "']")
        .write();

    scratch.file("bin/" + binFile);
    scratch.file(
        "bin/BUILD",
        binaryType + "(",
        "    name = 'bin',",
        "    srcs = ['src.m'],",
        "    deps = ['//lib:lib'],",
        "    " + attributeName + " = ['" + binFile + "'],",
        Joiner.on(',').join(extraAttributes),
        ")");
  }

  protected void checkCollectsResourceFilesTransitively(
      String targetLabel,
      Collection<String> binBundleMergeInputs,
      Collection<String> libBundleMergeInputs,
      ImmutableSetMultimap<String, Multiset<String>> filesByTarget)
      throws Exception {
    Action mergeBundleAction = bundleMergeAction(targetLabel);

    assertThat(Artifact.toRootRelativePaths(mergeBundleAction.getInputs()))
        .containsAllIn(binBundleMergeInputs);
  }

  protected void checkLinksDylibsTransitively(RuleType ruleType)
      throws Exception {
    createLibraryTargetWriter("//lib:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setList("sdk_dylibs", "libdy1", "libdy2")
        .write();
    ruleType.scratchTarget(scratch,
        "sdk_dylibs", "['libdy3']",
        "deps", "['//lib:lib']");

    CommandAction action = linkAction("//x:x");
    assertThat(Joiner.on(" ").join(action.getArguments())).contains("-ldy1 -ldy2 -ldy3");
  }

  protected BinaryFileWriteAction bundleMergeControlAction(String binaryLabelString)
      throws Exception {
    Label binaryLabel = Label.parseAbsolute(binaryLabelString);
    ConfiguredTarget binary = getConfiguredTarget(binaryLabelString);
    return (BinaryFileWriteAction) getGeneratingAction(
        getBinArtifact(binaryLabel.getName() + artifactName(".ipa-control"), binary));
  }

  protected BundleMergeProtos.Control bundleMergeControl(String binaryLabel)
      throws Exception {
    try (InputStream in = bundleMergeControlAction(binaryLabel).getSource()
        .openStream()) {
      return BundleMergeProtos.Control.parseFrom(in);
    }
  }

  protected void checkNoDebugSymbolFileWithoutAppleFlag(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch, "srcs", "['a.m']");
    ConfiguredTarget target = getConfiguredTarget("//x:x");

    Artifact plistArtifact = getBinArtifact("bin.app.dSYM/Contents/Info.plist", target);
    Artifact debugSymbolArtifact =
        getBinArtifact("bin.app.dSYM/Contents/Resources/DWARF/bin", target);
    CommandAction plistAction = (CommandAction) getGeneratingAction(plistArtifact);
    CommandAction debugSymbolAction = (CommandAction) getGeneratingAction(debugSymbolArtifact);
    CommandAction linkAction = (CommandAction) getGeneratingAction(getBinArtifact("x_bin", target));

    assertThat(linkAction.getArguments().get(2)).doesNotContain(DSYMUTIL);
    assertThat(plistAction).isNull();
    assertThat(debugSymbolAction).isNull();
  }

  protected ConfiguredTarget createTargetWithStoryboards(RuleType ruleType) throws Exception {
    scratch.file("x/1.storyboard");
    scratch.file("x/2.storyboard");
    scratch.file("x/subdir_for_no_reason/en.lproj/loc.storyboard");
    scratch.file("x/ja.lproj/loc.storyboard");
    ruleType.scratchTarget(scratch, "storyboards", "glob(['*.storyboard', '**/*.storyboard'])");
    return getConfiguredTarget("//x:x");
  }

  private ConfiguredTarget createTargetWithStoryboards(BinaryRuleTypePair ruleTypePair)
      throws Exception {
    scratch.file("x/1.storyboard");
    scratch.file("x/2.storyboard");
    scratch.file("x/subdir_for_no_reason/en.lproj/loc.storyboard");
    scratch.file("x/ja.lproj/loc.storyboard");
    ruleTypePair.scratchTargets(scratch, "storyboards",
        "glob(['*.storyboard', '**/*.storyboard'])");
    return getConfiguredTarget("//x:x");
  }

  private ConfiguredTarget createTargetWithSwift(RuleType ruleType) throws Exception {
    scratch.file("x/main.m");

    scratch.file("examples/rule/BUILD");
    scratch.file(
        "examples/rule/apple_rules.bzl",
        "def swift_rule_impl(ctx):",
        "  return struct(objc=apple_common.new_objc_provider(uses_swift=True))",
        "swift_rule = rule(implementation = swift_rule_impl, attrs = {})");

    scratch.file(
        "x/BUILD",
        "load('//examples/rule:apple_rules.bzl', 'swift_rule')",
        "swift_rule(name='swift_bin')",
        ruleType.getRuleTypeName() + "(",
        "    name = 'x',",
        "    srcs = ['main.m'],",
        "    deps = [':swift_bin'],",
        ")");

    return getConfiguredTarget("//x:x");
  }

  protected void checkProvidesStoryboardObjects(RuleType ruleType) throws Exception {
    useConfiguration();
    createTargetWithStoryboards(ruleType);
    ObjcProvider provider = providerForTarget("//x:x");
    ImmutableList<Artifact> storyboardInputs = ImmutableList.of(
        getSourceArtifact("x/1.storyboard"),
        getSourceArtifact("x/2.storyboard"),
        getSourceArtifact("x/subdir_for_no_reason/en.lproj/loc.storyboard"),
        getSourceArtifact("x/ja.lproj/loc.storyboard"));

    assertThat(provider.get(STORYBOARD))
        .containsExactlyElementsIn(storyboardInputs);
  }

  protected void checkRegistersStoryboardCompileActions(
      BinaryRuleTypePair ruleTypePair, DottedVersion minimumOsVersion,
      String platformName) throws Exception {
    checkRegistersStoryboardCompileActions(
        createTargetWithStoryboards(ruleTypePair), minimumOsVersion,
        ImmutableList.of(platformName));
  }

  protected void checkRegistersStoryboardCompileActions(RuleType ruleType,
      String platformName) throws Exception {
    checkRegistersStoryboardCompileActions(
        createTargetWithStoryboards(ruleType), DEFAULT_IOS_SDK_VERSION,
        ImmutableList.of(platformName));
  }

  private void checkRegistersStoryboardCompileActions(
      ConfiguredTarget target, DottedVersion minimumOsVersion, ImmutableList<String> targetDevices)
      throws Exception {
    Artifact storyboardZip = getBinArtifact("x/1.storyboard.zip", target);
    CommandAction compileAction = (CommandAction) getGeneratingAction(storyboardZip);
    assertThat(Artifact.toExecPaths(compileAction.getInputs()))
        .containsExactly(MOCK_IBTOOLWRAPPER_PATH, "x/1.storyboard");
    String archiveRoot = targetDevices.contains("watch") ? "." : "1.storyboardc";
    assertThat(compileAction.getOutputs()).containsExactly(storyboardZip);
    assertThat(compileAction.getArguments())
        .containsExactlyElementsIn(
            new Builder()
                .addDynamicString(MOCK_IBTOOLWRAPPER_PATH)
                .addExecPath(storyboardZip)
                .addDynamicString(archiveRoot) // archive root
                .add("--minimum-deployment-target", minimumOsVersion.toString())
                .add("--module")
                .add("x")
                .addAll(VectorArg.addBefore("--target-device").each(targetDevices))
                .add("x/1.storyboard")
                .build()
                .arguments())
        .inOrder();

    storyboardZip = getBinArtifact("x/ja.lproj/loc.storyboard.zip", target);
    compileAction = (CommandAction) getGeneratingAction(storyboardZip);
    assertThat(Artifact.toExecPaths(compileAction.getInputs()))
        .containsExactly(MOCK_IBTOOLWRAPPER_PATH, "x/ja.lproj/loc.storyboard");
    assertThat(compileAction.getOutputs()).containsExactly(storyboardZip);
    archiveRoot = targetDevices.contains("watch") ? "ja.lproj/" : "ja.lproj/loc.storyboardc";
    assertThat(compileAction.getArguments())
        .containsExactlyElementsIn(
            new Builder()
                .addDynamicString(MOCK_IBTOOLWRAPPER_PATH)
                .addExecPath(storyboardZip)
                .addDynamicString(archiveRoot) // archive root
                .add("--minimum-deployment-target", minimumOsVersion.toString())
                .add("--module")
                .add("x")
                .addAll(VectorArg.addBefore("--target-device").each(targetDevices))
                .add("x/ja.lproj/loc.storyboard")
                .build()
                .arguments())
        .inOrder();
  }

  protected void assertSwiftStdlibToolAction(
      ConfiguredTarget target,
      String platformName,
      String zipName,
      String bundlePath,
      String toolchain)
      throws Exception {
    String zipArtifactName = String.format("%s.%s.zip", target.getTarget().getName(), zipName);
    Artifact swiftLibsZip = getBinArtifact(zipArtifactName, target);
    Artifact binary = getBinArtifact("x_lipobin", target);
    SpawnAction toolAction = (SpawnAction) getGeneratingAction(swiftLibsZip);

    assertThat(Artifact.toExecPaths(toolAction.getInputs()))
        .containsExactly(binary.getExecPathString(), MOCK_SWIFTSTDLIBTOOLWRAPPER_PATH);
    assertThat(toolAction.getOutputs()).containsExactly(swiftLibsZip);

    CustomCommandLine.Builder expectedCommandLine =
        CustomCommandLine.builder().addDynamicString(MOCK_SWIFTSTDLIBTOOLWRAPPER_PATH);

    if (toolchain != null) {
      expectedCommandLine.add("--toolchain", toolchain);
    }

    expectedCommandLine
        .addExecPath("--output_zip_path", swiftLibsZip)
        .add("--bundle_path", bundlePath)
        .add("--platform", platformName)
        .addExecPath("--scan-executable", binary);

    assertThat(toolAction.getArguments()).isEqualTo(expectedCommandLine.build().arguments());
  }

  protected void checkRegisterSwiftSupportActions(
      RuleType ruleType, String platformName, String toolchain) throws Exception {
    checkRegisterSwiftSupportActions(createTargetWithSwift(ruleType), platformName, toolchain);
  }

  protected void checkRegisterSwiftSupportActions(
      RuleType ruleType, String platformName) throws Exception {
    checkRegisterSwiftSupportActions(createTargetWithSwift(ruleType), platformName, null);
  }

  protected void checkRegisterSwiftSupportActions(
      ConfiguredTarget target, String platformName, String toolchain) throws Exception {
    assertSwiftStdlibToolAction(
        target, platformName, "swiftsupport", "SwiftSupport/" + platformName, toolchain);
  }

  protected void checkRegisterSwiftSupportActions(
      ConfiguredTarget target, String platformName) throws Exception {
    assertSwiftStdlibToolAction(
        target, platformName, "swiftsupport", "SwiftSupport/" + platformName, null);
  }

  protected void checkRegisterSwiftStdlibActions(
      RuleType ruleType, String platformName, String toolchain) throws Exception {
    checkRegisterSwiftStdlibActions(createTargetWithSwift(ruleType), platformName, toolchain);
  }

  protected void checkRegisterSwiftStdlibActions(
      RuleType ruleType, String platformName) throws Exception {
    checkRegisterSwiftStdlibActions(createTargetWithSwift(ruleType), platformName, null);
  }

  protected void checkRegisterSwiftStdlibActions(
      ConfiguredTarget target, String platformName, String toolchain) throws Exception {
    assertSwiftStdlibToolAction(target, platformName, "swiftstdlib", "Frameworks", toolchain);
  }

  protected void checkRegisterSwiftStdlibActions(
      ConfiguredTarget target, String platformName) throws Exception {
    assertSwiftStdlibToolAction(target, platformName, "swiftstdlib", "Frameworks", null);
  }

  /**
   * Checks that a target at {@code //x:x}, which is already created, merges xcdatamodel zips
   * properly based on certain arbitrary and default values.
   */
  private void checkMergesXcdatamodelZips(String bundleDir, String binarysMergeZip)
      throws Exception {
    Action mergeBundleAction = bundleMergeAction("//x:x");
    Iterable<Artifact> mergeInputs = mergeBundleAction.getInputs();
    assertThat(Artifact.toRootRelativePaths(mergeInputs))
        .containsAllOf("x/x/foo.zip", binarysMergeZip);
    BundleMergeProtos.Control control = bundleMergeControl("//x:x");
    assertThat(control.getMergeZipList())
        .containsExactly(
            MergeZip.newBuilder()
                .setEntryNamePrefix(bundleDir + "/")
                .setSourcePath(
                    getFirstArtifactEndingWith(mergeInputs, "x/foo.zip").getExecPathString())
                .build(),
            MergeZip.newBuilder()
                .setEntryNamePrefix(bundleDir + "/")
                .setSourcePath(
                    getFirstArtifactEndingWith(mergeInputs, binarysMergeZip).getExecPathString())
                .build());
  }

  protected void checkMergesXcdatamodelZips(RuleType ruleType) throws Exception {
    createLibraryTargetWriter("//lib:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setList("datamodels", "foo.xcdatamodel/1")
        .write();
    ruleType.scratchTarget(scratch,
        "deps", "['//lib:lib']",
        "datamodels", "['bar.xcdatamodeld/barx.xcdatamodel/2']");
    checkMergesXcdatamodelZips(getBundlePathInsideIpa(ruleType), "x/x/bar.zip");
  }

  protected void checkMergesXcdatamodelZips(BinaryRuleTypePair ruleTypePair) throws Exception {
    createLibraryTargetWriter("//lib:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setList("datamodels", "foo.xcdatamodel/1")
        .write();
    ruleTypePair.scratchTargets(scratch,
        "deps", "['//lib:lib']",
        "datamodels", "['bar.xcdatamodeld/barx.xcdatamodel/2']");
    checkMergesXcdatamodelZips(ruleTypePair.getBundleDir(), "x/x/bar.zip");
  }

  protected void checkIncludesStoryboardOutputZipsAsMergeZips(RuleType ruleType) throws Exception {
    scratch.file("lib/libsb.storyboard");
    createLibraryTargetWriter("//lib:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setList("storyboards", "libsb.storyboard")
        .write();
    scratch.file("bndl/bndlsb.storyboard");
    scratch.file("bndl/BUILD",
        "objc_bundle_library(",
        "    name = 'bndl',",
        "    storyboards = ['bndlsb.storyboard'],",
        ")");

    scratch.file("x/xsb.storyboard");
    ruleType.scratchTarget(scratch,
        "storyboards", "['xsb.storyboard']",
        "deps", "['//lib:lib']",
        "bundles", "['//bndl:bndl']");

    Artifact libsbOutputZip = getBinArtifact("x/libsb.storyboard.zip", "//x:x");
    Artifact bndlsbOutputZip = getBinArtifact("bndl/bndlsb.storyboard.zip", "//bndl:bndl");
    Artifact xsbOutputZip = getBinArtifact("x/xsb.storyboard.zip", "//x:x");

    BundleMergeProtos.Control mergeControl = bundleMergeControl("//x:x");
    String prefix = getBundlePathInsideIpa(ruleType) + "/";
    assertThat(mergeControl.getMergeZipList()).containsExactly(
        BundleMergeProtos.MergeZip.newBuilder()
            .setEntryNamePrefix(prefix)
            .setSourcePath(libsbOutputZip.getExecPathString())
            .build(),
        BundleMergeProtos.MergeZip.newBuilder()
            .setEntryNamePrefix(prefix)
            .setSourcePath(xsbOutputZip.getExecPathString())
            .build());

    BundleMergeProtos.Control nestedMergeControl =
        Iterables.getOnlyElement(mergeControl.getNestedBundleList());
    assertThat(nestedMergeControl.getMergeZipList()).containsExactly(
        BundleMergeProtos.MergeZip.newBuilder()
            .setEntryNamePrefix(prefix + "bndl.bundle/")
            .setSourcePath(bndlsbOutputZip.getExecPathString())
            .build());

    Action mergeAction = bundleMergeAction("//x:x");
    assertThat(mergeAction.getInputs())
        .containsAllOf(libsbOutputZip, xsbOutputZip, bndlsbOutputZip);
  }

  protected void checkIncludesStoryboardOutputZipsAsMergeZips(BinaryRuleTypePair ruleTypePair,
      BuildConfiguration nestedConfiguration) throws Exception {
    scratch.file("lib/libsb.storyboard");
    createLibraryTargetWriter("//lib:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setList("storyboards", "libsb.storyboard")
        .write();

    scratch.file("bndl/bndlsb.storyboard");
    scratch.file("bndl/BUILD",
        "objc_bundle_library(",
        "    name = 'bndl',",
        "    storyboards = ['bndlsb.storyboard'],",
        ")");

    scratch.file("x/xsb.storyboard");
    ruleTypePair.scratchTargets(scratch,
        "storyboards", "['xsb.storyboard']",
        "deps", "['//lib:lib']",
        "bundles", "['//bndl:bndl']");

    Artifact libsbOutputZip = getBinArtifact("x/libsb.storyboard.zip", "//x:x");
    Artifact bndlsbOutputZip = getBinArtifact(
        "bndl/bndlsb.storyboard.zip", getConfiguredTarget("//bndl:bndl", nestedConfiguration));
    Artifact xsbOutputZip = getBinArtifact("x/xsb.storyboard.zip", "//x:x");

    String bundleDir = ruleTypePair.getBundleDir();
    Control mergeControl = bundleMergeControl("//x:x");
    assertThat(mergeControl.getMergeZipList()).containsExactly(
        MergeZip.newBuilder()
            .setEntryNamePrefix(bundleDir + "/")
            .setSourcePath(libsbOutputZip.getExecPathString())
            .build(),
        MergeZip.newBuilder()
            .setEntryNamePrefix(bundleDir + "/")
            .setSourcePath(xsbOutputZip.getExecPathString())
            .build());

    Control nestedMergeControl =
        Iterables.getOnlyElement(mergeControl.getNestedBundleList());
    assertThat(nestedMergeControl.getMergeZipList()).containsExactly(
        MergeZip.newBuilder()
            .setEntryNamePrefix(bundleDir + "/bndl.bundle/")
            .setSourcePath(bndlsbOutputZip.getExecPathString())
            .build());
  }

  protected List<String> rootedIncludePaths(
      BuildConfiguration configuration, String... unrootedPaths) {
    ImmutableList.Builder<String> rootedPaths = new ImmutableList.Builder<>();
    for (String unrootedPath : unrootedPaths) {
      rootedPaths.add(unrootedPath)
          .add(configuration.getGenfilesFragment().getRelative(unrootedPath).getSafePathString());
    }
    return rootedPaths.build();
  }

  protected void checkErrorsWrongFileTypeForSrcsWhenCompiling(RuleType ruleType)
      throws Exception {
    scratch.file("fg/BUILD",
        "filegroup(",
        "    name = 'fg',",
        "    srcs = ['non_matching', 'matchingh.h', 'matchingc.c'],",
        ")");
    checkError("x1", "x1",
        "does not match expected type: " + SRCS_TYPE,
        ruleType.target(scratch, "x1", "x1",
            "srcs", "['//fg:fg']"));
  }

  protected void prepareAlwayslinkCheck(RuleType ruleType) throws Exception {
    scratch.file(
        "imp1/BUILD",
        "objc_import(",
        "    name = 'imp1',",
        "    archives = ['imp1.a'],",
        "    alwayslink = 1,",
        ")");
    scratch.file("imp2/BUILD",
        "objc_import(",
        "    name = 'imp2',",
        "    archives = ['imp2.a'],",
        ")");
    createLibraryTargetWriter("//lib1:lib1")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .set("alwayslink", 1)
        .write();
    createLibraryTargetWriter("//lib2:lib2")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .set("alwayslink", 0)
        .write();
    ruleType.scratchTarget(scratch,
        "srcs", "['a.m']",
        "deps", "['//imp1:imp1', '//imp2:imp2', '//lib1:lib1', '//lib2:lib2']");
  }

  protected void checkForceLoadsAlwayslinkTargets(RuleType ruleType, ExtraLinkArgs extraLinkArgs)
      throws Exception {
    prepareAlwayslinkCheck(ruleType);
    CommandAction action = linkAction("//x:x");

    verifyObjlist(action, "x-linker.objlist",
        execPathEndingWith(action.getInputs(), "x/libx.a"),
        execPathEndingWith(action.getInputs(), "lib2.a"),
        execPathEndingWith(action.getInputs(), "imp2.a"));
    Iterable<String> forceLoadArchives = ImmutableList.of(
        execPathEndingWith(action.getInputs(), "imp1.a"),
        execPathEndingWith(action.getInputs(), "lib1.a"));
    assertThat(action.getArguments())
        .containsExactly(
            "/bin/bash",
            "-c",
            Joiner.on(" ")
                .join(
                    new ImmutableList.Builder<String>()
                        .add(MOCK_XCRUNWRAPPER_PATH)
                        .add(CLANG)
                        .add("-filelist")
                        .add(execPathEndingWith(action.getInputs(), "x-linker.objlist"))
                        .add("-mios-simulator-version-min=" + DEFAULT_IOS_SDK_VERSION)
                        .add("-arch")
                        .add("x86_64")
                        .add("-isysroot", AppleToolchain.sdkDir())
                        .add(
                            "-F", AppleToolchain.sdkDir() + AppleToolchain.DEVELOPER_FRAMEWORK_PATH)
                        .add("-F", frameworkDir(getConfiguredTarget("//x:x")))
                        .add("-Xlinker", "-objc_abi_version", "-Xlinker", "2")
                        .add("-Xlinker", "-rpath", "-Xlinker", "@executable_path/Frameworks")
                        .add("-fobjc-link-runtime")
                        .add("-ObjC")
                        .addAll(
                            Interspersing.beforeEach(
                                "-framework", SdkFramework.names(AUTOMATIC_SDK_FRAMEWORKS)))
                        .add("-o")
                        .addAll(Artifact.toExecPaths(action.getOutputs()))
                        .addAll(Interspersing.beforeEach("-force_load", forceLoadArchives))
                        .addAll(extraLinkArgs)
                        .build()))
        .inOrder();
  }

  protected void checkObjcCopts(RuleType ruleType) throws Exception {
    useConfiguration("--objccopt=-foo");

    scratch.file("x/a.m");
    ruleType.scratchTarget(scratch, "srcs", "['a.m']");
    List<String> args = compileAction("//x:x", "a.o").getArguments();
    assertThat(args).contains("-foo");
  }

  protected void checkObjcCopts_argumentOrdering(RuleType ruleType) throws Exception {
    useConfiguration("--objccopt=-foo");

    scratch.file("x/a.m");
    ruleType.scratchTarget(scratch,
        "srcs", "['a.m']",
        "copts", "['-bar']");
    List<String> args = compileAction("//x:x", "a.o").getArguments();
    assertThat(args).containsAllOf("-fobjc-arc", "-foo", "-bar").inOrder();
  }

  protected void checkClangCoptsForCompilationMode(RuleType ruleType, CompilationMode mode,
      CodeCoverageMode codeCoverageMode) throws Exception {
    ImmutableList.Builder<String> allExpectedCoptsBuilder = ImmutableList.<String>builder()
        .addAll(CompilationSupport.DEFAULT_COMPILER_FLAGS)
        .addAll(compilationModeCopts(mode));

    switch (codeCoverageMode) {
      case NONE:
        useConfiguration("--compilation_mode=" + compilationModeFlag(mode));
        break;
      case GCOV:
        allExpectedCoptsBuilder.addAll(CompilationSupport.CLANG_GCOV_COVERAGE_FLAGS);
        useConfiguration("--collect_code_coverage",
            "--compilation_mode=" + compilationModeFlag(mode));
        break;
      case LLVMCOV:
        allExpectedCoptsBuilder.addAll(CompilationSupport.CLANG_LLVM_COVERAGE_FLAGS);
        useConfiguration("--collect_code_coverage", "--experimental_use_llvm_covmap",
            "--compilation_mode=" + compilationModeFlag(mode));
        break;
    }
    scratch.file("x/a.m");
    ruleType.scratchTarget(scratch,
        "srcs", "['a.m']");

    CommandAction compileActionA = compileAction("//x:x", "a.o");

    assertThat(compileActionA.getArguments())
        .containsAllIn(allExpectedCoptsBuilder.build());
  }

  protected void checkClangCoptsForDebugModeWithoutGlib(RuleType ruleType) throws Exception {
     ImmutableList.Builder<String> allExpectedCoptsBuilder = ImmutableList.<String>builder()
        .addAll(CompilationSupport.DEFAULT_COMPILER_FLAGS)
        .addAll(ObjcConfiguration.DBG_COPTS);

    useConfiguration("--compilation_mode=dbg", "--objc_debug_with_GLIBCXX=false");
    scratch.file("x/a.m");
    ruleType.scratchTarget(scratch,
        "srcs", "['a.m']");

    CommandAction compileActionA = compileAction("//x:x", "a.o");

    assertThat(compileActionA.getArguments())
        .containsAllIn(allExpectedCoptsBuilder.build()).inOrder();

  }

  protected void checkLinkopts(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch, "linkopts", "['foo', 'bar']");

    CommandAction linkAction = linkAction("//x:x");
    String linkArgs = Joiner.on(" ").join(linkAction.getArguments());
    assertThat(linkArgs).contains("-Wl,foo");
    assertThat(linkArgs).contains("-Wl,bar");
  }

  protected void checkMergesPartialInfoplists(RuleType ruleType) throws Exception {
    scratch.file("x/primary-Info.plist");
    ruleType.scratchTarget(scratch,
        INFOPLIST_ATTR, "'primary-Info.plist'",
        "asset_catalogs", "['foo.xcassets/bar']");

    String targetName = "//x:x";
    ConfiguredTarget target = getConfiguredTarget(targetName);

    PlMergeProtos.Control control = plMergeControl(targetName);
    Artifact actoolPartial = getBinArtifact("x.actool-PartialInfo.plist", "//x:x");
    Artifact versionInfoplist = getBinArtifact("plists/x-version.plist", target);
    Artifact environmentInfoplist = getBinArtifact("plists/x-environment.plist", target);
    Artifact automaticInfoplist = getBinArtifact("plists/x-automatic.plist", target);

    assertPlistMergeControlUsesSourceFiles(
        control,
        ImmutableList.<String>of(
            "x/primary-Info.plist",
            versionInfoplist.getExecPathString(),
            environmentInfoplist.getExecPathString(),
            automaticInfoplist.getExecPathString(),
            actoolPartial.getExecPathString()));
  }

  protected void checkMergesPartialInfoplists(BinaryRuleTypePair ruleTypePair) throws Exception {
    scratch.file("x/primary-Info.plist");
    ruleTypePair.scratchTargets(scratch,
        "asset_catalogs", "['foo.xcassets/bar']",
        INFOPLIST_ATTR, "'primary-Info.plist'");

    String targetName = "//x:x";
    ConfiguredTarget target = getConfiguredTarget(targetName);
    PlMergeProtos.Control control = plMergeControl(targetName);


    Artifact merged = getBinArtifact("x-MergedInfo.plist", target);
    Artifact actoolPartial = getBinArtifact("x.actool-PartialInfo.plist", "//x:x");

    Artifact versionInfoplist = getBinArtifact("plists/x-version.plist", target);
    Artifact environmentInfoplist = getBinArtifact("plists/x-environment.plist", target);
    Artifact automaticInfoplist = getBinArtifact("plists/x-automatic.plist", target);

    assertPlistMergeControlUsesSourceFiles(
        control,
        ImmutableList.<String>of(
            "x/primary-Info.plist",
            versionInfoplist.getExecPathString(),
            environmentInfoplist.getExecPathString(),
            automaticInfoplist.getExecPathString(),
            actoolPartial.getExecPathString()));
    assertThat(control.getOutFile()).isEqualTo(merged.getExecPathString());
    assertThat(control.getVariableSubstitutionMapMap())
        .containsExactlyEntriesIn(getVariableSubstitutionArguments(ruleTypePair));
    assertThat(control.getFallbackBundleId()).isEqualTo("example.x");
  }

  private void addTransitiveDefinesUsage(RuleType topLevelRuleType) throws Exception {
    createLibraryTargetWriter("//lib1:lib1")
        .setAndCreateFiles("srcs", "a.m")
        .setList("defines", "A=foo", "B")
        .write();
    createLibraryTargetWriter("//lib2:lib2")
        .setAndCreateFiles("srcs", "a.m")
        .setList("deps", "//lib1:lib1")
        .setList("defines", "C=bar", "D")
        .write();

    topLevelRuleType.scratchTarget(scratch,
        "srcs", "['a.m']",
        "non_arc_srcs", "['b.m']",
        "deps", "['//lib2:lib2']",
        "defines", "['E=baz']",
        "copts", "['explicit_copt']");
  }

  protected void checkReceivesTransitivelyPropagatedDefines(RuleType ruleType) throws Exception {
    addTransitiveDefinesUsage(ruleType);
    assertContainsSublist(compileAction("//x:x", "a.o").getArguments(),
        ImmutableList.of("-DA=foo", "-DB", "-DC=bar", "-DD", "-DE=baz", "explicit_copt"));
    assertContainsSublist(compileAction("//x:x", "b.o").getArguments(),
        ImmutableList.of("-DA=foo", "-DB", "-DC=bar", "-DD", "-DE=baz", "explicit_copt"));
  }

  protected void checkDefinesFromCcLibraryDep(RuleType ruleType) throws Exception {
    useConfiguration();
    ScratchAttributeWriter.fromLabelString(this, "cc_library", "//dep:lib")
        .setList("srcs", "a.cc")
        .setList("defines", "foo", "bar")
        .write();

    ScratchAttributeWriter.fromLabelString(this, ruleType.getRuleTypeName(), "//objc:x")
        .setList("srcs", "a.m")
        .setList("deps", "//dep:lib")
        .write();

    CommandAction compileAction = compileAction("//objc:x", "a.o");
    assertThat(compileAction.getArguments()).containsAllOf("-Dfoo", "-Dbar");
  }

  protected void checkSdkIncludesUsedInCompileAction(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch,
        "sdk_includes", "['foo', 'bar/baz']",
        "srcs", "['a.m', 'b.m']");
    String sdkIncludeDir = AppleToolchain.sdkDir() + "/usr/include";
    assertThat(compileAction("//x:x", "a.o").getArguments())
        .containsAllOf(
            "-I", sdkIncludeDir + "/foo",
            "-I", sdkIncludeDir + "/bar/baz")
        .inOrder();
    assertThat(compileAction("//x:x", "b.o").getArguments())
        .containsAllOf(
            "-I", sdkIncludeDir + "/foo",
            "-I", sdkIncludeDir + "/bar/baz")
        .inOrder();
  }

  protected void checkSdkIncludesUsedInCompileActionsOfDependers(RuleType ruleType)
      throws Exception {
    ruleType.scratchTarget(scratch, "sdk_includes", "['foo', 'bar/baz']");
    // Add some dependers (including transitive depender //bin:bin) and make sure they use the flags
    // as well.
    createLibraryTargetWriter("//lib:lib")
        .setAndCreateFiles("srcs", "a.m")
        .setList("deps", "//x:x")
        .setList("sdk_includes", "from_lib")
        .write();
    createBinaryTargetWriter("//bin:bin")
        .setAndCreateFiles("srcs", "b.m")
        .setList("deps", "//lib:lib")
        .setList("sdk_includes", "from_bin")
        .write();
    String sdkIncludeDir = AppleToolchain.sdkDir() + "/usr/include";
    assertThat(compileAction("//lib:lib", "a.o").getArguments())
        .containsAllOf(
            "-I", sdkIncludeDir + "/from_lib",
            "-I", sdkIncludeDir + "/foo",
            "-I", sdkIncludeDir + "/bar/baz")
        .inOrder();
    assertThat(compileAction("//bin:bin", "b.o").getArguments())
        .containsAllOf(
            "-I", sdkIncludeDir + "/from_bin",
            "-I", sdkIncludeDir + "/from_lib",
            "-I", sdkIncludeDir + "/foo",
            "-I", sdkIncludeDir + "/bar/baz")
        .inOrder();
  }

  protected void checkCompileXibActions(
      BinaryRuleTypePair ruleTypePair, DottedVersion minimumOsVersion,
      String platformType) throws Exception {
    ruleTypePair.scratchTargets(scratch, "xibs", "['foo.xib', 'es.lproj/bar.xib']");
    checkCompileXibActions(minimumOsVersion, platformType);
  }

  protected void checkCompileXibActions(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch, "xibs", "['foo.xib', 'es.lproj/bar.xib']");
    checkCompileXibActions(DEFAULT_IOS_SDK_VERSION, "iphone");
  }

  private void checkCompileXibActions(DottedVersion minimumOsVersion,
      String platformType) throws Exception {
    scratch.file("x/foo.xib");
    scratch.file("x/es.lproj/bar.xib");
    ConfiguredTarget target = getConfiguredTarget("//x:x");
    Artifact fooNibZip = getBinArtifact("x/x/foo.nib.zip", target);
    Artifact barNibZip = getBinArtifact("x/x/es.lproj/bar.nib.zip", target);
    CommandAction fooCompile = (CommandAction) getGeneratingAction(fooNibZip);
    CommandAction barCompile = (CommandAction) getGeneratingAction(barNibZip);

    assertThat(Artifact.toExecPaths(fooCompile.getInputs()))
        .containsExactly(MOCK_IBTOOLWRAPPER_PATH, "x/foo.xib");
    assertThat(Artifact.toExecPaths(barCompile.getInputs()))
        .containsExactly(MOCK_IBTOOLWRAPPER_PATH, "x/es.lproj/bar.xib");

    assertThat(fooCompile.getArguments())
        .containsExactly(
            MOCK_IBTOOLWRAPPER_PATH,
            fooNibZip.getExecPathString(),
            "foo.nib", // archive root
            "--minimum-deployment-target", minimumOsVersion.toString(),
            "--module", "x",
            "--target-device", platformType,
            "x/foo.xib")
        .inOrder();
    assertThat(barCompile.getArguments())
        .containsExactly(
            MOCK_IBTOOLWRAPPER_PATH,
            barNibZip.getExecPathString(),
            "es.lproj/bar.nib", // archive root
            "--minimum-deployment-target", minimumOsVersion.toString(),
            "--module", "x",
            "--target-device", platformType,
            "x/es.lproj/bar.xib")
        .inOrder();
  }

  protected void checkNibZipsMergedIntoBundle(RuleType ruleType) throws Exception {
    scratch.file("lib/a.m");
    scratch.file("lib/lib.xib");
    scratch.file("lib/BUILD",
        "objc_library(",
        "    name = 'lib',",
        "    srcs = ['a.m'],",
        "    xibs = ['lib.xib'],",
        ")");
    scratch.file("x/foo.xib");
    scratch.file("x/es.lproj/x.xib");
    ruleType.scratchTarget(scratch,
        "xibs", "['foo.xib', 'es.lproj/x.xib']",
        "deps", "['//lib:lib']");

    ConfiguredTarget xTarget = getConfiguredTarget("//x:x");
    List<Artifact> nibZips = ImmutableList.of(
        getBinArtifact("x/lib/lib.nib.zip", xTarget),
        getBinArtifact("x/x/foo.nib.zip", xTarget),
        getBinArtifact("x/x/es.lproj/x.nib.zip", xTarget));
    List<BundleMergeProtos.MergeZip> mergeZips = new ArrayList<>();
    for (Artifact nibZip : nibZips) {
      mergeZips.add(BundleMergeProtos.MergeZip.newBuilder()
          .setEntryNamePrefix(getBundlePathInsideIpa(ruleType) + "/")
          .setSourcePath(nibZip.getExecPathString())
          .build());
    }

    assertThat(bundleMergeAction("//x:x").getInputs()).containsAllIn(nibZips);
    assertThat(bundleMergeControl("//x:x").getMergeZipList())
        .containsAllIn(mergeZips);
  }

  protected void checkNibZipsMergedIntoBundle(BinaryRuleTypePair ruleTypePair) throws Exception {
    scratch.file("lib/a.m");
    scratch.file("lib/lib.xib");
    scratch.file("lib/BUILD",
        "objc_library(",
        "    name = 'lib',",
        "    srcs = ['a.m'],",
        "    xibs = ['lib.xib'],",
        ")");
    scratch.file("x/foo.xib");
    scratch.file("x/es.lproj/bar.xib");
    ruleTypePair.scratchTargets(scratch,
        "xibs", "['foo.xib', 'es.lproj/bar.xib']",
        "deps", "['//lib:lib']");

    ConfiguredTarget bundlingTarget = getConfiguredTarget("//x:x");
    List<Artifact> nibZips = ImmutableList.of(
        getBinArtifact("x/lib/lib.nib.zip", bundlingTarget),
        getBinArtifact("x/x/foo.nib.zip", bundlingTarget),
        getBinArtifact("x/x/es.lproj/bar.nib.zip", bundlingTarget));
    List<BundleMergeProtos.MergeZip> mergeZips = new ArrayList<>();
    for (Artifact nibZip : nibZips) {
      mergeZips.add(BundleMergeProtos.MergeZip.newBuilder()
          .setEntryNamePrefix(ruleTypePair.getBundleDir() + "/")
          .setSourcePath(nibZip.getExecPathString())
          .build());
    }

    assertThat(bundleMergeAction("//x:x").getInputs()).containsAllIn(nibZips);
    assertThat(bundleMergeControl("//x:x").getMergeZipList())
        .containsAllIn(mergeZips);
  }

  protected void checkBinaryLipoActionMultiCpu(
      BinaryRuleTypePair ruleTypePair, ConfigurationDistinguisher configurationDistinguisher)
      throws Exception {
    useConfiguration("--ios_multi_cpus=i386,x86_64", "--cpu=ios_armv7");
    ruleTypePair.scratchTargets(scratch);

    CommandAction action = (CommandAction) getGeneratingAction(
        getBinArtifact("x_lipobin", getConfiguredTarget("//x:x")));

    String i386Prefix = configurationBin("i386", configurationDistinguisher);
    String x8664Prefix = configurationBin("x86_64", configurationDistinguisher);
    assertThat(Artifact.toExecPaths(action.getInputs()))
        .containsExactly(
            i386Prefix + "x/bin_bin",
            x8664Prefix + "x/bin_bin",
            MOCK_XCRUNWRAPPER_PATH);

    assertThat(action.getArguments())
        .containsExactly(MOCK_XCRUNWRAPPER_PATH, LIPO,
            "-create", i386Prefix + "x/bin_bin", x8664Prefix + "x/bin_bin",
            "-o", execPathEndingWith(action.getOutputs(), "x_lipobin"))
        .inOrder();
  }

  protected void checkBinaryActionMultiPlatform_fails(BinaryRuleTypePair ruleTypePair)
      throws Exception {
    useConfiguration(
        "--ios_multi_cpus=i386,x86_64,armv7,arm64", "--watchos_cpus=armv7k", "--cpu=ios_armv7");
    ruleTypePair.scratchTargets(scratch);

    try {
      getConfiguredTarget("//x:x");
      fail("Multiplatform binary should have failed");
    } catch (AssertionError expected) {
      assertThat(expected)
          .hasMessageThat()
          .contains(
              "--ios_multi_cpus does not currently allow values for both simulator and device "
              + "builds.");
    }
  }

  protected void checkMultiCpuResourceInheritance(BinaryRuleTypePair ruleTypePair)
      throws Exception {
    useConfiguration("--ios_multi_cpus=i386,x86_64");
    ruleTypePair.scratchTargets(scratch, "resources", "['foo.png']");

    assertThat(Artifact.toRootRelativePaths(bundleMergeAction("//x:x").getInputs()))
        .containsExactly(
            "x/foo.png",
            "x/x_lipobin",
            toolsRepoExecPath("tools/objc/bundlemerge"),
            "x/x.ipa-control",
            "x/x-MergedInfo.plist");
  }

  public void checkAllowVariousNonBlacklistedTypesInHeaders(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch, "hdrs", "['foo.foo', 'NoExtension', 'bar.inc', 'baz.hpp']");
    assertThat(view.hasErrors(getConfiguredTarget("//x:x"))).isFalse();
  }

  public void checkWarningForBlacklistedTypesInHeaders(RuleType ruleType) throws Exception {
    checkWarning("x1", "x1",
        "file 'foo.a' from target '//x1:foo.a' is not allowed in hdrs",
        ruleType.target(scratch, "x1", "x1", "hdrs", "['foo.a']"));
    checkWarning("x2", "x2",
        "file 'bar.o' from target '//x2:bar.o' is not allowed in hdrs",
        ruleType.target(scratch, "x2", "x2", "hdrs", "['bar.o']"));
  }

  public void checkCppSourceCompilesWithCppFlags(RuleType ruleType) throws Exception {
    useConfiguration();

    ruleType.scratchTarget(
        scratch, "srcs", "['a.mm', 'b.cc', 'c.mm', 'd.cxx', 'e.c', 'f.m', 'g.C']");
    assertThat(compileAction("//x:x", "a.o").getArguments()).contains("-stdlib=libc++");
    assertThat(compileAction("//x:x", "b.o").getArguments()).contains("-stdlib=libc++");
    assertThat(compileAction("//x:x", "c.o").getArguments()).contains("-stdlib=libc++");
    assertThat(compileAction("//x:x", "d.o").getArguments()).contains("-stdlib=libc++");
    assertThat(compileAction("//x:x", "e.o").getArguments()).doesNotContain("-stdlib=libc++");
    assertThat(compileAction("//x:x", "f.o").getArguments()).doesNotContain("-stdlib=libc++");
    assertThat(compileAction("//x:x", "g.o").getArguments()).contains("-stdlib=libc++");

    // Also test that --std=gnu++11 is provided whenever -stdlib=libc++ is.
    assertThat(compileAction("//x:x", "a.o").getArguments()).contains("-std=gnu++11");
    assertThat(compileAction("//x:x", "b.o").getArguments()).contains("-std=gnu++11");
    assertThat(compileAction("//x:x", "c.o").getArguments()).contains("-std=gnu++11");
    assertThat(compileAction("//x:x", "d.o").getArguments()).contains("-std=gnu++11");
    assertThat(compileAction("//x:x", "e.o").getArguments()).doesNotContain("-std=gnu++11");
    assertThat(compileAction("//x:x", "f.o").getArguments()).doesNotContain("-std=gnu++11");
    assertThat(compileAction("//x:x", "g.o").getArguments()).contains("-std=gnu++11");
  }

  public void checkBundleIdPassedAsFallbackId(RuleType ruleType) throws Exception {
    scratch.file("bin/a.m");
    scratch.file("bin/Info.plist");

    ruleType.scratchTarget(scratch,
        INFOPLIST_ATTR, "'Info.plist'");

    BundleMergeProtos.Control control = bundleMergeControl("//x:x");

    assertThat(control.hasPrimaryBundleIdentifier()).isFalse();
    assertThat(control.getFallbackBundleIdentifier()).isEqualTo("example.x");
  }

  public void checkBundleIdPassedAsPrimaryId(RuleType ruleType) throws Exception {
    scratch.file("bin/a.m");
    scratch.file("bin/Info.plist");

    ruleType.scratchTarget(scratch,
        INFOPLIST_ATTR, "'Info.plist'",
        BUNDLE_ID_ATTR, "'com.bundle.id'");

    BundleMergeProtos.Control control = bundleMergeControl("//x:x");

    assertThat(control.getPrimaryBundleIdentifier()).isEqualTo("com.bundle.id");
    assertThat(control.hasFallbackBundleIdentifier()).isFalse();
  }

  protected void checkPrimaryBundleIdInMergedPlist(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch,
        INFOPLIST_ATTR, "'Info.plist'",
        BUNDLE_ID_ATTR, "'com.bundle.id'");
    scratch.file("bin/Info.plist");
    checkBundleIdFlagsInPlistMergeAction(
        Optional.of("com.bundle.id"), getVariableSubstitutionArgumentsDefaultFormat(ruleType));
  }

  protected void checkPrimaryBundleIdInMergedPlist(BinaryRuleTypePair ruleTypePair)
      throws Exception {
    ruleTypePair.scratchTargets(scratch,
        INFOPLIST_ATTR, "'Info.plist'",
        BUNDLE_ID_ATTR, "'com.bundle.id'");
    scratch.file("bin/Info.plist");
    checkBundleIdFlagsInPlistMergeAction(
        Optional.of("com.bundle.id"), getVariableSubstitutionArguments(ruleTypePair));
  }

  protected void checkFallbackBundleIdInMergedPlist(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch,
        INFOPLIST_ATTR, "'Info.plist'");
    scratch.file("bin/Info.plist");

    checkBundleIdFlagsInPlistMergeAction(
        Optional.<String>absent(), getVariableSubstitutionArgumentsDefaultFormat(ruleType));
  }

  protected void checkFallbackBundleIdInMergedPlist(BinaryRuleTypePair ruleTypePair)
      throws Exception {
    ruleTypePair.scratchTargets(scratch,
        INFOPLIST_ATTR, "'Info.plist'");
    scratch.file("bin/Info.plist");

    checkBundleIdFlagsInPlistMergeAction(
        Optional.<String>absent(), getVariableSubstitutionArguments(ruleTypePair));
  }

  protected void checkBundleIdFlagsInPlistMergeAction(
      Optional<String> specifiedBundleId, Map<String, String> variableSubstitutions)
      throws Exception {
    checkBundleIdFlagsInPlistMergeAction(specifiedBundleId, variableSubstitutions,
        "example.x");
  }
  protected void checkBundleIdFlagsInPlistMergeAction(
      Optional<String> specifiedBundleId, Map<String, String> variableSubstitutions,
      String defaultBundleId) throws Exception {
    String targetName = "//x:x";
    PlMergeProtos.Control control = plMergeControl(targetName);
    ConfiguredTarget target = getConfiguredTarget(targetName);
    Artifact mergedPlist = getMergedInfoPlist(target);
    String bundleIdToCheck = specifiedBundleId.or(defaultBundleId);
    Artifact versionInfoplist = getBinArtifact("plists/x" + artifactName("-version.plist"), target);
    Artifact environmentInfoplist = getBinArtifact("plists/x" + artifactName("-environment.plist"),
        target);
    Artifact automaticInfoplist = getBinArtifact("plists/x" + artifactName("-automatic.plist"),
        target);

    assertPlistMergeControlUsesSourceFiles(
        control,
        ImmutableList.<String>of(
            "x/Info.plist",
            versionInfoplist.getExecPathString(),
            environmentInfoplist.getExecPathString(),
            automaticInfoplist.getExecPathString()));
    assertThat(control.getOutFile()).isEqualTo(mergedPlist.getExecPathString());
    assertThat(control.getVariableSubstitutionMapMap())
        .containsExactlyEntriesIn(variableSubstitutions);

    if (specifiedBundleId.isPresent()) {
      assertThat(control.hasPrimaryBundleId()).isTrue();
      assertThat(control.getPrimaryBundleId()).isEqualTo(bundleIdToCheck);
    } else {
      assertThat(control.hasFallbackBundleId()).isTrue();
      assertThat(control.getFallbackBundleId()).isEqualTo(bundleIdToCheck);
    }
  }

  protected void checkSigningSimulatorBuild(BinaryRuleTypePair ruleTypePair, boolean useMultiCpu)
      throws Exception {
    if (useMultiCpu) {
      useConfiguration("--ios_multi_cpus=i386,x86_64", "--cpu=ios_i386");
    } else {
      useConfiguration("--cpu=ios_i386");
    }

    ruleTypePair.scratchTargets(scratch);

    SpawnAction ipaGeneratingAction = (SpawnAction) ipaGeneratingAction();
    assertThat(ActionsTestUtil.baseArtifactNames(ipaGeneratingAction.getInputs()))
        .containsExactly("x.unprocessed.ipa");

    assertThat(normalizeBashArgs(ipaGeneratingAction.getArguments()))
        .containsAllOf("/usr/bin/codesign", "--sign", "\"-\"")
        .inOrder();

    BundleMergeProtos.Control control = bundleMergeControl("//x:x");
    assertThat(mobileProvisionProfiles(control)).isEmpty();
  }

  protected Action ipaGeneratingAction() throws Exception {
    return getGeneratingActionForLabel("//x:x.ipa");
  }

  protected void checkProvisioningProfileDeviceBuild(
      BinaryRuleTypePair ruleTypePair, boolean useMultiCpu) throws Exception {
    if (useMultiCpu) {
      useConfiguration("--ios_multi_cpus=armv7,arm64", "--cpu=ios_i386", "--watchos_cpus=armv7k");
    } else {
      useConfiguration("--cpu=ios_armv7", "--watchos_cpus=armv7k");
    }

    ruleTypePair.scratchTargets(scratch);

    Artifact provisioningProfile =
        getFileConfiguredTarget("//tools/objc:foo.mobileprovision").getArtifact();
    assertThat(ipaGeneratingAction().getInputs()).contains(provisioningProfile);

    BundleMergeProtos.Control control = bundleMergeControl("//x:x");
    Map<String, String> profiles = mobileProvisionProfiles(control);
    ImmutableMap<String, String> expectedProfiles = ImmutableMap.of(
        provisioningProfile.getExecPathString(),
        ReleaseBundlingSupport.PROVISIONING_PROFILE_BUNDLE_FILE);
    assertThat(profiles).isEqualTo(expectedProfiles);
  }

  protected void addCustomProvisioningProfile(BinaryRuleTypePair ruleTypePair,
      String provisioningProfileAttribute) throws Exception {
    scratch.file("custom/BUILD", "exports_files(['pp.mobileprovision'])");
    scratch.file("custom/pp.mobileprovision");
    ruleTypePair.scratchTargets(
        scratch, provisioningProfileAttribute, "'//custom:pp.mobileprovision'");
  }

  protected void checkProvisioningProfileUserSpecified(
      BinaryRuleTypePair ruleTypePair, boolean useMultiCpu) throws Exception {
    checkProvisioningProfileUserSpecified(ruleTypePair, useMultiCpu, PROVISIONING_PROFILE_ATTR);
  }
  protected void checkProvisioningProfileUserSpecified(
      BinaryRuleTypePair ruleTypePair, boolean useMultiCpu,
      String provisioningProfileAttribute) throws Exception {
    if (useMultiCpu) {
      useConfiguration("--ios_multi_cpus=armv7,arm64", "--cpu=ios_i386", "--watchos_cpus=armv7k");
    } else {
      useConfiguration("--cpu=ios_armv7", "--watchos_cpus=armv7k");
    }
    addCustomProvisioningProfile(ruleTypePair, provisioningProfileAttribute);

    Artifact defaultProvisioningProfile =
        getFileConfiguredTarget("//tools/objc:foo.mobileprovision").getArtifact();
    Artifact customProvisioningProfile =
        getFileConfiguredTarget("//custom:pp.mobileprovision").getArtifact();
    Action signingAction = ipaGeneratingAction();
    assertThat(signingAction.getInputs()).contains(customProvisioningProfile);
    assertThat(signingAction.getInputs()).doesNotContain(defaultProvisioningProfile);

    BundleMergeProtos.Control control = bundleMergeControl("//x:x");
    Map<String, String> profiles = mobileProvisionProfiles(control);
    Map<String, String> expectedProfiles = ImmutableMap.of(
        customProvisioningProfile.getExecPathString(),
        ReleaseBundlingSupport.PROVISIONING_PROFILE_BUNDLE_FILE);
    assertThat(profiles).isEqualTo(expectedProfiles);
  }

  protected void checkMergeBundleAction(BinaryRuleTypePair ruleTypePair) throws Exception {
    ruleTypePair.scratchTargets(scratch,
        INFOPLIST_ATTR, "'Info.plist'");
    SpawnAction action = bundleMergeAction("//x:x");
    assertThat(Artifact.toRootRelativePaths(action.getInputs()))
        .containsExactly(
            MOCK_BUNDLEMERGE_PATH,
            "x/x_lipobin",
            "x/x.ipa-control",
            "x/x-MergedInfo.plist");
    assertThat(Artifact.toRootRelativePaths(action.getOutputs()))
        .containsExactly("x/x.unprocessed.ipa");
    assertNotRequiresDarwin(action);
    assertThat(action.getEnvironment()).isEmpty();
    assertThat(action.getArguments())
        .containsExactly(
            MOCK_BUNDLEMERGE_PATH,
            execPathEndingWith(action.getInputs(), "x.ipa-control"))
        .inOrder();
  }

  protected void checkCollectsAssetCatalogsTransitively(BinaryRuleTypePair ruleTypePair)
      throws Exception {
    scratch.file("lib/ac.xcassets/foo");
    scratch.file("lib/ac.xcassets/bar");
    createLibraryTargetWriter("//lib:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .set("asset_catalogs", "glob(['ac.xcassets/**'])")
        .write();

    scratch.file("x/ac.xcassets/baz");
    scratch.file("x/ac.xcassets/42");
    ruleTypePair.scratchTargets(scratch,
        "deps", "['//lib:lib']",
        "asset_catalogs", "glob(['ac.xcassets/**'])");

    // Test that the actoolzip Action has arguments and inputs obtained from dependencies.
    SpawnAction actoolZipAction = actoolZipActionForIpa("//x:x");
    assertThat(Artifact.toExecPaths(actoolZipAction.getInputs())).containsExactly(
        "lib/ac.xcassets/foo", "lib/ac.xcassets/bar", "x/ac.xcassets/baz", "x/ac.xcassets/42",
        MOCK_ACTOOLWRAPPER_PATH);
    assertContainsSublist(actoolZipAction.getArguments(),
        ImmutableList.of("lib/ac.xcassets", "x/ac.xcassets"));
  }

  protected void checkMergeActionsWithAssetCatalog(BinaryRuleTypePair ruleTypePair)
      throws Exception {
    Artifact actoolZipOut = getBinArtifact("x.actool.zip", "//x:x");
    assertThat(bundleMergeAction("//x:x").getInputs()).contains(actoolZipOut);

    BundleMergeProtos.Control mergeControl = bundleMergeControl("//x:x");
    assertThat(mergeControl.getMergeZipList())
        .containsExactly(MergeZip.newBuilder()
            .setEntryNamePrefix(ruleTypePair.getBundleDir() + "/")
            .setSourcePath(actoolZipOut.getExecPathString())
            .build());
  }

  protected void checkBundleablesAreMerged(
      String bundlingTarget, ListMultimap<String, String> artifactAndBundlePaths) throws Exception {
    BundleMergeProtos.Control control = bundleMergeControl(bundlingTarget);
    Action mergeBundleAction = bundleMergeAction(bundlingTarget);
    List<BundleFile> expectedBundleFiles = new ArrayList<>();
    for (Map.Entry<String, String> path : artifactAndBundlePaths.entries()) {
      Artifact artifact = getFirstArtifactEndingWith(mergeBundleAction.getInputs(), path.getKey());
      expectedBundleFiles.add(BundleFile.newBuilder()
          .setBundlePath(path.getValue())
          .setSourceFile(artifact.getExecPathString())
          .setExternalFileAttribute(BundleableFile.DEFAULT_EXTERNAL_FILE_ATTRIBUTE)
          .build());
    }
    assertThat(control.getBundleFileList()).containsAllIn(expectedBundleFiles);
  }

  protected void checkNestedBundleInformationPropagatedToDependers(RuleType ruleType)
      throws Exception {
    scratch.file("bndl/bndl-Info.plist");
    scratch.file("bndl/bndl.png");
    scratch.file("bndl/BUILD",
        "objc_bundle_library(",
        "    name = 'bndl',",
        "    infoplist = 'bndl-Info.plist',",
        "    resources = ['bndl.png'],",
        ")");

    ruleType.scratchTarget(scratch, "bundles", "['//bndl:bndl']");

    scratch.file("bin/bin.m");
    scratch.file("bin/BUILD",
        "objc_binary(",
        "    name = 'bin',",
        "    srcs = ['bin.m'],",
        "    deps = ['//x:x'],",
        ")");

    assertThat(bundleMergeAction("//bin:bin").getInputs())
        .containsAllOf(
            getSourceArtifact("bndl/bndl-Info.plist"), getSourceArtifact("bndl/bndl.png"));

    BundleMergeProtos.Control binControl = bundleMergeControl("//bin:bin");
    BundleMergeProtos.Control bundleControl =
        Iterables.getOnlyElement(binControl.getNestedBundleList());

    assertThat(bundleControl.getBundleInfoPlistFile()).isEqualTo("bndl/bndl-Info.plist");

    assertThat(bundleControl.getBundleFileList())
        .containsExactly(BundleMergeProtos.BundleFile.newBuilder()
            .setBundlePath("bndl.png")
            .setSourceFile("bndl/bndl.png")
            .setExternalFileAttribute(BundleableFile.DEFAULT_EXTERNAL_FILE_ATTRIBUTE)
            .build());
  }

  protected void checkConvertStringsAction(BinaryRuleTypePair ruleTypePair) throws Exception {
    scratch.file("lib/foo.strings");
    scratch.file("lib/es.lproj/bar.strings");
    ruleTypePair.scratchTargets(scratch, "strings", "['foo.strings', 'es.lproj/bar.strings']");

    ConfiguredTarget target = getConfiguredTarget("//x:x");
    Artifact binaryFoo = getBinArtifact("x/foo.strings.binary", target);
    Artifact binaryBar = getBinArtifact("x/es.lproj/bar.strings.binary", target);

    CommandAction fooAction = (CommandAction) getGeneratingAction(binaryFoo);
    CommandAction barAction = (CommandAction) getGeneratingAction(binaryBar);

    assertThat(fooAction.getOutputs())
        .containsExactly(binaryFoo);
    assertThat(Artifact.toExecPaths(fooAction.getInputs()))
        .containsExactly("x/foo.strings", MOCK_XCRUNWRAPPER_PATH);

    assertThat(barAction.getOutputs())
        .containsExactly(binaryBar);
    assertThat(Artifact.toExecPaths(barAction.getInputs()))
        .containsExactly("x/es.lproj/bar.strings", MOCK_XCRUNWRAPPER_PATH);
  }

  protected void checkMomczipActions(
      BinaryRuleTypePair ruleTypePair, DottedVersion minimumOsVersion) throws Exception {
    createLibraryTargetWriter("//lib:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setList("datamodels", "foo.xcdatamodel/1")
        .write();
    ruleTypePair.scratchTargets(scratch,
        "deps", "['//lib:lib']",
        "datamodels", "['bar.xcdatamodeld/barx.xcdatamodel/2']");

    AppleConfiguration configuration =
        getTargetConfiguration().getFragment(AppleConfiguration.class);

    Action mergeBundleAction = bundleMergeAction("//x:x");
    Artifact fooMomZip = getFirstArtifactEndingWith(mergeBundleAction.getInputs(), "x/foo.zip");
    CommandAction fooMomczipAction = (CommandAction) getGeneratingAction(fooMomZip);
    Artifact barMomZip = getFirstArtifactEndingWith(mergeBundleAction.getInputs(), "x/bar.zip");
    CommandAction barMomczipAction = (CommandAction) getGeneratingAction(barMomZip);

    assertThat(Artifact.toExecPaths(fooMomczipAction.getInputs()))
        .containsExactly("lib/foo.xcdatamodel/1", MOCK_MOMCWRAPPER_PATH);
    assertThat(fooMomczipAction.getOutputs()).containsExactly(fooMomZip);
    assertThat(Artifact.toExecPaths(barMomczipAction.getInputs()))
        .containsExactly("x/bar.xcdatamodeld/barx.xcdatamodel/2", MOCK_MOMCWRAPPER_PATH);
    assertThat(barMomczipAction.getOutputs()).containsExactly(barMomZip);

    ImmutableList<String> commonMomcZipArguments = ImmutableList.of(
        "-XD_MOMC_SDKROOT=" + AppleToolchain.sdkDir(),
        "-XD_MOMC_IOS_TARGET_VERSION=" + minimumOsVersion,
        "-MOMC_PLATFORMS",
            configuration.getMultiArchPlatform(PlatformType.IOS).getLowerCaseNameInPlist(),
        "-XD_MOMC_TARGET_VERSION=10.6");

    assertThat(fooMomczipAction.getArguments())
        .isEqualTo(
            new ImmutableList.Builder<String>()
                .add(MOCK_MOMCWRAPPER_PATH)
                .add(fooMomZip.getExecPathString())
                .add("foo.mom")
                .addAll(commonMomcZipArguments)
                .add("lib/foo.xcdatamodel")
                .build());
    assertThat(barMomczipAction.getArguments())
        .isEqualTo(
            new ImmutableList.Builder<String>()
                .add(MOCK_MOMCWRAPPER_PATH)
                .add(barMomZip.getExecPathString())
                .add("bar.momd")
                .addAll(commonMomcZipArguments)
                .add("x/bar.xcdatamodeld")
                .build());
  }

  protected void addCommonResources(BinaryRuleTypePair ruleTypePair) throws Exception {
    scratch.file("x/Model.xcdatamodeld/Model-1.0.xcdatamodel/contents");
    ruleTypePair.scratchTargets(scratch,
        "strings", "['foo.strings']",
        "xibs", "['bar.xib']",
        "storyboards", "['baz.storyboard']",
        "datamodels", "glob(['Model.xcdatamodeld/**'])");
  }

  protected void checkMultiCpuCompiledResources(BinaryRuleTypePair ruleTypePair) throws Exception {
    useConfiguration("--ios_multi_cpus=armv7,arm64", "--watchos_cpus=armv7k");
    addCommonResources(ruleTypePair);

    BundleMergeProtos.Control topControl = bundleMergeControl("//x:x");
    ImmutableList.Builder<String> bundlePaths = ImmutableList.builder();
    for (BundleMergeProtos.BundleFile file : topControl.getBundleFileList()) {
      bundlePaths.add(file.getBundlePath());
    }
    assertThat(bundlePaths.build()).containsNoDuplicates();

    ImmutableList.Builder<String> mergeZipNames = ImmutableList.builder();
    for (BundleMergeProtos.MergeZip zip : topControl.getMergeZipList()) {
      mergeZipNames.add(Iterables.getLast(Splitter.on('/').split(zip.getSourcePath())));
    }
    assertThat(mergeZipNames.build()).containsNoDuplicates();
  }

  protected void checkMultiCpuCompiledResourcesFromGenrule(BinaryRuleTypePair ruleTypePair)
      throws Exception {
    useConfiguration("--ios_multi_cpus=armv7,arm64", "--watchos_cpus=armv7k");

    String targets =
        ruleTypePair.targets(scratch, "x", "strings", "['Resources/en.lproj/foo.strings']");
    scratch.file("x/foo.strings");
    scratch.file("x/BUILD",
        "genrule(",
        "    name = 'gen',",
        "    srcs = ['foo.strings'],",
        "    outs = ['Resources/en.lproj/foo.strings'],",
        "    cmd = 'cp $(location foo.strings) $(location Resources/en.lproj/foo.strings)'",
        ")",
        targets);

    BundleMergeProtos.Control topControl = bundleMergeControl("//x:x");
    ImmutableList.Builder<String> bundlePaths = ImmutableList.builder();
    for (BundleMergeProtos.BundleFile file : topControl.getBundleFileList()) {
      bundlePaths.add(file.getBundlePath());
    }
    assertThat(bundlePaths.build()).containsNoDuplicates();
  }

  protected void checkMultiCpuGeneratedResourcesFromGenrule(BinaryRuleTypePair ruleTypePair)
      throws Exception {
    useConfiguration("--ios_multi_cpus=armv7,arm64", "--watchos_cpus=armv7k");

    String targets = ruleTypePair.targets(scratch, "x", "resources", "[':gen']");
    scratch.file(
        "x/BUILD",
        "genrule(",
        "    name = 'gen',",
        "    srcs = ['foo'],",
        "    outs = ['foo.res'],",
        "    cmd = 'cp $(location foo) $(location foo.res)'",
        ")",
        targets);

    BundleMergeProtos.Control topControl = bundleMergeControl("//x:x");
    ImmutableList.Builder<String> bundlePaths = ImmutableList.builder();
    for (BundleMergeProtos.BundleFile file : topControl.getBundleFileList()) {
      bundlePaths.add(file.getBundlePath());
    }
    assertThat(bundlePaths.build()).containsNoDuplicates();
  }

  protected void checkTwoStringsOneBundlePath(BinaryRuleTypePair ruleTypePair, String errorTarget)
      throws Exception {
    String targets = ruleTypePair.targets(scratch, "x",
        "strings", "['Resources/en.lproj/foo.strings', 'FooBar/en.lproj/foo.strings']");
    checkTwoStringsOneBundlePath(targets, errorTarget);
  }

  protected void checkTwoStringsOneBundlePath(RuleType ruleType) throws Exception {
    String targets = ruleType.target(scratch, "x", "bndl",
        "strings", "['Resources/en.lproj/foo.strings', 'FooBar/en.lproj/foo.strings']");
    checkTwoStringsOneBundlePath(targets, "bndl");
  }

  private void checkTwoStringsOneBundlePath(String targets, String errorTarget) throws Exception {
    checkError(
        "x",
        errorTarget,
        "Two files map to the same path [en.lproj/foo.strings] in this bundle but come from "
            + "different locations: //x:Resources/en.lproj/foo.strings and "
            + "//x:FooBar/en.lproj/foo.strings",
        targets);
  }

  protected void checkTwoResourcesOneBundlePath(RuleType ruleType) throws Exception {
    String targets = ruleType.target(scratch, "x", "bndl", "resources", "['baz/foo', 'bar/foo']");
    checkTwoResourcesOneBundlePath(targets, "bndl");
  }

  protected void checkTwoResourcesOneBundlePath(BinaryRuleTypePair ruleTypePair, String errorTarget)
      throws Exception {
    String targets = ruleTypePair.targets(scratch, "x", "resources", "['baz/foo', 'bar/foo']");
    checkTwoResourcesOneBundlePath(targets, errorTarget);
  }

  private void checkTwoResourcesOneBundlePath(String targets, String errorTarget) throws Exception {
    checkError(
        "x",
        errorTarget,
        "Two files map to the same path [foo] in this bundle but come from "
            + "different locations: //x:baz/foo and //x:bar/foo",
        targets);
  }

  protected void checkSameStringsTwice(RuleType ruleType) throws Exception {
    String targets =
        ruleType.target(
            scratch,
            "x",
            "bndl",
            "resources",
            "['Resources/en.lproj/foo.strings']",
            "strings",
            "['Resources/en.lproj/foo.strings']");
    checkSameStringsTwice(targets, "bndl");
  }

  protected void checkSameStringsTwice(BinaryRuleTypePair ruleTypePair, String errorTarget)
      throws Exception {
    String targets =
        ruleTypePair.targets(
            scratch,
            "x",
            "resources",
            "['Resources/en.lproj/foo.strings']",
            "strings",
            "['Resources/en.lproj/foo.strings']");
    checkSameStringsTwice(targets, errorTarget);
  }

  private void checkSameStringsTwice(String targets, String errorTarget) throws Exception {
    checkError(
        "x",
        errorTarget,
        "The same file was included multiple times in this rule: x/Resources/en.lproj/foo.strings",
        targets);
  }

  protected Artifact getMergedInfoPlist(ConfiguredTarget target) {
    return getBinArtifact(target.getLabel().getName() + artifactName("-MergedInfo.plist"),
        target);
  }

  protected void checkTargetHasDebugSymbols(RuleType ruleType) throws Exception {
    useConfiguration("--apple_generate_dsym");
    ruleType.scratchTarget(scratch);

    Iterable<Artifact> filesToBuild =
        getConfiguredTarget("//x:x").getProvider(FileProvider.class).getFilesToBuild();
    assertThat(filesToBuild)
        .containsAllOf(
            getBinArtifact("x.app.dSYM/Contents/Resources/DWARF/x_bin", "//x:x"),
            getBinArtifact("x.app.dSYM/Contents/Info.plist", "//x:x"));
  }

  protected void checkTargetHasCpuSpecificDsymFiles(RuleType ruleType) throws Exception {
    useConfiguration("--ios_multi_cpus=armv7,arm64", "--apple_generate_dsym");
    ruleType.scratchTarget(scratch);

    List<Artifact> debugArtifacts = new ArrayList<>();
    debugArtifacts.add(getBinArtifact("x.app.dSYM/Contents/Resources/DWARF/x_armv7", "//x:x"));
    debugArtifacts.add(getBinArtifact("x.app.dSYM/Contents/Resources/DWARF/x_arm64", "//x:x"));

    Iterable<Artifact> filesToBuild =
        getConfiguredTarget("//x:x").getProvider(FileProvider.class).getFilesToBuild();
    assertThat(filesToBuild).containsAllIn(debugArtifacts);
  }

  protected void checkTargetHasDsymPlist(RuleType ruleType) throws Exception {
    useConfiguration("--ios_multi_cpus=armv7,arm64", "--apple_generate_dsym");
    ruleType.scratchTarget(scratch);

    Iterable<Artifact> filesToBuild =
        getConfiguredTarget("//x:x").getProvider(FileProvider.class).getFilesToBuild();
    assertThat(filesToBuild).contains(getBinArtifact("x.app.dSYM/Contents/Info.plist", "//x:x"));
  }

  protected void checkCcDependency(BinaryRuleTypePair ruleTypePair,
      ConfigurationDistinguisher configurationDistinguisher) throws Exception {
    ruleTypePair.scratchTargets(scratch, "deps", "['//lib:cclib']");
    checkCcDependency(configurationDistinguisher, "bin");
  }

  /**
   * @param extraAttrs pairs of key-value strings (must be an even number) which will be added as
   *     extra attributes to the target generated for this test
   */
  protected void checkCcDependency(RuleType ruleType, String... extraAttrs) throws Exception {
    List<String> attrs =
        ImmutableList.<String>builder()
            .add("deps")
            .add("['//lib:cclib']")
            .add(extraAttrs)
            .build();
    ruleType.scratchTarget(scratch, attrs.toArray(new String[0]));
    checkCcDependency(ConfigurationDistinguisher.UNKNOWN, "x");
  }

  private void checkCcDependency(
      ConfigurationDistinguisher configurationDistinguisher, String targetName) throws Exception {
    useConfiguration("--cpu=ios_i386");

    scratch.file("lib/BUILD",
        "cc_library(",
        "    name = 'cclib',",
        "    srcs = ['dep.c'],",
        ")");

    Action appLipoAction = lipoBinAction("//x:x");

    CommandAction binBinAction = (CommandAction) getGeneratingAction(
        getFirstArtifactEndingWith(appLipoAction.getInputs(), targetName + "_bin"));

    verifyObjlist(
        binBinAction, String.format("%s-linker.objlist", targetName),
        "lib/libcclib.a", String.format("x/lib%s.a", targetName));

    assertThat(Artifact.toExecPaths(binBinAction.getInputs()))
        .containsAllOf(
            iosConfigurationCcDepsBin("i386", configurationDistinguisher) + "lib/libcclib.a",
            iosConfigurationCcDepsBin("i386", configurationDistinguisher)
                + String.format("x/lib%s.a", targetName),
            iosConfigurationCcDepsBin("i386", configurationDistinguisher)
                + String.format("x/%s-linker.objlist", targetName));
  }

  protected void checkCcDependencyMultiArch(BinaryRuleTypePair ruleTypePair,
      ConfigurationDistinguisher configurationDistinguisher) throws Exception {
    useConfiguration("--ios_multi_cpus=armv7,arm64");

    scratch.file("lib/BUILD",
        "cc_library(",
        "    name = 'cclib',",
        "    srcs = ['dep.c'],",
        ")");
    ruleTypePair.scratchTargets(scratch, "deps", "['//lib:cclib']");

    CommandAction appLipoAction = (CommandAction) getGeneratingAction(
        getBinArtifact("x_lipobin", getConfiguredTarget("//x:x", targetConfig)));

    assertThat(Artifact.toExecPaths(appLipoAction.getInputs()))
        .containsAllOf(
            configurationBin("armv7", configurationDistinguisher) + "x/bin_bin",
            configurationBin("arm64", configurationDistinguisher) + "x/bin_bin");

    ImmutableSet.Builder<Artifact> binInputs = ImmutableSet.builder();
    for (Artifact bin : appLipoAction.getInputs()) {
      CommandAction binAction = (CommandAction) getGeneratingAction(bin);
      if (binAction != null) {
        binInputs.addAll(binAction.getInputs());
        verifyObjlist(binAction, "x/bin-linker.objlist",
            "x/libbin.a", "lib/libcclib.a");
      }
    }

    assertThat(Artifact.toExecPaths(binInputs.build()))
        .containsAllOf(
            configurationBin("armv7", configurationDistinguisher) + "x/libbin.a",
            configurationBin("arm64", configurationDistinguisher) + "x/libbin.a",
            configurationBin("armv7", configurationDistinguisher)
                + "lib/libcclib.a",
            configurationBin("arm64", configurationDistinguisher)
                + "lib/libcclib.a",
            configurationBin("armv7", configurationDistinguisher)
                + "x/bin-linker.objlist",
            configurationBin("arm64", configurationDistinguisher)
                + "x/bin-linker.objlist");
  }

  protected void checkGenruleDependency(BinaryRuleTypePair ruleTypePair) throws Exception {
    checkGenruleDependency(ruleTypePair.targets(scratch, "x", "srcs", "['gen.m']"));
  }

  protected void checkGenruleDependency(RuleType ruleType) throws Exception {
    checkGenruleDependency(ruleType.target(scratch, "x", "bin", "srcs", "['gen.m']"));
  }

  private void checkGenruleDependency(String targets) throws Exception {
    scratch.file("x/BUILD",
        "genrule(",
        "    name = 'gen',",
        "    srcs = [],",
        "    outs = ['gen.m'],",
        "    cmd = '\\'\\' > $(location gen.m)'",
        ")",
        targets);

    CommandAction binBinAction = (CommandAction)
        getGeneratingAction(getConfiguredTarget("//x:bin"), "x/bin_bin");
    Artifact libBin = getFirstArtifactEndingWith(binBinAction.getInputs(), "libbin.a");
    Action libBinAction = getGeneratingAction(libBin);
    Action genOAction =
        getGeneratingAction(Iterables.getOnlyElement(inputsEndingWith(libBinAction, ".o")));

    assertThat(Artifact.toExecPaths(genOAction.getInputs()))
        .contains(
            configurationGenfiles(
                    "x86_64",
                    ConfigurationDistinguisher.UNKNOWN,
                    defaultMinimumOs(ConfigurationDistinguisher.UNKNOWN))
                + "/x/gen.m");
  }

  protected void checkGenruleDependencyMultiArch(BinaryRuleTypePair ruleTypePair,
      ConfigurationDistinguisher configurationDistinguisher) throws Exception {
    useConfiguration("--ios_multi_cpus=armv7,arm64");
    String targets = ruleTypePair.targets(scratch, "x", "srcs", "['gen.m']");
    scratch.file("x/BUILD",
        "genrule(",
        "    name = 'gen',",
        "    srcs = [],",
        "    outs = ['gen.m'],",
        "    cmd = '\\'\\' > $(location gen.m)'",
        ")",
        targets);

    CommandAction appLipoAction = (CommandAction) getGeneratingAction(
        getBinArtifact("x_lipobin", getConfiguredTarget("//x:x", targetConfig)));

    assertThat(Artifact.toExecPaths(appLipoAction.getInputs()))
        .containsExactly(
            configurationBin("armv7", configurationDistinguisher) + "x/bin_bin",
            configurationBin("arm64", configurationDistinguisher) + "x/bin_bin",
            MOCK_XCRUNWRAPPER_PATH);
  }

  protected void checkGenruleWithoutJavaCcDependency(BinaryRuleTypePair ruleTypePair)
      throws Exception {
    useConfiguration("--ios_multi_cpus=armv7,arm64");

    String targets = ruleTypePair.targets(scratch, "x", "srcs", "['gen.m']");
    scratch.file("x/BUILD",
        "genrule(",
        "    name = 'gen',",
        "    srcs = [],",
        "    outs = ['gen.m'],",
        "    cmd = 'echo \"\" > $(location gen.m)'",
        ")",
        targets);

    CommandAction appLipoAction = (CommandAction) getGeneratingAction(
        getBinArtifact("x_lipobin", getConfiguredTarget("//x:x", targetConfig)));

    for (Artifact binBin : appLipoAction.getInputs()) {
      CommandAction binBinAction = (CommandAction) getGeneratingAction(binBin);
      if (binBinAction == null) {
        continue;
      }
      Action libBinAction =
          getGeneratingAction(getFirstArtifactEndingWith(binBinAction.getInputs(), "libbin.a"));
      Action genOAction =
          getGeneratingAction(Iterables.getOnlyElement(inputsEndingWith(libBinAction, ".o")));
      Action genMAction =
          getGeneratingAction(getFirstArtifactEndingWith(genOAction.getInputs(), "gen.m"));
      assertThat(genMAction).isNotInstanceOf(FailAction.class);
    }
  }

  protected void checkCcDependencyWithProtoDependency(BinaryRuleTypePair ruleTypePair,
      ConfigurationDistinguisher configurationDistinguisher) throws Exception {
    MockProtoSupport.setup(mockToolsConfig);
    useConfiguration("--cpu=ios_i386");
    scratch.file("lib/BUILD",
        "proto_library(",
        "    name = 'protolib',",
        "    srcs = ['foo.proto'],",
        "    cc_api_version = 1,",
        ")",
        "",
        "cc_library(",
        "    name = 'cclib',",
        "    srcs = ['dep.c'],",
        "    deps = [':protolib'],",
        ")");
    ruleTypePair.scratchTargets(scratch, "deps", "['//lib:cclib']");

    Action appLipoAction = lipoBinAction("//x:x");

     CommandAction binBinAction = (CommandAction) getGeneratingAction(
        getFirstArtifactEndingWith(appLipoAction.getInputs(), "bin_bin"));

     String i386Prefix = iosConfigurationCcDepsBin("i386", configurationDistinguisher);
     ImmutableList<String> archiveFilenames = ImmutableList.of(
         i386Prefix + "lib/libcclib.a",
         i386Prefix + "x/libbin.a",
         i386Prefix + "lib/libprotolib.a",
         i386Prefix + "net/proto/libproto.a");

     verifyObjlist(binBinAction, "x/bin-linker.objlist",
         archiveFilenames.toArray(new String[archiveFilenames.size()]));

    assertThat(Artifact.toExecPaths(binBinAction.getInputs()))
        .containsAllIn(
            ImmutableList.builder()
                .addAll(archiveFilenames)
                .add(i386Prefix + "x/bin-linker.objlist")
                .build());
  }

  protected void checkCcDependencyAndJ2objcDependency(BinaryRuleTypePair ruleTypePair,
      ConfigurationDistinguisher configurationDistinguisher) throws Exception {
    MockProtoSupport.setup(mockToolsConfig);
    MockJ2ObjcSupport.setup(mockToolsConfig);
    useConfiguration("--cpu=ios_i386");

    scratch.file("lib/BUILD",
        "java_library(",
        "    name = 'javalib',",
        "    srcs = ['foo.java'],",
        ")",
        "",
        "j2objc_library(",
        "    name = 'j2objclib',",
        "    deps = [':javalib'],",
        ")",
        "",
        "cc_library(",
        "    name = 'cclib',",
        "    srcs = ['dep.c'],",
        ")");
    ruleTypePair.scratchTargets(scratch, "deps", "['//lib:cclib', '//lib:j2objclib']");

    Action appLipoAction = getGeneratingAction(
        getBinArtifact("x_lipobin", getConfiguredTarget("//x:x", targetConfig)));

    CommandAction binBinAction = (CommandAction) getGeneratingAction(
        getFirstArtifactEndingWith(appLipoAction.getInputs(), "bin_bin"));

    String i386Prefix = iosConfigurationCcDepsBin("i386", configurationDistinguisher);
    ImmutableList<String> archiveFilenames =
        ImmutableList.of(
            i386Prefix + "lib/libcclib.a",
            i386Prefix + "x/libbin.a",
            i386Prefix + "lib/libjavalib_j2objc.a",
            i386Prefix + toolsRepoExecPath("third_party/java/j2objc/libjre_core_lib.a"));

    verifyObjlist(binBinAction, "x/bin-linker.objlist",
        archiveFilenames.toArray(new String[archiveFilenames.size()]));

    assertThat(Artifact.toExecPaths(binBinAction.getInputs()))
        .containsAllIn(
            ImmutableList.builder()
                .addAll(archiveFilenames)
                .add(i386Prefix + "x/bin-linker.objlist")
                .build());
  }

  protected void checkCcDependencyWithProtoDependencyMultiArch(BinaryRuleTypePair ruleTypePair,
      ConfigurationDistinguisher configurationDistinguisher) throws Exception {
    MockProtoSupport.setup(mockToolsConfig);
    useConfiguration("--ios_multi_cpus=armv7,arm64");
    scratch.file("lib/BUILD",
        "proto_library(",
        "    name = 'protolib',",
        "    srcs = ['foo.proto'],",
        "    cc_api_version = 1,",
        ")",
        "",
        "cc_library(",
        "    name = 'cclib',",
        "    srcs = ['dep.c'],",
        "    deps = [':protolib'],",
        ")");
    ruleTypePair.scratchTargets(scratch, "deps", "['//lib:cclib']");

    Action appLipoAction = lipoBinAction("//x:x");

    assertThat(Artifact.toExecPaths(appLipoAction.getInputs()))
        .containsExactly(
            configurationBin("armv7", configurationDistinguisher) + "x/bin_bin",
            configurationBin("arm64", configurationDistinguisher) + "x/bin_bin",
            MOCK_XCRUNWRAPPER_PATH);
  }

  protected void checkBinaryStripAction(RuleType ruleType, String... extraItems) throws Exception {
    ruleType.scratchTarget(scratch);

    useConfiguration("--compilation_mode=opt", "--objc_enable_binary_stripping");
    ConfiguredTarget binaryTarget = getConfiguredTarget("//x:x");
    Artifact strippedBinary = getBinArtifact("x_bin", binaryTarget);
    Artifact unstrippedBinary = getBinArtifact("x_bin_unstripped", binaryTarget);
    CommandAction symbolStripAction = (CommandAction) getGeneratingAction(strippedBinary);
    boolean isTestRule = ruleType.getRuleTypeName().endsWith("_test");

    ImmutableList.Builder<String> expectedSymbolStripArgs = ImmutableList.<String>builder()
        .add(MOCK_XCRUNWRAPPER_PATH)
        .add(STRIP);

    expectedSymbolStripArgs.add(extraItems);

    expectedSymbolStripArgs.add(
        "-o", strippedBinary.getExecPathString(), unstrippedBinary.getExecPathString());

    assertThat(symbolStripAction.getArguments())
        .containsExactlyElementsIn(expectedSymbolStripArgs.build())
        .inOrder();

    CommandAction linkAction = (CommandAction) getGeneratingAction(unstrippedBinary);

    String args = Joiner.on(" ").join(linkAction.getArguments());
    if (isTestRule) {
      assertThat(args).doesNotContain("-dead_strip");
      assertThat(args).doesNotContain("-no_dead_strip_inits_and_terms");
    } else {
      assertThat(args).contains("-dead_strip");
      assertThat(args).contains("-no_dead_strip_inits_and_terms");
    }

    assertThat(compileAction("//x:x", "a.o").getArguments()).contains("-g");
  }

  protected void checkLaunchStoryboardIncluded(BinaryRuleTypePair ruleTypePair) throws Exception {
    useConfiguration("--ios_minimum_os=8.1");
    ruleTypePair.scratchTargets(scratch, "launch_storyboard", "'launch.storyboard'");
    ConfiguredTarget target = getConfiguredTarget("//x:x");

    Artifact storyboardZip = getBinArtifact("x/launch.storyboard.zip", target);
    CommandAction storyboardCompile = (CommandAction) getGeneratingAction(storyboardZip);

    assertThat(Artifact.toExecPaths(storyboardCompile.getInputs()))
        .containsExactly(MOCK_IBTOOLWRAPPER_PATH, "x/launch.storyboard");

    assertThat(storyboardCompile.getArguments())
        .isEqualTo(
            new CustomCommandLine.Builder()
                .addDynamicString(MOCK_IBTOOLWRAPPER_PATH)
                .addExecPath(storyboardZip)
                .add("launch.storyboardc")
                .add("--minimum-deployment-target")
                .add("8.1")
                .add("--module")
                .add("x")
                .add("--target-device")
                .add("iphone")
                .add("x/launch.storyboard")
                .build()
                .arguments());

    assertGeneratesLaunchStoryboardPlist(target, "launch");
    assertUsesLaunchStoryboardPlist(target);
    assertMergesLaunchStoryboard(ruleTypePair, storyboardZip);
  }

  protected void checkLaunchStoryboardXib(BinaryRuleTypePair ruleTypePair) throws Exception {
    useConfiguration("--ios_minimum_os=8.1");
    ruleTypePair.scratchTargets(scratch, "launch_storyboard", "'launch.xib'");
    ConfiguredTarget target = getConfiguredTarget("//x:x");

    Artifact nibZip = getBinArtifact("x/x/launch.nib.zip", target);
    CommandAction nibCompile = (CommandAction) getGeneratingAction(nibZip);

    assertThat(Artifact.toExecPaths(nibCompile.getInputs()))
        .containsExactly(MOCK_IBTOOLWRAPPER_PATH, "x/launch.xib");

    assertThat(nibCompile.getArguments())
        .containsExactly(
            MOCK_IBTOOLWRAPPER_PATH,
            nibZip.getExecPathString(),
            "launch.nib",
            "--minimum-deployment-target", "8.1",
            "--module", "x",
            "--target-device", "iphone",
            "x/launch.xib")
        .inOrder();

    assertGeneratesLaunchStoryboardPlist(target, "launch");
    assertUsesLaunchStoryboardPlist(target);
    assertMergesLaunchStoryboard(ruleTypePair, nibZip);
  }

  private void assertGeneratesLaunchStoryboardPlist(ConfiguredTarget target, String baseName)
      throws Exception {
    Artifact storyboardPlist =
        getBinArtifact("plists/" + target.getLabel().getName() + "-launchstoryboard.plist", target);

    FileWriteAction plistAction = (FileWriteAction) getGeneratingAction(storyboardPlist);

    assertThat(plistAction.getFileContents())
        .contains("\"UILaunchStoryboardName\" = \"" + baseName + "\"");
  }

  private void assertUsesLaunchStoryboardPlist(ConfiguredTarget target) throws Exception {
    Artifact storyboardPlist =
        getBinArtifact("plists/" + target.getLabel().getName() + "-launchstoryboard.plist", target);
    PlMergeProtos.Control plMergeControl = plMergeControl(target.getLabel().getCanonicalForm());

    assertThat(plMergeControl.getSourceFileList()).contains(storyboardPlist.getExecPathString());
  }

  private void assertMergesLaunchStoryboard(BinaryRuleTypePair ruleTypePair, Artifact storyboardZip)
      throws Exception {
    assertThat(bundleMergeAction("//x:x").getInputs()).contains(storyboardZip);
    assertThat(bundleMergeControl("//x:x").getMergeZipList())
        .contains(
            BundleMergeProtos.MergeZip.newBuilder()
                .setEntryNamePrefix(ruleTypePair.getBundleDir() + "/")
                .setSourcePath(storyboardZip.getExecPathString())
                .build());
  }

  protected void checkLaunchStoryboardLproj(BinaryRuleTypePair ruleTypePair) throws Exception {
    useConfiguration("--ios_minimum_os=8.1");
    ruleTypePair.scratchTargets(
        scratch, "launch_storyboard", "'superfluous_dir/en.lproj/launch.storyboard'");
    ConfiguredTarget target = getConfiguredTarget("//x:x");

    Artifact storyboardZip = getBinArtifact("x/en.lproj/launch.storyboard.zip", target);
    CommandAction storyboardCompile = (CommandAction) getGeneratingAction(storyboardZip);

    assertThat(storyboardCompile.getInputs())
        .contains(getSourceArtifact("x/superfluous_dir/en.lproj/launch.storyboard"));

    assertThat(storyboardCompile.getArguments())
        .containsAllOf(
            "en.lproj/launch.storyboardc", "x/superfluous_dir/en.lproj/launch.storyboard");

    assertGeneratesLaunchStoryboardPlist(target, "launch");
    assertMergesLaunchStoryboard(ruleTypePair, storyboardZip);
  }

  protected void checkAutomaticPlistEntries(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch, INFOPLIST_ATTR, RuleType.OMIT_REQUIRED_ATTR);

    ConfiguredTarget target = getConfiguredTarget("//x:x");
    Artifact automaticInfoplist = getBinArtifact("plists/x-automatic.plist", target);
    FileWriteAction automaticInfoplistAction =
        (FileWriteAction) getGeneratingAction(automaticInfoplist);

    NSDictionary foundAutomaticEntries =
        (NSDictionary)
            PropertyListParser.parse(
                automaticInfoplistAction.getFileContents().getBytes(Charset.defaultCharset()));

    assertThat(foundAutomaticEntries.keySet())
        .containsExactly(
            "UIDeviceFamily",
            "DTPlatformName",
            "DTSDKName",
            "CFBundleSupportedPlatforms",
            "MinimumOSVersion");
  }

  protected void checkMultipleInfoPlists(RuleType ruleType) throws Exception {
    scratch.file("x/a.plist");
    scratch.file("x/b.plist");
    ruleType.scratchTarget(scratch, "infoplists", "['a.plist', 'b.plist']");

    String targetName = "//x:x";
    PlMergeProtos.Control control = plMergeControl(targetName);

    assertThat(control.getSourceFileList())
        .contains(getSourceArtifact("x/a.plist").getExecPathString());
    assertThat(control.getSourceFileList())
        .contains(getSourceArtifact("x/b.plist").getExecPathString());
  }

  protected void checkInfoplistAndInfoplistsTogether(RuleType ruleType) throws Exception {
    scratch.file("x/a.plist");
    scratch.file("x/b.plist");
    scratch.file("x/c.plist");
    ruleType.scratchTarget(scratch, "infoplists", "['a.plist', 'b.plist']", INFOPLIST_ATTR,
        "'c.plist'");

    String targetName = "//x:x";
    PlMergeProtos.Control control = plMergeControl(targetName);

    assertThat(control.getSourceFileList())
        .contains(getSourceArtifact("x/a.plist").getExecPathString());
    assertThat(control.getSourceFileList())
        .contains(getSourceArtifact("x/b.plist").getExecPathString());
    assertThat(control.getSourceFileList())
        .contains(getSourceArtifact("x/c.plist").getExecPathString());
  }

  protected void checkBundleMergeInputContainsPlMergeOutput(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch, INFOPLIST_ATTR, RuleType.OMIT_REQUIRED_ATTR);

    Artifact mergedPlist = getMergedInfoPlist(getConfiguredTarget("//x:x"));
    CommandAction mergeAction = (CommandAction) getGeneratingAction(mergedPlist);

    assertThat(bundleMergeAction("//x:x").getInputs()).containsAllIn(mergeAction.getOutputs());
  }

  /**
   * Computes expected variable substitutions for "x" with full bundle name.
   */
  protected Map<String, String> getVariableSubstitutionArgumentsDefaultFormat(RuleType ruleType) {
    return constructVariableSubstitutions("x", getBundlePathInsideIpa(ruleType));
  }

  /**
   * Return the expected name for the bundle.
   */
  protected String getBundleNameWithExtension(RuleType ruleType) {
    return "x." + ruleType.bundleExtension();
  }

  /**
   * Return the expected bundle path inside an Ipa.
   */
  protected String getBundlePathInsideIpa(RuleType ruleType) {
    return "Payload/" + getBundleNameWithExtension(ruleType);
  }

  /**
   * Computes expected variable substitutions from a ruleTypePair
   */
  protected Map<String, String> getVariableSubstitutionArguments(BinaryRuleTypePair ruleTypePair) {
    return constructVariableSubstitutions(
        ruleTypePair.getBundleName(), ruleTypePair.getBundleDir());
  }

  private Map<String, String> constructVariableSubstitutions(String bundleName, String bundleDir) {
    return new ImmutableMap.Builder<String, String>()
        .put("EXECUTABLE_NAME", bundleName)
        .put("BUNDLE_NAME", bundleDir.split("/")[1])
        .put("PRODUCT_NAME", bundleName)
        .build();
  }

  protected void assertPlistMergeControlUsesSourceFiles(
      PlMergeProtos.Control control, Iterable<String> sourceFilePaths) throws Exception {
    Iterable<String> allSourceFiles =
        Iterables.concat(control.getSourceFileList(), control.getImmutableSourceFileList());
    assertThat(allSourceFiles).containsAllIn(sourceFilePaths);
  }

  private BinaryFileWriteAction plMergeAction(String binaryLabelString) throws Exception {
    Label binaryLabel = Label.parseAbsolute(binaryLabelString);
    ConfiguredTarget binary = getConfiguredTarget(binaryLabelString);
    return (BinaryFileWriteAction)
        getGeneratingAction(getBinArtifact(binaryLabel.getName()
            + artifactName(".plmerge-control"), binary));
  }

  protected PlMergeProtos.Control plMergeControl(String binaryLabelString) throws Exception {
    InputStream in = plMergeAction(binaryLabelString).getSource().openStream();
    return PlMergeProtos.Control.parseFrom(in);
  }

  protected void setArtifactPrefix(String artifactPrefix) {
    this.artifactPrefix = artifactPrefix;
  }

  private String artifactName(String artifactName) {
    if (artifactPrefix != null) {
      return String.format("-%s%s", artifactPrefix, artifactName);
    }
    return artifactName;
  }

  /**
   * Normalizes arguments to a bash action into a space-separated list.
   *
   * <p>Bash actions' arguments have two parts, the bash invocation ({@code "/bin/bash", "-c"}) and
   * the command executed in the bash shell, as a single string. This method merges all these
   * arguments and splits them on {@code ' '}.
   */
  protected List<String> normalizeBashArgs(List<String> args) {
    return Splitter.on(' ').splitToList(Joiner.on(' ').join(args));
  }

  /** Returns the directory where objc modules will be cached. */
  protected String getModulesCachePath() throws InterruptedException {
    return getAppleCrosstoolConfiguration().getGenfilesFragment()
        + "/"
        + CompilationSupport.OBJC_MODULE_CACHE_DIR_NAME;
  }

  /**
   * Verifies that the given rule supports the minimum_os attribute, and adds compile and link
   * args to set the minimum os appropriately, including compile args for dependencies.
   *
   * @param ruleType the rule to test
   */
  protected void checkMinimumOsLinkAndCompileArg(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch,
        "deps", "['//package:objcLib']",
        "minimum_os_version", "'5.4'");
    scratch.file("package/BUILD",
        "objc_library(name = 'objcLib', srcs = [ 'b.m' ])");
    useConfiguration("--xcode_version=5.8");

    CommandAction linkAction = linkAction("//x:x");
    CommandAction objcLibArchiveAction = (CommandAction) getGeneratingAction(
        getFirstArtifactEndingWith(linkAction.getInputs(), "libobjcLib.a"));
    CommandAction objcLibCompileAction = (CommandAction) getGeneratingAction(
        getFirstArtifactEndingWith(objcLibArchiveAction.getInputs(), "b.o"));

    String linkArgs = Joiner.on(" ").join(linkAction.getArguments());
    String compileArgs = Joiner.on(" ").join(objcLibCompileAction.getArguments());
    assertThat(linkArgs).contains("-mios-simulator-version-min=5.4");
    assertThat(compileArgs).contains("-mios-simulator-version-min=5.4");
  }

  /**
   * Verifies that the given rule supports the minimum_os attribute under the watchOS platform
   * type, and adds compile and link args to set the minimum os appropriately for watchos,
   * including compile args for dependencies.
   *
   * @param ruleType the rule to test
   */
  protected void checkMinimumOsLinkAndCompileArg_watchos(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch,
        "deps", "['//package:objcLib']",
        "platform_type", "'watchos'",
        "minimum_os_version", "'5.4'");
    scratch.file("package/BUILD",
        "objc_library(name = 'objcLib', srcs = [ 'b.m' ])");
    useConfiguration("--xcode_version=5.8");

    CommandAction linkAction = linkAction("//x:x");
    CommandAction objcLibArchiveAction = (CommandAction) getGeneratingAction(
        getFirstArtifactEndingWith(linkAction.getInputs(), "libobjcLib.a"));
    CommandAction objcLibCompileAction = (CommandAction) getGeneratingAction(
        getFirstArtifactEndingWith(objcLibArchiveAction.getInputs(), "b.o"));

    String linkArgs = Joiner.on(" ").join(linkAction.getArguments());
    String compileArgs = Joiner.on(" ").join(objcLibCompileAction.getArguments());
    assertThat(linkArgs).contains("-mwatchos-simulator-version-min=5.4");
    assertThat(compileArgs).contains("-mwatchos-simulator-version-min=5.4");
  }

  /**
   * Verifies that the given rule throws a sensible error if the minimum_os attribute has a bad
   * value.
   */
  protected void checkMinimumOs_invalid_nonVersion(RuleType ruleType) throws Exception {
    checkError("x", "x",
        String.format(
            MultiArchSplitTransitionProvider.INVALID_VERSION_STRING_ERROR_FORMAT,
            "foobar"),
        ruleType.target(scratch, "x", "x", "minimum_os_version", "'foobar'"));
  }

  /**
   * Verifies that the given rule throws a sensible error if the minimum_os attribute has a bad
   * value.
   */
  protected void checkMinimumOs_invalid_containsAlphabetic(RuleType ruleType) throws Exception {
    checkError("x", "x",
        String.format(
            MultiArchSplitTransitionProvider.INVALID_VERSION_STRING_ERROR_FORMAT,
            "4.3alpha"),
        ruleType.target(scratch, "x", "x", "minimum_os_version", "'4.3alpha'"));
  }

  /**
   * Verifies that the given rule throws a sensible error if the minimum_os attribute has a bad
   * value.
   */
  protected void checkMinimumOs_invalid_tooManyComponents(RuleType ruleType) throws Exception {
    checkError("x", "x",
        String.format(
            MultiArchSplitTransitionProvider.INVALID_VERSION_STRING_ERROR_FORMAT,
            "4.3.1"),
        ruleType.target(scratch, "x", "x", "minimum_os_version", "'4.3.1'"));
  }

  protected void checkDylibDependencies(RuleType ruleType,
      ExtraLinkArgs extraLinkArgs) throws Exception {
    ruleType.scratchTarget(scratch,
        "srcs", "['a.m']",
        "dylibs", "['//fx:framework_import']");

    scratch.file("fx/MyFramework.framework/MyFramework");
    scratch.file("fx/BUILD",
        "objc_framework(",
        "    name = 'framework_import',",
        "    framework_imports = glob(['MyFramework.framework/*']),",
        "    is_dynamic = 1,",
        ")");
    useConfiguration("--ios_multi_cpus=i386,x86_64");

    Action lipobinAction = lipoBinAction("//x:x");

    String i386Bin =
        configurationBin("i386", ConfigurationDistinguisher.APPLEBIN_IOS)
            + "x/x_bin";
    String i386Filelist =
        configurationBin("i386", ConfigurationDistinguisher.APPLEBIN_IOS)
            + "x/x-linker.objlist";
    String x8664Bin =
        configurationBin("x86_64", ConfigurationDistinguisher.APPLEBIN_IOS)
            + "x/x_bin";
    String x8664Filelist =
        configurationBin("x86_64", ConfigurationDistinguisher.APPLEBIN_IOS)
            + "x/x-linker.objlist";

    Artifact i386BinArtifact = getFirstArtifactEndingWith(lipobinAction.getInputs(), i386Bin);
    Artifact i386FilelistArtifact =
        getFirstArtifactEndingWith(getGeneratingAction(i386BinArtifact).getInputs(), i386Filelist);
    Artifact x8664BinArtifact = getFirstArtifactEndingWith(lipobinAction.getInputs(), x8664Bin);
    Artifact x8664FilelistArtifact =
        getFirstArtifactEndingWith(getGeneratingAction(x8664BinArtifact).getInputs(),
            x8664Filelist);

    ImmutableList<String> archiveNames =
        ImmutableList.of("x/libx.a", "lib1/liblib1.a", "lib2/liblib2.a");
    verifyLinkAction(i386BinArtifact, i386FilelistArtifact, "i386", archiveNames,
        ImmutableList.of(PathFragment.create("fx/MyFramework")), extraLinkArgs);
    verifyLinkAction(x8664BinArtifact, x8664FilelistArtifact,
        "x86_64", archiveNames,  ImmutableList.of(PathFragment.create("fx/MyFramework")),
        extraLinkArgs);
  }

  protected void checkLipoBinaryAction(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch,
        "srcs", "['a.m']");

    useConfiguration("--ios_multi_cpus=i386,x86_64");

    CommandAction action = (CommandAction) lipoBinAction("//x:x");
    String i386Bin =
        configurationBin("i386", ConfigurationDistinguisher.APPLEBIN_IOS) + "x/x_bin";
    String x8664Bin =
        configurationBin("x86_64", ConfigurationDistinguisher.APPLEBIN_IOS) + "x/x_bin";

    assertThat(Artifact.toExecPaths(action.getInputs()))
        .containsExactly(i386Bin, x8664Bin, MOCK_XCRUNWRAPPER_PATH);

    assertThat(action.getArguments())
        .containsExactly(MOCK_XCRUNWRAPPER_PATH, LIPO,
            "-create", i386Bin, x8664Bin,
            "-o", execPathEndingWith(action.getOutputs(), "x_lipobin"))
        .inOrder();

    assertThat(Artifact.toRootRelativePaths(action.getOutputs()))
        .containsExactly("x/x_lipobin");
    assertRequiresDarwin(action);
  }

  protected void checkMultiarchCcDep(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch,
        "srcs", "['a.m']", "deps", "['//package:cclib']");
    scratch.file("package/BUILD",
        "cc_library(name = 'cclib', srcs = ['dep.c'])");

    useConfiguration("--ios_multi_cpus=i386,x86_64");

    Action appLipoAction = actionProducingArtifact("//x:x", "_lipobin");
    String i386Prefix =
        configurationBin("i386", ConfigurationDistinguisher.APPLEBIN_IOS);
    String x8664Prefix =
        configurationBin("x86_64", ConfigurationDistinguisher.APPLEBIN_IOS);

    CommandAction i386BinAction = (CommandAction) getGeneratingAction(
        getFirstArtifactEndingWith(appLipoAction.getInputs(), i386Prefix + "x/x_bin"));

    CommandAction x8664BinAction = (CommandAction) getGeneratingAction(
        getFirstArtifactEndingWith(appLipoAction.getInputs(), x8664Prefix + "x/x_bin"));

    verifyObjlist(
        i386BinAction, "x/x-linker.objlist",
        "x/libx.a", "package/libcclib.a");
    verifyObjlist(
        x8664BinAction, "x/x-linker.objlist",
        "x/libx.a", "package/libcclib.a");

    assertThat(Artifact.toExecPaths(i386BinAction.getInputs()))
        .containsAllOf(
            i386Prefix + "x/libx.a",
            i386Prefix + "package/libcclib.a",
            i386Prefix + "x/x-linker.objlist");
    assertThat(Artifact.toExecPaths(x8664BinAction.getInputs()))
        .containsAllOf(
            x8664Prefix + "x/libx.a",
            x8664Prefix + "package/libcclib.a",
            x8664Prefix + "x/x-linker.objlist");
  }

  protected void checkLinkActionsWithSrcs(RuleType ruleType,
      ExtraLinkArgs extraLinkArgs) throws Exception {
    createLibraryTargetWriter("//lib1:lib1")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "hdr.h")
        .write();
    createLibraryTargetWriter("//lib2:lib2")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setAndCreateFiles("hdrs", "hdr.h")
        .write();
    ruleType.scratchTarget(scratch,
        "srcs", "['a.m']",
        "deps", "['//lib1:lib1', '//lib2:lib2']");
    useConfiguration("--ios_multi_cpus=i386,x86_64");

    Action lipobinAction = lipoBinAction("//x:x");

    String i386Bin =
        configurationBin("i386", ConfigurationDistinguisher.APPLEBIN_IOS)
            + "x/x_bin";
    String i386Filelist =
        configurationBin("i386", ConfigurationDistinguisher.APPLEBIN_IOS)
            + "x/x-linker.objlist";
    String x8664Bin =
        configurationBin("x86_64", ConfigurationDistinguisher.APPLEBIN_IOS)
            + "x/x_bin";
    String x8664Filelist =
        configurationBin("x86_64", ConfigurationDistinguisher.APPLEBIN_IOS)
            + "x/x-linker.objlist";

    Artifact i386BinArtifact = getFirstArtifactEndingWith(lipobinAction.getInputs(), i386Bin);
    Artifact i386FilelistArtifact =
        getFirstArtifactEndingWith(getGeneratingAction(i386BinArtifact).getInputs(), i386Filelist);
    Artifact x8664BinArtifact = getFirstArtifactEndingWith(lipobinAction.getInputs(), x8664Bin);
    Artifact x8664FilelistArtifact =
        getFirstArtifactEndingWith(getGeneratingAction(x8664BinArtifact).getInputs(),
            x8664Filelist);

    ImmutableList<String> archiveNames =
        ImmutableList.of("x/libx.a", "lib1/liblib1.a", "lib2/liblib2.a");
    verifyLinkAction(i386BinArtifact, i386FilelistArtifact, "i386", archiveNames,
        ImmutableList.<PathFragment>of(), extraLinkArgs);
    verifyLinkAction(x8664BinArtifact, x8664FilelistArtifact,
        "x86_64", archiveNames,  ImmutableList.<PathFragment>of(), extraLinkArgs);
  }

  // Regression test for b/32310268.
  protected void checkAliasedLinkoptsThroughObjcLibrary(RuleType ruleType) throws Exception {
    useConfiguration("--cpu=ios_i386");

    scratch.file("bin/BUILD",
        "objc_library(",
        "    name = 'objclib',",
        "    srcs = ['objcdep.c'],",
        "    deps = ['cclib'],",
        ")",
        "alias(",
        "    name = 'cclib',",
        "    actual = 'cclib_real',",
        ")",
        "cc_library(",
        "    name = 'cclib_real',",
        "    srcs = ['ccdep.c'],",
        "    linkopts = ['-somelinkopt'],",
        ")");

    ruleType.scratchTarget(scratch,
        "srcs", "['main.m']",
        "deps", "['//bin:objclib']");

    // Frameworks should get placed together with no duplicates.
    assertThat(Joiner.on(" ").join(linkAction("//x").getArguments()))
        .contains("-somelinkopt");
  }

  protected void checkCcDependencyLinkoptsArePropagatedToLinkAction(
      RuleType ruleType) throws Exception {
    useConfiguration("--cpu=ios_i386");

    scratch.file("bin/BUILD",
        "cc_library(",
        "    name = 'cclib1',",
        "    srcs = ['dep1.c'],",
        "    linkopts = ['-framework F1', '-framework F2', '-Wl,--other-opt'],",
        ")",
        "cc_library(",
        "    name = 'cclib2',",
        "    srcs = ['dep2.c'],",
        "    linkopts = ['-another-opt', '-framework F2'],",
        "    deps = ['cclib1'],",
        ")",
        "cc_library(",
        "    name = 'cclib3',",
        "    srcs = ['dep2.c'],",
        "    linkopts = ['-one-more-opt', '-framework UIKit'],",
        "    deps = ['cclib1'],",
        ")");

    ruleType.scratchTarget(scratch,
        "srcs", "['main.m']",
        "deps", "['//bin:cclib2', '//bin:cclib3']");

    // Frameworks from the CROSSTOOL "apply_implicit_frameworks" feature should be present.
    assertThat(Joiner.on(" ").join(linkAction("//x").getArguments()))
        .contains("-framework Foundation -framework UIKit");
    // Frameworks included in linkopts by the user should get placed together with no duplicates.
    // (They may duplicate the ones inserted by the CROSSTOOL feature, but we don't test that here.)
    assertThat(Joiner.on(" ").join(linkAction("//x").getArguments()))
        .contains("-framework F2 -framework F1");
    // Linkopts should also be grouped together.
    assertThat(Joiner.on(" ").join(linkAction("//x").getArguments()))
        .contains("-another-opt -Wl,--other-opt -one-more-opt");
  }

  protected void checkObjcProviderLinkInputsInLinkAction(RuleType ruleType) throws Exception {
    useConfiguration("--cpu=ios_i386");

    scratch.file("bin/defs.bzl",
        "def _custom_rule_impl(ctx):",
        "  return struct(objc=apple_common.new_objc_provider(",
        "      link_inputs=depset(ctx.files.link_inputs)))",
        "custom_rule = rule(",
        "    _custom_rule_impl,",
        "    attrs={'link_inputs': attr.label_list(allow_files=True)},",
        ")");

    scratch.file("bin/input.txt");

    scratch.file("bin/BUILD",
        "load('//bin:defs.bzl', 'custom_rule')",
        "custom_rule(",
        "    name = 'custom',",
        "    link_inputs = ['input.txt'],",
        ")");

    ruleType.scratchTarget(scratch,
        "srcs", "['main.m']",
        "deps", "['//bin:custom']");

    Artifact inputFile = getSourceArtifact("bin/input.txt");
    assertThat(linkAction("//x").getInputs()).contains(inputFile);
  }

  protected void checkAppleSdkVersionEnv(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch);

    CommandAction action = linkAction("//x:x");

    assertAppleSdkVersionEnv(action);
  }

  protected void checkNonDefaultAppleSdkVersionEnv(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch);
    useConfiguration("--ios_sdk_version=8.1");

    CommandAction action = linkAction("//x:x");

    assertAppleSdkVersionEnv(action, "8.1");
  }

  protected void checkAppleSdkDefaultPlatformEnv(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch);
    CommandAction action = linkAction("//x:x");

    assertAppleSdkPlatformEnv(action, "iPhoneSimulator");
  }

  protected void checkAppleSdkIphoneosPlatformEnv(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch);
    useConfiguration(
        "--cpu=ios_arm64");

    CommandAction action = linkAction("//x:x");

    assertAppleSdkPlatformEnv(action, "iPhoneOS");
  }

  protected void checkAppleSdkWatchsimulatorPlatformEnv(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch,
        "platform_type", "'watchos'");
    useConfiguration("--watchos_cpus=i386");

    Action lipoAction = actionProducingArtifact("//x:x", "_lipobin");

    String i386Bin = configurationBin("i386", ConfigurationDistinguisher.APPLEBIN_WATCHOS)
        + "x/x_bin";
    Artifact binArtifact = getFirstArtifactEndingWith(lipoAction.getInputs(), i386Bin);
    CommandAction linkAction = (CommandAction) getGeneratingAction(binArtifact);

    assertAppleSdkPlatformEnv(linkAction, "WatchSimulator");
  }

  protected void checkAppleSdkWatchosPlatformEnv(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch,
        "platform_type", "'watchos'");
    useConfiguration("--watchos_cpus=armv7k");

    Action lipoAction = actionProducingArtifact("//x:x", "_lipobin");

    String armv7kBin =
        configurationBin("armv7k", ConfigurationDistinguisher.APPLEBIN_WATCHOS)
        + "x/x_bin";
    Artifact binArtifact = getFirstArtifactEndingWith(lipoAction.getInputs(), armv7kBin);
    CommandAction linkAction = (CommandAction) getGeneratingAction(binArtifact);

    assertAppleSdkPlatformEnv(linkAction, "WatchOS");
  }

  protected void checkAppleSdkTvsimulatorPlatformEnv(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch,
        "platform_type", "'tvos'");
    useConfiguration("--tvos_cpus=x86_64");

    CommandAction linkAction = linkAction("//x:x");

    assertAppleSdkPlatformEnv(linkAction, "AppleTVSimulator");
  }

  protected void checkAppleSdkTvosPlatformEnv(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch,
        "platform_type", "'tvos'");
    useConfiguration("--tvos_cpus=arm64");

    CommandAction linkAction = linkAction("//x:x");

    assertAppleSdkPlatformEnv(linkAction, "AppleTVOS");
  }

  protected void checkLinkMinimumOSVersion(ConfigurationDistinguisher distinguisher, String arch,
      String minOSVersionOption) throws Exception {
    CommandAction linkAction = linkAction("//x:x");

    assertThat(Joiner.on(" ").join(linkAction.getArguments())).contains(minOSVersionOption);
  }

  protected void checkWatchSimulatorDepCompile(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch,
        "deps", "['//package:objcLib']",
        "platform_type", "'watchos'");
    scratch.file("package/BUILD",
        "objc_library(name = 'objcLib', srcs = [ 'b.m' ])");

    Action lipoAction = actionProducingArtifact("//x:x", "_lipobin");

    String i386Bin = configurationBin("i386", ConfigurationDistinguisher.APPLEBIN_WATCHOS)
        + "x/x_bin";
    Artifact binArtifact = getFirstArtifactEndingWith(lipoAction.getInputs(), i386Bin);
    CommandAction linkAction = (CommandAction) getGeneratingAction(binArtifact);
    CommandAction objcLibCompileAction = (CommandAction) getGeneratingAction(
        getFirstArtifactEndingWith(linkAction.getInputs(), "libobjcLib.a"));

    assertAppleSdkPlatformEnv(objcLibCompileAction, "WatchSimulator");
    assertThat(objcLibCompileAction.getArguments()).containsAllOf("-arch_only", "i386").inOrder();
  }

  protected void checkWatchSimulatorLinkAction(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch,
        "deps", "['//package:objcLib']",
        "platform_type", "'watchos'");
    scratch.file("package/BUILD",
        "objc_library(name = 'objcLib', srcs = [ 'b.m' ])");

    // Tests that ios_multi_cpus and cpu are completely ignored.
    useConfiguration("--ios_multi_cpus=x86_64", "--cpu=ios_x86_64", "--watchos_cpus=i386");

    Action lipoAction = actionProducingArtifact("//x:x", "_lipobin");

    String i386Bin = configurationBin("i386", ConfigurationDistinguisher.APPLEBIN_WATCHOS)
        + "x/x_bin";
    Artifact binArtifact = getFirstArtifactEndingWith(lipoAction.getInputs(), i386Bin);
    CommandAction linkAction = (CommandAction) getGeneratingAction(binArtifact);

    assertAppleSdkPlatformEnv(linkAction, "WatchSimulator");
    assertThat(normalizeBashArgs(linkAction.getArguments()))
        .containsAllOf("-arch", "i386").inOrder();
  }

  protected void checkWatchSimulatorLipoAction(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch,
        "platform_type", "'watchos'");

    // Tests that ios_multi_cpus and cpu are completely ignored.
    useConfiguration("--ios_multi_cpus=x86_64", "--cpu=ios_x86_64", "--watchos_cpus=i386,armv7k");

    CommandAction action = (CommandAction) lipoBinAction("//x:x");
    String i386Bin = configurationBin("i386", ConfigurationDistinguisher.APPLEBIN_WATCHOS)
        + "x/x_bin";
    String armv7kBin = configurationBin("armv7k", ConfigurationDistinguisher.APPLEBIN_WATCHOS)
        + "x/x_bin";

    assertThat(Artifact.toExecPaths(action.getInputs()))
        .containsExactly(i386Bin, armv7kBin, MOCK_XCRUNWRAPPER_PATH);

    assertContainsSublist(action.getArguments(), ImmutableList.of(
        MOCK_XCRUNWRAPPER_PATH, LIPO, "-create"));
    assertThat(action.getArguments()).containsAllOf(armv7kBin, i386Bin);
    assertContainsSublist(action.getArguments(), ImmutableList.of(
        "-o", execPathEndingWith(action.getOutputs(), "x_lipobin")));

    assertThat(Artifact.toRootRelativePaths(action.getOutputs()))
        .containsExactly("x/x_lipobin");
    assertAppleSdkPlatformEnv(action, "WatchOS");
    assertRequiresDarwin(action);
  }

  protected void checkXcodeVersionEnv(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch);
    useConfiguration("--xcode_version=5.8");

    CommandAction action = linkAction("//x:x");

    assertXcodeVersionEnv(action, "5.8");
  }

  protected void checkNoSrcs(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch,
        "deps", "['//package:objcLib']");
    scratch.file("package/BUILD",
        "objc_library(name = 'objcLib', srcs = [ 'b.m' ])");
    useConfiguration("--xcode_version=5.8");

    CommandAction action = linkAction("//x:x");
    assertThat(Artifact.toRootRelativePaths(action.getInputs())).containsAllOf(
        "x/libx.a", "package/libobjcLib.a", "x/x-linker.objlist");
  }

  public void checkLinkingRuleCanUseCrosstool(RuleType ruleType) throws Exception {
    useConfiguration(ObjcCrosstoolMode.ALL);
    ruleType.scratchTarget(scratch, "srcs", "['a.m']");
    ConfiguredTarget target = getConfiguredTarget("//x:x");

    // If bin is indeed using the c++ backend, then its archive action should be a CppLinkAction.
    Action action =
        getGeneratingAction(getBinArtifact("lib" + target.getLabel().getName() + ".a", target));
    assertThat(action).isInstanceOf(CppLinkAction.class);
  }

  public void checkLinkingRuleCanUseCrosstool_singleArch(RuleType ruleType) throws Exception {
    useConfiguration(ObjcCrosstoolMode.ALL);
    ruleType.scratchTarget(scratch, "srcs", "['a.m']");

    // If bin is indeed using the c++ backend, then its archive action should be a CppLinkAction.
    Action lipobinAction = lipoBinAction("//x:x");
    Artifact bin = getFirstArtifactEndingWith(lipobinAction.getInputs(), "_bin");
    Action linkAction = getGeneratingAction(bin);
    Artifact archive = getFirstArtifactEndingWith(linkAction.getInputs(), ".a");
    Action archiveAction = getGeneratingAction(archive);
    assertThat(archiveAction).isInstanceOf(CppLinkAction.class);
  }

  public void checkLinkingRuleCanUseCrosstool_multiArch(RuleType ruleType) throws Exception {
    useConfiguration(ObjcCrosstoolMode.ALL, "--ios_multi_cpus=i386,x86_64");
    ruleType.scratchTarget(scratch, "srcs", "['a.m']");

    // If bin is indeed using the c++ backend, then its archive action should be a CppLinkAction.
    Action lipobinAction = lipoBinAction("//x:x");
    Artifact bin = getFirstArtifactEndingWith(lipobinAction.getInputs(), "_bin");
    Action linkAction = getGeneratingAction(bin);
    Artifact archive = getFirstArtifactEndingWith(linkAction.getInputs(), ".a");
    Action archiveAction = getGeneratingAction(archive);
    assertThat(archiveAction).isInstanceOf(CppLinkAction.class);
  }

  protected void scratchFrameworkSkylarkStub(String bzlPath) throws Exception {
    PathFragment pathFragment = PathFragment.create(bzlPath);
    scratch.file(pathFragment.getParentDirectory() + "/BUILD");
    scratch.file(
        bzlPath,
        "def framework_stub_impl(ctx):",
        "  bin_provider = ctx.attr.binary[apple_common.AppleDylibBinary]",
        "  my_provider = apple_common.new_dynamic_framework_provider(",
        "      objc = bin_provider.objc,",
        "      binary = bin_provider.binary,",
        "      framework_files = depset([bin_provider.binary]),",
        "      framework_dirs = depset(['_frameworks/stubframework.framework']))",
        "  return struct(providers = [my_provider])",
        "framework_stub_rule = rule(",
        "    framework_stub_impl,",
        // Both 'binary' and 'deps' are needed because ObjcProtoAspect is applied transitively
        // along attribute 'deps' only.
        "    attrs = {'binary': attr.label(mandatory=True,",
        "                                  providers=[apple_common.AppleDylibBinary]),",
        "             'deps': attr.label_list(providers=[apple_common.AppleDylibBinary])},",
        "    fragments = ['apple', 'objc'],",
        ")");
  }

  private void assertAvoidDepsObjects(RuleType ruleType) throws Exception {
    /*
     * The target tree for ease of understanding:
     * x depends on "avoidLib" as a dylib and "objcLib" as a static dependency.
     *
     *               (    objcLib    )
     *              /              \
     *       (   avoidLib   )     (   baseLib   )
     *        /                    /           \
     * (avoidLibDep)              /            (baseLibDep)
     *        \                  /
     *        (   avoidLibDepTwo   )
     *
     * All libraries prefixed with "avoid" shouldn't be statically linked in the top level target.
     */
    ruleType.scratchTarget(scratch,
        "deps", "['//package:objcLib']",
        "dylibs", "['//package:avoidLib']");
    scratchFrameworkSkylarkStub("frameworkstub/framework_stub.bzl");
    scratch.file("package/BUILD",
        "load('//frameworkstub:framework_stub.bzl', 'framework_stub_rule')",
        "objc_library(name = 'objcLib', srcs = [ 'b.m' ],",
        "    deps = [':avoidLibDep', ':baseLib'])",
        "objc_library(name = 'baseLib', srcs = [ 'base.m' ],",
        "    deps = [':baseLibDep', ':avoidLibDepTwo'])",
        "objc_library(name = 'baseLibDep', srcs = [ 'basedep.m' ],",
        "    sdk_frameworks = ['BaseSDK'], resources = [':base.png'])",
        "framework_stub_rule(name = 'avoidLib', binary = ':avoidLibBinary')",
        "apple_binary(name = 'avoidLibBinary', binary_type = 'dylib', srcs = [ 'c.m' ],",
        "    platform_type = 'ios',",
        "    deps = [':avoidLibDep'])",
        "objc_library(name = 'avoidLibDep', srcs = [ 'd.m' ], deps = [':avoidLibDepTwo'])",
        "objc_library(name = 'avoidLibDepTwo', srcs = [ 'e.m' ],",
        "    sdk_frameworks = ['AvoidSDK'], resources = [':avoid.png'])");

    Action lipobinAction = lipoBinAction("//x:x");
    Artifact binArtifact = getFirstArtifactEndingWith(lipobinAction.getInputs(), "x/x_bin");

    Action action = getGeneratingAction(binArtifact);

    assertThat(getFirstArtifactEndingWith(action.getInputs(), "x/libx.a")).isNotNull();
    assertThat(getFirstArtifactEndingWith(action.getInputs(), "package/libobjcLib.a")).isNotNull();
    assertThat(getFirstArtifactEndingWith(action.getInputs(), "package/libbaseLib.a")).isNotNull();
    assertThat(getFirstArtifactEndingWith(action.getInputs(), "package/libbaseLibDep.a"))
        .isNotNull();
    assertThat(getFirstArtifactEndingWith(action.getInputs(), "package/libavoidLib.a")).isNull();
    assertThat(getFirstArtifactEndingWith(action.getInputs(), "package/libavoidLibDepTwo.a"))
        .isNull();
    assertThat(getFirstArtifactEndingWith(action.getInputs(), "package/libavoidLibDep.a")).isNull();
  }

  public void checkAvoidDepsObjectsWithCrosstool(RuleType ruleType) throws Exception {
    useConfiguration(ObjcCrosstoolMode.ALL, "--ios_multi_cpus=i386,x86_64");
    assertAvoidDepsObjects(ruleType);
  }

  public void checkAvoidDepsObjects(RuleType ruleType) throws Exception {
    useConfiguration("--ios_multi_cpus=i386,x86_64");
    assertAvoidDepsObjects(ruleType);
  }

  /**
   * Verifies that if apple_binary A depends on a dylib B1 which then depends on a dylib B2,
   * that the symbols from B2 are not present in A.
   */
  public void checkAvoidDepsThroughDylib(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch,
        "deps", "['//package:ObjcLib']",
        "dylibs", "['//package:dylib1']");
    scratchFrameworkSkylarkStub("frameworkstub/framework_stub.bzl");
    scratch.file("package/BUILD",
        "load('//frameworkstub:framework_stub.bzl', 'framework_stub_rule')",
        "objc_library(name = 'ObjcLib', srcs = [ 'ObjcLib.m' ],",
        "    deps = [':Dylib1Lib', ':Dylib2Lib'])",
        "objc_library(name = 'Dylib1Lib', srcs = [ 'Dylib1Lib.m' ])",
        "objc_library(name = 'Dylib2Lib', srcs = [ 'Dylib2Lib.m' ])",
        "framework_stub_rule(name = 'dylib1', binary = ':dylib1Binary')",
        "apple_binary(name = 'dylib1Binary', binary_type = 'dylib', srcs = [ 'Dylib1Bin.m' ],",
        "    platform_type = 'ios',",
        "    deps = [':Dylib1Lib'], dylibs = ['//package:dylib2'])",
        "framework_stub_rule(name = 'dylib2', binary = ':dylib2Binary')",
        "apple_binary(name = 'dylib2Binary', binary_type = 'dylib', srcs = [ 'Dylib2Bin.m' ],",
        "    platform_type = 'ios',",
        "    deps = [':Dylib2Lib'])",
        "apple_binary(name = 'alternate', srcs = [ 'alternate.m' ],",
        "    platform_type = 'ios',",
        "    deps = ['//package:ObjcLib'])");

    Action lipobinAction = lipoBinAction("//x:x");
    Artifact binArtifact = getFirstArtifactEndingWith(lipobinAction.getInputs(), "x/x_bin");

    Action linkAction = getGeneratingAction(binArtifact);

    assertThat(getFirstArtifactEndingWith(linkAction.getInputs(),
        "package/libObjcLib.a")).isNotNull();
    assertThat(getFirstArtifactEndingWith(linkAction.getInputs(),
        "package/libDylib1Lib.a")).isNull();
    assertThat(getFirstArtifactEndingWith(linkAction.getInputs(),
        "package/libDylib2Lib.a")).isNull();

    // Sanity check that the identical binary without dylibs would be fully linked.
    Action alternateLipobinAction = lipoBinAction("//package:alternate");
    Artifact alternateBinArtifact = getFirstArtifactEndingWith(alternateLipobinAction.getInputs(),
        "package/alternate_bin");
    Action alternateLinkAction = getGeneratingAction(alternateBinArtifact);

    assertThat(getFirstArtifactEndingWith(alternateLinkAction.getInputs(),
        "package/libObjcLib.a")).isNotNull();
    assertThat(getFirstArtifactEndingWith(alternateLinkAction.getInputs(),
        "package/libDylib1Lib.a")).isNotNull();
    assertThat(getFirstArtifactEndingWith(alternateLinkAction.getInputs(),
        "package/libDylib2Lib.a")).isNotNull();
  }

  /**
   * Tests that direct cc_library dependencies of a dylib (and their dependencies) are correctly
   * removed from the main binary.
   */
  // transitively avoided, even if it is not present in deps.
  public void checkAvoidDepsObjects_avoidViaCcLibrary(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch,
        "deps", "['//package:objcLib']",
        "dylibs", "['//package:avoidLib']");
    scratchFrameworkSkylarkStub("frameworkstub/framework_stub.bzl");
    scratch.file("package/BUILD",
        "load('//frameworkstub:framework_stub.bzl', 'framework_stub_rule')",
        "framework_stub_rule(name = 'avoidLib', binary = ':avoidLibBinary')",
        "apple_binary(name = 'avoidLibBinary', binary_type = 'dylib', srcs = [ 'c.m' ],",
        "    platform_type = 'ios',",
        "    deps = [':avoidCclib'])",
        "cc_library(name = 'avoidCclib', srcs = ['cclib.c'], deps = [':avoidObjcLib'])",
        "objc_library(name = 'objcLib', srcs = [ 'b.m' ], deps = [':avoidObjcLib'])",
        "objc_library(name = 'avoidObjcLib', srcs = [ 'c.m' ])");

    Action lipobinAction = actionProducingArtifact("//x:x", "_lipobin");
    Artifact binArtifact = getFirstArtifactEndingWith(lipobinAction.getInputs(), "x/x_bin");

    Action action = getGeneratingAction(binArtifact);

    assertThat(getFirstArtifactEndingWith(action.getInputs(), "package/libobjcLib.a")).isNotNull();
    assertThat(getFirstArtifactEndingWith(action.getInputs(), "package/libavoidObjcLib.a"))
        .isNull();
  }

  public void checkFilesToCompileOutputGroup(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch, "srcs", "['a.m']");
    ConfiguredTarget target = getConfiguredTarget("//x:x");
    assertThat(
            ActionsTestUtil.baseNamesOf(
                getOutputGroup(target, OutputGroupProvider.FILES_TO_COMPILE)))
        .isEqualTo("a.o");
  }

  protected void checkCustomModuleMap(RuleType ruleType) throws Exception {
    useConfiguration("--experimental_objc_enable_module_maps");
    ruleType.scratchTarget(scratch, "srcs", "['a.m']", "deps", "['//z:testModuleMap']");
    scratch.file("x/a.m");
    scratch.file("z/b.m");
    scratch.file("z/b.h");
    scratch.file("y/module.modulemap", "module my_module_b { export *\n header b.h }");
    scratch.file("z/BUILD",
        "objc_library(",
            "name = 'testModuleMap',",
            "hdrs = ['b.h'],",
            "srcs = ['b.m'],",
            "module_map = '//y:mm'",
         ")");
    scratch.file("y/BUILD",
        "filegroup(",
            "name = 'mm',",
            "srcs = ['module.modulemap']",
        ")");

    CommandAction compileActionA = compileAction("//z:testModuleMap", "b.o");
    assertThat(compileActionA.getArguments()).doesNotContain("-fmodule-maps");
    assertThat(compileActionA.getArguments()).doesNotContain("-fmodule-name");

    ObjcProvider provider = providerForTarget("//z:testModuleMap");
    assertThat(Artifact.toExecPaths(provider.get(MODULE_MAP)))
        .containsExactly("y/module.modulemap");

    provider = providerForTarget("//x:x");
    assertThat(Artifact.toExecPaths(provider.get(MODULE_MAP))).contains("y/module.modulemap");
  }

  /**
   * Verifies that the given rule supports different minimum_os attribute values for two targets
   * in the same build, and adds compile args to set the minimum os appropriately for
   * dependencies of each.
   *
   * @param ruleType the rule to test
   * @param multiArchArtifactSuffix the suffix of the artifact that the rule-under-test produces
   * @param singleArchArtifactSuffix the suffix of the single-architecture artifact that is an
   *     input to the rule-under-test's generating action
   */
  protected void checkMinimumOsDifferentTargets(RuleType ruleType, String multiArchArtifactSuffix,
      String singleArchArtifactSuffix) throws Exception {
    ruleType.scratchTarget("nine", "nine", scratch,
        "deps", "['//package:objcLib']",
        "minimum_os_version", "'9.0'");
    ruleType.scratchTarget("eight", "eight", scratch,
        "deps", "['//package:objcLib']",
        "minimum_os_version", "'8.0'");
    scratch.file("package/BUILD",
        "genrule(name = 'root', srcs = ['//nine:nine', '//eight:eight'], outs = ['genout'],",
        "    cmd = 'touch genout')",
        "objc_library(name = 'objcLib', srcs = [ 'b.m' ])");

    ConfiguredTarget rootTarget = getConfiguredTarget("//package:root");
    Artifact rootArtifact = getGenfilesArtifact("genout", rootTarget);

    Action genruleAction = getGeneratingAction(rootArtifact);
    Action eightLipoAction = getGeneratingAction(
        getFirstArtifactEndingWith(genruleAction.getInputs(), "eight" + multiArchArtifactSuffix));
    Action nineLipoAction = getGeneratingAction(
        getFirstArtifactEndingWith(genruleAction.getInputs(), "nine" + multiArchArtifactSuffix));
    Artifact eightBin =
        getFirstArtifactEndingWith(eightLipoAction.getInputs(), singleArchArtifactSuffix);
    Artifact nineBin =
        getFirstArtifactEndingWith(nineLipoAction.getInputs(), singleArchArtifactSuffix);

    CommandAction eightLinkAction = (CommandAction) getGeneratingAction(eightBin);
    CommandAction nineLinkAction = (CommandAction) getGeneratingAction(nineBin);

    CommandAction eightObjcLibArchiveAction = (CommandAction) getGeneratingAction(
        getFirstArtifactEndingWith(eightLinkAction.getInputs(), "libobjcLib.a"));
    CommandAction eightObjcLibCompileAction = (CommandAction) getGeneratingAction(
        getFirstArtifactEndingWith(eightObjcLibArchiveAction.getInputs(), "b.o"));
    CommandAction nineObjcLibArchiveAction = (CommandAction) getGeneratingAction(
        getFirstArtifactEndingWith(nineLinkAction.getInputs(), "libobjcLib.a"));
    CommandAction nineObjcLibCompileAction = (CommandAction) getGeneratingAction(
        getFirstArtifactEndingWith(nineObjcLibArchiveAction.getInputs(), "b.o"));

    assertThat(Joiner.on(" ").join(eightObjcLibCompileAction.getArguments()))
        .contains("-mios-simulator-version-min=8.0");
    assertThat(Joiner.on(" ").join(nineObjcLibCompileAction.getArguments()))
        .contains("-mios-simulator-version-min=9.0");
  }

  /** Returns the full label string for labels within the main tools repository. */
  protected static String toolsRepoLabel(String label) {
    return TestConstants.TOOLS_REPOSITORY + label;
  }

  /**
   * Returns the full exec path string for exec paths of targets within the main tools repository.
   */
  protected static String toolsRepoExecPath(String execPath) {
    return TestConstants.TOOLS_REPOSITORY_PATH_PREFIX + execPath;
  }
}
