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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth8.assertThat;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.getFirstArtifactEndingWith;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.getFirstDerivedArtifactEndingWith;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.MODULE_MAP;

import com.google.common.base.Joiner;
import com.google.common.base.Optional;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.truth.Correspondence;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandAction;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptionsView;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.CoreOptions.OutputDirectoryNamingScheme;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.config.transitions.SplitTransition;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.analysis.util.ScratchAttributeWriter;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.util.MockObjcSupport;
import com.google.devtools.build.lib.rules.apple.AppleCommandLineOptions;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration.ConfigurationDistinguisher;
import com.google.devtools.build.lib.rules.apple.ApplePlatform.PlatformType;
import com.google.devtools.build.lib.rules.apple.AppleToolchain;
import com.google.devtools.build.lib.rules.apple.DottedVersion;
import com.google.devtools.build.lib.rules.cpp.CcCompilationContext;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.rules.cpp.CppLinkAction;
import com.google.devtools.build.lib.rules.objc.CompilationSupport.ExtraLinkArgs;
import com.google.devtools.build.lib.skyframe.BuildConfigurationFunction;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.OptionsParsingException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Pattern;
import javax.annotation.Nullable;

/**
 * Superclass for all Obj-C rule tests.
 *
 * <p>TODO(matvore): split this up into more helper classes, especially the check... methods, which
 * are many and not shared by all objc_ rules.
 *
 * <p>TODO(matvore): find a more concise way to repeat common tests (in particular, those which
 * simply call a check... method) across several rule types.
 */
public abstract class ObjcRuleTestCase extends BuildViewTestCase {
  private static final Correspondence<String, String> MATCHES_REGEX =
      Correspondence.from((a, b) -> Pattern.matches(b, a), "matches");
  protected static final ImmutableList<String> FASTBUILD_COPTS = ImmutableList.of("-O0", "-DDEBUG");

  protected static final DottedVersion DEFAULT_IOS_SDK_VERSION =
      DottedVersion.fromStringUnchecked(AppleCommandLineOptions.DEFAULT_IOS_SDK_VERSION);

  protected static final String OUTPUTDIR = TestConstants.PRODUCT_NAME + "-out//bin";

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
   * Returns the bin dir for artifacts built for a given Apple architecture (as set by a
   * configuration transition) and configuration distinguisher but the global default for {@code
   * --cpu} and the platform default for minimum OS.
   *
   * @param arch the given Apple architecture which artifacts are built under this configuration.
   *     Note this will likely be different than the value of {@code --cpu}
   * @param configurationDistinguisher the configuration distinguisher used to describe the a
   *     configuration transition
   */
  protected String configurationBin(
      String arch, ConfigurationDistinguisher configurationDistinguisher) {
    return configurationBin(arch, configurationDistinguisher, null, CompilationMode.FASTBUILD);
  }

  /**
   * Returns the bin dir for artifacts built for a given Apple architecture and minimum OS version
   * (as set by a configuration transition) and configuration distinguisher but the global default
   * for {@code --cpu}.
   *
   * @param arch the given Apple architecture which artifacts are built under this configuration.
   *     Note this will likely be different than the value of {@code --cpu}.
   * @param configurationDistinguisher the configuration distinguisher used to describe the a
   *     configuration transition
   * @param minOsVersion the minimum os version for which to compile artifacts in the configuration
   * @param compilationMode the compilation mode used during the build
   */
  protected String configurationBin(
      String arch,
      ConfigurationDistinguisher configurationDistinguisher,
      DottedVersion minOsVersion,
      CompilationMode compilationMode) {
    return configurationDir(arch, configurationDistinguisher, minOsVersion, compilationMode)
        + "bin/";
  }

  /**
   * Returns the genfiles dir for artifacts built for a given Apple architecture and minimum OS
   * version (as set by a configuration transition) and configuration distinguisher but the global
   * default for {@code --cpu}.
   *
   * @param arch the given Apple architecture which artifacts are built under this configuration.
   *     Note this will likely be different than the value of {@code --cpu}.
   * @param configurationDistinguisher the configuration distinguisher used to describe the a
   *     configuration transition
   * @param minOsVersion the minimum os version for which to compile artifacts in the configuration
   */
  protected String configurationGenfiles(
      String arch,
      ConfigurationDistinguisher configurationDistinguisher,
      DottedVersion minOsVersion) {
    return configurationDir(
            arch, configurationDistinguisher, minOsVersion, CompilationMode.FASTBUILD)
        + getTargetConfiguration()
            .getGenfilesDirectory(RepositoryName.MAIN)
            .getExecPath()
            .getBaseName();
  }

  @SuppressWarnings("MissingCasesInEnumSwitch")
  private String configurationDir(
      String arch,
      ConfigurationDistinguisher configurationDistinguisher,
      DottedVersion minOsVersion,
      CompilationMode compilationMode) {
    String minOsSegment = minOsVersion == null ? "" : "-min" + minOsVersion;
    String modeSegment = compilationModeFlag(compilationMode);

    String hash = "";
    if (targetConfig.getOptions().get(CoreOptions.class).outputDirectoryNamingScheme
        == OutputDirectoryNamingScheme.DIFF_AGAINST_BASELINE) {
      PlatformType platformType = null;
      switch (configurationDistinguisher) {
        case APPLEBIN_IOS:
          platformType = PlatformType.IOS;
          break;
        case APPLEBIN_WATCHOS:
          platformType = PlatformType.WATCHOS;
          break;
      }
      BuildOptions transitionedConfig = targetConfig.cloneOptions();
      transitionedConfig.get(CoreOptions.class).cpu = platformType + "_" + arch;
      transitionedConfig.get(AppleCommandLineOptions.class).configurationDistinguisher =
          configurationDistinguisher;
      transitionedConfig.get(AppleCommandLineOptions.class).applePlatformType = platformType;
      transitionedConfig.get(AppleCommandLineOptions.class).appleSplitCpu = arch;
      hash =
          "-"
              + BuildConfigurationFunction.computeNameFragmentWithDiff(
                  transitionedConfig, targetConfig.getOptions());
    }

    switch (configurationDistinguisher) {
      case UNKNOWN:
        return String.format("%s-out/ios_%s-%s/", TestConstants.PRODUCT_NAME, arch, modeSegment);
      case APPLE_CROSSTOOL:
        return String.format(
            "%1$s-out/apl-ios_%2$s%3$s-%4$s/",
            TestConstants.PRODUCT_NAME, arch, minOsSegment, modeSegment);
      case APPLEBIN_IOS:
        return String.format(
            "%1$s-out/ios-%2$s%4$s-%3$s-ios_%2$s-%5$s%6$s/",
            TestConstants.PRODUCT_NAME,
            arch,
            configurationDistinguisher.toString().toLowerCase(Locale.US),
            minOsSegment,
            modeSegment,
            hash);
      case APPLEBIN_WATCHOS:
        return String.format(
            "%1$s-out/watchos-%2$s%4$s-%3$s-watchos_%2$s-%5$s%6$s/",
            TestConstants.PRODUCT_NAME,
            arch,
            configurationDistinguisher.toString().toLowerCase(Locale.US),
            minOsSegment,
            modeSegment,
            hash);
      default:
        throw new AssertionError();
    }
  }

  protected String execPathEndingWith(Iterable<Artifact> artifacts, String suffix) {
    return getFirstArtifactEndingWith(artifacts, suffix).getExecPathString();
  }

  @Override
  protected void initializeMockClient() throws IOException {
    super.initializeMockClient();
    MockObjcSupport.setup(mockToolsConfig);
  }

  /** Creates an {@code objc_library} target writer for the label indicated by the given String. */
  protected ScratchAttributeWriter createLibraryTargetWriter(String labelString) {
    return ScratchAttributeWriter.fromLabelString(this, "objc_library", labelString);
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

  protected static ImmutableList<String> compilationModeCopts(CompilationMode mode) {
    switch (mode) {
      case DBG:
        return ImmutableList.copyOf(ObjcConfiguration.DBG_COPTS);
      case OPT:
        return ObjcConfiguration.OPT_COPTS;
      case FASTBUILD:
        return FASTBUILD_COPTS;
    }
    throw new AssertionError();
  }

  /** Override this to trigger platform-based Apple toolchain resolution. */
  protected boolean platformBasedToolchains() {
    return false;
  }

  @Override
  protected void useConfiguration(String... args) throws Exception {
    ImmutableList<String> extraArgs;
    if (platformBasedToolchains()) {
      extraArgs = MockObjcSupport.requiredObjcPlatformFlags();
    } else {
      extraArgs = MockObjcSupport.requiredObjcCrosstoolFlags();
    }
    args = Arrays.copyOf(args, args.length + extraArgs.size());
    for (int i = 0; i < extraArgs.size(); i++) {
      args[(args.length - extraArgs.size()) + i] = extraArgs.get(i);
    }
    super.useConfiguration(args);
  }

  protected void useConfigurationWithCustomXcode(String... args) throws Exception {
    ImmutableList<String> extraArgs;
    if (platformBasedToolchains()) {
      extraArgs = MockObjcSupport.requiredObjcPlatformFlagsNoXcodeConfig();
    } else {
      extraArgs = MockObjcSupport.requiredObjcCrosstoolFlagsNoXcodeConfig();
    }
    args = Arrays.copyOf(args, args.length + extraArgs.size());
    for (int i = 0; i < extraArgs.size(); i++) {
      args[(args.length - extraArgs.size()) + i] = extraArgs.get(i);
    }
    super.useConfiguration(args);
  }

  /** Asserts that an action specifies the given requirement. */
  protected void assertHasRequirement(Action action, String executionRequirement) {
    assertThat(action.getExecutionInfo()).containsKey(executionRequirement);
  }

  /** Asserts that an action does not specify the given requirement. */
  protected void assertNotHasRequirement(Action action, String executionRequirement) {
    assertThat(action.getExecutionInfo()).doesNotContainKey(executionRequirement);
  }

  /**
   * Returns the arguments to pass to clang for specifying module map artifact location and module
   * name.
   *
   * @param packagePath the path to the package this target is in
   * @param targetName the name of the target
   */
  protected List<String> moduleMapArtifactArguments(String packagePath, String targetName) {
    Artifact moduleMapArtifact =
        getGenfilesArtifact(
            targetName + ".modulemaps/module.modulemap", packagePath + ":" + targetName);
    String moduleName = packagePath + ":" + targetName;

    return ImmutableList.of(
        "-iquote",
        moduleMapArtifact.getExecPath().getParentDirectory().toString(),
        "-fmodule-name=" + moduleName);
  }

  /**
   * Returns all child configurations resulting from a given split transition on a given
   * configuration.
   */
  protected List<BuildConfigurationValue> getSplitConfigurations(
      BuildConfigurationValue configuration, SplitTransition splitTransition)
      throws InterruptedException, OptionsParsingException, InvalidConfigurationException {
    ImmutableList.Builder<BuildConfigurationValue> splitConfigs = ImmutableList.builder();

    BuildOptionsView fragmentRestrictedOptions =
        new BuildOptionsView(configuration.getOptions(), splitTransition.requiresOptionFragments());
    for (BuildOptions splitOptions :
        splitTransition.split(fragmentRestrictedOptions, eventCollector).values()) {
      splitConfigs.add(getSkyframeExecutor().getConfigurationForTesting(reporter, splitOptions));
    }

    return splitConfigs.build();
  }

  /**
   * Verifies a {@code -filelist} file's contents.
   *
   * @param originalAction the action which uses the filelist artifact
   * @param inputArchives path suffixes of the expected contents of the filelist
   */
  protected void verifyObjlist(Action originalAction, String... inputArchives) throws Exception {
    ImmutableList.Builder<String> execPaths = ImmutableList.builder();
    for (String inputArchive : inputArchives) {
      execPaths.add(execPathEndingWith(originalAction.getInputs().toList(), inputArchive));
    }
    assertThat(paramFileArgsForAction(originalAction)).containsExactlyElementsIn(execPaths.build());
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
      assertThat(getFirstArtifactEndingWith(binAction.getInputs(), inputArchive)).isNotNull();
    }
    ImmutableList.Builder<String> frameworkPathFragmentParents = ImmutableList.builder();
    ImmutableList.Builder<String> frameworkPathBaseNames = ImmutableList.builder();

    ImmutableList<String> expectedCommandLineFragments =
        ImmutableList.<String>builder()
            .add("-mios-simulator-version-min=" + DEFAULT_IOS_SDK_VERSION)
            .add("-arch " + arch)
            .add("-isysroot " + AppleToolchain.sdkDir())
            .addAll(frameworkPathFragmentParents.build())
            .add("-Xlinker -objc_abi_version -Xlinker 2")
            .add("-fobjc-link-runtime")
            .add("-ObjC")
            .addAll(
                Interspersing.beforeEach(
                    "-framework", CompilationSupport.AUTOMATIC_SDK_FRAMEWORKS.toList()))
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

  protected void assertAppleSdkVersionEnv(Map<String, String> env) throws Exception {
    assertAppleSdkVersionEnv(env, DEFAULT_IOS_SDK_VERSION);
  }

  protected void assertAppleSdkVersionEnv(Map<String, String> env, DottedVersion versionNumber) {
    assertThat(env).containsEntry("APPLE_SDK_VERSION_OVERRIDE", versionNumber.toString());
  }

  protected void assertAppleSdkVersionEnv(CommandAction action) throws Exception {
    assertAppleSdkVersionEnv(action, DEFAULT_IOS_SDK_VERSION.toString());
  }

  protected void assertAppleSdkVersionEnv(CommandAction action, String versionString)
      throws ActionExecutionException {
    assertThat(action.getIncompleteEnvironmentForTesting())
        .containsEntry("APPLE_SDK_VERSION_OVERRIDE", versionString);
  }

  protected void assertAppleSdkPlatformEnv(CommandAction action, String platformName)
      throws ActionExecutionException {
    assertThat(action.getIncompleteEnvironmentForTesting())
        .containsEntry("APPLE_SDK_PLATFORM", platformName);
  }

  protected void assertXcodeVersionEnv(CommandAction action, String versionNumber)
      throws ActionExecutionException {
    assertThat(action.getIncompleteEnvironmentForTesting())
        .containsEntry("XCODE_VERSION_OVERRIDE", versionNumber);
  }

  protected CcInfo ccInfoForTarget(String label) throws Exception {
    CcInfo ccInfo = getConfiguredTarget(label).get(CcInfo.PROVIDER);
    if (ccInfo != null) {
      return ccInfo;
    }
    AppleExecutableBinaryInfo executableProvider =
        getConfiguredTarget(label).get(AppleExecutableBinaryInfo.STARLARK_CONSTRUCTOR);
    if (executableProvider != null) {
      return executableProvider.getDepsCcInfo();
    }
    return null;
  }

  protected CommandAction archiveAction(String label) throws Exception {
    ConfiguredTarget target = getConfiguredTarget(label);
    return (CommandAction)
        getGeneratingAction(getBinArtifact("lib" + target.getLabel().getName() + ".a", target));
  }

  protected Iterable<Artifact> inputsEndingWith(Action action, final String suffix) {
    return Iterables.filter(
        action.getInputs().toList(), artifact -> artifact.getExecPathString().endsWith(suffix));
  }

  /**
   * Asserts that the given action can specify execution requirements, and requires execution on
   * darwin.
   */
  protected void assertRequiresDarwin(Action action) {
    assertHasRequirement(action, ExecutionRequirements.REQUIRES_DARWIN);
  }

  protected ConfiguredTarget addBinWithTransitiveDepOnFrameworkImport() throws Exception {
    ConfiguredTarget lib = addLibWithDepOnFrameworkImport();
    scratch.file(
        "bin/BUILD",
        "load('//test_starlark:apple_binary_starlark.bzl', 'apple_binary_starlark')",
        "apple_binary_starlark(",
        "    name = 'bin',",
        "    platform_type = 'ios',",
        "    deps = ['" + lib.getLabel().toString() + "'],",
        ")");
    return getConfiguredTarget("//bin:bin");
  }

  private ConfiguredTarget addLibWithDepOnFrameworkImport() throws Exception {
    scratch.file(
        "fx/defs.bzl",
        "def _custom_static_framework_import_impl(ctx):",
        "    return [",
        "        apple_common.new_objc_provider(),",
        "        CcInfo(",
        "            compilation_context=cc_common.create_compilation_context(",
        "                framework_includes=depset(ctx.attr.framework_search_paths)",
        "            ),",
        "        )",
        "    ]",
        "custom_static_framework_import = rule(",
        "    _custom_static_framework_import_impl,",
        "    attrs={'framework_search_paths': attr.string_list()},",
        ")");
    scratch.file(
        "fx/BUILD",
        "load(':defs.bzl', 'custom_static_framework_import')",
        "custom_static_framework_import(",
        "    name = 'fx',",
        "    framework_search_paths = ['fx'],",
        ")");
    return createLibraryTargetWriter("//lib:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setList("deps", "//fx:fx")
        .write();
  }

  protected static void addAppleBinaryStarlarkRule(Scratch scratch) throws Exception {
    scratch.file("test_starlark/BUILD");
    RepositoryName toolsRepo = TestConstants.TOOLS_REPOSITORY;
    String toolsLoc = toolsRepo + "//tools/objc";
    scratch.file(
        TestConstants.TOOLS_REPOSITORY_SCRATCH + "tools/objc/dummy/BUILD",
        "objc_library(name = 'dummy_lib', srcs = ['objc_dummy.mm'], alwayslink = False)");

    scratch.file(
        "test_starlark/apple_binary_starlark.bzl",
        "_apple_platform_transition_inputs = [",
        "    '//command_line_option:apple_crosstool_top',",
        "    '//command_line_option:cpu',",
        "    '//command_line_option:ios_multi_cpus',",
        "    '//command_line_option:macos_cpus',",
        "    '//command_line_option:tvos_cpus',",
        "    '//command_line_option:watchos_cpus',",
        "]",
        "_apple_rule_base_transition_outputs = [",
        "    '//command_line_option:apple configuration distinguisher',",
        "    '//command_line_option:apple_platform_type',",
        "    '//command_line_option:apple_platforms',",
        "    '//command_line_option:apple_split_cpu',",
        "    '//command_line_option:compiler',",
        "    '//command_line_option:cpu',",
        "    '//command_line_option:crosstool_top',",
        "    '//command_line_option:fission',",
        "    '//command_line_option:grte_top',",
        "]",
        "def _command_line_options(*, environment_arch = None, platform_type, settings):",
        "    output_dictionary = {",
        "        '//command_line_option:apple configuration distinguisher':",
        "            'applebin_' + platform_type,",
        "        '//command_line_option:apple_platform_type': platform_type,",
        "        '//command_line_option:apple_platforms': [],",
        "        '//command_line_option:apple_split_cpu': environment_arch,",
        "        '//command_line_option:compiler': None,",
        "        '//command_line_option:cpu': ",
        "            'darwin_' + environment_arch if platform_type == 'macos'",
        "            else platform_type + '_' +  environment_arch,",
        "        '//command_line_option:crosstool_top': ",
        "            settings['//command_line_option:apple_crosstool_top'],",
        "        '//command_line_option:fission': [],",
        "        '//command_line_option:grte_top': None,",
        "    }",
        "    return output_dictionary",
        "def _apple_platform_split_transition_impl(settings, attr):",
        "    output_dictionary = {}",
        "    platform_type = attr.platform_type",
        "    if platform_type == 'ios':",
        "       environment_archs = settings['//command_line_option:ios_multi_cpus']",
        "    else:",
        "       environment_archs = settings['//command_line_option:%s_cpus' % platform_type]",
        "    if not environment_archs and platform_type == 'ios':",
        "       cpu_value = settings['//command_line_option:cpu']",
        "       if cpu_value.startswith('ios_'):",
        "           environment_archs = [cpu_value[4:]]",
        "    if not environment_archs:",
        "        environment_archs = ['x86_64']",
        "    for environment_arch in environment_archs:",
        "        found_cpu = 'ios_{}'.format(environment_arch)",
        "        output_dictionary[found_cpu] = _command_line_options(",
        "            environment_arch = environment_arch,",
        "            platform_type = platform_type,",
        "            settings = settings,",
        "        )",
        "    return output_dictionary",
        "apple_platform_split_transition = transition(",
        "    implementation = _apple_platform_split_transition_impl,",
        "    inputs = _apple_platform_transition_inputs,",
        "    outputs = _apple_rule_base_transition_outputs,",
        ")",
        "def apple_binary_starlark_impl(ctx):",
        "    all_avoid_deps = list(ctx.attr.avoid_deps)",
        "    binary_type = ctx.attr.binary_type",
        "    bundle_loader = ctx.attr.bundle_loader",
        "    linkopts = []",
        "    link_inputs = []",
        "    variables_extension = {}",
        "    variables_extension.update(ctx.attr.string_variables_extension)",
        "    variables_extension.update(ctx.attr.string_list_variables_extension)",
        "    if binary_type == 'dylib':",
        "        linkopts.append('-dynamiclib')",
        "    elif binary_type == 'loadable_bundle':",
        "        linkopts.extend(['-bundle', '-Xlinker', '-rpath', '-Xlinker',"
            + " '@loader_path/Frameworks'])",
        "    if ctx.attr.bundle_loader:",
        "        bundle_loader = ctx.attr.bundle_loader",
        "        bundle_loader_file = bundle_loader[apple_common.AppleExecutableBinary].binary",
        "        all_avoid_deps.append(bundle_loader)",
        "        linkopts.extend(['-bundle_loader', bundle_loader_file.path])",
        "        link_inputs.append(bundle_loader_file)",
        "    link_result = apple_common.link_multi_arch_binary(",
        "        ctx = ctx,",
        "        avoid_deps = all_avoid_deps,",
        "        extra_linkopts = linkopts,",
        "        extra_link_inputs = link_inputs,",
        "        extra_requested_features = ctx.attr.extra_requested_features,",
        "        extra_disabled_features = ctx.attr.extra_disabled_features,",
        "        stamp = ctx.attr.stamp,",
        "        variables_extension = variables_extension,",
        "    )",
        "    processed_binary = ctx.actions.declare_file('{}_lipobin'.format(ctx.label.name))",
        "    lipo_inputs = [output.binary for output in link_result.outputs]",
        "    if len(lipo_inputs) > 1:",
        "        apple_env = {}",
        "        xcode_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig]",
        "        apple_env.update(apple_common.apple_host_system_env(xcode_config))",
        "        apple_env.update(",
        "            apple_common.target_apple_env(",
        "                xcode_config,",
        "                ctx.fragments.apple.single_arch_platform,",
        "            ),",
        "        )",
        "        args = ctx.actions.args()",
        "        args.add('-create')",
        "        args.add_all(lipo_inputs)",
        "        args.add('-output', processed_binary)",
        "        ctx.actions.run(",
        "            arguments = [args],",
        "            env = apple_env,",
        "            executable = '/usr/bin/lipo',",
        "            execution_requirements = xcode_config.execution_info(),",
        "            inputs = lipo_inputs,",
        "            outputs = [processed_binary],",
        "        )",
        "    else:",
        "        ctx.actions.symlink(target_file = lipo_inputs[0], output = processed_binary)",
        "    providers = [",
        "        DefaultInfo(files=depset([processed_binary])),",
        "        OutputGroupInfo(**link_result.output_groups),",
        "        link_result.debug_outputs_provider,",
        "    ]",
        "    if binary_type == 'executable':",
        "        providers.append(apple_common.new_executable_binary_provider(",
        "            binary = processed_binary,",
        "            cc_info = link_result.cc_info,",
        "            objc = link_result.objc,",
        "        ))",
        "    return providers",
        "apple_binary_starlark = rule(",
        "    apple_binary_starlark_impl,",
        "    attrs = {",
        "        '_child_configuration_dummy': attr.label(",
        "            cfg=apple_platform_split_transition,",
        "            default=Label('" + toolsRepo + "//tools/cpp:current_cc_toolchain'),),",
        "        '_dummy_lib': attr.label(",
        "            default = Label('" + toolsLoc + "/dummy:dummy_lib'),),",
        "        '_grep_includes': attr.label(",
        "            cfg = 'exec',",
        "            allow_single_file = True,",
        "            executable = True,",
        "            default = Label('" + toolsRepo + "//tools/cpp:grep-includes'),",
        "        ),",
        "        '_j2objc_dead_code_pruner': attr.label(",
        "            executable = True,",
        "            allow_files=True,",
        "            cfg = config.exec('j2objc'),",
        "            default = Label('" + toolsLoc + ":j2objc_dead_code_pruner_binary'),),",
        "        '_xcode_config': attr.label(",
        "            default=configuration_field(",
        "                fragment='apple', name='xcode_config_label'),),",
        "        'additional_linker_inputs': attr.label_list(",
        "            allow_files = True,",
        "        ),",
        "        'avoid_deps': attr.label_list(default=[]),",
        "        'binary_type': attr.string(",
        "             default = 'executable',",
        "             values = ['dylib', 'executable', 'loadable_bundle']",
        "        ),",
        "        'bundle_loader': attr.label(),",
        "        'deps': attr.label_list(",
        "             cfg=apple_platform_split_transition,",
        "        ),",
        "        'linkopts': attr.string_list(),",
        "        'extra_requested_features': attr.string_list(),",
        "        'extra_disabled_features': attr.string_list(),",
        "        'platform_type': attr.string(),",
        "        'minimum_os_version': attr.string(),",
        "        'stamp': attr.int(values=[-1,0,1],default=-1),",
        "        'string_variables_extension': attr.string_dict(),",
        "        'string_list_variables_extension': attr.string_list_dict(),",
        "        '_allowlist_function_transition': attr.label(",
        "            default = '//tools/allowlists/function_transition_allowlist',",
        "        ),",
        "    },",
        "    exec_groups = {",
        "        'j2objc': exec_group()",
        "    },",
        "    fragments = ['apple', 'objc', 'cpp',],",
        ")");
    scratch.overwriteFile(
        "tools/allowlists/function_transition_allowlist/BUILD",
        "package_group(",
        "    name = 'function_transition_allowlist',",
        "    packages = [",
        "        '//...',",
        "    ],",
        ")");
  }

  protected CommandAction compileAction(String ownerLabel, String objFileName) throws Exception {
    Action archiveAction = archiveAction(ownerLabel);
    return (CommandAction)
        getGeneratingAction(
            getFirstArtifactEndingWith(archiveAction.getInputs(), "/" + objFileName));
  }

  /**
   * Verifies simply that some rule type creates the {@link CompilationArtifacts} object
   * successfully; in particular, makes sure it is not ignoring attributes. If the scope of {@link
   * CompilationArtifacts} expands, make sure this method tests it properly.
   *
   * <p>This test only makes sure the attributes are not being ignored - it does not test any other
   * functionality in depth, which is covered by other unit tests.
   */
  protected void checkPopulatesCompilationArtifacts(RuleType ruleType) throws Exception {
    scratch.file("x/a.m");
    scratch.file("x/b.m");
    scratch.file("x/c.pch");
    ruleType.scratchTarget(scratch, "srcs", "['a.m']", "non_arc_srcs", "['b.m']", "pch", "'c.pch'");
    ImmutableList<String> includeFlags = ImmutableList.of("-include", "x/c.pch");
    assertContainsSublist(compileAction("//x:x", "a.o").getArguments(), includeFlags);
    assertContainsSublist(compileAction("//x:x", "b.o").getArguments(), includeFlags);
  }

  protected void checkProvidesHdrsAndIncludes(RuleType ruleType, Optional<String> privateHdr)
      throws Exception {
    scratch.file("x/a.h");
    ruleType.scratchTarget(scratch, "hdrs", "['a.h']", "includes", "['incdir']");
    CcCompilationContext ccCompilationContext =
        getConfiguredTarget("//x:x").get(CcInfo.PROVIDER).getCcCompilationContext();
    ImmutableList<String> declaredIncludeSrcs =
        ccCompilationContext.getDeclaredIncludeSrcs().toList().stream()
            .map(x -> removeConfigFragment(x.getExecPathString()))
            .collect(toImmutableList());
    if (privateHdr.isPresent()) {
      assertThat(declaredIncludeSrcs)
          .containsExactly(
              getSourceArtifact("x/a.h").getExecPathString(),
              getSourceArtifact(privateHdr.get()).getExecPathString());
    } else {
      assertThat(declaredIncludeSrcs)
          .containsExactly(getSourceArtifact("x/a.h").getExecPathString());
    }
    assertThat(
            ccCompilationContext.getIncludeDirs().stream()
                .map(x -> removeConfigFragment(x.toString())))
        .containsExactly(PathFragment.create("x/incdir").toString(), OUTPUTDIR + "/x/incdir");
  }

  protected void checkCompilesWithHdrs(RuleType ruleType) throws Exception {
    scratch.file("x/a.m");
    scratch.file("x/a.h");
    ruleType.scratchTarget(scratch, "srcs", "['a.m']", "hdrs", "['a.h']");
    assertThat(compileAction("//x:x", "a.o").getPossibleInputsForTesting().toList())
        .contains(getSourceArtifact("x/a.h"));
  }

  protected ImmutableList<Artifact> getAllObjectFilesLinkedInBin(Artifact bin) {
    ImmutableList.Builder<Artifact> objects = ImmutableList.builder();
    CommandAction binAction = (CommandAction) getGeneratingAction(bin);
    for (Artifact binActionArtifact : binAction.getInputs().toList()) {
      if (binActionArtifact.getRootRelativePath().getPathString().endsWith(".a")) {
        CommandAction linkAction = (CommandAction) getGeneratingAction(binActionArtifact);
        for (Artifact linkActionArtifact : linkAction.getInputs().toList()) {
          if (linkActionArtifact.getRootRelativePath().getPathString().endsWith(".o")) {
            objects.add(linkActionArtifact);
          }
        }
      }
    }
    return objects.build();
  }

  /**
   * Ensures that all middleman artifacts in the action input are expanded so that the real inputs
   * are also included.
   */
  protected Iterable<Artifact> getExpandedActionInputs(Action action) {
    List<Artifact> containedArtifacts = new ArrayList<>();
    for (Artifact input : action.getInputs().toList()) {
      if (input.isMiddlemanArtifact()) {
        Action middlemanAction = getGeneratingAction(input);
        Iterables.addAll(containedArtifacts, getExpandedActionInputs(middlemanAction));
      }
      containedArtifacts.add(input);
    }
    return containedArtifacts;
  }

  protected void checkFrameworkDepLinkFlags(RuleType ruleType, ExtraLinkArgs extraLinkArgs)
      throws Exception {
    scratch.file(
        "libs/defs.bzl",
        "def _custom_static_framework_import_impl(ctx):",
        "  return [",
        "      apple_common.new_objc_provider(",
        "          static_framework_file=depset(ctx.files.link_inputs)",
        "      ),",
        "      CcInfo(",
        "          linking_context = cc_common.create_linking_context(",
        "              linker_inputs = depset(direct = [",
        "                  cc_common.create_linker_input(",
        "                      owner = ctx.label,",
        "                      libraries = depset([",
        "                          cc_common.create_library_to_link(",
        "                              actions = ctx.actions,",
        "                              static_library = library,",
        "                          ) for library in ctx.files.link_inputs",
        "                      ]),",
        "                  ),",
        "              ]),",
        "          ),",
        "      ),",
        "  ]",
        "custom_static_framework_import = rule(",
        "    _custom_static_framework_import_impl,",
        "    attrs={'link_inputs': attr.label_list(allow_files=True)},",
        ")");
    scratch.file("libs/buzzbuzz.framework/buzzbuzz");
    scratch.file(
        "libs/BUILD",
        "load(':defs.bzl', 'custom_static_framework_import')",
        "objc_library(",
        "    name = 'objc_lib',",
        "    srcs = ['a.m'],",
        "    deps = [':my_framework'],",
        ")",
        "custom_static_framework_import(",
        "    name = 'my_framework',",
        "    link_inputs = glob(['buzzbuzz.framework/*']),",
        ")");

    ruleType.scratchTarget(scratch, "deps", "['//libs:objc_lib']");

    CommandAction linkAction = linkAction("//x:x");
    Artifact binArtifact = getFirstArtifactEndingWith(linkAction.getOutputs(), "x_bin");
    Artifact objList = getFirstArtifactEndingWith(linkAction.getInputs(), "x-linker.objlist");

    verifyLinkAction(
        binArtifact,
        objList,
        "x86_64",
        ImmutableList.of("libobjc_lib.a"),
        ImmutableList.of(PathFragment.create("libs/buzzbuzz")),
        extraLinkArgs);
  }

  protected void checkBundleLoaderIsCorrectlyPassedToTheLinker(RuleType ruleType) throws Exception {
    if (!"apple_binary_starlark".equals(ruleType.getRuleTypeName())) {
      addAppleBinaryStarlarkRule(scratch);
    }
    scratch.file(
        "bin/BUILD",
        "load('//test_starlark:apple_binary_starlark.bzl', 'apple_binary_starlark')",
        "objc_library(",
        "    name = 'lib',",
        "    srcs = ['a.m'],",
        ")",
        "apple_binary_starlark(",
        "    name = 'bin',",
        "    deps = [':lib'],",
        "    platform_type = 'ios',",
        ")");

    ruleType.scratchTarget(
        scratch, "binary_type", "'loadable_bundle'", "bundle_loader", "'//bin:bin'");
    ConfiguredTarget binTarget = getConfiguredTarget("//bin:bin");

    CommandAction linkAction = linkAction("//x:x");
    assertThat(Joiner.on(" ").join(linkAction.getArguments()))
        .contains("-bundle_loader " + getBinArtifact("bin_lipobin", binTarget).getExecPath());
    assertThat(Joiner.on(" ").join(linkAction.getArguments()))
        .contains("-Xlinker -rpath -Xlinker @loader_path/Frameworks");
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
        Artifact binArtifact = getFirstArtifactEndingWith(lipoAction.getInputs(), "fl.a");
        linkAction = (CommandAction) getGeneratingAction(binArtifact);
      }
    }
    return linkAction;
  }

  protected Action actionProducingArtifact(String targetLabel, String artifactSuffix)
      throws Exception {
    ConfiguredTarget libraryTarget = getConfiguredTarget(targetLabel);
    Label parsedLabel = Label.parseCanonical(targetLabel);
    Artifact linkedLibrary = getBinArtifact(parsedLabel.getName() + artifactSuffix, libraryTarget);
    return getGeneratingAction(linkedLibrary);
  }

  protected List<String> rootedIncludePaths(String... unrootedPaths) {
    ImmutableList.Builder<String> rootedPaths = new ImmutableList.Builder<>();
    for (String unrootedPath : unrootedPaths) {
      rootedPaths
          .add(unrootedPath)
          .add(
              removeConfigFragment(
                  PathFragment.create(TestConstants.PRODUCT_NAME + "-out/any-config-fragment/bin")
                      .getRelative(unrootedPath)
                      .getSafePathString()));
    }
    return rootedPaths.build();
  }

  protected void checkClangCoptsForCompilationMode(
      RuleType ruleType, CompilationMode mode, CodeCoverageMode codeCoverageMode) throws Exception {
    ImmutableList.Builder<String> allExpectedCoptsBuilder =
        ImmutableList.<String>builder()
            .addAll(CompilationSupport.DEFAULT_COMPILER_FLAGS)
            .addAll(compilationModeCopts(mode));

    switch (codeCoverageMode) {
      case NONE:
        useConfiguration(
            "--apple_platform_type=ios", "--compilation_mode=" + compilationModeFlag(mode));
        break;
      case GCOV:
        allExpectedCoptsBuilder.addAll(CompilationSupport.CLANG_GCOV_COVERAGE_FLAGS);
        useConfiguration(
            "--apple_platform_type=ios",
            "--collect_code_coverage",
            "--compilation_mode=" + compilationModeFlag(mode));
        break;
      case LLVMCOV:
        allExpectedCoptsBuilder.addAll(CompilationSupport.CLANG_LLVM_COVERAGE_FLAGS);
        useConfiguration(
            "--apple_platform_type=ios",
            "--collect_code_coverage",
            "--experimental_use_llvm_covmap",
            "--compilation_mode=" + compilationModeFlag(mode));
        break;
    }
    scratch.file("x/a.m");
    ruleType.scratchTarget(scratch, "srcs", "['a.m']");

    CommandAction compileActionA = compileAction("//x:x", "a.o");

    assertThat(compileActionA.getArguments())
        .containsAtLeastElementsIn(allExpectedCoptsBuilder.build());
  }

  protected void checkClangCoptsForDebugModeWithoutGlib(RuleType ruleType) throws Exception {
    ImmutableList.Builder<String> allExpectedCoptsBuilder =
        ImmutableList.<String>builder()
            .addAll(CompilationSupport.DEFAULT_COMPILER_FLAGS)
            .addAll(ObjcConfiguration.DBG_COPTS);

    useConfiguration(
        "--apple_platform_type=ios", "--compilation_mode=dbg", "--objc_debug_with_GLIBCXX=false");
    scratch.file("x/a.m");
    ruleType.scratchTarget(scratch, "srcs", "['a.m']");

    CommandAction compileActionA = compileAction("//x:x", "a.o");

    assertThat(compileActionA.getArguments())
        .containsAtLeastElementsIn(allExpectedCoptsBuilder.build())
        .inOrder();
    assertThat(compileActionA.getArguments()).doesNotContain("-D_GLIBCXX_DEBUG");
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

    topLevelRuleType.scratchTarget(
        scratch,
        "srcs",
        "['a.m']",
        "non_arc_srcs",
        "['b.m']",
        "deps",
        "['//lib2:lib2']",
        "defines",
        "['E=baz']",
        "copts",
        "['explicit_copt']");
  }

  protected void checkReceivesTransitivelyPropagatedDefines(RuleType ruleType) throws Exception {
    addTransitiveDefinesUsage(ruleType);
    List<String> expectedArgs =
        ImmutableList.of("-DA=foo", "-DB", "-DC=bar", "-DD", "-DE=baz", "explicit_copt");
    List<String> compileActionAArgs = compileAction("//x:x", "a.o").getArguments();
    List<String> compileActionBArgs = compileAction("//x:x", "b.o").getArguments();
    for (String expectedArg : expectedArgs) {
      assertThat(compileActionAArgs).contains(expectedArg);
      assertThat(compileActionBArgs).contains(expectedArg);
    }
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
    assertThat(compileAction.getArguments()).containsAtLeast("-Dfoo", "-Dbar");
  }

  protected void checkSdkIncludesUsedInCompileAction(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch, "sdk_includes", "['foo', 'bar/baz']", "srcs", "['a.m', 'b.m']");
    String sdkIncludeDir = AppleToolchain.sdkDir() + "/usr/include";
    // we remove spaces, since the legacy rules put a space after "-I" in include paths.
    String compileActionACommandLine =
        Joiner.on(" ").join(compileAction("//x:x", "a.o").getArguments()).replace(" ", "");
    assertThat(compileActionACommandLine).contains("-I" + sdkIncludeDir + "/foo");
    assertThat(compileActionACommandLine).contains("-I" + sdkIncludeDir + "/bar/baz");

    String compileActionBCommandLine =
        Joiner.on(" ").join(compileAction("//x:x", "b.o").getArguments()).replace(" ", "");
    assertThat(compileActionBCommandLine).contains("-I" + sdkIncludeDir + "/foo");
    assertThat(compileActionBCommandLine).contains("-I" + sdkIncludeDir + "/bar/baz");
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
    createLibraryTargetWriter("//bin:main_lib")
        .setAndCreateFiles("srcs", "b.m")
        .setList("deps", "//lib:lib")
        .setList("sdk_includes", "from_bin")
        .write();
    String sdkIncludeDir = AppleToolchain.sdkDir() + "/usr/include";

    // We remove spaces because the crosstool case does not use spaces for include paths.
    String compileAArgs =
        Joiner.on("").join(compileAction("//lib:lib", "a.o").getArguments()).replace(" ", "");
    assertThat(compileAArgs).contains("-I" + sdkIncludeDir + "/from_lib");
    assertThat(compileAArgs).contains("-I" + sdkIncludeDir + "/foo");
    assertThat(compileAArgs).contains("-I" + sdkIncludeDir + "/bar/baz");

    String compileBArgs =
        Joiner.on("").join(compileAction("//bin:main_lib", "b.o").getArguments()).replace(" ", "");
    assertThat(compileBArgs).contains("-I" + sdkIncludeDir + "/from_bin");
    assertThat(compileBArgs).contains("-I" + sdkIncludeDir + "/from_lib");
    assertThat(compileBArgs).contains("-I" + sdkIncludeDir + "/foo");
    assertThat(compileBArgs).contains("-I" + sdkIncludeDir + "/bar/baz");
  }

  public void checkAllowVariousNonBlacklistedTypesInHeaders(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch, "hdrs", "['foo.foo', 'NoExtension', 'bar.inc', 'baz.hpp']");
    assertThat(view.hasErrors(getConfiguredTarget("//x:x"))).isFalse();
  }

  public void checkWarningForBlacklistedTypesInHeaders(RuleType ruleType) throws Exception {
    checkWarning(
        "x1",
        "x1",
        "file 'foo.a' from target '//x1:foo.a' is not allowed in hdrs",
        ruleType.target(scratch, "x1", "x1", "hdrs", "['foo.a']"));
    checkWarning(
        "x2",
        "x2",
        "file 'bar.o' from target '//x2:bar.o' is not allowed in hdrs",
        ruleType.target(scratch, "x2", "x2", "hdrs", "['bar.o']"));
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

  /**
   * Verifies that the given rule supports the minimum_os attribute, and adds compile and link args
   * to set the minimum os appropriately, including compile args for dependencies.
   *
   * @param ruleType the rule to test
   */
  protected void checkMinimumOsLinkAndCompileArg(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch, "deps", "['//package:objcLib']", "minimum_os_version", "'5.4'");
    scratch.file("package/BUILD", "objc_library(name = 'objcLib', srcs = [ 'b.m' ])");
    useConfiguration("--xcode_version=5.8");

    CommandAction linkAction = linkAction("//x:x");
    CommandAction objcLibArchiveAction =
        (CommandAction)
            getGeneratingAction(getFirstArtifactEndingWith(linkAction.getInputs(), "libobjcLib.a"));
    CommandAction objcLibCompileAction =
        (CommandAction)
            getGeneratingAction(
                getFirstArtifactEndingWith(objcLibArchiveAction.getInputs(), "b.o"));

    String linkArgs = Joiner.on(" ").join(linkAction.getArguments());
    String compileArgs = Joiner.on(" ").join(objcLibCompileAction.getArguments());
    assertThat(linkArgs).contains("-mios-simulator-version-min=5.4");
    assertThat(compileArgs).contains("-mios-simulator-version-min=5.4");
  }

  /**
   * Verifies that the given rule supports the minimum_os attribute under the watchOS platform type,
   * and adds compile and link args to set the minimum os appropriately for watchos, including
   * compile args for dependencies.
   *
   * @param ruleType the rule to test
   */
  protected void checkMinimumOsLinkAndCompileArg_watchos(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(
        scratch,
        "deps",
        "['//package:objcLib']",
        "platform_type",
        "'watchos'",
        "minimum_os_version",
        "'5.4'");
    scratch.file("package/BUILD", "objc_library(name = 'objcLib', srcs = [ 'b.m' ])");
    useConfiguration("--xcode_version=5.8");

    CommandAction linkAction = linkAction("//x:x");
    CommandAction objcLibArchiveAction =
        (CommandAction)
            getGeneratingAction(getFirstArtifactEndingWith(linkAction.getInputs(), "libobjcLib.a"));
    CommandAction objcLibCompileAction =
        (CommandAction)
            getGeneratingAction(
                getFirstArtifactEndingWith(objcLibArchiveAction.getInputs(), "b.o"));

    String linkArgs = Joiner.on(" ").join(linkAction.getArguments());
    String compileArgs = Joiner.on(" ").join(objcLibCompileAction.getArguments());
    assertThat(linkArgs).contains("-mwatchos-simulator-version-min=5.4");
    assertThat(compileArgs).contains("-mwatchos-simulator-version-min=5.4");
  }

  private void checkAvoidDepsDependenciesSetupFramework() throws Exception {
    scratch.file(
        "fx/defs.bzl",
        "def _custom_dynamic_framework_import_impl(ctx):",
        "    toolchain = ctx.attr._cc_toolchain[cc_common.CcToolchainInfo]",
        "    feature_configuration = cc_common.configure_features(",
        "        ctx = ctx,",
        "        cc_toolchain=toolchain,",
        "        requested_features = ctx.features,",
        "        unsupported_features = ctx.disabled_features,",
        "    )",
        "    objc_provider = apple_common.new_objc_provider(",
        "        dynamic_framework_file=depset(ctx.files.link_inputs),",
        "    )",
        "    cc_info = CcInfo(",
        "        linking_context = cc_common.create_linking_context(",
        "            linker_inputs = depset(direct = [",
        "                cc_common.create_linker_input(",
        "                    owner = ctx.label,",
        "                    libraries = depset([",
        "                        cc_common.create_library_to_link(",
        "                            actions = ctx.actions,",
        "                            cc_toolchain = toolchain,",
        "                            feature_configuration = feature_configuration,",
        "                            dynamic_library = library,",
        "                        ) for library in ctx.files.link_inputs",
        "                    ]),",
        "                ),",
        "            ]),",
        "        ),",
        "    )",
        "    return [",
        "        cc_info,",
        "        objc_provider,",
        "        apple_common.new_dynamic_framework_provider(",
        "            cc_info = cc_info,",
        "            objc = objc_provider,",
        "        ),",
        "    ]",
        "custom_dynamic_framework_import = rule(",
        "    _custom_dynamic_framework_import_impl,",
        "    attrs={",
        "        'link_inputs': attr.label_list(allow_files=True),",
        "        '_cc_toolchain': attr.label(default = '//fx:alias'),",
        "    },",
        "    fragments = ['cpp'],",
        ")");
    scratch.file("fx/MyFramework.framework/MyFramework");
    scratch.file(
        "fx/BUILD",
        "load(':defs.bzl', 'custom_dynamic_framework_import')",
        "cc_toolchain_alias(name='alias')",
        "custom_dynamic_framework_import(",
        "    name = 'framework_import',",
        "    link_inputs = ['MyFramework.framework/MyFramework'],",
        ")");
  }

  protected void checkAvoidDepsDependencies(RuleType ruleType, ExtraLinkArgs extraLinkArgs)
      throws Exception {
    useConfiguration("--ios_multi_cpus=i386,x86_64");

    ruleType.scratchTarget(scratch, "avoid_deps", "['//fx:framework_import']");

    checkAvoidDepsDependenciesSetupFramework();

    Action lipobinAction = lipoBinAction("//x:x");

    String i386Bin = configurationBin("i386", ConfigurationDistinguisher.APPLEBIN_IOS) + "x/x_bin";
    String i386Filelist =
        configurationBin("i386", ConfigurationDistinguisher.APPLEBIN_IOS) + "x/x-linker.objlist";
    String x8664Bin =
        configurationBin("x86_64", ConfigurationDistinguisher.APPLEBIN_IOS) + "x/x_bin";
    String x8664Filelist =
        configurationBin("x86_64", ConfigurationDistinguisher.APPLEBIN_IOS) + "x/x-linker.objlist";

    Artifact i386BinArtifact = getFirstArtifactEndingWith(lipobinAction.getInputs(), i386Bin);
    Artifact i386FilelistArtifact =
        getFirstArtifactEndingWith(getGeneratingAction(i386BinArtifact).getInputs(), i386Filelist);
    Artifact x8664BinArtifact = getFirstArtifactEndingWith(lipobinAction.getInputs(), x8664Bin);
    Artifact x8664FilelistArtifact =
        getFirstArtifactEndingWith(
            getGeneratingAction(x8664BinArtifact).getInputs(), x8664Filelist);

    verifyLinkAction(
        i386BinArtifact,
        i386FilelistArtifact,
        "i386",
        ImmutableList.of(),
        ImmutableList.of(PathFragment.create("fx/MyFramework")),
        extraLinkArgs);
    verifyLinkAction(
        x8664BinArtifact,
        x8664FilelistArtifact,
        "x86_64",
        ImmutableList.of(),
        ImmutableList.of(PathFragment.create("fx/MyFramework")),
        extraLinkArgs);
  }

  protected void checkLipoBinaryAction(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch);

    useConfiguration("--ios_multi_cpus=i386,x86_64");

    CommandAction action = (CommandAction) lipoBinAction("//x:x");
    String i386Bin = configurationBin("i386", ConfigurationDistinguisher.APPLEBIN_IOS) + "x/x_bin";
    String x8664Bin =
        configurationBin("x86_64", ConfigurationDistinguisher.APPLEBIN_IOS) + "x/x_bin";

    assertThat(Artifact.asExecPaths(action.getInputs()))
        .comparingElementsUsing(MATCHES_REGEX)
        .containsExactly(i386Bin, x8664Bin);

    assertThat(action.getArguments())
        .comparingElementsUsing(MATCHES_REGEX)
        .containsExactly(
            "/usr/bin/lipo",
            "-create",
            i386Bin,
            x8664Bin,
            "-output",
            execPathEndingWith(action.getOutputs(), "x_lipobin"))
        .inOrder();

    assertThat(Artifact.toRootRelativePaths(action.getOutputs())).containsExactly("x/x_lipobin");
    assertRequiresDarwin(action);
  }

  protected void checkMultiarchCcDep(RuleType ruleType) throws Exception {
    useConfiguration("--ios_multi_cpus=i386,x86_64");
    ruleType.scratchTarget(scratch, "deps", "['//package:cclib']");
    scratch.file("package/BUILD", "cc_library(name = 'cclib', srcs = ['dep.c'])");

    Action appLipoAction = actionProducingArtifact("//x:x", "_lipobin");
    String i386Prefix = configurationBin("i386", ConfigurationDistinguisher.APPLEBIN_IOS);
    String x8664Prefix = configurationBin("x86_64", ConfigurationDistinguisher.APPLEBIN_IOS);

    CommandAction i386BinAction =
        (CommandAction)
            getGeneratingAction(
                getFirstArtifactEndingWith(appLipoAction.getInputs(), i386Prefix + "x/x_bin"));

    CommandAction x8664BinAction =
        (CommandAction)
            getGeneratingAction(
                getFirstArtifactEndingWith(appLipoAction.getInputs(), x8664Prefix + "x/x_bin"));

    verifyObjlist(i386BinAction, "package/libcclib.a");
    verifyObjlist(x8664BinAction, "package/libcclib.a");

    assertThat(Artifact.asExecPaths(i386BinAction.getInputs()))
        .containsAtLeast(i386Prefix + "package/libcclib.a", i386Prefix + "x/x-linker.objlist");
    assertThat(Artifact.asExecPaths(x8664BinAction.getInputs()))
        .containsAtLeast(x8664Prefix + "package/libcclib.a", x8664Prefix + "x/x-linker.objlist");
  }

  // Regression test for b/32310268.
  protected void checkAliasedLinkoptsThroughObjcLibrary(RuleType ruleType) throws Exception {
    useConfiguration("--cpu=ios_i386");
    scratch.file(
        "bin/BUILD",
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

    ruleType.scratchTarget(scratch, "deps", "['//bin:objclib']");

    // Frameworks should get placed together with no duplicates.
    assertThat(Joiner.on(" ").join(linkAction("//x").getArguments())).contains("-somelinkopt");
  }

  protected void checkCcDependencyLinkoptsArePropagatedToLinkAction(RuleType ruleType)
      throws Exception {
    useConfiguration("--cpu=ios_i386");

    scratch.file(
        "bin/BUILD",
        "cc_library(",
        "    name = 'cclib1',",
        "    srcs = ['dep1.c'],",
        "    linkopts = ['-framework', 'F1', '-framework', 'F2', '-weak_framework', 'F3',"
            + " '-Wl,--other-opt'],",
        ")",
        "cc_library(",
        "    name = 'cclib2',",
        "    srcs = ['dep2.c'],",
        "    linkopts = ['-another-opt', '-framework', 'F2', '-weak_framework', 'F3'],",
        "    deps = ['cclib1'],",
        ")",
        "cc_library(",
        "    name = 'cclib3',",
        "    srcs = ['dep2.c'],",
        "    linkopts = ['-one-more-opt', '-framework', 'UIKit'],",
        "    deps = ['cclib1'],",
        ")");

    ruleType.scratchTarget(scratch, "deps", "['//bin:cclib2', '//bin:cclib3']");

    List<String> linkArgs = linkAction("//x").getArguments();
    String linkArgsCommandLine = Joiner.on(" ").join(linkArgs);
    // Frameworks from the CROSSTOOL "apply_implicit_frameworks" feature should be present.
    assertThat(linkArgsCommandLine).contains("-framework Foundation");
    assertThat(linkArgsCommandLine).contains("-framework UIKit");
    // Frameworks included in linkopts by the user should get placed together with no duplicates.
    // (They may duplicate the ones inserted by the CROSSTOOL feature, but we don't test that here.)
    assertThat(linkArgsCommandLine).contains("-framework F1");
    assertThat(linkArgsCommandLine).contains("-framework F2");
    assertThat(linkArgsCommandLine).contains("-weak_framework F3");
    assertThat(linkArgsCommandLine.indexOf("-framework F2"))
        .isEqualTo(linkArgsCommandLine.lastIndexOf("-framework F2"));
    assertThat(linkArgsCommandLine.indexOf("-weak_framework F3"))
        .isEqualTo(linkArgsCommandLine.lastIndexOf("-weak_framework F3"));
    assertThat(linkArgs)
        .containsAtLeast("-another-opt", "-one-more-opt", "-Wl,--other-opt")
        .inOrder();
  }

  protected void checkObjcLibraryLinkoptsArePropagatedToLinkAction(RuleType ruleType)
      throws Exception {
    useConfiguration("--cpu=ios_i386");

    scratch.file(
        "bin/BUILD",
        "objc_library(",
        "    name = 'objclib1',",
        "    srcs = ['dep1.m'],",
        "    linkopts = ['-framework', 'F1', '-framework', 'F2', '-weak_framework', 'F3',"
            + " '-Wl,--other-opt'],",
        ")",
        "objc_library(",
        "    name = 'objclib2',",
        "    srcs = ['dep2.m'],",
        "    linkopts = ['-another-opt', '-framework', 'F2', '-weak_framework', 'F3'],",
        "    deps = [':objclib1'],",
        ")",
        "objc_library(",
        "    name = 'objclib3',",
        "    srcs = ['dep3.m'],",
        "    linkopts = ['-one-more-opt', '-framework', 'UIKit'],",
        "    deps = [':objclib1'],",
        ")");

    ruleType.scratchTarget(scratch, "deps", "['//bin:objclib2', '//bin:objclib3']");

    List<String> linkArgs = linkAction("//x").getArguments();

    String linkArgsCommandLine = Joiner.on(" ").join(linkArgs);
    // Frameworks from the CROSSTOOL "apply_implicit_frameworks" feature should be present.
    assertThat(linkArgsCommandLine).contains("-framework Foundation");
    assertThat(linkArgsCommandLine).contains("-framework UIKit");
    // Frameworks included in linkopts by the user should get placed together with no duplicates.
    // (They may duplicate the ones inserted by the CROSSTOOL feature, but we don't test that here.)
    assertThat(linkArgsCommandLine).contains("-framework F1");
    assertThat(linkArgsCommandLine).contains("-framework F2");
    assertThat(linkArgsCommandLine).contains("-weak_framework F3");
    assertThat(linkArgsCommandLine.indexOf("-framework F2"))
        .isEqualTo(linkArgsCommandLine.lastIndexOf("-framework F2"));
    assertThat(linkArgsCommandLine.indexOf("-weak_framework F3"))
        .isEqualTo(linkArgsCommandLine.lastIndexOf("-weak_framework F3"));
    assertThat(linkArgs)
        .containsAtLeast("-another-opt", "-one-more-opt", "-Wl,--other-opt")
        .inOrder();
  }

  protected void checkLinkInputsInLinkAction(RuleType ruleType) throws Exception {
    useConfiguration("--cpu=ios_i386");

    scratch.file(
        "bin/defs.bzl",
        "def _custom_rule_impl(ctx):",
        "    linker_input = cc_common.create_linker_input(",
        "        owner = ctx.label,",
        "        additional_inputs = depset(direct = ctx.files.link_inputs),",
        "    )",
        "    return [",
        "        apple_common.new_objc_provider(),",
        "        CcInfo(",
        "            linking_context = cc_common.create_linking_context(",
        "                linker_inputs = depset(direct = [linker_input]),",
        "            ),",
        "        ),",
        "    ]",
        "custom_rule = rule(",
        "    _custom_rule_impl,",
        "    attrs={'link_inputs': attr.label_list(allow_files=True)},",
        ")");

    scratch.file("bin/input.txt");

    scratch.file(
        "bin/BUILD",
        "load('//bin:defs.bzl', 'custom_rule')",
        "custom_rule(",
        "    name = 'custom',",
        "    link_inputs = ['input.txt'],",
        ")");

    ruleType.scratchTarget(scratch, "deps", "['//bin:custom']");

    Artifact inputFile = getSourceArtifact("bin/input.txt");
    assertThat(linkAction("//x").getInputs().toList()).contains(inputFile);
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
    useConfiguration("--apple_platform_type=ios", "--cpu=ios_arm64");

    CommandAction action = linkAction("//x:x");

    assertAppleSdkPlatformEnv(action, "iPhoneOS");
  }

  protected void checkAppleSdkWatchsimulatorPlatformEnv(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch, "platform_type", "'watchos'");
    useConfiguration("--watchos_cpus=i386");

    Action lipoAction = actionProducingArtifact("//x:x", "_lipobin");

    String i386Bin =
        configurationBin("i386", ConfigurationDistinguisher.APPLEBIN_WATCHOS) + "x/x_bin";
    Artifact binArtifact = getFirstArtifactEndingWith(lipoAction.getInputs(), i386Bin);
    CommandAction linkAction = (CommandAction) getGeneratingAction(binArtifact);

    assertAppleSdkPlatformEnv(linkAction, "WatchSimulator");
  }

  protected void checkAppleSdkWatchosPlatformEnv(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch, "platform_type", "'watchos'");
    useConfiguration("--watchos_cpus=armv7k");

    Action lipoAction = actionProducingArtifact("//x:x", "_lipobin");

    String armv7kBin =
        configurationBin("armv7k", ConfigurationDistinguisher.APPLEBIN_WATCHOS) + "x/x_bin";
    Artifact binArtifact = getFirstArtifactEndingWith(lipoAction.getInputs(), armv7kBin);
    CommandAction linkAction = (CommandAction) getGeneratingAction(binArtifact);

    assertAppleSdkPlatformEnv(linkAction, "WatchOS");
  }

  protected void checkAppleSdkTvsimulatorPlatformEnv(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch, "platform_type", "'tvos'");
    useConfiguration("--tvos_cpus=x86_64");

    CommandAction linkAction = linkAction("//x:x");

    assertAppleSdkPlatformEnv(linkAction, "AppleTVSimulator");
  }

  protected void checkAppleSdkTvosPlatformEnv(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch, "platform_type", "'tvos'");
    useConfiguration("--tvos_cpus=arm64");

    CommandAction linkAction = linkAction("//x:x");

    assertAppleSdkPlatformEnv(linkAction, "AppleTVOS");
  }

  protected void checkLinkMinimumOSVersion(String minOSVersionOption) throws Exception {
    CommandAction linkAction = linkAction("//x:x");

    assertThat(Joiner.on(" ").join(linkAction.getArguments())).contains(minOSVersionOption);
  }

  protected void checkWatchSimulatorLinkAction(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch, "deps", "['//package:objcLib']", "platform_type", "'watchos'");
    scratch.file("package/BUILD", "objc_library(name = 'objcLib', srcs = [ 'b.m' ])");

    // Tests that ios_multi_cpus and cpu are completely ignored.
    useConfiguration("--ios_multi_cpus=x86_64", "--cpu=ios_x86_64", "--watchos_cpus=i386");

    Action lipoAction = actionProducingArtifact("//x:x", "_lipobin");

    String i386Bin =
        configurationBin("i386", ConfigurationDistinguisher.APPLEBIN_WATCHOS) + "x/x_bin";
    Artifact binArtifact = getFirstArtifactEndingWith(lipoAction.getInputs(), i386Bin);
    CommandAction linkAction = (CommandAction) getGeneratingAction(binArtifact);

    assertAppleSdkPlatformEnv(linkAction, "WatchSimulator");
    assertThat(normalizeBashArgs(linkAction.getArguments()))
        .containsAtLeast("-arch", "i386")
        .inOrder();
  }

  protected void checkWatchSimulatorLipoAction(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch, "platform_type", "'watchos'");

    // Tests that ios_multi_cpus and cpu are completely ignored.
    useConfiguration("--ios_multi_cpus=x86_64", "--cpu=ios_x86_64", "--watchos_cpus=i386,armv7k");

    CommandAction action = (CommandAction) lipoBinAction("//x:x");
    String i386Bin =
        configurationBin("i386", ConfigurationDistinguisher.APPLEBIN_WATCHOS) + "x/x_bin";
    String armv7kBin =
        configurationBin("armv7k", ConfigurationDistinguisher.APPLEBIN_WATCHOS) + "x/x_bin";

    assertThat(Artifact.asExecPaths(action.getInputs()))
        .comparingElementsUsing(MATCHES_REGEX)
        .containsExactly(i386Bin, armv7kBin);

    assertThat(action.getArguments())
        .comparingElementsUsing(MATCHES_REGEX)
        .containsAtLeast("/usr/bin/lipo", "-create")
        .inOrder();
    assertThat(action.getArguments()).containsAtLeast(armv7kBin, i386Bin);
    assertContainsSublist(
        action.getArguments(),
        ImmutableList.of("-output", execPathEndingWith(action.getOutputs(), "x_lipobin")));

    assertThat(Artifact.toRootRelativePaths(action.getOutputs())).containsExactly("x/x_lipobin");
    assertAppleSdkPlatformEnv(action, "MacOSX");
    assertRequiresDarwin(action);
  }

  protected void checkXcodeVersionEnv(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch);
    useConfiguration("--xcode_version=5.8");

    CommandAction action = linkAction("//x:x");

    assertXcodeVersionEnv(action, "5.8");
  }

  public void checkLinkingRuleCanUseCrosstool_singleArch(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch);

    // If bin is indeed using the c++ backend, then its archive action should be a CppLinkAction.
    Action lipobinAction = lipoBinAction("//x:x");
    Artifact bin = getFirstDerivedArtifactEndingWith(lipobinAction.getInputs(), "_bin");
    Action linkAction = getGeneratingAction(bin);
    Artifact archive = getFirstDerivedArtifactEndingWith(linkAction.getInputs(), ".a");
    Action archiveAction = getGeneratingAction(archive);
    assertThat(archiveAction).isInstanceOf(CppLinkAction.class);
  }

  public void checkLinkingRuleCanUseCrosstool_multiArch(RuleType ruleType) throws Exception {
    useConfiguration("--ios_multi_cpus=i386,x86_64");
    ruleType.scratchTarget(scratch);

    // If bin is indeed using the c++ backend, then its archive action should be a CppLinkAction.
    Action lipobinAction = lipoBinAction("//x:x");
    Artifact bin = getFirstDerivedArtifactEndingWith(lipobinAction.getInputs(), "_bin");
    Action linkAction = getGeneratingAction(bin);
    Artifact archive = getFirstDerivedArtifactEndingWith(linkAction.getInputs(), ".a");
    Action archiveAction = getGeneratingAction(archive);
    assertThat(archiveAction).isInstanceOf(CppLinkAction.class);
  }

  protected void scratchFrameworkStarlarkStub(String bzlPath) throws Exception {
    PathFragment pathFragment = PathFragment.create(bzlPath);
    scratch.file(pathFragment.getParentDirectory() + "/BUILD");
    scratch.file(
        bzlPath,
        "def framework_stub_impl(ctx):",
        "  bin_provider = ctx.attr.binary[apple_common.AppleExecutableBinary]",
        "  my_provider = apple_common.new_dynamic_framework_provider(",
        "      cc_info = bin_provider.cc_info,",
        "      objc = bin_provider.objc,",
        "      binary = bin_provider.binary,",
        "      framework_files = depset([bin_provider.binary]),",
        "      framework_dirs = depset(['_frameworks/stubframework.framework']))",
        "  return struct(providers = [my_provider])",
        "framework_stub_rule = rule(",
        "    framework_stub_impl,",
        "    attrs = {'binary': attr.label(mandatory=True,",
        "                                  providers=[apple_common.AppleExecutableBinary])},",
        "    fragments = ['apple', 'objc'],",
        ")");
  }

  private void assertAvoidDepsObjects(RuleType ruleType) throws Exception {
    /*
     * The target tree for ease of understanding:
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
    ruleType.scratchTarget(
        scratch, "deps", "['//package:objcLib']", "avoid_deps", "['//package:avoidLib']");
    if (!"apple_binary_starlark".equals(ruleType.getRuleTypeName())) {
      addAppleBinaryStarlarkRule(scratch);
    }
    scratchFrameworkStarlarkStub("frameworkstub/framework_stub.bzl");
    scratch.file(
        "package/BUILD",
        "load('//test_starlark:apple_binary_starlark.bzl', 'apple_binary_starlark')",
        "load('//frameworkstub:framework_stub.bzl', 'framework_stub_rule')",
        "objc_library(name = 'objcLib', srcs = [ 'b.m' ],",
        "    deps = [':avoidLibDep', ':baseLib'])",
        "objc_library(name = 'baseLib', srcs = [ 'base.m' ],",
        "    deps = [':baseLibDep', ':avoidLibDepTwo'])",
        "objc_library(name = 'baseLibDep', srcs = [ 'basedep.m' ],",
        "    sdk_frameworks = ['BaseSDK'])",
        "framework_stub_rule(name = 'avoidLib', binary = ':avoidLibBinary')",
        "apple_binary_starlark(name = 'avoidLibBinary', binary_type = 'executable',",
        "    platform_type = 'ios',",
        "    deps = [':avoidLibDep'])",
        "objc_library(name = 'avoidLibDep', srcs = [ 'd.m' ], deps = [':avoidLibDepTwo'])",
        "objc_library(name = 'avoidLibDepTwo', srcs = [ 'e.m' ],",
        "    sdk_frameworks = ['AvoidSDK'])");

    Action lipobinAction = lipoBinAction("//x:x");
    Artifact binArtifact = getFirstArtifactEndingWith(lipobinAction.getInputs(), "x/x_bin");

    Action action = getGeneratingAction(binArtifact);

    assertThat(getFirstArtifactEndingWith(action.getInputs(), "package/libobjcLib.a")).isNotNull();
    assertThat(getFirstArtifactEndingWith(action.getInputs(), "package/libbaseLib.a")).isNotNull();
    assertThat(getFirstArtifactEndingWith(action.getInputs(), "package/libbaseLibDep.a"))
        .isNotNull();
    assertThat(getFirstArtifactEndingWith(action.getInputs(), "package/libavoidLib.a")).isNull();
    assertThat(getFirstArtifactEndingWith(action.getInputs(), "package/libavoidLibDepTwo.a"))
        .isNull();
    assertThat(getFirstArtifactEndingWith(action.getInputs(), "package/libavoidLibDep.a")).isNull();
  }

  public void checkAvoidDepsObjects(RuleType ruleType) throws Exception {
    useConfiguration("--ios_multi_cpus=i386,x86_64");
    assertAvoidDepsObjects(ruleType);
  }

  private void assertAvoidDepsObjcLibraries(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(
        scratch, "deps", "['//package:objcLib']", "avoid_deps", "['//package:avoidLib']");
    scratch.file(
        "package/BUILD",
        "objc_library(name = 'objcLib', srcs = [ 'b.m' ], deps = [':avoidLib', ':baseLib'])",
        "objc_library(name = 'baseLib', srcs = [ 'base.m' ])",
        "objc_library(name = 'avoidLib', srcs = [ 'c.m' ])");

    Action lipobinAction = lipoBinAction("//x:x");
    Artifact binArtifact = getFirstArtifactEndingWith(lipobinAction.getInputs(), "x/x_bin");

    Action action = getGeneratingAction(binArtifact);

    assertThat(getFirstArtifactEndingWith(action.getInputs(), "package/libobjcLib.a")).isNotNull();
    assertThat(getFirstArtifactEndingWith(action.getInputs(), "package/libbaseLib.a")).isNotNull();
    assertThat(getFirstArtifactEndingWith(action.getInputs(), "package/libavoidLib.a")).isNull();
  }

  public void checkAvoidDepsObjcLibraries(RuleType ruleType) throws Exception {
    useConfiguration("--ios_multi_cpus=i386,x86_64");
    assertAvoidDepsObjcLibraries(ruleType);
  }

  public void assertAvoidDepsObjcLibrariesAvoidViaCcLibrary(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(
        scratch, "deps", "['//package:objcLib']", "avoid_deps", "['//package:avoidCclib']");
    scratch.file(
        "package/BUILD",
        "cc_library(name = 'avoidCclib', srcs = ['cclib.c'], deps = [':avoidLib'])",
        "objc_library(name = 'objcLib', srcs = [ 'b.m' ], deps = [':avoidLib'])",
        "objc_library(name = 'avoidLib', srcs = [ 'c.m' ])");

    Action lipobinAction = lipoBinAction("//x:x");
    Artifact binArtifact = getFirstArtifactEndingWith(lipobinAction.getInputs(), "x/x_bin");

    Action action = getGeneratingAction(binArtifact);

    assertThat(getFirstArtifactEndingWith(action.getInputs(), "package/libavoidCclib.a")).isNull();
    assertThat(getFirstArtifactEndingWith(action.getInputs(), "package/libobjcLib.a")).isNotNull();
    assertThat(getFirstArtifactEndingWith(action.getInputs(), "package/libavoidLib.a")).isNull();
  }

  public void checkAvoidDepsObjcLibrariesAvoidViaCcLibrary(RuleType ruleType) throws Exception {
    useConfiguration("--ios_multi_cpus=i386,x86_64");
    assertAvoidDepsObjcLibrariesAvoidViaCcLibrary(ruleType);
  }

  /**
   * Verifies that if apple_binary_starlark A has an avoid deps dep on B1 which then depends on an
   * apple_binary_starlark B2, that the symbols from B2 are not present in A.
   */
  public void checkAvoidDepsThroughAvoidDep(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(
        scratch, "deps", "['//package:ObjcLib']", "avoid_deps", "['//package:dylib1']");
    if (!"apple_binary_starlark".equals(ruleType.getRuleTypeName())) {
      addAppleBinaryStarlarkRule(scratch);
    }
    scratchFrameworkStarlarkStub("frameworkstub/framework_stub.bzl");
    scratch.file(
        "package/BUILD",
        "load('//frameworkstub:framework_stub.bzl', 'framework_stub_rule')",
        "load('//test_starlark:apple_binary_starlark.bzl', 'apple_binary_starlark')",
        "objc_library(name = 'ObjcLib', srcs = [ 'ObjcLib.m' ],",
        "    deps = [':Dylib1Lib', ':Dylib2Lib'])",
        "objc_library(name = 'Dylib1Lib', srcs = [ 'Dylib1Lib.m' ])",
        "objc_library(name = 'Dylib2Lib', srcs = [ 'Dylib2Lib.m' ])",
        "framework_stub_rule(name = 'dylib1', binary = ':dylib1Binary')",
        "apple_binary_starlark(name = 'dylib1Binary', binary_type = 'executable',",
        "    platform_type = 'ios',",
        "    deps = [':Dylib1Lib'], avoid_deps = ['//package:dylib2'])",
        "framework_stub_rule(name = 'dylib2', binary = ':dylib2Binary')",
        "apple_binary_starlark(name = 'dylib2Binary', binary_type = 'executable',",
        "    platform_type = 'ios',",
        "    deps = [':Dylib2Lib'])",
        "apple_binary_starlark(name = 'alternate',",
        "    platform_type = 'ios',",
        "    deps = ['//package:ObjcLib'])");

    Action lipobinAction = lipoBinAction("//x:x");
    Artifact binArtifact = getFirstArtifactEndingWith(lipobinAction.getInputs(), "x/x_bin");

    Action linkAction = getGeneratingAction(binArtifact);

    assertThat(getFirstArtifactEndingWith(linkAction.getInputs(), "package/libObjcLib.a"))
        .isNotNull();
    assertThat(getFirstArtifactEndingWith(linkAction.getInputs(), "package/libDylib1Lib.a"))
        .isNull();
    assertThat(getFirstArtifactEndingWith(linkAction.getInputs(), "package/libDylib2Lib.a"))
        .isNull();

    // Check that the identical binary without avoid_deps would be fully linked.
    Action alternateLipobinAction = lipoBinAction("//package:alternate");
    Artifact alternateBinArtifact =
        getFirstArtifactEndingWith(alternateLipobinAction.getInputs(), "package/alternate_bin");
    Action alternateLinkAction = getGeneratingAction(alternateBinArtifact);

    assertThat(getFirstArtifactEndingWith(alternateLinkAction.getInputs(), "package/libObjcLib.a"))
        .isNotNull();
    assertThat(
            getFirstArtifactEndingWith(alternateLinkAction.getInputs(), "package/libDylib1Lib.a"))
        .isNotNull();
    assertThat(
            getFirstArtifactEndingWith(alternateLinkAction.getInputs(), "package/libDylib2Lib.a"))
        .isNotNull();
  }

  /**
   * Tests that direct cc_library dependencies of an avoid_dep (and its dependencies) are correctly
   * removed from the main binary.
   */
  // transitively avoided, even if it is not present in deps.
  public void checkAvoidDepsObjectsAvoidViaCcLibrary(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(
        scratch, "deps", "['//package:objcLib']", "avoid_deps", "['//package:avoidLib']");
    if (!"apple_binary_starlark".equals(ruleType.getRuleTypeName())) {
      addAppleBinaryStarlarkRule(scratch);
    }
    scratchFrameworkStarlarkStub("frameworkstub/framework_stub.bzl");
    scratch.file(
        "package/BUILD",
        "load('//frameworkstub:framework_stub.bzl', 'framework_stub_rule')",
        "load('//test_starlark:apple_binary_starlark.bzl', 'apple_binary_starlark')",
        "framework_stub_rule(name = 'avoidLib', binary = ':avoidLibBinary')",
        "apple_binary_starlark(name = 'avoidLibBinary', binary_type = 'executable',",
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

  public void checkAvoidDepsSubtractsImportedLibrary(RuleType ruleType) throws Exception {
    if (!ruleType.getRuleTypeName().equals("apple_binary_starlark")) {
      addAppleBinaryStarlarkRule(scratch);
    }

    ruleType.scratchTarget(
        scratch, "deps", "['//libs:objc_lib']", "avoid_deps", "['//libs:objc_avoid_lib']");

    scratch.file(
        "libs/defs.bzl",
        "def _custom_library_impl(ctx):",
        "  return [",
        "      apple_common.new_objc_provider(",
        "          library=depset([ctx.file.library]),",
        "      ), CcInfo()",
        "  ]",
        "custom_library = rule(",
        "    _custom_library_impl,",
        "    attrs={'library': attr.label(allow_single_file=True)},",
        ")",
        "def _custom_static_framework_import_impl(ctx):",
        "  return [",
        "      apple_common.new_objc_provider(",
        "          imported_library=depset([ctx.file.library]),",
        "      ), CcInfo()",
        "  ]",
        "custom_static_framework_import = rule(",
        "    _custom_static_framework_import_impl,",
        "    attrs={'library': attr.label(allow_single_file=True)},",
        ")");

    scratch.file(
        "libs/BUILD",
        "load('//test_starlark:apple_binary_starlark.bzl', 'apple_binary_starlark')",
        "load(':defs.bzl', 'custom_library', 'custom_static_framework_import')",
        "objc_library(",
        "    name = 'objc_lib',",
        "    srcs = ['a.m'],",
        "    deps = [':framework_library'],",
        ")",
        "custom_library(",
        "    name = 'framework_library',",
        "    library = 'buzzbuzz.framework/buzzbuzz',",
        ")",
        "objc_library(",
        "    name = 'objc_avoid_lib',",
        "    srcs = ['b.m'],",
        "    deps = [':framework'],",
        ")",
        "custom_static_framework_import(",
        "    name = 'framework',",
        "    library = 'buzzbuzz.framework/buzzbuzz',",
        ")");

    Artifact binArtifact =
        getFirstArtifactEndingWith(lipoBinAction("//x:x").getInputs(), "x/x_bin");
    Action action = getGeneratingAction(binArtifact);
    assertThat(Artifact.toRootRelativePaths(action.getInputs()))
        .doesNotContain("libs/buzzbuzz.framework/buzzbuzz");
  }

  public void checkFilesToCompileOutputGroup(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch);
    ConfiguredTarget target = getConfiguredTarget("//x:x");
    assertThat(
            ActionsTestUtil.baseNamesOf(getOutputGroup(target, OutputGroupInfo.FILES_TO_COMPILE)))
        .isEqualTo("a.o");
  }

  protected void checkCustomModuleMap(RuleType ruleType) throws Exception {
    ruleType.scratchTarget(scratch, "deps", "['//z:testModuleMap']");
    scratch.file("z/b.m");
    scratch.file("z/b.h");
    scratch.file("y/module.modulemap", "module my_module_b { export *\n header b.h }");
    scratch.file(
        "z/BUILD",
        "objc_library(",
        "name = 'testModuleMap',",
        "hdrs = ['b.h'],",
        "srcs = ['b.m'],",
        "module_map = '//y:mm'",
        ")");
    scratch.file("y/BUILD", "filegroup(", "name = 'mm',", "srcs = ['module.modulemap']", ")");

    CommandAction compileActionA = compileAction("//z:testModuleMap", "b.o");
    assertThat(compileActionA.getArguments()).doesNotContain("-fmodule-maps");
    assertThat(compileActionA.getArguments()).doesNotContain("-fmodule-name");

    ObjcProvider provider =
        getConfiguredTarget("//z:testModuleMap").get(ObjcProvider.STARLARK_CONSTRUCTOR);
    assertThat(Artifact.asExecPaths(provider.get(MODULE_MAP)))
        .containsExactly("y/module.modulemap");

    provider = getConfiguredTarget("//x:x").get(ObjcProvider.STARLARK_CONSTRUCTOR);
    assertThat(Artifact.asExecPaths(provider.get(MODULE_MAP))).contains("y/module.modulemap");
  }

  /**
   * Verifies that the given rule supports different minimum_os attribute values for two targets in
   * the same build, and adds compile args to set the minimum os appropriately for dependencies of
   * each.
   *
   * @param ruleType the rule to test
   * @param multiArchArtifactSuffix the suffix of the artifact that the rule-under-test produces
   * @param singleArchArtifactSuffix the suffix of the single-architecture artifact that is an input
   *     to the rule-under-test's generating action
   */
  protected void checkMinimumOsDifferentTargets(
      RuleType ruleType, String multiArchArtifactSuffix, String singleArchArtifactSuffix)
      throws Exception {
    ruleType.scratchTarget(
        "nine", "nine", scratch, "deps", "['//package:objcLib']", "minimum_os_version", "'9.0'");
    ruleType.scratchTarget(
        "eight", "eight", scratch, "deps", "['//package:objcLib']", "minimum_os_version", "'8.0'");
    scratch.file(
        "package/BUILD",
        "genrule(name = 'root', srcs = ['//nine:nine', '//eight:eight'], outs = ['genout'],",
        "    cmd = 'touch genout')",
        "objc_library(name = 'objcLib', srcs = [ 'b.m' ])");

    ConfiguredTarget rootTarget = getConfiguredTarget("//package:root");
    Artifact rootArtifact = getGenfilesArtifact("genout", rootTarget);

    Action genruleAction = getGeneratingAction(rootArtifact);
    Action eightLipoAction =
        getGeneratingAction(
            getFirstArtifactEndingWith(
                genruleAction.getInputs(), "eight" + multiArchArtifactSuffix));
    Action nineLipoAction =
        getGeneratingAction(
            getFirstArtifactEndingWith(
                genruleAction.getInputs(), "nine" + multiArchArtifactSuffix));
    Artifact eightBin =
        getFirstArtifactEndingWith(eightLipoAction.getInputs(), singleArchArtifactSuffix);
    Artifact nineBin =
        getFirstArtifactEndingWith(nineLipoAction.getInputs(), singleArchArtifactSuffix);

    CommandAction eightLinkAction = (CommandAction) getGeneratingAction(eightBin);
    CommandAction nineLinkAction = (CommandAction) getGeneratingAction(nineBin);

    CommandAction eightObjcLibArchiveAction =
        (CommandAction)
            getGeneratingAction(
                getFirstArtifactEndingWith(eightLinkAction.getInputs(), "libobjcLib.a"));
    CommandAction eightObjcLibCompileAction =
        (CommandAction)
            getGeneratingAction(
                getFirstArtifactEndingWith(eightObjcLibArchiveAction.getInputs(), "b.o"));
    CommandAction nineObjcLibArchiveAction =
        (CommandAction)
            getGeneratingAction(
                getFirstArtifactEndingWith(nineLinkAction.getInputs(), "libobjcLib.a"));
    CommandAction nineObjcLibCompileAction =
        (CommandAction)
            getGeneratingAction(
                getFirstArtifactEndingWith(nineObjcLibArchiveAction.getInputs(), "b.o"));

    assertThat(Joiner.on(" ").join(eightObjcLibCompileAction.getArguments()))
        .contains("-mios-simulator-version-min=8.0");
    assertThat(Joiner.on(" ").join(nineObjcLibCompileAction.getArguments()))
        .contains("-mios-simulator-version-min=9.0");
  }

  protected void verifyDrops32BitIosArchitecture(RuleType ruleType) throws Exception {
    scratch.file(
        "libs/BUILD", "objc_library(", "    name = 'objc_lib',", "    srcs = ['a.m'],", ")");

    ruleType.scratchTarget(
        scratch,
        "deps",
        "['//libs:objc_lib']",
        "platform_type",
        "'ios'",
        "minimum_os_version",
        "'11.0'"); // Does not support 32-bit architectures.

    useConfiguration("--ios_multi_cpus=armv7,arm64,i386,x86_64");

    Action lipoAction = actionProducingArtifact("//x:x", "_lipobin");

    getSingleArchBinary(lipoAction, "arm64");
    getSingleArchBinary(lipoAction, "x86_64");
    assertThat(getSingleArchBinaryIfAvailable(lipoAction, "armv7")).isNull();
    assertThat(getSingleArchBinaryIfAvailable(lipoAction, "i386")).isNull();
  }

  protected void verifyDrops32BitWatchArchitecture(RuleType ruleType) throws Exception {
    scratch.file(
        "libs/BUILD", "objc_library(", "    name = 'objc_lib',", "    srcs = ['a.m'],", ")");

    ruleType.scratchTarget(
        scratch,
        "deps",
        "['//libs:objc_lib']",
        "platform_type",
        "'watchos'",
        "minimum_os_version",
        "'9.0'"); // Does not support 32-bit architectures.

    useConfiguration("--watchos_cpus=armv7k,arm64_32,i386,x86_64");

    Action lipoAction = actionProducingArtifact("//x:x", "_lipobin");

    getSingleArchBinary(lipoAction, "arm64_32");
    getSingleArchBinary(lipoAction, "x86_64");
    assertThat(getSingleArchBinaryIfAvailable(lipoAction, "armv7k")).isNull();
    assertThat(getSingleArchBinaryIfAvailable(lipoAction, "i386")).isNull();
  }

  /**
   * Returns the full exec path string for exec paths of targets within the main tools repository.
   */
  protected static String toolsRepoExecPath(String execPath) {
    return TestConstants.TOOLS_REPOSITORY_PATH_PREFIX + execPath;
  }

  @Nullable
  protected Artifact getSingleArchBinaryIfAvailable(Action lipoAction, String arch) {
    for (Artifact archBinary : lipoAction.getInputs().toList()) {
      String execPath = archBinary.getExecPathString();
      if (execPath.endsWith("_bin") && execPath.contains(arch)) {
        return archBinary;
      }
    }
    return null;
  }

  protected Artifact getSingleArchBinary(Action lipoAction, String arch) {
    Artifact result = getSingleArchBinaryIfAvailable(lipoAction, arch);
    if (result != null) {
      return result;
    } else {
      throw new AssertionError("Lipo action does not contain an input binary from arch " + arch);
    }
  }

  protected void scratchFeatureFlagTestLib() throws Exception {
    scratch.file(
        "lib/BUILD",
        "config_feature_flag(",
        "  name = 'flag1',",
        "  allowed_values = ['on', 'off'],",
        "  default_value = 'off',",
        ")",
        "config_setting(",
        "  name = 'flag1@on',",
        "  flag_values = {':flag1': 'on'},",
        "  transitive_configs = [':flag1'],",
        ")",
        "config_feature_flag(",
        "  name = 'flag2',",
        "  allowed_values = ['on', 'off'],",
        "  default_value = 'off',",
        ")",
        "config_setting(",
        "  name = 'flag2@on',",
        "  flag_values = {':flag2': 'on'},",
        "  transitive_configs = [':flag2'],",
        ")",
        "objc_library(",
        "  name = 'objcLib',",
        "  srcs = select({",
        "    ':flag1@on': ['flag1on.m'],",
        "    '//conditions:default': ['flag1off.m'],",
        "  }) + select({",
        "    ':flag2@on': ['flag2on.m'],",
        "    '//conditions:default': ['flag2off.m'],",
        "  }),",
        "  copts = select({",
        "    ':flag1@on': ['-FLAG_1_ON'],",
        "    '//conditions:default': ['-FLAG_1_OFF'],",
        "  }) + select({",
        "    ':flag2@on': ['-FLAG_2_ON'],",
        "    '//conditions:default': ['-FLAG_2_OFF'],",
        "  }),",
        "  transitive_configs = [':flag1', ':flag2'],",
        ")");
  }

  protected String removeConfigFragment(String text) {
    return text.replaceAll("-out/.*/bin", "-out//bin").replaceAll("-out/.*/gen", "-out//gen");
  }

  protected List<String> removeConfigFragment(List<String> text) {
    return text.stream().map(this::removeConfigFragment).collect(toImmutableList());
  }

  protected static Iterable<String> getArifactPathsOfLibraries(ConfiguredTarget target) {
    return Artifact.toRootRelativePaths(
        target
            .get(CcInfo.PROVIDER)
            .getCcLinkingContext()
            .getStaticModeParamsForDynamicLibraryLibraries());
  }

  protected static Iterable<String> getArifactPathsOfHeaders(ConfiguredTarget target) {
    return Artifact.toRootRelativePaths(
        target.get(CcInfo.PROVIDER).getCcCompilationContext().getDeclaredIncludeSrcs());
  }
}
