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
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.getFirstArtifactEndingWith;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandAction;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.packages.util.MockObjcSupport;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration.ConfigurationDistinguisher;
import com.google.devtools.build.lib.rules.cpp.CcCommon;
import com.google.devtools.build.lib.rules.cpp.CcCommon.Language;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.rules.cpp.CcLinkingContext;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppHelper;
import com.google.devtools.build.lib.rules.cpp.CppRuleClasses;
import com.google.devtools.build.lib.rules.objc.CompilationSupport.ExtraLinkArgs;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import net.starlark.java.eval.Dict;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test cases for the Starlark Apple Linking API, {@code apple_common.link_multi_arch_binary}. */
@RunWith(JUnit4.class)
public class AppleBinaryStarlarkApiTest extends ObjcRuleTestCase {
  static final RuleType RULE_TYPE =
      new RuleType("apple_binary_starlark") {
        @Override
        Iterable<String> requiredAttributes(
            Scratch scratch, String packageDir, Set<String> alreadyAdded) throws IOException {
          ImmutableList.Builder<String> attributes = new ImmutableList.Builder<>();
          if (!alreadyAdded.contains("deps")) {
            String depPackageDir = packageDir + "_defaultDep";
            scratch.file(depPackageDir + "/a.m");
            scratch.file(depPackageDir + "/private.h");
            scratch.file(
                depPackageDir + "/BUILD",
                "objc_library(name = 'lib_dep', srcs = ['a.m', 'private.h'])");
            attributes.add("deps = ['//" + depPackageDir + ":" + "lib_dep']");
          }
          if (!alreadyAdded.contains("platform_type")) {
            attributes.add("platform_type = 'ios'");
          }
          if (!alreadyAdded.contains("binary_type")) {
            attributes.add("binary_type = 'executable'");
          }
          return attributes.build();
        }

        @Override
        public String starlarkLoadPrerequisites() {
          return "load('//test_starlark:apple_binary_starlark.bzl', 'apple_binary_starlark')";
        }
      };

  private static final String COCOA_FRAMEWORK_FLAG = "-framework Cocoa";
  private static final String FOUNDATION_FRAMEWORK_FLAG = "-framework Foundation";
  private static final String UIKIT_FRAMEWORK_FLAG = "-framework UIKit";
  private static final ImmutableSet<String> IMPLICIT_NON_MAC_FRAMEWORK_FLAGS =
      ImmutableSet.of(FOUNDATION_FRAMEWORK_FLAG, UIKIT_FRAMEWORK_FLAG);
  private static final ImmutableSet<String> IMPLICIT_MAC_FRAMEWORK_FLAGS =
      ImmutableSet.of(FOUNDATION_FRAMEWORK_FLAG);
  private static final ImmutableSet<String> COCOA_FEATURE_FLAGS =
      ImmutableSet.of(COCOA_FRAMEWORK_FLAG);

  @Before
  public final void setup() throws Exception {
    addAppleBinaryStarlarkRule(scratch);
  }

  @Before
  public void setupMyInfo() throws Exception {
    scratch.file("myinfo/myinfo.bzl", "MyInfo = provider()");

    scratch.file("myinfo/BUILD");

    invalidatePackages();
  }

  private StructImpl getMyInfoFromTarget(ConfiguredTarget configuredTarget) throws Exception {
    Provider.Key key =
        new StarlarkProvider.Key(Label.parseCanonical("//myinfo:myinfo.bzl"), "MyInfo");
    return (StructImpl) configuredTarget.get(key);
  }

  @Test
  public void testOutputDirectoryWithMandatoryMinimumVersion() throws Exception {
    scratch.file(
        "a/BUILD",
        "load('//test_starlark:apple_binary_starlark.bzl', 'apple_binary_starlark')",
        "apple_binary_starlark(name='a', platform_type='ios', deps=['b'],"
            + " minimum_os_version='7.0')",
        "objc_library(name='b', srcs=['b.c'])");

    useConfiguration("ios_cpus=i386");
    ConfiguredTarget a = getConfiguredTarget("//a:a");
    ConfiguredTarget b = getDirectPrerequisite(a, "//a:b");

    PathFragment aPath = getConfiguration(a).getOutputDirectory(RepositoryName.MAIN).getExecPath();
    PathFragment bPath = getConfiguration(b).getOutputDirectory(RepositoryName.MAIN).getExecPath();

    assertThat(aPath.getPathString()).doesNotMatch("-min[0-9]");
    assertThat(bPath.getPathString()).contains("-min7.0-");
  }

  @Test
  public void testMandatoryMinimumOsVersionSet() throws Exception {
    getRuleType()
        .scratchTarget(scratch, "minimum_os_version", "'8.0'", "platform_type", "'watchos'");
    getConfiguredTarget("//x:x");
  }

  @Test
  public void testLocalXcodeSetsLocalOnlyRequirementLipo() throws Exception {
    scratch.file(
        "xcode/BUILD",
        "xcode_version(",
        "    name = 'version10_1_0',",
        "    version = '10.1.0',",
        "    aliases = ['10.1' ,'10.1.0'],",
        "    default_ios_sdk_version = '12.1',",
        "    default_tvos_sdk_version = '12.1',",
        "    default_macos_sdk_version = '10.14',",
        "    default_watchos_sdk_version = '5.1',",
        ")",
        "xcode_version(",
        "    name = 'version10_2_1',",
        "    version = '10.2.1',",
        "    aliases = ['10.2.1' ,'10.2'],",
        "    default_ios_sdk_version = '12.2',",
        "    default_tvos_sdk_version = '12.2',",
        "    default_macos_sdk_version = '10.14',",
        "    default_watchos_sdk_version = '5.2',",
        ")",
        "available_xcodes(",
        "    name= 'local',",
        "    versions = [':version10_1_0'],",
        "    default = ':version10_1_0',",
        ")",
        "available_xcodes(",
        "    name= 'remote',",
        "    versions = [':version10_2_1'],",
        "    default = ':version10_2_1',",
        ")",
        "xcode_config(",
        "    name = 'my_config',",
        "    local_versions = ':local',",
        "    remote_versions = ':remote',",
        ")");
    getRuleType().scratchTarget(scratch, "platform_type", "'watchos'");

    useConfigurationWithCustomXcode(
        "--xcode_version=10.1",
        "--xcode_version_config=//xcode:my_config",
        "--watchos_cpus=i386,armv7k",
        "--watchos_sdk_version=2.1");
    CommandAction action = (CommandAction) lipoBinAction("//x:x");
    assertHasRequirement(action, ExecutionRequirements.REQUIREMENTS_SET);
    assertHasRequirement(action, ExecutionRequirements.NO_REMOTE);
  }

  @Test
  public void testRemoteXcodeSetsLocalOnlyRequirementLipo() throws Exception {
    scratch.file(
        "xcode/BUILD",
        "xcode_version(",
        "    name = 'version10_1_0',",
        "    version = '10.1.0',",
        "    aliases = ['10.1' ,'10.1.0'],",
        "    default_ios_sdk_version = '12.1',",
        "    default_tvos_sdk_version = '12.1',",
        "    default_macos_sdk_version = '10.14',",
        "    default_watchos_sdk_version = '5.1',",
        ")",
        "xcode_version(",
        "    name = 'version10_2_1',",
        "    version = '10.2.1',",
        "    aliases = ['10.2.1' ,'10.2'],",
        "    default_ios_sdk_version = '12.2',",
        "    default_tvos_sdk_version = '12.2',",
        "    default_macos_sdk_version = '10.14',",
        "    default_watchos_sdk_version = '5.2',",
        ")",
        "available_xcodes(",
        "    name= 'local',",
        "    versions = [':version10_1_0'],",
        "    default = ':version10_1_0',",
        ")",
        "available_xcodes(",
        "    name= 'remote',",
        "    versions = [':version10_2_1'],",
        "    default = ':version10_2_1',",
        ")",
        "xcode_config(",
        "    name = 'my_config',",
        "    local_versions = ':local',",
        "    remote_versions = ':remote',",
        ")");
    getRuleType().scratchTarget(scratch, "platform_type", "'watchos'");

    useConfigurationWithCustomXcode(
        "--xcode_version=10.2.1",
        "--xcode_version_config=//xcode:my_config",
        "--watchos_cpus=i386,armv7k",
        "--watchos_sdk_version=2.1");
    CommandAction action = (CommandAction) lipoBinAction("//x:x");
    assertHasRequirement(action, ExecutionRequirements.REQUIREMENTS_SET);
    assertHasRequirement(action, ExecutionRequirements.NO_LOCAL);
    assertNotHasRequirement(action, ExecutionRequirements.NO_REMOTE);
  }

  @Test
  public void testLocalXcodeSetsRemoteOnlyRequirementLipo() throws Exception {
    scratch.file(
        "xcode/BUILD",
        "xcode_version(",
        "    name = 'version10_1_0',",
        "    version = '10.1.0',",
        "    aliases = ['10.1' ,'10.1.0'],",
        "    default_ios_sdk_version = '12.1',",
        "    default_tvos_sdk_version = '12.1',",
        "    default_macos_sdk_version = '10.14',",
        "    default_watchos_sdk_version = '5.1',",
        ")",
        "xcode_version(",
        "    name = 'version10_2_1',",
        "    version = '10.2.1',",
        "    aliases = ['10.2.1' ,'10.2'],",
        "    default_ios_sdk_version = '12.2',",
        "    default_tvos_sdk_version = '12.2',",
        "    default_macos_sdk_version = '10.14',",
        "    default_watchos_sdk_version = '5.2',",
        ")",
        "available_xcodes(",
        "    name= 'local',",
        "    versions = [':version10_1_0'],",
        "    default = ':version10_1_0',",
        ")",
        "available_xcodes(",
        "    name= 'remote',",
        "    versions = [':version10_2_1'],",
        "    default = ':version10_2_1',",
        ")",
        "xcode_config(",
        "    name = 'my_config',",
        "    local_versions = ':local',",
        "    remote_versions = ':remote',",
        ")");
    getRuleType().scratchTarget(scratch, "platform_type", "'watchos'");

    useConfigurationWithCustomXcode(
        "--xcode_version=10.2.1",
        "--xcode_version_config=//xcode:my_config",
        "--watchos_cpus=i386,armv7k",
        "--watchos_sdk_version=2.1");
    CommandAction action = (CommandAction) lipoBinAction("//x:x");
    assertHasRequirement(action, ExecutionRequirements.REQUIREMENTS_SET);
    assertHasRequirement(action, ExecutionRequirements.NO_LOCAL);
    assertNotHasRequirement(action, ExecutionRequirements.NO_REMOTE);
  }

  @Test
  public void testMutualXcodeNoLocalityRequirementsLipo() throws Exception {
    scratch.file(
        "xcode/BUILD",
        "xcode_version(",
        "    name = 'version10_1_0',",
        "    version = '10.1.0',",
        "    aliases = ['10.1' ,'10.1.0'],",
        "    default_ios_sdk_version = '12.1',",
        "    default_tvos_sdk_version = '12.1',",
        "    default_macos_sdk_version = '10.14',",
        "    default_watchos_sdk_version = '5.1',",
        ")",
        "available_xcodes(",
        "    name= 'local',",
        "    versions = [':version10_1_0'],",
        "    default = ':version10_1_0',",
        ")",
        "available_xcodes(",
        "    name= 'remote',",
        "    versions = [':version10_1_0'],",
        "    default = ':version10_1_0',",
        ")",
        "xcode_config(",
        "    name = 'my_config',",
        "    local_versions = ':local',",
        "    remote_versions = ':remote',",
        ")");
    getRuleType().scratchTarget(scratch, "platform_type", "'watchos'");

    useConfigurationWithCustomXcode(
        "--xcode_version=10.1",
        "--xcode_version_config=//xcode:my_config",
        "--watchos_cpus=i386,armv7k",
        "--watchos_sdk_version=2.1");
    CommandAction action = (CommandAction) lipoBinAction("//x:x");
    assertHasRequirement(action, ExecutionRequirements.REQUIREMENTS_SET);
    assertNotHasRequirement(action, ExecutionRequirements.NO_LOCAL);
    assertNotHasRequirement(action, ExecutionRequirements.NO_REMOTE);
  }

  @Test
  public void testSymlinkInsteadOfLipoSingleArch() throws Exception {
    getRuleType().scratchTarget(scratch);

    SymlinkAction action = (SymlinkAction) lipoBinAction("//x:x");
    CommandAction linkAction = linkAction("//x:x");

    assertThat(action.getInputs().toList())
        .containsExactly(Iterables.getOnlyElement(linkAction.getOutputs()));
  }

  @Test
  public void testCcDependencyLinkoptsArePropagatedToLinkActionPreMigration() throws Exception {
    checkCcDependencyLinkoptsArePropagatedToLinkAction(
        getRuleType(), /* linkingInfoMigration= */ false);
  }

  @Test
  public void testCcDependencyLinkoptsArePropagatedToLinkActionPostMigration() throws Exception {
    checkCcDependencyLinkoptsArePropagatedToLinkAction(
        getRuleType(), /* linkingInfoMigration= */ true);
  }

  @Test
  public void testObjcLibraryLinkoptsArePropagatedToLinkActionPreMigration() throws Exception {
    checkObjcLibraryLinkoptsArePropagatedToLinkAction(
        getRuleType(), /* linkingInfoMigration= */ false);
  }

  @Test
  public void testObjcLibraryLinkoptsArePropagatedToLinkActionPostMigration() throws Exception {
    checkObjcLibraryLinkoptsArePropagatedToLinkAction(
        getRuleType(), /* linkingInfoMigration= */ true);
  }

  /** Returns the path to the dSYM binary artifact for given architecture and compilation mode. */
  protected String dsymBinaryPath(String arch, CompilationMode mode) throws Exception {
    return configurationBin(arch, ConfigurationDistinguisher.APPLEBIN_IOS, null, mode)
        + "examples/apple_starlark/bin_bin.dwarf";
  }

  /** Returns the path to the linkmap artifact for a given architecture. */
  protected String linkmapPath(String arch) throws Exception {
    return configurationBin(arch, ConfigurationDistinguisher.APPLEBIN_IOS)
        + "examples/apple_starlark/bin.linkmap";
  }

  @Test
  public void testProvider_executable() throws Exception {
    scratch.file("examples/rule/BUILD");
    scratch.file(
        "examples/rule/apple_rules.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "def _test_rule_impl(ctx):",
        "   dep = ctx.attr.deps[0]",
        "   provider = dep[apple_common.AppleExecutableBinary]",
        "   return MyInfo(",
        "      binary = provider.binary,",
        "      cc_info = provider.cc_info,",
        "      objc = provider.objc,",
        "   )",
        "test_rule = rule(implementation = _test_rule_impl,",
        "   attrs = {",
        "   'deps': attr.label_list(allow_files = False, mandatory = False,)",
        "})");

    scratch.file(
        "examples/apple_starlark/BUILD",
        "load('//test_starlark:apple_binary_starlark.bzl', 'apple_binary_starlark')",
        "load('//examples/rule:apple_rules.bzl', 'test_rule')",
        "package(default_visibility = ['//visibility:public'])",
        "apple_binary_starlark(",
        "    name = 'bin',",
        "    deps = [':lib'],",
        "    binary_type = 'executable',",
        "    platform_type = 'ios',",
        ")",
        "objc_library(",
        "    name = 'lib',",
        "    srcs = ['a.m'],",
        ")",
        "test_rule(",
        "    name = 'my_target',",
        "    deps = [':bin'],",
        ")");

    useConfiguration("--ios_multi_cpus=armv7,arm64");
    ConfiguredTarget starlarkTarget = getConfiguredTarget("//examples/apple_starlark:my_target");
    StructImpl myInfo = getMyInfoFromTarget(starlarkTarget);

    assertThat(myInfo.getValue("binary")).isInstanceOf(Artifact.class);
    assertThat(myInfo.getValue("objc")).isInstanceOf(ObjcProvider.class);
    assertThat(myInfo.getValue("cc_info")).isInstanceOf(CcInfo.class);
  }

  private void checkDuplicateLinkopts() throws Exception {
    getRuleType().scratchTarget(scratch, "linkopts", "['-foo', 'bar', '-foo', 'baz']");

    CommandAction linkAction = linkAction("//x:x");
    String linkArgs = Joiner.on(" ").join(linkAction.getArguments());
    assertThat(linkArgs).contains("-Wl,-foo -Wl,bar");
    assertThat(linkArgs).contains("-Wl,-foo -Wl,baz");
  }

  @Test
  public void testDuplicateLinkoptsPreMigration() throws Exception {
    useConfiguration(linkingInfoMigrationFlag(false));
    checkDuplicateLinkopts();
  }

  @Test
  public void testDuplicateLinkoptsPostMigration() throws Exception {
    useConfiguration(linkingInfoMigrationFlag(true));
    checkDuplicateLinkopts();
  }

  @Test
  public void testCanUseCrosstool_singleArch() throws Exception {
    checkLinkingRuleCanUseCrosstool_singleArch(getRuleType());
  }

  @Test
  public void testCanUseCrosstool_multiArch() throws Exception {
    checkLinkingRuleCanUseCrosstool_multiArch(getRuleType());
  }

  @Test
  public void testAppleSdkIphoneosPlatformEnv() throws Exception {
    checkAppleSdkIphoneosPlatformEnv(getRuleType());
  }

  @Test
  public void testXcodeVersionEnv() throws Exception {
    checkXcodeVersionEnv(getRuleType());
  }

  @Test
  public void testLinksImplicitFrameworksWithCrosstoolIos() throws Exception {
    useConfiguration("--ios_multi_cpus=x86_64", "--ios_sdk_version=10.0", "--ios_minimum_os=8.0");
    getRuleType().scratchTarget(scratch, "platform_type", "'ios'");

    Action lipoAction = actionProducingArtifact("//x:x", "_lipobin");
    Artifact binArtifact = getFirstArtifactEndingWith(lipoAction.getInputs(), "x/x_bin");
    CommandAction linkAction = (CommandAction) getGeneratingAction(binArtifact);

    assertThat(linkAction.getArguments())
        .containsAtLeastElementsIn(IMPLICIT_NON_MAC_FRAMEWORK_FLAGS);
  }

  @Test
  public void testLinksImplicitFrameworksWithCrosstoolWatchos() throws Exception {
    useConfiguration(
        "--watchos_cpus=i386", "--watchos_sdk_version=3.0", "--watchos_minimum_os=2.0");
    getRuleType().scratchTarget(scratch, "platform_type", "'watchos'");

    Action lipoAction = actionProducingArtifact("//x:x", "_lipobin");
    Artifact binArtifact = getFirstArtifactEndingWith(lipoAction.getInputs(), "x/x_bin");
    CommandAction linkAction = (CommandAction) getGeneratingAction(binArtifact);

    assertThat(linkAction.getArguments())
        .containsAtLeastElementsIn(IMPLICIT_NON_MAC_FRAMEWORK_FLAGS);
  }

  @Test
  public void testLinksImplicitFrameworksWithCrosstoolTvos() throws Exception {
    useConfiguration("--tvos_cpus=x86_64", "--tvos_sdk_version=10.1", "--tvos_minimum_os=10.0");
    getRuleType().scratchTarget(scratch, "platform_type", "'tvos'");

    Action lipoAction = actionProducingArtifact("//x:x", "_lipobin");
    Artifact binArtifact = getFirstArtifactEndingWith(lipoAction.getInputs(), "x/x_bin");
    CommandAction linkAction = (CommandAction) getGeneratingAction(binArtifact);

    assertThat(linkAction.getArguments())
        .containsAtLeastElementsIn(IMPLICIT_NON_MAC_FRAMEWORK_FLAGS);
  }

  @Test
  public void testLinksImplicitFrameworksWithCrosstoolMacos() throws Exception {
    useConfiguration(
        "--macos_cpus=x86_64", "--macos_sdk_version=10.11", "--macos_minimum_os=10.11");
    getRuleType().scratchTarget(scratch, "platform_type", "'macos'");

    Action lipoAction = actionProducingArtifact("//x:x", "_lipobin");
    Artifact binArtifact = getFirstArtifactEndingWith(lipoAction.getInputs(), "x/x_bin");
    CommandAction linkAction = (CommandAction) getGeneratingAction(binArtifact);

    assertThat(linkAction.getArguments()).containsAtLeastElementsIn(IMPLICIT_MAC_FRAMEWORK_FLAGS);
    assertThat(linkAction.getArguments())
        .containsNoneOf(COCOA_FRAMEWORK_FLAG, UIKIT_FRAMEWORK_FLAG);
  }

  @Test
  public void testLinkCocoaFeatureWithCrosstoolMacos() throws Exception {
    useConfiguration(
        "--macos_cpus=x86_64", "--macos_sdk_version=10.11", "--macos_minimum_os=10.11");
    getRuleType().scratchTarget(scratch, "platform_type", "'macos'", "features", "['link_cocoa']");

    Action lipoAction = actionProducingArtifact("//x:x", "_lipobin");
    Artifact binArtifact = getFirstArtifactEndingWith(lipoAction.getInputs(), "x/x_bin");
    CommandAction linkAction = (CommandAction) getGeneratingAction(binArtifact);

    assertThat(linkAction.getArguments()).containsAtLeastElementsIn(IMPLICIT_MAC_FRAMEWORK_FLAGS);
    assertThat(linkAction.getArguments()).containsAtLeastElementsIn(COCOA_FEATURE_FLAGS);
    assertThat(linkAction.getArguments()).doesNotContain(UIKIT_FRAMEWORK_FLAG);
  }

  @Test
  public void testAliasedLinkoptsThroughObjcLibraryPreMigration() throws Exception {
    checkAliasedLinkoptsThroughObjcLibrary(getRuleType(), /* linkingInfoMigration= */ false);
  }

  @Test
  public void testAliasedLinkoptsThroughObjcLibraryPostMigration() throws Exception {
    checkAliasedLinkoptsThroughObjcLibrary(getRuleType(), /* linkingInfoMigration= */ true);
  }

  @Test
  public void testLinkInputsInLinkActionPreMigration() throws Exception {
    checkLinkInputsInLinkAction(getRuleType(), /* linkingInfoMigration= */ false);
  }

  @Test
  public void testLinkInputsInLinkActionPostMigration() throws Exception {
    checkLinkInputsInLinkAction(getRuleType(), /* linkingInfoMigration= */ true);
  }

  @Test
  public void testAppleSdkVersionEnv() throws Exception {
    checkAppleSdkVersionEnv(getRuleType());
  }

  @Test
  public void testNonDefaultAppleSdkVersionEnv() throws Exception {
    checkNonDefaultAppleSdkVersionEnv(getRuleType());
  }

  @Test
  public void testAppleSdkDefaultPlatformEnv() throws Exception {
    checkAppleSdkDefaultPlatformEnv(getRuleType());
  }

  @Test
  public void testAvoidDepsThroughAvoidDepPreMigration() throws Exception {
    checkAvoidDepsThroughAvoidDep(getRuleType(), /* linkingInfoMigration= */ false);
  }

  @Test
  public void testAvoidDepsThroughAvoidDepPostMigration() throws Exception {
    checkAvoidDepsThroughAvoidDep(getRuleType(), /* linkingInfoMigration= */ true);
  }

  @Test
  public void testAvoidDepsObjectsAvoidViaCcLibraryPreMigration() throws Exception {
    checkAvoidDepsObjectsAvoidViaCcLibrary(getRuleType(), /* linkingInfoMigration= */ false);
  }

  @Test
  public void testAvoidDepsObjectsAvoidViaCcLibraryPostMigration() throws Exception {
    checkAvoidDepsObjectsAvoidViaCcLibrary(getRuleType(), /* linkingInfoMigration= */ true);
  }

  @Test
  public void testAvoidDepsSubtractsImportedLibraryPreMigration() throws Exception {
    checkAvoidDepsSubtractsImportedLibrary(getRuleType(), /* linkingInfoMigration= */ false);
  }

  @Test
  public void testAvoidDepsSubtractsImportedLibraryPostMigration() throws Exception {
    checkAvoidDepsSubtractsImportedLibrary(getRuleType(), /* linkingInfoMigration= */ true);
  }

  @Test
  public void testBundleLoaderIsCorrectlyPassedToTheLinkerPreMigration() throws Exception {
    checkBundleLoaderIsCorrectlyPassedToTheLinker(getRuleType(), /* linkingInfoMigration= */ false);
  }

  @Test
  public void testBundleLoaderIsCorrectlyPassedToTheLinkerPostMigration() throws Exception {
    checkBundleLoaderIsCorrectlyPassedToTheLinker(getRuleType(), /* linkingInfoMigration= */ true);
  }

  @Test
  public void testLipoBinaryAction() throws Exception {
    checkLipoBinaryAction(getRuleType());
  }

  @Test
  public void testLinkActionHasCorrectIosSimulatorMinVersion() throws Exception {
    getRuleType().scratchTarget(scratch, "platform_type", "'ios'");
    useConfiguration("--ios_multi_cpus=x86_64", "--ios_sdk_version=10.0", "--ios_minimum_os=8.0");
    checkLinkMinimumOSVersion("-mios-simulator-version-min=8.0");
  }

  @Test
  public void testLinkActionHasCorrectIosMinVersion() throws Exception {
    getRuleType().scratchTarget(scratch, "platform_type", "'ios'");
    useConfiguration("--ios_multi_cpus=arm64", "--ios_sdk_version=10.0", "--ios_minimum_os=8.0");
    checkLinkMinimumOSVersion("-miphoneos-version-min=8.0");
  }

  private void checkDylibBinaryType() throws Exception {
    getRuleType().scratchTarget(scratch, "binary_type", "'dylib'");

    CommandAction linkAction = linkAction("//x:x");
    assertThat(Joiner.on(" ").join(linkAction.getArguments())).contains("-dynamiclib");
  }

  @Test
  public void testDylibBinaryTypePreMigration() throws Exception {
    useConfiguration(linkingInfoMigrationFlag(/* linkingInfoMigration= */ false));
    checkDylibBinaryType();
  }

  @Test
  public void testDylibBinaryTypePostMigration() throws Exception {
    useConfiguration(linkingInfoMigrationFlag(/* linkingInfoMigration= */ true));
    checkDylibBinaryType();
  }

  private void checkBinaryTypeIsCorrectlySetToBundle() throws Exception {
    getRuleType().scratchTarget(scratch, "binary_type", "'loadable_bundle'");

    CommandAction linkAction = linkAction("//x:x");
    assertThat(Joiner.on(" ").join(linkAction.getArguments())).contains("-bundle");
  }

  @Test
  public void testBinaryTypeIsCorrectlySetToBundlePreMigration() throws Exception {
    useConfiguration(linkingInfoMigrationFlag(/* linkingInfoMigration= */ false));
    checkBinaryTypeIsCorrectlySetToBundle();
  }

  @Test
  public void testBinaryTypeIsCorrectlySetToBundlePostMigration() throws Exception {
    useConfiguration(linkingInfoMigrationFlag(/* linkingInfoMigration= */ true));
    checkBinaryTypeIsCorrectlySetToBundle();
  }

  @Test
  public void testMultiarchPreMigration() throws Exception {
    checkMultiarchCcDep(getRuleType(), /* linkingInfoMigration= */ false);
  }

  @Test
  public void testMultiarchPostMigration() throws Exception {
    checkMultiarchCcDep(getRuleType(), /* linkingInfoMigration= */ true);
  }

  @Test
  public void testWatchSimulatorLipoAction() throws Exception {
    checkWatchSimulatorLipoAction(getRuleType());
  }

  @Test
  public void testFrameworkDepLinkFlagsPreMigration() throws Exception {
    checkFrameworkDepLinkFlags(
        getRuleType(), new ExtraLinkArgs(), /* linkingInfoMigration= */ false);
  }

  @Test
  public void testFrameworkDepLinkFlagsPostMigration() throws Exception {
    checkFrameworkDepLinkFlags(
        getRuleType(), new ExtraLinkArgs(), /* linkingInfoMigration= */ true);
  }

  @Test
  public void testAvoidDepsDependenciesPreMigration() throws Exception {
    checkAvoidDepsDependencies(
        getRuleType(), new ExtraLinkArgs(), /* linkingInfoMigration= */ false);
  }

  @Test
  public void testAvoidDepsDependenciesPostMigration() throws Exception {
    checkAvoidDepsDependencies(
        getRuleType(), new ExtraLinkArgs(), /* linkingInfoMigration= */ true);
  }

  @Test
  public void testMinimumOs() throws Exception {
    checkMinimumOsLinkAndCompileArg(getRuleType());
  }

  @Test
  public void testMinimumOs_watchos() throws Exception {
    checkMinimumOsLinkAndCompileArg_watchos(getRuleType());
  }

  @Test
  public void testPlatformTypeIsConfigurable() throws Exception {
    scratch.file(
        "examples/BUILD",
        "load('//test_starlark:apple_binary_starlark.bzl', 'apple_binary_starlark')",
        "package(default_visibility = ['//visibility:public'])",
        "apple_binary_starlark(",
        "    name = 'bin',",
        "    deps = [':objc_lib'],",
        "    platform_type = select({",
        "        ':watch_setting': 'watchos',",
        "        '//conditions:default': 'ios',",
        "    }),",
        ")",
        "objc_library(",
        "    name = 'objc_lib',",
        "    srcs = ['a.m'],",
        ")",
        "config_setting(",
        "    name = 'watch_setting',",
        "    values = {'define': 'use_watch=1'},",
        ")");

    useConfiguration(
        "--define=use_watch=1", "--ios_multi_cpus=armv7,arm64", "--watchos_cpus=armv7k");

    Action lipoAction = actionProducingArtifact("//examples:bin", "_lipobin");

    assertThat(getSingleArchBinary(lipoAction, "armv7k")).isNotNull();
  }

  private Dict<String, Dict<String, Artifact>> generateAppleDebugOutputsStarlarkProviderMap()
      throws Exception {
    scratch.file("examples/rule/BUILD");
    scratch.file(
        "examples/rule/apple_rules.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "def _test_rule_impl(ctx):",
        "   dep = ctx.attr.deps[0]",
        "   provider = dep[apple_common.AppleDebugOutputs]",
        "   return MyInfo(",
        "      outputs_map=provider.outputs_map,",
        "   )",
        "test_rule = rule(implementation = _test_rule_impl,",
        "   attrs = {",
        "   'deps': attr.label_list(",
        "       allow_files = False,",
        "       mandatory = False,",
        "       providers = [apple_common.AppleDebugOutputs],",
        "    )",
        "})");

    scratch.file(
        "examples/apple_starlark/BUILD",
        "load('//test_starlark:apple_binary_starlark.bzl', 'apple_binary_starlark')",
        "load('//examples/rule:apple_rules.bzl', 'test_rule')",
        "package(default_visibility = ['//visibility:public'])",
        "apple_binary_starlark(",
        "    name = 'bin',",
        "    deps = [':lib'],",
        "    platform_type = 'ios',",
        ")",
        "objc_library(",
        "    name = 'lib',",
        "    srcs = ['a.m'],",
        ")",
        "test_rule(",
        "    name = 'my_target',",
        "    deps = [':bin'],",
        ")");
    ConfiguredTarget starlarkTarget = getConfiguredTarget("//examples/apple_starlark:my_target");

    // This cast is safe: struct providers are represented as Dict.
    @SuppressWarnings("unchecked")
    Dict<String, Dict<String, Artifact>> outputMap =
        (Dict<String, Dict<String, Artifact>>)
            getMyInfoFromTarget(starlarkTarget).getValue("outputs_map");
    return outputMap;
  }

  private void checkAppleDebugSymbolProviderDsymEntries(
      Map<String, Dict<String, Artifact>> outputMap, CompilationMode compilationMode)
      throws Exception {
    assertThat(outputMap).containsKey("arm64");
    assertThat(outputMap).containsKey("armv7");

    Map<String, Artifact> arm64 = outputMap.get("arm64");
    String expectedArm64Path = dsymBinaryPath("arm64", compilationMode);
    assertThat(arm64.get("dsym_binary").getExecPathString()).isEqualTo(expectedArm64Path);

    Map<String, Artifact> armv7 = outputMap.get("armv7");
    String expectedArmv7Path = dsymBinaryPath("armv7", compilationMode);
    assertThat(armv7.get("dsym_binary").getExecPathString()).isEqualTo(expectedArmv7Path);

    Map<String, Artifact> x8664 = outputMap.get("x86_64");
    String expectedx8664Path = dsymBinaryPath("x86_64", compilationMode);
    assertThat(x8664.get("dsym_binary").getExecPathString()).isEqualTo(expectedx8664Path);
  }

  private void checkAppleDebugSymbolProviderLinkMapEntries(
      Map<String, Dict<String, Artifact>> outputMap) throws Exception {
    assertThat(outputMap).containsKey("arm64");
    assertThat(outputMap).containsKey("armv7");

    Map<String, Artifact> arm64 = outputMap.get("arm64");
    assertThat(arm64.get("linkmap").getExecPathString()).isEqualTo(linkmapPath("arm64"));

    Map<String, Artifact> armv7 = outputMap.get("armv7");
    assertThat(armv7.get("linkmap").getExecPathString()).isEqualTo(linkmapPath("armv7"));

    Map<String, Artifact> x8664 = outputMap.get("x86_64");
    assertThat(x8664.get("linkmap").getExecPathString()).isEqualTo(linkmapPath("x86_64"));
  }

  @Test
  public void testAppleDebugSymbolProviderWithDsymsExposedToStarlark() throws Exception {
    useConfiguration("--apple_generate_dsym", "--ios_multi_cpus=armv7,arm64,x86_64");
    checkAppleDebugSymbolProviderDsymEntries(
        generateAppleDebugOutputsStarlarkProviderMap(), CompilationMode.FASTBUILD);
  }

  @Test
  public void testAppleDebugSymbolProviderWithAutoDsymDbgAndDsymsExposedToStarlark()
      throws Exception {
    useConfiguration(
        "--compilation_mode=dbg",
        "--apple_enable_auto_dsym_dbg",
        "--ios_multi_cpus=armv7,arm64,x86_64");
    checkAppleDebugSymbolProviderDsymEntries(
        generateAppleDebugOutputsStarlarkProviderMap(), CompilationMode.DBG);
  }

  @Test
  public void testAppleDebugSymbolProviderWithLinkMapsExposedToStarlark() throws Exception {
    useConfiguration(
        "--objc_generate_linkmap",
        "--ios_multi_cpus=armv7,arm64,x86_64");
    checkAppleDebugSymbolProviderLinkMapEntries(generateAppleDebugOutputsStarlarkProviderMap());
  }

  @Test
  public void testAppleDebugSymbolProviderWithDsymsAndLinkMapsExposedToStarlark() throws Exception {
    useConfiguration(
        "--objc_generate_linkmap",
        "--apple_generate_dsym",
        "--ios_multi_cpus=armv7,arm64,x86_64");

    Dict<String, Dict<String, Artifact>> outputMap = generateAppleDebugOutputsStarlarkProviderMap();
    checkAppleDebugSymbolProviderDsymEntries(outputMap, CompilationMode.FASTBUILD);
    checkAppleDebugSymbolProviderLinkMapEntries(outputMap);
  }

  @Test
  public void testInstrumentedFilesProviderContainsDepsAndBundleLoaderFiles() throws Exception {
    useConfiguration("--collect_code_coverage");
    scratch.file(
        "examples/BUILD",
        "load('//test_starlark:apple_binary_starlark.bzl', 'apple_binary_starlark')",
        "package(default_visibility = ['//visibility:public'])",
        "apple_binary_starlark(",
        "    name = 'bin',",
        "    deps = [':lib'],",
        "    platform_type = 'ios',",
        ")",
        "apple_binary_starlark(",
        "    name = 'bundle',",
        "    deps = [':bundle_lib'],",
        "    binary_type = 'loadable_bundle',",
        "    bundle_loader = ':bin',",
        "    platform_type = 'ios',",
        ")",
        "objc_library(",
        "    name = 'lib',",
        "    srcs = ['lib.m'],",
        ")",
        "objc_library(",
        "    name = 'bundle_lib',",
        "    srcs = ['bundle_lib.m'],",
        ")");

    ConfiguredTarget bundleTarget = getConfiguredTarget("//examples:bundle");
    InstrumentedFilesInfo instrumentedFilesProvider =
        bundleTarget.get(InstrumentedFilesInfo.STARLARK_CONSTRUCTOR);
    assertThat(instrumentedFilesProvider).isNotNull();

    assertThat(Artifact.toRootRelativePaths(instrumentedFilesProvider.getInstrumentedFiles()))
        .containsAtLeast("examples/lib.m", "examples/bundle_lib.m");
  }

  @Test
  public void testAppleSdkWatchsimulatorPlatformEnv() throws Exception {
    checkAppleSdkWatchsimulatorPlatformEnv(getRuleType());
  }

  @Test
  public void testAppleSdkWatchosPlatformEnv() throws Exception {
    checkAppleSdkWatchosPlatformEnv(getRuleType());
  }

  @Test
  public void testAppleSdkTvsimulatorPlatformEnv() throws Exception {
    checkAppleSdkTvsimulatorPlatformEnv(getRuleType());
  }

  @Test
  public void testAppleSdkTvosPlatformEnv() throws Exception {
    checkAppleSdkTvosPlatformEnv(getRuleType());
  }

  @Test
  public void testLinkActionHasCorrectWatchosSimulatorMinVersion() throws Exception {
    getRuleType().scratchTarget(scratch, "platform_type", "'watchos'");
    useConfiguration(
        "--watchos_cpus=i386", "--watchos_sdk_version=3.0", "--watchos_minimum_os=2.0");
    checkLinkMinimumOSVersion("-mwatchos-simulator-version-min=2.0");
  }

  @Test
  public void testLinkActionHasCorrectWatchosMinVersion() throws Exception {
    getRuleType().scratchTarget(scratch, "platform_type", "'watchos'");
    useConfiguration(
        "--watchos_cpus=armv7k", "--watchos_sdk_version=3.0", "--watchos_minimum_os=2.0");
    checkLinkMinimumOSVersion("-mwatchos-version-min=2.0");
  }

  @Test
  public void testLinkActionHasCorrectTvosSimulatorMinVersion() throws Exception {
    getRuleType().scratchTarget(scratch, "platform_type", "'tvos'");
    useConfiguration("--tvos_cpus=x86_64", "--tvos_sdk_version=10.1", "--tvos_minimum_os=10.0");
    checkLinkMinimumOSVersion("-mtvos-simulator-version-min=10.0");
  }

  @Test
  public void testLinkActionHasCorrectTvosMinVersion() throws Exception {
    getRuleType().scratchTarget(scratch, "platform_type", "'tvos'");
    useConfiguration("--tvos_cpus=arm64", "--tvos_sdk_version=10.1", "--tvos_minimum_os=10.0");
    checkLinkMinimumOSVersion("-mtvos-version-min=10.0");
  }

  @Test
  public void testWatchSimulatorLinkAction() throws Exception {
    checkWatchSimulatorLinkAction(getRuleType());
  }

  @Test
  public void testAvoidDepsObjectsPreMigration() throws Exception {
    checkAvoidDepsObjects(getRuleType(), /* linkingInfoMigration= */ false);
  }

  @Test
  public void testAvoidDepsObjectsPostMigration() throws Exception {
    checkAvoidDepsObjects(getRuleType(), /* linkingInfoMigration= */ true);
  }

  @Test
  public void testAvoidDepsObjcLibrariesPreMigration() throws Exception {
    checkAvoidDepsObjcLibraries(getRuleType(), /* linkingInfoMigration= */ false);
  }

  @Test
  public void testAvoidDepsObjcLibrariesPostMigration() throws Exception {
    checkAvoidDepsObjcLibraries(getRuleType(), /* linkingInfoMigration= */ true);
  }

  @Test
  public void testAvoidDepsObjcLibrariesAvoidViaCcLibraryPreMigration() throws Exception {
    checkAvoidDepsObjcLibrariesAvoidViaCcLibrary(getRuleType(), /* linkingInfoMigration= */ false);
  }

  @Test
  public void testAvoidDepsObjcLibrariesAvoidViaCcLibraryPostMigration() throws Exception {
    checkAvoidDepsObjcLibrariesAvoidViaCcLibrary(getRuleType(), /* linkingInfoMigration= */ true);
  }

  @Test
  public void testBundleLoaderPropagatesAppleExecutableBinaryProvider() throws Exception {
    scratch.file(
        "bin/BUILD",
        "load('//test_starlark:apple_binary_starlark.bzl', 'apple_binary_starlark')",
        "apple_binary_starlark(",
        "    name = 'bin',",
        "    deps = [':lib'],",
        "    platform_type = 'ios',",
        ")",
        "objc_library(",
        "    name = 'lib',",
        "    srcs = ['a.m'],",
        ")");
    scratch.file(
        "test/BUILD",
        "load('//test_starlark:apple_binary_starlark.bzl', 'apple_binary_starlark')",
        "apple_binary_starlark(",
        "    name = 'test',",
        "    deps = [':lib'],",
        "    binary_type = 'loadable_bundle',",
        "    bundle_loader = '//bin:bin',",
        "    platform_type = 'ios',",
        ")",
        "objc_library(",
        "    name = 'lib',",
        "    srcs = ['a.m'],",
        ")");
    ConfiguredTarget binTarget = getConfiguredTarget("//bin:bin");
    AppleExecutableBinaryInfo executableBinaryProvider =
        binTarget.get(AppleExecutableBinaryInfo.STARLARK_CONSTRUCTOR);
    assertThat(executableBinaryProvider).isNotNull();

    CommandAction testLinkAction = linkAction("//test:test");
    assertThat(testLinkAction.getInputs().toList())
        .contains(executableBinaryProvider.getAppleExecutableBinary());
  }

  @Test
  public void testLoadableBundleBinaryAddsRpathLinkOptWithNoBundleLoader() throws Exception {
    scratch.file(
        "test/BUILD",
        "load('//test_starlark:apple_binary_starlark.bzl', 'apple_binary_starlark')",
        "apple_binary_starlark(",
        "    name = 'test',",
        "    deps = [':lib'],",
        "    binary_type = 'loadable_bundle',",
        "    platform_type = 'ios',",
        ")",
        "objc_library(",
        "    name = 'lib',",
        "    srcs = ['a.m'],",
        ")");

    CommandAction testLinkAction = linkAction("//test:test");
    assertThat(Joiner.on(" ").join(testLinkAction.getArguments()))
        .contains("@loader_path/Frameworks");
  }

  @Test
  public void testLoadableBundleBinaryAddsRpathLinkOptWithBundleLoader() throws Exception {
    scratch.file(
        "bin/BUILD",
        "load('//test_starlark:apple_binary_starlark.bzl', 'apple_binary_starlark')",
        "apple_binary_starlark(",
        "    name = 'bin',",
        "    deps = [':lib'],",
        "    platform_type = 'ios',",
        ")",
        "objc_library(",
        "    name = 'lib',",
        "    srcs = ['a.m'],",
        ")");
    scratch.file(
        "test/BUILD",
        "load('//test_starlark:apple_binary_starlark.bzl', 'apple_binary_starlark')",
        "apple_binary_starlark(",
        "    name = 'test',",
        "    deps = [':lib'],",
        "    binary_type = 'loadable_bundle',",
        "    bundle_loader = '//bin:bin',",
        "    platform_type = 'ios',",
        ")",
        "objc_library(",
        "    name = 'lib',",
        "    srcs = ['a.m'],",
        ")");

    CommandAction testLinkAction = linkAction("//test:test");
    assertThat(Joiner.on(" ").join(testLinkAction.getArguments()))
        .contains("@loader_path/Frameworks");
  }

  @Test
  public void testMinimumOsDifferentTargets() throws Exception {
    checkMinimumOsDifferentTargets(getRuleType(), "_lipobin", "_bin");
  }

  @Test
  public void testDrops32BitIosArchitecture() throws Exception {
    verifyDrops32BitIosArchitecture(getRuleType());
  }

  @Test
  public void testDrops32BitWatchArchitecture() throws Exception {
    verifyDrops32BitWatchArchitecture(getRuleType());
  }

  @Test
  public void testFeatureFlags_offByDefault() throws Exception {
    useConfiguration("--enforce_transitive_configs_for_config_feature_flag");
    scratchFeatureFlagTestLib();
    scratch.file(
        "test/BUILD",
        "load('//test_starlark:apple_binary_starlark.bzl', 'apple_binary_starlark')",
        "apple_binary_starlark(",
        "    name = 'bin',",
        "    deps = ['//lib:objcLib'],",
        "    platform_type = 'ios',",
        "    transitive_configs = ['//lib:flag1', '//lib:flag2'],",
        ")");

    CommandAction linkAction = linkAction("//test:bin");
    CommandAction objcLibArchiveAction =
        (CommandAction)
            getGeneratingAction(getFirstArtifactEndingWith(linkAction.getInputs(), "libobjcLib.a"));

    CommandAction flag1offCompileAction =
        (CommandAction)
            getGeneratingAction(
                getFirstArtifactEndingWith(objcLibArchiveAction.getInputs(), "flag1off.o"));
    CommandAction flag2offCompileAction =
        (CommandAction)
            getGeneratingAction(
                getFirstArtifactEndingWith(objcLibArchiveAction.getInputs(), "flag2off.o"));

    String compileArgs1 = Joiner.on(" ").join(flag1offCompileAction.getArguments());
    String compileArgs2 = Joiner.on(" ").join(flag2offCompileAction.getArguments());
    assertThat(compileArgs1).contains("FLAG_1_OFF");
    assertThat(compileArgs1).contains("FLAG_2_OFF");
    assertThat(compileArgs2).contains("FLAG_1_OFF");
    assertThat(compileArgs2).contains("FLAG_2_OFF");
  }

  @Test
  public void testExecutableObjcProvider() throws Exception {
    scratch.file("testlib/BUILD", "objc_library(", "    name = 'lib',", "    srcs = ['a.m'],", ")");

    getRuleType()
        .scratchTarget(scratch, "binary_type", "'executable'", "deps", "['//testlib:lib']");

    ObjcProvider objcProvider = objcProviderForTarget("//x:x");
    assertThat(Artifact.toRootRelativePaths(objcProvider.get(ObjcProvider.LIBRARY)))
        .contains("testlib/liblib.a");
    CcLinkingContext ccLinkingContext = ccInfoForTarget("//x:x").getCcLinkingContext();
    assertThat(
            Artifact.toRootRelativePaths(
                ccLinkingContext.getStaticModeParamsForDynamicLibraryLibraries()))
        .contains("testlib/liblib.a");
  }

  private void checkIncludesLinkstampFiles() throws Exception {
    scratch.file(
        "test/BUILD",
        "load('//test_starlark:apple_binary_starlark.bzl', 'apple_binary_starlark')",
        "apple_binary_starlark(",
        "  name = 'bin',",
        "  platform_type = 'macos',",
        "  deps = [':lib'],",
        ")",
        "cc_library(",
        "  name = 'lib',",
        "  linkstamp = 'some_linkstamp.cc',",
        ")");
    CommandAction linkAction = linkAction("//test:bin");
    assertThat(paramFileArgsForAction(linkAction))
        .contains(execPathEndingWith(linkAction.getInputs().toList(), "some_linkstamp.o"));
  }

  @Test
  public void testIncludesLinkstampFilesPreMigration() throws Exception {
    useConfiguration(linkingInfoMigrationFlag(/* linkingInfoMigration= */ false));
    checkIncludesLinkstampFiles();
  }

  @Test
  public void testIncludesLinkstampFilesPostMigration() throws Exception {
    useConfiguration(linkingInfoMigrationFlag(/* linkingInfoMigration= */ true));
    checkIncludesLinkstampFiles();
  }

  @Test
  public void testProcessHeadersInDependencies() throws Exception {
    MockObjcSupport.setupCcToolchainConfig(
        mockToolsConfig, MockObjcSupport.darwinX86_64().withFeatures(CppRuleClasses.PARSE_HEADERS));
    useConfiguration("--features=parse_headers", "--process_headers_in_dependencies");
    ConfiguredTarget x =
        scratchConfiguredTarget(
            "foo",
            "x",
            "load('//test_starlark:apple_binary_starlark.bzl', 'apple_binary_starlark')",
            "apple_binary_starlark(name = 'x', platform_type = 'macos', deps = [':y', ':z'])",
            "cc_library(name = 'y', hdrs = ['y.h'])",
            "objc_library(name = 'z', hdrs = ['z.h'])");
    String validation = ActionsTestUtil.baseNamesOf(getOutputGroup(x, OutputGroupInfo.VALIDATION));
    assertThat(validation).contains("y.h.processed");
    assertThat(validation).contains("z.h.processed");
  }

  protected RuleType getRuleType() {
    return RULE_TYPE;
  }

  private void checkExpandedLinkopts() throws Exception {
    scratch.file(
        "a/BUILD",
        "load('//test_starlark:apple_binary_starlark.bzl', 'apple_binary_starlark')",
        "objc_library(",
        "    name = 'lib',",
        "    srcs = ['a.m'],",
        ")",
        "genrule(name = 'linker', cmd='generate', outs=['a.lds'])",
        "apple_binary_starlark(",
        "    name='bin',",
        "    platform_type = 'ios',",
        "    deps = [':lib'],",
        "    linkopts=['@$(location a.lds)'],",
        "    additional_linker_inputs=['a.lds'])");

    ConfiguredTarget target = getConfiguredTarget("//a:bin");
    CommandAction action = linkAction("//a:bin");

    assertThat(Joiner.on(" ").join(action.getArguments()))
        .contains(
            String.format(
                "-Wl,@%s/a/a.lds",
                getRuleContext(target).getGenfilesDirectory().getExecPath().getPathString()));
  }

  @Test
  public void testExpandedLinkoptsPreMigration() throws Exception {
    useConfiguration(linkingInfoMigrationFlag(/* linkingInfoMigration= */ false));
    checkExpandedLinkopts();
  }

  @Test
  public void testExpandedLinkoptsPostMigration() throws Exception {
    useConfiguration(linkingInfoMigrationFlag(/* linkingInfoMigration= */ true));
    checkExpandedLinkopts();
  }

  private void checkProvidesLinkerScriptToLinkAction() throws Exception {
    scratch.file(
        "a/BUILD",
        "load('//test_starlark:apple_binary_starlark.bzl', 'apple_binary_starlark')",
        "objc_library(",
        "    name = 'lib',",
        "    srcs = ['a.m'],",
        ")",
        "apple_binary_starlark(",
        "    name='bin',",
        "    platform_type = 'ios',",
        "    deps = [':lib'],",
        "    linkopts=['@$(location a.lds)'],",
        "    additional_linker_inputs=['a.lds'])");

    CommandAction action = linkAction("//a:bin");

    assertThat(ActionsTestUtil.baseArtifactNames(action.getInputs())).contains("a.lds");
  }

  @Test
  public void testProvidesLinkerScriptToLinkActionPreMigration() throws Exception {
    useConfiguration(linkingInfoMigrationFlag(/* linkingInfoMigration= */ false));
    checkProvidesLinkerScriptToLinkAction();
  }

  @Test
  public void testProvidesLinkerScriptToLinkActionPostMigration() throws Exception {
    useConfiguration(linkingInfoMigrationFlag(/* linkingInfoMigration= */ true));
    checkProvidesLinkerScriptToLinkAction();
  }

  private void checkRuntimeLib() throws Exception {
    MockObjcSupport.setupCcToolchainConfig(
        mockToolsConfig,
        MockObjcSupport.darwinX86_64().withFeatures(CppRuleClasses.STATIC_LINK_CPP_RUNTIMES));
    scratch.file(
        "a/BUILD",
        "load('//test_starlark:apple_binary_starlark.bzl', 'apple_binary_starlark')",
        "objc_library(",
        "    name = 'lib',",
        "    srcs = ['a.m'],",
        ")",
        "apple_binary_starlark(",
        "    name='bin',",
        "    platform_type = 'macos',",
        "    deps = [':lib'])");

    ConfiguredTarget libTarget = getConfiguredTarget("//a:lib");
    RuleContext libRuleContext = getRuleContext(libTarget);
    CcToolchainProvider toolchain =
        CppHelper.getToolchain(libRuleContext, libRuleContext.getPrerequisite("$cc_toolchain"));
    FeatureConfiguration featureConfiguration =
        CcCommon.configureFeaturesOrThrowEvalException(
            /* requestedFeatures= */ ImmutableSet.of(),
            /* unsupportedFeatures= */ ImmutableSet.of(),
            Language.OBJC,
            toolchain,
            libRuleContext.getFragment(CppConfiguration.class));
    ImmutableList<Artifact> staticRuntimes =
        toolchain.getStaticRuntimeLinkInputs(featureConfiguration).toList();
    CommandAction action = linkAction("//a:bin");

    assertThat(paramFileArgsForAction(action))
        .containsAtLeastElementsIn(ActionsTestUtil.execPaths(staticRuntimes));
  }

  @Test
  public void testRuntimeLibPreMigration() throws Exception {
    useConfiguration(linkingInfoMigrationFlag(/* linkingInfoMigration= */ false));
    checkRuntimeLib();
  }

  @Test
  public void testRuntimeLibPostMigration() throws Exception {
    useConfiguration(linkingInfoMigrationFlag(/* linkingInfoMigration= */ true));
    checkRuntimeLib();
  }
}
