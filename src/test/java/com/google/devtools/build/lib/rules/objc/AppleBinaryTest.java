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
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandAction;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.packages.util.MockObjcSupport;
import com.google.devtools.build.lib.packages.util.MockProtoSupport;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration.ConfigurationDistinguisher;
import com.google.devtools.build.lib.rules.cpp.CppRuleClasses;
import com.google.devtools.build.lib.rules.objc.AppleBinary.BinaryType;
import com.google.devtools.build.lib.rules.objc.CompilationSupport.ExtraLinkArgs;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import net.starlark.java.eval.Dict;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test case for apple_binary. */
@RunWith(JUnit4.class)
public class AppleBinaryTest extends ObjcRuleTestCase {
  static final RuleType RULE_TYPE = new RuleType("apple_binary") {
    @Override
    Iterable<String> requiredAttributes(Scratch scratch, String packageDir,
        Set<String> alreadyAdded) throws IOException {
      ImmutableList.Builder<String> attributes = new ImmutableList.Builder<>();
      if (!alreadyAdded.contains("deps")) {
        String depPackageDir = packageDir + "_defaultDep";
        scratch.file(depPackageDir + "/a.m");
        scratch.file(depPackageDir + "/private.h");
        scratch.file(depPackageDir + "/BUILD",
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
  public void setupMyInfo() throws Exception {
    scratch.file("myinfo/myinfo.bzl", "MyInfo = provider()");

    scratch.file("myinfo/BUILD");

    MockProtoSupport.setupWorkspace(scratch);
    invalidatePackages();
  }

  private StructImpl getMyInfoFromTarget(ConfiguredTarget configuredTarget) throws Exception {
    Provider.Key key =
        new StarlarkProvider.Key(
            Label.parseAbsolute("//myinfo:myinfo.bzl", ImmutableMap.of()), "MyInfo");
    return (StructImpl) configuredTarget.get(key);
  }

  @Test
  public void testOutputDirectoryWithMandatoryMinimumVersion() throws Exception {
    scratch.file("a/BUILD",
        "apple_binary(name='a', platform_type='ios', deps=['b'], minimum_os_version='7.0')",
        "objc_library(name='b', srcs=['b.c'])");

    useConfiguration(
        "--experimental_apple_mandatory_minimum_version",
        "ios_cpus=i386");
    ConfiguredTarget a = getConfiguredTarget("//a:a");
    ConfiguredTarget b = getDirectPrerequisite(a, "//a:b");

    PathFragment aPath = getConfiguration(a).getOutputDirectory(RepositoryName.MAIN).getExecPath();
    PathFragment bPath = getConfiguration(b).getOutputDirectory(RepositoryName.MAIN).getExecPath();

    assertThat(aPath.getPathString()).doesNotMatch("-min[0-9]");
    assertThat(bPath.getPathString()).contains("-min7.0-");
  }

  @Test
  public void testMandatoryMinimumVersionEnforced() throws Exception {
    scratch.file("a/BUILD", "apple_binary(name='a', platform_type='ios')");

    useConfiguration("--experimental_apple_mandatory_minimum_version");
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//a:a");
    assertContainsEvent("This attribute must be explicitly specified");
  }

  @Test
  public void testMandatoryMinimumOsVersionUnset() throws Exception {
    getRuleType().scratchTarget(scratch,
        "platform_type", "'watchos'");
    useConfiguration("--experimental_apple_mandatory_minimum_version");
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//x:x");
    assertContainsEvent("must be explicitly specified");
  }

  @Test
  public void testMandatoryMinimumOsVersionSet() throws Exception {
    getRuleType().scratchTarget(scratch,
        "minimum_os_version", "'8.0'",
        "platform_type", "'watchos'");
    useConfiguration("--experimental_apple_mandatory_minimum_version");
    getConfiguredTarget("//x:x");
  }

  @Test
  public void testLipoActionEnv() throws Exception {
    getRuleType().scratchTarget(scratch,
        "platform_type", "'watchos'");

    useConfiguration("--watchos_cpus=i386,armv7k", "--xcode_version=7.3",
        "--watchos_sdk_version=2.1");

    CommandAction action = (CommandAction) lipoBinAction("//x:x");
    assertAppleSdkVersionEnv(action, "2.1");
    assertAppleSdkPlatformEnv(action, "WatchOS");
    assertXcodeVersionEnv(action, "7.3");
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
  public void testLipoActionEnv_sdkVersionPadding() throws Exception {
    getRuleType().scratchTarget(scratch,
        "platform_type", "'watchos'");

    useConfiguration("--watchos_cpus=i386,armv7k",
        "--xcode_version=7.3", "--watchos_sdk_version=2");

    CommandAction action = (CommandAction) lipoBinAction("//x:x");
    assertAppleSdkVersionEnv(action, "2.0");
  }

  @Test
  public void testCcDependencyLinkoptsArePropagatedToLinkAction() throws Exception {
    checkCcDependencyLinkoptsArePropagatedToLinkAction(getRuleType());
  }

  @Test
  public void testUnknownPlatformType() throws Exception {
    checkError(
        "package",
        "test",
        String.format(MultiArchSplitTransitionProvider.UNSUPPORTED_PLATFORM_TYPE_ERROR_FORMAT,
            "meow_meow_os"),
        "apple_binary(name = 'test', platform_type = 'meow_meow_os')");
  }

  @Test
  public void testProtoDylibDeps() throws Exception {
    checkProtoDedupingDeps(BinaryType.DYLIB);
  }

  @Test
  public void testProtoBundleLoaderDeps() throws Exception {
    checkProtoDedupingDeps(BinaryType.LOADABLE_BUNDLE);
  }

  /**
   * Test scenario where all proto symbols are contained within the lower level dependency. There is
   * an implicit dependency hierarchy between the different apple_binary types (executable,
   * loadable_bundle, dylib) which looks like this (top level binary types can depend on lower level
   * binary types):
   *
   *              loadable_bundle
   *            /                \
   *        dylib(s)          executable
   *                              |
   *                            dylibs
   *
   * The mechanism to remove duplicate dependencies between dylibs and executable binaries works the
   * same way between executable and loadable_bundle binaries; the only difference is through which
   * attribute the dependency is declared (dylibs vs bundle_loader). This test scenario sets up
   * dependencies for low level and top level binaries and checks that the correct files are linked
   * for each of the binaries.
   *
   * @param depBinaryType either {@link BinaryType#DYLIB} or {@link BinaryType#LOADABLE_BUNDLE}, as
   *     this deduping test is applicable for either
   */
  private void checkProtoDedupingDeps(BinaryType depBinaryType) throws Exception {
    MockObjcSupport.setupObjcProtoLibrary(scratch);
    scratch.file("x/filter_a.pbascii");
    scratch.file("x/filter_b.pbascii");
    scratch.file(
        "protos/BUILD",
        TestConstants.LOAD_PROTO_LIBRARY,
        "load('//objc_proto_library:objc_proto_library.bzl', 'objc_proto_library')",
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
        "    srcs = ['data_a.proto', 'data_c.proto', 'data_d.proto'],",
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

    if (depBinaryType == BinaryType.DYLIB) {
      scratchFrameworkStarlarkStub("frameworkstub/framework_stub.bzl");
      scratch.file(
          "depBinary/BUILD",
          "load('//frameworkstub:framework_stub.bzl', 'framework_stub_rule')",
          "apple_binary(",
          "    name = 'apple_low_level_binary',",
          "    deps = ['//libs:objc_lib'],",
          "    platform_type = 'ios',",
          "    binary_type = 'dylib',",
          ")",
          "framework_stub_rule(",
          "  name = 'low_level_framework',",
          "  deps = [':apple_low_level_binary'],",
          "  binary = ':apple_low_level_binary')");
      getRuleType().scratchTarget(
          scratch,
          "binary_type",
          "'executable'",
          "deps",
          "['//libs:objc_lib']",
          "dylibs",
          "['//depBinary:low_level_framework']");

    } else {
      assertThat(depBinaryType == BinaryType.LOADABLE_BUNDLE).isTrue();
      scratch.file(
          "depBinary/BUILD",
          "apple_binary(",
          "    name = 'apple_low_level_binary',",
          "    deps = ['//libs:objc_lib'],",
          "    platform_type = 'ios',",
          "    binary_type = 'executable',",
          ")");
      getRuleType().scratchTarget(
          scratch,
          "binary_type",
          "'loadable_bundle'",
          "deps",
          "['//libs:objc_lib']",
          "bundle_loader",
          "'//depBinary:apple_low_level_binary'");
    }

    // The proto libraries objc_lib depends on should be linked into the low level binary but not x.
    Artifact lowLevelBinary =
        Iterables.getOnlyElement(linkAction("//depBinary:apple_low_level_binary").getOutputs());
    ImmutableList<Artifact> lowLevelObjectFiles = getAllObjectFilesLinkedInBin(lowLevelBinary);
    assertThat(getFirstArtifactEndingWith(lowLevelObjectFiles, "DataA.pbobjc.o")).isNotNull();
    assertThat(getFirstArtifactEndingWith(lowLevelObjectFiles, "DataB.pbobjc.o")).isNotNull();
    assertThat(getFirstArtifactEndingWith(lowLevelObjectFiles, "DataC.pbobjc.o")).isNotNull();
    assertThat(getFirstArtifactEndingWith(lowLevelObjectFiles, "DataD.pbobjc.o")).isNotNull();

    Artifact bin = Iterables.getOnlyElement(linkAction("//x:x").getOutputs());
    ImmutableList<Artifact> binObjectFiles = getAllObjectFilesLinkedInBin(bin);
    assertThat(getFirstArtifactEndingWith(binObjectFiles, "DataA.pbobjc.o")).isNull();
    assertThat(getFirstArtifactEndingWith(binObjectFiles, "DataB.pbobjc.o")).isNull();
    assertThat(getFirstArtifactEndingWith(binObjectFiles, "DataC.pbobjc.o")).isNull();
    assertThat(getFirstArtifactEndingWith(binObjectFiles, "DataD.pbobjc.o")).isNull();
  }

  @Test
  public void testProtoDylibDepsPartial() throws Exception {
    checkProtoDedupingDepsPartial(AppleBinary.BinaryType.DYLIB);
  }

  @Test
  public void testProtoBundleLoaderDepsPartial() throws Exception {
    checkProtoDedupingDepsPartial(AppleBinary.BinaryType.LOADABLE_BUNDLE);
  }

  /**
   * Test scenario where proto symbols are mixed between the low and top level binaries. There is
   * an implicit dependency hierarchy between the different apple_binary types (executable,
   * loadable_bundle, dylib) which looks like this (top level binary types can depend on lower level
   * binary types):
   *
   *              loadable_bundle
   *            /                \
   *        dylib(s)          executable
   *                              |
   *                            dylibs
   *
   * The mechanism to remove duplicate dependencies between dylibs and executable binaries works the
   * same way between executable and loadable_bundle binaries; the only difference is through which
   * attribute the dependency is declared (dylibs vs bundle_loader). This test scenario sets up
   * dependencies for low level and top level binaries and checks that the correct files are linked
   * for each of the binaries.
   *
   * @param depBinaryType either {@link BinaryType#DYLIB} or {@link BinaryType#LOADABLE_BUNDLE}, as
   *     this deduping test is applicable for either
   */
  private void checkProtoDedupingDepsPartial(BinaryType depBinaryType) throws Exception {
    MockObjcSupport.setupObjcProtoLibrary(scratch);
    scratch.file("x/filter_a.pbascii");
    scratch.file("x/filter_b.pbascii");
    scratch.file(
        "protos/BUILD",
        TestConstants.LOAD_PROTO_LIBRARY,
        "load('//objc_proto_library:objc_proto_library.bzl', 'objc_proto_library')",
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
        "    srcs = ['data_a.proto', 'data_c.proto', 'data_d.proto'],",
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
        "    name = 'main_lib',",
        "    srcs = ['a.m'],",
        "    deps = ['//protos:objc_protos_a', '//protos:objc_protos_b']",
        ")",
        "objc_library(",
        "    name = 'apple_low_level_lib',",
        "    srcs = ['b.m'],",
        "    deps = ['//protos:objc_protos_a']",
        ")");

    if (depBinaryType == BinaryType.DYLIB) {
      scratchFrameworkStarlarkStub("frameworkstub/framework_stub.bzl");
      scratch.file(
          "depBinary/BUILD",
          "load('//frameworkstub:framework_stub.bzl', 'framework_stub_rule')",
          "apple_binary(",
          "    name = 'apple_low_level_binary',",
          "    deps = ['//libs:apple_low_level_lib'],",
          "    platform_type = 'ios',",
          "    binary_type = 'dylib',",
          ")",
          "framework_stub_rule(",
          "  name = 'low_level_framework',",
          "  deps = [':apple_low_level_binary'],",
          "  binary = ':apple_low_level_binary')");
      getRuleType().scratchTarget(
          scratch,
          "binary_type",
          "'executable'",
          "deps",
          "['//libs:main_lib']",
          "dylibs",
          "['//depBinary:low_level_framework']");

    } else {
      assertThat(depBinaryType == BinaryType.LOADABLE_BUNDLE).isTrue();
      scratch.file(
          "depBinary/BUILD",
          "apple_binary(",
          "    name = 'apple_low_level_binary',",
          "    deps = ['//libs:apple_low_level_lib'],",
          "    platform_type = 'ios',",
          "    binary_type = 'executable',",
          ")");
      getRuleType().scratchTarget(
          scratch,
          "binary_type",
          "'loadable_bundle'",
          "deps",
          "['//libs:main_lib']",
          "bundle_loader",
          "'//depBinary:apple_low_level_binary'");
    }

    // The proto libraries objc_lib depends on should be linked into the low level binary but not x.
    Artifact lowLevelBinary =
        Iterables.getOnlyElement(linkAction("//depBinary:apple_low_level_binary").getOutputs());
    ImmutableList<Artifact> lowLevelObjectFiles = getAllObjectFilesLinkedInBin(lowLevelBinary);
    assertThat(getFirstArtifactEndingWith(lowLevelObjectFiles, "DataA.pbobjc.o")).isNotNull();
    assertThat(getFirstArtifactEndingWith(lowLevelObjectFiles, "DataB.pbobjc.o")).isNotNull();
    assertThat(getFirstArtifactEndingWith(lowLevelObjectFiles, "DataC.pbobjc.o")).isNull();
    assertThat(getFirstArtifactEndingWith(lowLevelObjectFiles, "DataD.pbobjc.o")).isNull();

    Artifact bin = Iterables.getOnlyElement(linkAction("//x:x").getOutputs());
    ImmutableList<Artifact> binObjectFiles = getAllObjectFilesLinkedInBin(bin);

    assertThat(getFirstArtifactEndingWith(binObjectFiles, "DataA.pbobjc.o")).isNull();
    assertThat(getFirstArtifactEndingWith(binObjectFiles, "DataB.pbobjc.o")).isNull();
    assertThat(getFirstArtifactEndingWith(binObjectFiles, "DataC.pbobjc.o")).isNotNull();
    assertThat(getFirstArtifactEndingWith(binObjectFiles, "DataD.pbobjc.o")).isNotNull();
  }

  @Test
  public void testProtoDepsViaDylib() throws Exception {
    checkProtoDisjointDeps(BinaryType.DYLIB);
  }

  @Test
  public void testProtoDepsViaBundleLoader() throws Exception {
    checkProtoDisjointDeps(BinaryType.LOADABLE_BUNDLE);
  }

  /**
   * Test scenario where a proto in the top level binary depends on a proto in a low level binary.
   * There is an implicit dependency hierarchy between the different apple_binary types (executable,
   * loadable_bundle, dylib) which looks like this (top level binary types can depend on lower level
   * binary types):
   *
   *              loadable_bundle
   *            /                \
   *        dylib(s)          executable
   *                              |
   *                            dylibs
   *
   * The mechanism to remove duplicate dependencies between dylibs and executable binaries works the
   * same way between executable and loadable_bundle binaries; the only difference is through which
   * attribute the dependency is declared (dylibs vs bundle_loader). This test scenario sets up
   * dependencies for low level and top level binaries and checks that the correct files are linked
   * for each of the binaries.
   *
   * @param depBinaryType either {@link BinaryType#DYLIB} or {@link BinaryType#LOADABLE_BUNDLE}, as
   *     this deduping test is applicable for either
   */
  private void checkProtoDisjointDeps(BinaryType depBinaryType) throws Exception {
    MockObjcSupport.setupObjcProtoLibrary(scratch);
    scratch.file("x/filter_a.pbascii");
    scratch.file("x/filter_b.pbascii");
    scratch.file(
        "protos/BUILD",
        TestConstants.LOAD_PROTO_LIBRARY,
        "load('//objc_proto_library:objc_proto_library.bzl', 'objc_proto_library')",
        "proto_library(",
        "    name = 'protos_main',",
        "    srcs = ['data_a.proto', 'data_b.proto'],",
        ")",
        "proto_library(",
        "    name = 'protos_low_level',",
        "    srcs = ['data_b.proto'],",
        ")",
        "objc_proto_library(",
        "    name = 'objc_protos_main',",
        "    portable_proto_filters = ['filter_a.pbascii'],",
        "    deps = [':protos_main'],",
        ")",
        "objc_proto_library(",
        "    name = 'objc_protos_low_level',",
        "    portable_proto_filters = ['filter_b.pbascii'],",
        "    deps = [':protos_low_level'],",
        ")");
    scratch.file(
        "libs/BUILD",
        "objc_library(",
        "    name = 'main_lib',",
        "    srcs = ['a.m'],",
        "    deps = ['//protos:objc_protos_main',]",
        ")",
        "objc_library(",
        "    name = 'apple_low_level_lib',",
        "    srcs = ['a.m'],",
        "    deps = ['//protos:objc_protos_low_level',]",
        ")");

    if (depBinaryType == BinaryType.DYLIB) {
      scratchFrameworkStarlarkStub("frameworkstub/framework_stub.bzl");
      scratch.file(
          "depBinary/BUILD",
          "load('//frameworkstub:framework_stub.bzl', 'framework_stub_rule')",
          "apple_binary(",
          "    name = 'apple_low_level_binary',",
          "    deps = ['//libs:apple_low_level_lib'],",
          "    platform_type = 'ios',",
          "    binary_type = 'dylib',",
          ")",
          "framework_stub_rule(",
          "  name = 'low_level_framework',",
          "  deps = [':apple_low_level_binary'],",
          "  binary = ':apple_low_level_binary')");
      getRuleType().scratchTarget(
          scratch,
          "binary_type",
          "'executable'",
          "deps",
          "['//libs:main_lib']",
          "dylibs",
          "['//depBinary:low_level_framework']");

    } else {
      assertThat(depBinaryType == BinaryType.LOADABLE_BUNDLE).isTrue();
      scratch.file(
          "depBinary/BUILD",
          "apple_binary(",
          "    name = 'apple_low_level_binary',",
          "    deps = ['//libs:apple_low_level_lib'],",
          "    platform_type = 'ios',",
          "    binary_type = 'executable',",
          ")");
      getRuleType().scratchTarget(
          scratch,
          "binary_type",
          "'loadable_bundle'",
          "deps",
          "['//libs:main_lib']",
          "bundle_loader",
          "'//depBinary:apple_low_level_binary'");
    }

    // The proto libraries objc_lib depends on should be linked into apple_dylib but not x.
    Artifact lowLevelBinary =
        Iterables.getOnlyElement(linkAction("//depBinary:apple_low_level_binary").getOutputs());
    ImmutableList<Artifact> lowLevelObjectFiles = getAllObjectFilesLinkedInBin(lowLevelBinary);
    assertThat(getFirstArtifactEndingWith(lowLevelObjectFiles, "DataA.pbobjc.o")).isNull();
    assertThat(getFirstArtifactEndingWith(lowLevelObjectFiles, "DataB.pbobjc.o")).isNotNull();

    Artifact bin = Iterables.getOnlyElement(linkAction("//x:x").getOutputs());
    ImmutableList<Artifact> binObjectFiles = getAllObjectFilesLinkedInBin(bin);
    assertThat(getFirstArtifactEndingWith(binObjectFiles, "DataA.pbobjc.o")).isNotNull();
    assertThat(getFirstArtifactEndingWith(binObjectFiles, "DataB.pbobjc.o")).isNull();
    Action dataAObjectAction =
        getGeneratingAction(getFirstArtifactEndingWith(binObjectFiles, "DataA.pbobjc.o"));
    assertThat(
            getFirstArtifactEndingWith(
                getExpandedActionInputs(dataAObjectAction), "DataB.pbobjc.h"))
        .isNotNull();
  }

  @Test
  public void testProtoBundlingWithTargetsWithNoDeps() throws Exception {
    checkProtoBundlingWithTargetsWithNoDeps(getRuleType());
  }

  @Test
  public void testProtoBundlingDoesNotHappen() throws Exception {
    useConfiguration("--noenable_apple_binary_native_protos");
    checkProtoBundlingDoesNotHappen(getRuleType());
  }

  @Test
  public void testAvoidDepsObjectsWithCrosstool() throws Exception {
    checkAvoidDepsObjectsWithCrosstool(getRuleType());
  }

  @Test
  public void testBundleLoaderCantBeSetWithoutBundleBinaryType() throws Exception {
    getRuleType().scratchTarget(scratch);
    checkError(
        "bundle", "bundle", AppleBinary.BUNDLE_LOADER_NOT_IN_BUNDLE_ERROR,
        "apple_binary(",
        "    name = 'bundle',",
        "    bundle_loader = '//x:x',",
        "    platform_type = 'ios',",
        ")");
  }

  /** Returns the bcsymbolmap artifact for given architecture and compilation mode. */
  protected Artifact bitcodeSymbol(String arch, CompilationMode mode) throws Exception {
    SpawnAction lipoAction = (SpawnAction) lipoBinAction("//examples/apple_starlark:bin");

    String bin =
        configurationBin(arch, ConfigurationDistinguisher.APPLEBIN_IOS, null, mode)
            + "examples/apple_starlark/bin_bin";
    Artifact binArtifact = getFirstArtifactEndingWith(lipoAction.getInputs(), bin);
    CommandAction linkAction = (CommandAction) getGeneratingAction(binArtifact);
    return getFirstArtifactEndingWith(linkAction.getOutputs(), "bcsymbolmap");
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
  @SuppressWarnings("unchecked")
  public void testProvider_dylib() throws Exception {
    scratch.file("examples/rule/BUILD");
    scratch.file(
        "examples/rule/apple_rules.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "def _test_rule_impl(ctx):",
        "   dep = ctx.attr.deps[0]",
        "   provider = dep[apple_common.AppleDylibBinary]",
        "   return MyInfo(",
        "      binary = provider.binary,",
        "      objc = provider.objc,",
        "      dep_dir = dir(dep),",
        "   )",
        "test_rule = rule(implementation = _test_rule_impl,",
        "   attrs = {",
        "   'deps': attr.label_list(allow_files = False, mandatory = False,)",
        "})");

    scratch.file(
        "examples/apple_starlark/BUILD",
        "package(default_visibility = ['//visibility:public'])",
        "load('//examples/rule:apple_rules.bzl', 'test_rule')",
        "apple_binary(",
        "    name = 'bin',",
        "    deps = [':lib'],",
        "    binary_type = '" + BinaryType.DYLIB + "',",
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

    List<String> depProviders = (List<String>) myInfo.getValue("dep_dir");
    assertThat(depProviders).doesNotContain("AppleExecutableBinary");
    assertThat(depProviders).doesNotContain("AppleLoadableBundleBinary");
  }

  @Test
  @SuppressWarnings("unchecked")
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
        "      objc = provider.objc,",
        "      dep_dir = dir(dep),",
        "   )",
        "test_rule = rule(implementation = _test_rule_impl,",
        "   attrs = {",
        "   'deps': attr.label_list(allow_files = False, mandatory = False,)",
        "})");

    scratch.file(
        "examples/apple_starlark/BUILD",
        "package(default_visibility = ['//visibility:public'])",
        "load('//examples/rule:apple_rules.bzl', 'test_rule')",
        "apple_binary(",
        "    name = 'bin',",
        "    deps = [':lib'],",
        "    binary_type = '" + BinaryType.EXECUTABLE + "',",
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

    List<String> depProviders = (List<String>) myInfo.getValue("dep_dir");
    assertThat(depProviders).doesNotContain("AppleDylibBinary");
    assertThat(depProviders).doesNotContain("AppleLoadableBundleBinary");
  }

  @Test
  @SuppressWarnings("unchecked")
  public void testProvider_loadableBundle() throws Exception {
    scratch.file("examples/rule/BUILD");
    scratch.file(
        "examples/rule/apple_rules.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "def _test_rule_impl(ctx):",
        "   dep = ctx.attr.deps[0]",
        "   provider = dep[apple_common.AppleLoadableBundleBinary]",
        "   return MyInfo(",
        "      binary = provider.binary,",
        "      dep_dir = dir(dep),",
        "   )",
        "test_rule = rule(implementation = _test_rule_impl,",
        "   attrs = {",
        "   'deps': attr.label_list(allow_files = False, mandatory = False,)",
        "})");

    scratch.file(
        "examples/apple_starlark/BUILD",
        "package(default_visibility = ['//visibility:public'])",
        "load('//examples/rule:apple_rules.bzl', 'test_rule')",
        "apple_binary(",
        "    name = 'bin',",
        "    deps = ['lib'],",
        "    binary_type = '" + BinaryType.LOADABLE_BUNDLE + "',",
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

    assertThat((Artifact) myInfo.getValue("binary")).isNotNull();

    List<String> depProviders = (List<String>) myInfo.getValue("dep_dir");
    assertThat(depProviders).doesNotContain("AppleExecutableBinary");
    assertThat(depProviders).doesNotContain("AppleDylibBinary");
  }

  @Test
  public void testDuplicateLinkopts() throws Exception {
    getRuleType().scratchTarget(scratch, "linkopts", "['-foo', 'bar', '-foo', 'baz']");

    CommandAction linkAction = linkAction("//x:x");
    String linkArgs = Joiner.on(" ").join(linkAction.getArguments());
    assertThat(linkArgs).contains("-Wl,-foo -Wl,bar");
    assertThat(linkArgs).contains("-Wl,-foo -Wl,baz");
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
    useConfiguration(
        "--ios_multi_cpus=x86_64",
        "--ios_sdk_version=10.0",
        "--ios_minimum_os=8.0");
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
        "--watchos_cpus=i386",
        "--watchos_sdk_version=3.0",
        "--watchos_minimum_os=2.0");
    getRuleType().scratchTarget(scratch, "platform_type", "'watchos'");

    Action lipoAction = actionProducingArtifact("//x:x", "_lipobin");
    Artifact binArtifact = getFirstArtifactEndingWith(lipoAction.getInputs(), "x/x_bin");
    CommandAction linkAction = (CommandAction) getGeneratingAction(binArtifact);

    assertThat(linkAction.getArguments())
        .containsAtLeastElementsIn(IMPLICIT_NON_MAC_FRAMEWORK_FLAGS);
  }

  @Test
  public void testLinksImplicitFrameworksWithCrosstoolTvos() throws Exception {
    useConfiguration(
        "--tvos_cpus=x86_64",
        "--tvos_sdk_version=10.1",
        "--tvos_minimum_os=10.0");
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
        "--macos_cpus=x86_64",
        "--macos_sdk_version=10.11",
        "--macos_minimum_os=10.11");
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
        "--macos_cpus=x86_64",
        "--macos_sdk_version=10.11",
        "--macos_minimum_os=10.11");
    getRuleType().scratchTarget(
        scratch, "platform_type", "'macos'", "features", "['link_cocoa']");

    Action lipoAction = actionProducingArtifact("//x:x", "_lipobin");
    Artifact binArtifact = getFirstArtifactEndingWith(lipoAction.getInputs(), "x/x_bin");
    CommandAction linkAction = (CommandAction) getGeneratingAction(binArtifact);

    assertThat(linkAction.getArguments()).containsAtLeastElementsIn(IMPLICIT_MAC_FRAMEWORK_FLAGS);
    assertThat(linkAction.getArguments()).containsAtLeastElementsIn(COCOA_FEATURE_FLAGS);
    assertThat(linkAction.getArguments()).doesNotContain(UIKIT_FRAMEWORK_FLAG);
  }

  @Test
  public void testAliasedLinkoptsThroughObjcLibrary() throws Exception {
    checkAliasedLinkoptsThroughObjcLibrary(getRuleType());
  }

  @Test
  public void testObjcProviderLinkInputsInLinkAction() throws Exception {
    checkObjcProviderLinkInputsInLinkAction(getRuleType());
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
  public void testAvoidDepsThroughDylib() throws Exception {
    checkAvoidDepsThroughDylib(getRuleType());
  }

  @Test
  public void testAvoidDepsObjects_avoidViaCcLibrary() throws Exception {
    checkAvoidDepsObjects_avoidViaCcLibrary(getRuleType());
  }

  @Test
  public void testBundleLoaderIsCorrectlyPassedToTheLinker() throws Exception {
    checkBundleLoaderIsCorrectlyPassedToTheLinker(getRuleType());
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

  @Test
  public void testWatchSimulatorDepCompile() throws Exception {
    checkWatchSimulatorDepCompile(getRuleType());
  }

  @Test
  public void testDylibBinaryType() throws Exception {
    getRuleType().scratchTarget(scratch, "binary_type", "'dylib'");

    CommandAction linkAction = linkAction("//x:x");
    assertThat(Joiner.on(" ").join(linkAction.getArguments())).contains("-dynamiclib");
  }

  @Test
  public void testBinaryTypeIsCorrectlySetToBundle() throws Exception {
    getRuleType().scratchTarget(scratch, "binary_type", "'loadable_bundle'");

    CommandAction linkAction = linkAction("//x:x");
    assertThat(Joiner.on(" ").join(linkAction.getArguments())).contains("-bundle");
  }

  @Test
  public void testMultiarchCcDep() throws Exception {
    checkMultiarchCcDep(getRuleType());
  }

  @Test
  public void testWatchSimulatorLipoAction() throws Exception {
    checkWatchSimulatorLipoAction(getRuleType());
  }

  @Test
  public void testFrameworkDepLinkFlagsPostCleanup() throws Exception {
    checkFrameworkDepLinkFlags(getRuleType(), new ExtraLinkArgs());
  }

  @Test
  public void testDylibDependenciesPostCleanup() throws Exception {
    checkDylibDependencies(getRuleType(), new ExtraLinkArgs());
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
  public void testMinimumOs_invalid_nonVersion() throws Exception {
    checkMinimumOs_invalid_nonVersion(getRuleType());
  }

  @Test
  public void testMinimumOs_invalid_containsAlphabetic() throws Exception {
    checkMinimumOs_invalid_containsAlphabetic(getRuleType());
  }

  @Test
  public void testMinimumOs_invalid_tooManyComponents() throws Exception {
    checkMinimumOs_invalid_tooManyComponents(getRuleType());
  }

  @Test
  public void testGenfilesProtoGetsCorrectPath() throws Exception {
    MockObjcSupport.setupObjcProtoLibrary(scratch);
    scratch.file("x/filter.pbascii");
    scratch.file(
        "examples/BUILD",
        TestConstants.LOAD_PROTO_LIBRARY,
        "load('//objc_proto_library:objc_proto_library.bzl', 'objc_proto_library')",
        "package(default_visibility = ['//visibility:public'])",
        "apple_binary(",
        "    name = 'bin',",
        "    deps = [':objc_protos'],",
        "    platform_type = 'ios',",
        ")",
        "objc_proto_library(",
        "    name = 'objc_protos',",
        "    portable_proto_filters = ['filter.pbascii'],",
        "    deps = [':protos'],",
        ")",
        "proto_library(",
        "    name = 'protos',",
        "    srcs = ['genfile.proto'],",
        ")",
        "genrule(",
        "    name = 'copy_proto',",
        "    srcs = ['original.proto'],",
        "    outs = ['genfile.proto'],",
        "    cmd = '/bin/cp $< $@',",
        ")");

    useConfiguration("--ios_multi_cpus=armv7,arm64");

    Action lipoAction = actionProducingArtifact("//examples:bin", "_lipobin");
    ArrayList<String> genfileRoots = new ArrayList<>();

    for (Artifact archBinary : lipoAction.getInputs().toList()) {
      if (archBinary.getExecPathString().endsWith("bin_bin")) {
        Artifact protoLib =
            getFirstArtifactEndingWith(
                getGeneratingAction(archBinary).getInputs(), "BundledProtos.a");
        Artifact protoObject =
            getFirstArtifactEndingWith(
                getGeneratingAction(protoLib).getInputs(), "Genfile.pbobjc.o");
        Artifact protoObjcSource =
            getFirstArtifactEndingWith(
                getGeneratingAction(protoObject).getInputs(), "Genfile.pbobjc.m");
        Artifact protoSource =
            getFirstArtifactEndingWith(
                getGeneratingAction(protoObjcSource).getInputs(), "genfile.proto");
        genfileRoots.add(protoSource.getRoot().getExecPathString());
      }
    }

    // Make sure there are genrules for both arm64 and armv7 configurations.
    Collections.sort(genfileRoots);
    assertThat(genfileRoots).hasSize(2);
    assertThat(genfileRoots.get(0)).contains("arm64");
    assertThat(genfileRoots.get(1)).contains("armv7");
  }

  @Test
  public void testDifferingProtoDepsPerArchitecture() throws Exception {
    MockObjcSupport.setupObjcProtoLibrary(scratch);
    scratch.file("x/filter.pbascii");
    scratch.file(
        "examples/BUILD",
        TestConstants.LOAD_PROTO_LIBRARY,
        "load('//objc_proto_library:objc_proto_library.bzl', 'objc_proto_library')",
        "package(default_visibility = ['//visibility:public'])",
        "apple_binary(",
        "    name = 'bin',",
        "    deps = [':objc_protos'],",
        "    platform_type = 'ios',",
        ")",
        "objc_proto_library(",
        "    name = 'objc_protos',",
        "    portable_proto_filters = ['filter.pbascii'],",
        "    deps = [':protos'],",
        ")",
        "proto_library(",
        "    name = 'protos',",
        "    srcs = select({",
        "        ':armv7': [ 'one.proto', ],",
        "        '//conditions:default': [ 'two.proto', ],",
        "    }),",
        ")",
        "config_setting(",
        "    name = 'armv7',",
        "    values = {'apple_split_cpu': 'armv7'},",
        ")");

    useConfiguration("--ios_multi_cpus=armv7,arm64");

    Action lipoAction = actionProducingArtifact("//examples:bin", "_lipobin");

    Artifact armv7Binary = getSingleArchBinary(lipoAction, "armv7");
    Artifact arm64Binary = getSingleArchBinary(lipoAction, "arm64");

    Artifact armv7ProtoLib =
        getFirstArtifactEndingWith(getGeneratingAction(armv7Binary).getInputs(), "BundledProtos.a");
    Artifact armv7ProtoObject =
        getFirstArtifactEndingWith(
            getGeneratingAction(armv7ProtoLib).getInputs(), "One.pbobjc.o");
    Artifact armv7ProtoObjcSource =
        getFirstArtifactEndingWith(
            getGeneratingAction(armv7ProtoObject).getInputs(), "One.pbobjc.m");
    assertThat(getFirstArtifactEndingWith(
        getGeneratingAction(armv7ProtoObjcSource).getInputs(), "one.proto")).isNotNull();

    Artifact arm64ProtoLib =
        getFirstArtifactEndingWith(getGeneratingAction(arm64Binary).getInputs(), "BundledProtos.a");
    Artifact arm64ProtoObject =
        getFirstArtifactEndingWith(
            getGeneratingAction(arm64ProtoLib).getInputs(), "Two.pbobjc.o");
    Artifact arm64ProtoObjcSource =
        getFirstArtifactEndingWith(
            getGeneratingAction(arm64ProtoObject).getInputs(), "Two.pbobjc.m");
    assertThat(getFirstArtifactEndingWith(
        getGeneratingAction(arm64ProtoObjcSource).getInputs(), "two.proto")).isNotNull();
  }

  @Test
  public void testPlatformTypeIsConfigurable() throws Exception {
    scratch.file(
        "examples/BUILD",
        "package(default_visibility = ['//visibility:public'])",
        "apple_binary(",
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

    useConfiguration("--define=use_watch=1",
        "--ios_multi_cpus=armv7,arm64",
        "--watchos_cpus=armv7k");

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
        "package(default_visibility = ['//visibility:public'])",
        "load('//examples/rule:apple_rules.bzl', 'test_rule')",
        "apple_binary(",
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

  private void checkAppleDebugSymbolProvider_DsymEntries(
      Map<String, Dict<String, Artifact>> outputMap, CompilationMode compilationMode)
      throws Exception {
    assertThat(outputMap).containsKey("arm64");
    assertThat(outputMap).containsKey("armv7");

    Map<String, Artifact> arm64 = outputMap.get("arm64");
    assertThat(arm64).containsEntry("bitcode_symbols", bitcodeSymbol("arm64", compilationMode));
    String expectedArm64Path = dsymBinaryPath("arm64", compilationMode);
    assertThat(arm64.get("dsym_binary").getExecPathString()).isEqualTo(expectedArm64Path);

    Map<String, Artifact> armv7 = outputMap.get("armv7");
    assertThat(armv7).containsEntry("bitcode_symbols", bitcodeSymbol("armv7", compilationMode));
    String expectedArmv7Path = dsymBinaryPath("armv7", compilationMode);
    assertThat(armv7.get("dsym_binary").getExecPathString()).isEqualTo(expectedArmv7Path);

    Map<String, Artifact> x8664 = outputMap.get("x86_64");
    // Simulator build has bitcode disabled.
    assertThat(x8664).doesNotContainKey("bitcode_symbols");
    String expectedx8664Path = dsymBinaryPath("x86_64", compilationMode);
    assertThat(x8664.get("dsym_binary").getExecPathString()).isEqualTo(expectedx8664Path);
  }

  private void checkAppleDebugSymbolProvider_LinkMapEntries(
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
    useConfiguration(
        "--apple_bitcode=embedded", "--apple_generate_dsym", "--ios_multi_cpus=armv7,arm64,x86_64");
    checkAppleDebugSymbolProvider_DsymEntries(
        generateAppleDebugOutputsStarlarkProviderMap(), CompilationMode.FASTBUILD);
  }

  @Test
  public void testAppleDebugSymbolProviderWithAutoDsymDbgAndDsymsExposedToStarlark()
      throws Exception {
    useConfiguration(
        "--apple_bitcode=embedded",
        "--compilation_mode=dbg",
        "--apple_enable_auto_dsym_dbg",
        "--ios_multi_cpus=armv7,arm64,x86_64");
    checkAppleDebugSymbolProvider_DsymEntries(
        generateAppleDebugOutputsStarlarkProviderMap(), CompilationMode.DBG);
  }

  @Test
  public void testAppleDebugSymbolProviderWithLinkMapsExposedToStarlark() throws Exception {
    useConfiguration(
        "--apple_bitcode=embedded",
        "--objc_generate_linkmap",
        "--ios_multi_cpus=armv7,arm64,x86_64");
    checkAppleDebugSymbolProvider_LinkMapEntries(generateAppleDebugOutputsStarlarkProviderMap());
  }

  @Test
  public void testAppleDebugSymbolProviderWithDsymsAndLinkMapsExposedToStarlark() throws Exception {
    useConfiguration(
        "--apple_bitcode=embedded",
        "--objc_generate_linkmap",
        "--apple_generate_dsym",
        "--ios_multi_cpus=armv7,arm64,x86_64");

    Dict<String, Dict<String, Artifact>> outputMap = generateAppleDebugOutputsStarlarkProviderMap();
    checkAppleDebugSymbolProvider_DsymEntries(outputMap, CompilationMode.FASTBUILD);
    checkAppleDebugSymbolProvider_LinkMapEntries(outputMap);
  }

  @Test
  public void testInstrumentedFilesProviderContainsDepsAndBundleLoaderFiles() throws Exception {
    useConfiguration("--collect_code_coverage");
    scratch.file(
        "examples/BUILD",
        "package(default_visibility = ['//visibility:public'])",
        "apple_binary(",
        "    name = 'bin',",
        "    deps = [':lib'],",
        "    platform_type = 'ios',",
        ")",
        "apple_binary(",
        "    name = 'bundle',",
        "    deps = [':bundle_lib'],",
        "    binary_type = '" + BinaryType.LOADABLE_BUNDLE + "',",
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
    getRuleType().scratchTarget(scratch,
        "platform_type", "'watchos'");
    useConfiguration(
        "--watchos_cpus=i386", "--watchos_sdk_version=3.0", "--watchos_minimum_os=2.0");
    checkLinkMinimumOSVersion("-mwatchos-simulator-version-min=2.0");
  }

  @Test
  public void testLinkActionHasCorrectWatchosMinVersion() throws Exception {
    getRuleType().scratchTarget(scratch,
        "platform_type", "'watchos'");
    useConfiguration(
        "--watchos_cpus=armv7k", "--watchos_sdk_version=3.0", "--watchos_minimum_os=2.0");
    checkLinkMinimumOSVersion("-mwatchos-version-min=2.0");
  }

  @Test
  public void testLinkActionHasCorrectTvosSimulatorMinVersion() throws Exception {
    getRuleType().scratchTarget(scratch,
        "platform_type", "'tvos'");
    useConfiguration(
        "--tvos_cpus=x86_64", "--tvos_sdk_version=10.1", "--tvos_minimum_os=10.0");
    checkLinkMinimumOSVersion("-mtvos-simulator-version-min=10.0");
  }

  @Test
  public void testLinkActionHasCorrectTvosMinVersion() throws Exception {
    getRuleType().scratchTarget(scratch,
        "platform_type", "'tvos'");
    useConfiguration(
        "--tvos_cpus=arm64", "--tvos_sdk_version=10.1", "--tvos_minimum_os=10.0");
    checkLinkMinimumOSVersion("-mtvos-version-min=10.0");
  }

  @Test
  public void testWatchSimulatorLinkAction() throws Exception {
    checkWatchSimulatorLinkAction(getRuleType());
  }

  @Test
  public void testProtoBundlingAndLinking() throws Exception {
    checkProtoBundlingAndLinking(getRuleType());
  }

  @Test
  public void testAvoidDepsObjects() throws Exception {
    checkAvoidDepsObjects(getRuleType());
  }

  @Test
  public void testBundleLoaderPropagatesAppleExecutableBinaryProvider() throws Exception {
    scratch.file(
        "bin/BUILD",
        "apple_binary(",
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
        "apple_binary(",
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
        "apple_binary(",
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
        "apple_binary(",
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
        "apple_binary(",
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
  public void testCustomModuleMap() throws Exception {
    checkCustomModuleMap(getRuleType());
  }

  @Test
  public void testMinimumOsDifferentTargets() throws Exception {
    checkMinimumOsDifferentTargets(getRuleType(), "_lipobin", "_bin");
  }

  @Test
  public void testDrops32BitArchitecture() throws Exception {
    verifyDrops32BitArchitecture(getRuleType());
  }

  @Test
  public void testFeatureFlags_offByDefault() throws Exception {
    useConfiguration("--enforce_transitive_configs_for_config_feature_flag");
    scratchFeatureFlagTestLib();
    scratch.file(
        "test/BUILD",
        "apple_binary(",
        "    name = 'bin',",
        "    deps = ['//lib:objcLib'],",
        "    platform_type = 'ios',",
        "    transitive_configs = ['//lib:flag1', '//lib:flag2'],",
        ")");

    CommandAction linkAction = linkAction("//test:bin");
    CommandAction objcLibArchiveAction = (CommandAction) getGeneratingAction(
        getFirstArtifactEndingWith(linkAction.getInputs(), "libobjcLib.a"));

    CommandAction flag1offCompileAction = (CommandAction) getGeneratingAction(
        getFirstArtifactEndingWith(objcLibArchiveAction.getInputs(), "flag1off.o"));
    CommandAction flag2offCompileAction = (CommandAction) getGeneratingAction(
        getFirstArtifactEndingWith(objcLibArchiveAction.getInputs(), "flag2off.o"));

    String compileArgs1 = Joiner.on(" ").join(flag1offCompileAction.getArguments());
    String compileArgs2 = Joiner.on(" ").join(flag2offCompileAction.getArguments());
    assertThat(compileArgs1).contains("FLAG_1_OFF");
    assertThat(compileArgs1).contains("FLAG_2_OFF");
    assertThat(compileArgs2).contains("FLAG_1_OFF");
    assertThat(compileArgs2).contains("FLAG_2_OFF");
  }

  @Test
  public void testFeatureFlags_oneFlagOn() throws Exception {
    useConfiguration("--enforce_transitive_configs_for_config_feature_flag");
    scratchFeatureFlagTestLib();
    scratch.file(
        "test/BUILD",
        "apple_binary(",
        "    name = 'bin',",
        "    deps = ['//lib:objcLib'],",
        "    platform_type = 'ios',",
        "    feature_flags = {",
        "      '//lib:flag2': 'on',",
        "    },",
        "    transitive_configs = ['//lib:flag1', '//lib:flag2'],",
        ")");

    CommandAction linkAction = linkAction("//test:bin");
    CommandAction objcLibArchiveAction = (CommandAction) getGeneratingAction(
        getFirstArtifactEndingWith(linkAction.getInputs(), "libobjcLib.a"));

    CommandAction flag1offCompileAction = (CommandAction) getGeneratingAction(
        getFirstArtifactEndingWith(objcLibArchiveAction.getInputs(), "flag1off.o"));
    CommandAction flag2onCompileAction = (CommandAction) getGeneratingAction(
        getFirstArtifactEndingWith(objcLibArchiveAction.getInputs(), "flag2on.o"));

    String compileArgs1 = Joiner.on(" ").join(flag1offCompileAction.getArguments());
    String compileArgs2 = Joiner.on(" ").join(flag2onCompileAction.getArguments());
    assertThat(compileArgs1).contains("FLAG_1_OFF");
    assertThat(compileArgs1).contains("FLAG_2_ON");
    assertThat(compileArgs2).contains("FLAG_1_OFF");
    assertThat(compileArgs2).contains("FLAG_2_ON");
  }

  @Test
  public void testFeatureFlags_allFlagsOn() throws Exception {
    useConfiguration("--enforce_transitive_configs_for_config_feature_flag");
    scratchFeatureFlagTestLib();
    scratch.file(
        "test/BUILD",
        "apple_binary(",
        "    name = 'bin',",
        "    deps = ['//lib:objcLib'],",
        "    platform_type = 'ios',",
        "    feature_flags = {",
        "      '//lib:flag1': 'on',",
        "      '//lib:flag2': 'on',",
        "    },",
        "    transitive_configs = ['//lib:flag1', '//lib:flag2'],",
        ")");

    CommandAction linkAction = linkAction("//test:bin");
    CommandAction objcLibArchiveAction = (CommandAction) getGeneratingAction(
        getFirstArtifactEndingWith(linkAction.getInputs(), "libobjcLib.a"));

    CommandAction flag1onCompileAction = (CommandAction) getGeneratingAction(
        getFirstArtifactEndingWith(objcLibArchiveAction.getInputs(), "flag1on.o"));
    CommandAction flag2onCompileAction = (CommandAction) getGeneratingAction(
        getFirstArtifactEndingWith(objcLibArchiveAction.getInputs(), "flag2on.o"));

    String compileArgs1 = Joiner.on(" ").join(flag1onCompileAction.getArguments());
    String compileArgs2 = Joiner.on(" ").join(flag2onCompileAction.getArguments());
    assertThat(compileArgs1).contains("FLAG_1_ON");
    assertThat(compileArgs1).contains("FLAG_2_ON");
    assertThat(compileArgs2).contains("FLAG_1_ON");
    assertThat(compileArgs2).contains("FLAG_2_ON");
  }

  @Test
  public void testLoadableBundleObjcProvider() throws Exception {
    scratch.file(
        "testlib/BUILD",
        "objc_library(",
        "    name = 'lib',",
        "    srcs = ['a.m'],",
        "    sdk_frameworks = ['TestFramework'],",
        ")");

    getRuleType().scratchTarget(scratch,
        "binary_type", "'loadable_bundle'",
        "deps", "['//testlib:lib']");

    ObjcProvider objcProvider = providerForTarget("//x:x");
    assertThat(objcProvider.sdkFramework().toList()).contains("TestFramework");
  }

  @Test
  public void testIncludesLinkstampFiles() throws Exception {
    scratch.file(
        "test/BUILD",
        "apple_binary(",
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
  public void testProcessHeadersInDependencies() throws Exception {
    MockObjcSupport.setupCcToolchainConfig(
        mockToolsConfig, MockObjcSupport.darwinX86_64().withFeatures(CppRuleClasses.PARSE_HEADERS));
    useConfiguration("--features=parse_headers", "--process_headers_in_dependencies");
    ConfiguredTarget x =
        scratchConfiguredTarget(
            "foo",
            "x",
            "apple_binary(name = 'x', platform_type = 'macos', deps = [':y', ':z'])",
            "cc_library(name = 'y', hdrs = ['y.h'])",
            "objc_library(name = 'z', hdrs = ['z.h'])");
    String validation = ActionsTestUtil.baseNamesOf(getOutputGroup(x, OutputGroupInfo.VALIDATION));
    assertThat(validation).contains("y.h.processed");
    assertThat(validation).contains("z.h.processed");
  }

  protected RuleType getRuleType() {
    return RULE_TYPE;
  }
}
