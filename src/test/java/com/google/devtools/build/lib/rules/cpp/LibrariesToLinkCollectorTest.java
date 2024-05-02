// Copyright 2022 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.rules.cpp.LibrariesToLinkCollector.getRelative;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.packages.util.Crosstool.CcToolchainConfig;
import com.google.devtools.build.lib.packages.util.ResourceLoader;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class LibrariesToLinkCollectorTest extends BuildViewTestCase {

  @Test
  public void getRalitive_returnsRelativePaths() {
    assertThat(getRelative(PathFragment.create("foo"), PathFragment.create("foo/bar/baz")))
        .isEqualTo(PathFragment.create("bar/baz"));
    assertThat(getRelative(PathFragment.create("foo/bar"), PathFragment.create("foo/bar/baz")))
        .isEqualTo(PathFragment.create("baz"));
    assertThat(getRelative(PathFragment.create(""), PathFragment.create("foo")))
        .isEqualTo(PathFragment.create("foo"));
    assertThat(getRelative(PathFragment.create(""), PathFragment.create("foo/bar")))
        .isEqualTo(PathFragment.create("foo/bar"));
    assertThat(
            getRelative(PathFragment.create("foo/bar"), PathFragment.create("foo/bar"))
                .getPathString())
        .isEmpty();

    assertThat(getRelative(PathFragment.create("foo/bar/baz"), PathFragment.create("foo")))
        .isEqualTo(PathFragment.create("../.."));
    assertThat(getRelative(PathFragment.create("foo/bar/baz"), PathFragment.create("foo/bar")))
        .isEqualTo(PathFragment.create(".."));
    assertThat(getRelative(PathFragment.create("foo"), PathFragment.create("")))
        .isEqualTo(PathFragment.create(".."));
    assertThat(getRelative(PathFragment.create("foo/bar"), PathFragment.create("")))
        .isEqualTo(PathFragment.create("../.."));
    assertThat(getRelative(PathFragment.create("foo/baz"), PathFragment.create("foo/bar")))
        .isEqualTo(PathFragment.create("../bar"));
    assertThat(getRelative(PathFragment.create("bar"), PathFragment.create("foo")))
        .isEqualTo(PathFragment.create("../foo"));

    assertThat(getRelative(PathFragment.create("foo"), PathFragment.create("../foo")))
        .isEqualTo(PathFragment.create("../../foo"));
    assertThat(getRelative(PathFragment.create(".."), PathFragment.create("../../foo")))
        .isEqualTo(PathFragment.create("../foo"));
    assertThat(getRelative(PathFragment.create("../bar"), PathFragment.create("../../foo")))
        .isEqualTo(PathFragment.create("../../foo"));
  }

  @Test
  public void getRelative_throwsOnInvalidCases() {
    assertThrows(
        IllegalArgumentException.class,
        () -> getRelative(PathFragment.create("/bar"), PathFragment.create("/foo")));
    assertThrows(
        IllegalArgumentException.class,
        () -> getRelative(PathFragment.create(".."), PathFragment.create("")));
    assertThrows(
        IllegalArgumentException.class,
        () -> getRelative(PathFragment.create("../../bar"), PathFragment.create("../foo")));
  }

  /* TODO: Add an integration test (maybe in cc_integration_test.sh) when a modular toolchain config
  is available.*/
  @Test
  public void dynamicLink_siblingLayout_externalBinary_rpath() throws Exception {
    FileSystemUtils.appendIsoLatin1(
        scratch.resolve("WORKSPACE"), "local_repository(name = 'src', path = 'src')");
    invalidatePackages();

    scratch.file("src/WORKSPACE");
    scratch.file(
        "src/test/BUILD",
        """
        cc_binary(
            name = "foo",
            srcs = [
                "some-dir/bar.so",
                "some-other-dir/qux.so",
            ],
        )
        """);
    scratch.file("src/test/some-dir/bar.so");
    scratch.file("src/test/some-other-dir/qux.so");

    analysisMock.ccSupport().setupCcToolchainConfig(mockToolsConfig, CcToolchainConfig.builder());
    mockToolsConfig.create(
        "toolchain/cc_toolchain_config.bzl",
        ResourceLoader.readFromResources(
            "com/google/devtools/build/lib/analysis/mock/cc_toolchain_config.bzl"));
    scratch.file(
        "toolchain/BUILD",
        "load(':cc_toolchain_config.bzl', 'cc_toolchain_config')",
        "filegroup(",
        "   name = 'empty',",
        ")",
        "filegroup(",
        "    name = 'static_runtime',",
        "    srcs = ['librt.a'],",
        ")",
        "filegroup(",
        "    name = 'dynamic_runtime',",
        "    srcs = ['librt.so'],",
        ")",
        "filegroup(",
        "    name = 'files',",
        "    srcs = ['librt.a', 'librt.so'],",
        ")",
        "cc_toolchain(",
        "    name = 'c_toolchain',",
        "    toolchain_identifier = 'toolchain-identifier-k8',",
        "    toolchain_config = ':k8-compiler_config',",
        "    all_files = ':files',",
        "    ar_files = ':empty',",
        "    as_files = ':empty',",
        "    compiler_files = ':empty',",
        "    dwp_files = ':empty',",
        "    linker_files = ':empty',",
        "    strip_files = ':empty',",
        "    objcopy_files = ':empty',",
        "    dynamic_runtime_lib = ':dynamic_runtime',",
        "    static_runtime_lib = ':static_runtime',",
        ")",
        CcToolchainConfig.builder()
            .withFeatures(
                CppRuleClasses.STATIC_LINK_CPP_RUNTIMES,
                CppRuleClasses.SUPPORTS_DYNAMIC_LINKER,
                "runtime_library_search_directories")
            .build()
            .getCcToolchainConfigRule(),
        "toolchain(",
        "  name = 'toolchain',",
        "  toolchain_type = '" + TestConstants.TOOLS_REPOSITORY + "//tools/cpp:toolchain_type',",
        "  toolchain = ':c_toolchain',",
        ")");
    scratch.file("toolchain/librt.a");
    scratch.file("toolchain/librt.so");
    analysisMock.ccSupport().setupCcToolchainConfig(mockToolsConfig, CcToolchainConfig.builder());

    setBuildLanguageOptions("--experimental_sibling_repository_layout");
    useConfiguration(
        "--extra_toolchains=//toolchain:toolchain",
        "--dynamic_mode=fully",
        "--incompatible_enable_cc_toolchain_resolution",
        "--platforms=" + TestConstants.PLATFORM_LABEL,
        "--experimental_platform_in_output_dir",
        String.format(
            "--experimental_override_name_platform_in_output_dir=%s=k8",
            TestConstants.PLATFORM_LABEL));

    ConfiguredTarget target = getConfiguredTarget("@src//test:foo");
    assertThat(target).isNotNull();
    Artifact binary = getExecutable(target);
    SpawnAction linkAction = (SpawnAction) getGeneratingAction(binary);
    assertThat(linkAction).isNotNull();

    String workspace = getTarget("//toolchain:toolchain").getPackage().getWorkspaceName();
    List<String> linkArgs = linkAction.getArguments();
    assertThat(linkArgs)
        .contains(
            "--runtime_library=../../../../k8-fastbuild/bin/_solib__toolchain_Cc_Utoolchain/");
    assertThat(linkArgs)
        .contains(
            "--runtime_library=foo.runfiles/" + workspace + "/_solib__toolchain_Cc_Utoolchain/");
    assertThat(linkArgs)
        .contains("--runtime_library=../../" + workspace + "/_solib__toolchain_Cc_Utoolchain/");
  }

  @Test
  public void dynamicLink_siblingLayout_externalToolchain_rpath() throws Exception {
    FileSystemUtils.appendIsoLatin1(
        scratch.resolve("WORKSPACE"), "local_repository(name = 'toolchain', path = 'toolchain')");
    invalidatePackages();

    scratch.file(
        "src/test/BUILD",
        """
        cc_binary(
            name = "foo",
            srcs = [
                "some-dir/bar.so",
                "some-other-dir/qux.so",
            ],
        )
        """);
    scratch.file("src/test/some-dir/bar.so");
    scratch.file("src/test/some-other-dir/qux.so");

    // The cc_toolchain_config.bzl cannot be placed in the external "toolchain" repo (b/269187186)
    mockToolsConfig.create(
        "cc_toolchain_config.bzl",
        ResourceLoader.readFromResources(
            "com/google/devtools/build/lib/analysis/mock/cc_toolchain_config.bzl"));
    scratch.file("BUILD");

    scratch.file("toolchain/WORKSPACE");
    scratch.file(
        "toolchain/BUILD",
        "load('@//:cc_toolchain_config.bzl', 'cc_toolchain_config')",
        "filegroup(",
        "   name = 'empty',",
        ")",
        "filegroup(",
        "    name = 'static_runtime',",
        "    srcs = ['librt.a'],",
        ")",
        "filegroup(",
        "    name = 'dynamic_runtime',",
        "    srcs = ['librt.so'],",
        ")",
        "filegroup(",
        "    name = 'files',",
        "    srcs = ['librt.a', 'librt.so'],",
        ")",
        "cc_toolchain(",
        "    name = 'c_toolchain',",
        "    toolchain_identifier = 'toolchain-identifier-k8',",
        "    toolchain_config = ':k8-compiler_config',",
        "    all_files = ':files',",
        "    ar_files = ':empty',",
        "    as_files = ':empty',",
        "    compiler_files = ':empty',",
        "    dwp_files = ':empty',",
        "    linker_files = ':empty',",
        "    strip_files = ':empty',",
        "    objcopy_files = ':empty',",
        "    dynamic_runtime_lib = ':dynamic_runtime',",
        "    static_runtime_lib = ':static_runtime',",
        ")",
        CcToolchainConfig.builder()
            .withFeatures(
                CppRuleClasses.STATIC_LINK_CPP_RUNTIMES,
                CppRuleClasses.SUPPORTS_DYNAMIC_LINKER,
                "runtime_library_search_directories")
            .build()
            .getCcToolchainConfigRule(),
        "toolchain(",
        "  name = 'toolchain',",
        "  toolchain_type = '" + TestConstants.TOOLS_REPOSITORY + "//tools/cpp:toolchain_type',",
        "  toolchain = ':c_toolchain',",
        ")");
    scratch.file("toolchain/librt.a");
    scratch.file("toolchain/librt.so");
    analysisMock.ccSupport().setupCcToolchainConfig(mockToolsConfig, CcToolchainConfig.builder());

    setBuildLanguageOptions("--experimental_sibling_repository_layout");
    useConfiguration(
        "--extra_toolchains=@toolchain//:toolchain",
        "--dynamic_mode=fully",
        "--incompatible_enable_cc_toolchain_resolution",
        "--platforms=" + TestConstants.PLATFORM_LABEL,
        "--experimental_platform_in_output_dir",
        String.format(
            "--experimental_override_name_platform_in_output_dir=%s=k8",
            TestConstants.PLATFORM_LABEL));

    ConfiguredTarget target = getConfiguredTarget("//src/test:foo");
    assertThat(target).isNotNull();
    Artifact binary = getExecutable(target);
    SpawnAction linkAction = (SpawnAction) getGeneratingAction(binary);
    assertThat(linkAction).isNotNull();

    List<String> linkArgs = linkAction.getArguments();
    assertThat(linkArgs)
        .contains(
            "--runtime_library=../../../../toolchain/k8-fastbuild/bin/_solib___Cc_Utoolchain/");
    assertThat(linkArgs)
        .contains("--runtime_library=foo.runfiles/toolchain/_solib___Cc_Utoolchain/");
    assertThat(linkArgs).contains("--runtime_library=../../../toolchain/_solib___Cc_Utoolchain/");
  }
}
