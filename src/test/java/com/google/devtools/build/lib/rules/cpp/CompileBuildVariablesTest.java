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

package com.google.devtools.build.lib.rules.cpp;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.PathMapper;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.packages.util.Crosstool.CcToolchainConfig;
import com.google.devtools.build.lib.packages.util.MockToolsConfig;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests that {@code CppCompileAction} is populated with the correct build variables. */
@RunWith(JUnit4.class)
public class CompileBuildVariablesTest extends BuildViewTestCase {
  /** Name of the build variable for the sysroot path variable name. */
  public static final String SYSROOT_VARIABLE_NAME = "sysroot";

  /** Name of the build variable for the minimum_os_version being targeted. */
  public static final String MINIMUM_OS_VERSION_VARIABLE_NAME = "minimum_os_version";

  private CppCompileAction getCppCompileAction(ConfiguredTarget target, final String name)
      throws Exception {
    return (CppCompileAction)
        getGeneratingAction(
            Iterables.find(
                getGeneratingAction(getFilesToBuild(target).getSingleton()).getInputs().toList(),
                (artifact) -> artifact.getExecPath().getBaseName().startsWith(name)));
  }

  /** Returns active build variables for a compile action of given type for given target. */
  protected CcToolchainVariables getCompileBuildVariables(String label, String name)
      throws Exception {
    return getCppCompileAction(getConfiguredTarget(label), name)
        .getCompileCommandLine()
        .getVariables();
  }

  @Test
  public void testPresenceOfIsUsingFissionVariable() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder().withFeatures(CppRuleClasses.PER_OBJECT_DEBUG_INFO));
    useConfiguration("--fission=yes");

    scratch.file(
        "x/BUILD",
        "load('@rules_cc//cc:cc_binary.bzl', 'cc_binary')",
        "cc_binary(name = 'bin', srcs = ['bin.cc'])");
    scratch.file("x/bin.cc");

    CcToolchainVariables variables = getCompileBuildVariables("//x:bin", "bin");

    assertThat(
            variables.getStringVariable(
                CompileBuildVariables.IS_USING_FISSION.getVariableName(), PathMapper.NOOP))
        .isNotNull();
  }

  @Test
  public void testPresenceOfIsUsingFissionAndPerDebugObjectFileVariablesWithThinlto()
      throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder()
                .withFeatures(
                    "fission_flags_for_lto_backend",
                    CppRuleClasses.PER_OBJECT_DEBUG_INFO,
                    CppRuleClasses.SUPPORTS_START_END_LIB,
                    CppRuleClasses.THIN_LTO));
    useConfiguration("--fission=yes", "--features=thin_lto");

    scratch.file(
        "x/BUILD",
        "load('@rules_cc//cc:cc_binary.bzl', 'cc_binary')",
        "cc_binary(name = 'bin', srcs = ['bin.cc'])");
    scratch.file("x/bin.cc");

    RuleConfiguredTarget target = (RuleConfiguredTarget) getConfiguredTarget("//x:bin");
    LtoBackendAction backendAction =
        (LtoBackendAction)
            target.getActions().stream()
                .filter(a -> a.getMnemonic().equals("CcLtoBackendCompile"))
                .findFirst()
                .get();
    CppCompileAction bitcodeAction =
        (CppCompileAction)
            target.getActions().stream()
                .filter(a -> a.getMnemonic().equals("CppCompile"))
                .findFirst()
                .get();

    // We don't pass per_object_debug_info_file to bitcode compiles
    assertThat(
            bitcodeAction
                .getCompileCommandLine()
                .getVariables()
                .isAvailable(CompileBuildVariables.IS_USING_FISSION.getVariableName()))
        .isTrue();
    assertThat(
            bitcodeAction
                .getCompileCommandLine()
                .getVariables()
                .isAvailable(CompileBuildVariables.PER_OBJECT_DEBUG_INFO_FILE.getVariableName()))
        .isFalse();

    // We do pass per_object_debug_info_file to backend compiles
    assertThat(backendAction.getArguments()).contains("-<PER_OBJECT_DEBUG_INFO_FILE>");
    assertThat(backendAction.getArguments()).contains("-<IS_USING_FISSION>");
  }

  @Test
  public void testPresenceOfPerObjectDebugFileBuildVariableUsingLegacyFields() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder().withFeatures(CppRuleClasses.PER_OBJECT_DEBUG_INFO));
    useConfiguration("--fission=yes");

    scratch.file(
        "x/BUILD",
        "load('@rules_cc//cc:cc_binary.bzl', 'cc_binary')",
        "cc_binary(name = 'bin', srcs = ['bin.cc'])");
    scratch.file("x/bin.cc");

    CcToolchainVariables variables = getCompileBuildVariables("//x:bin", "bin");

    assertThat(
            variables.getStringVariable(
                CompileBuildVariables.PER_OBJECT_DEBUG_INFO_FILE.getVariableName(),
                PathMapper.NOOP))
        .isNotNull();
  }

  @Test
  public void testPresenceOfMinOsVersionBuildVariable() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig, CcToolchainConfig.builder().withFeatures("min_os_version_flag"));
    useConfiguration("--minimum_os_version=6");
    scratch.file(
        "x/BUILD",
        "load('@rules_cc//cc:cc_binary.bzl', 'cc_binary')",
        "cc_binary(name = 'bin', srcs = ['bin.cc'])");
    scratch.file("x/bin.cc");

    CcToolchainVariables variables = getCompileBuildVariables("//x:bin", "bin");
    assertThat(variables.getStringVariable(MINIMUM_OS_VERSION_VARIABLE_NAME, PathMapper.NOOP))
        .isEqualTo("6");
  }

  @Test
  public void testExternalIncludePathsVariable() throws Exception {
    if (!analysisMock.isThisBazel()) {
      return;
    }
    AnalysisMock.get()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder().withFeatures(CppRuleClasses.EXTERNAL_INCLUDE_PATHS));
    useConfiguration(
        "--features=external_include_paths",
        "--platforms=" + TestConstants.PLATFORM_LABEL,
        "--experimental_platform_in_output_dir",
        String.format(
            "--experimental_override_name_platform_in_output_dir=%s=k8",
            TestConstants.PLATFORM_LABEL));
    scratch.appendFile(
        "MODULE.bazel",
        "bazel_dep(name = 'pkg')",
        "local_path_override(module_name = 'pkg', path = '/foo')");
    getSkyframeExecutor()
        .invalidateFilesUnderPathForTesting(
            reporter,
            new ModifiedFileSet.Builder().modify(PathFragment.create("MODULE.bazel")).build(),
            Root.fromPath(rootDirectory));

    scratch.file("/foo/MODULE.bazel", "module(name = 'pkg')");
    AnalysisMock.get().ccSupport().setup(new MockToolsConfig(scratch.resolve("/foo")));
    scratch.file(
        "/foo/third_party/BUILD",
        """
        load("@rules_cc//cc:cc_library.bzl", "cc_library")
        cc_library(
            name = "foo",
            hdrs = ["foo.hpp"],
        )

        cc_library(
            name = "foo2",
            hdrs = ["foo.hpp"],
            include_prefix = "prf",
        )
        """);
    scratch.file(
        "x/BUILD",
        """
        load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
        load("@rules_cc//cc:cc_library.bzl", "cc_library")
        cc_library(
            name = "bar",
            hdrs = ["bar.hpp"],
        )

        cc_binary(
            name = "bin",
            srcs = ["bin.cc"],
            deps = [
                "bar",
                "@pkg//third_party:foo",
                "@pkg//third_party:foo2",
            ],
        )
        """);

    CcToolchainVariables variables = getCompileBuildVariables("//x:bin", "bin");

    ImmutableList.Builder<String> entries =
        ImmutableList.<String>builder()
            .add(
                "/k8-fastbuild/bin/external/pkg+/third_party/_virtual_includes/foo2",
                "external/pkg+",
                "/k8-fastbuild/bin/external/pkg+");
    if (analysisMock.isThisBazel()) {
      entries.add("external/bazel_tools", "/k8-fastbuild/bin/external/bazel_tools");
    }

    assertThat(
            CcToolchainVariables.toStringList(
                    variables,
                    CompileBuildVariables.EXTERNAL_INCLUDE_PATHS.getVariableName(),
                    PathMapper.NOOP)
                .stream()
                .map(x -> removeOutDirectory(x))
                .collect(ImmutableList.toImmutableList()))
        .containsExactlyElementsIn(entries.build());
  }

  private String removeOutDirectory(String s) {
    return s.replace("blaze-out", "").replace("bazel-out", "");
  }
}
