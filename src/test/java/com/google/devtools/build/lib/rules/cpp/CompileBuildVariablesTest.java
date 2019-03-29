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

import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.packages.util.Crosstool.CcToolchainConfig;
import com.google.devtools.build.lib.packages.util.MockCcSupport;
import com.google.devtools.build.lib.packages.util.MockPlatformSupport;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests that {@code CppCompileAction} is populated with the correct build variables. */
@RunWith(JUnit4.class)
public class CompileBuildVariablesTest extends BuildViewTestCase {

  private CppCompileAction getCppCompileAction(final String label, final String name) throws
      Exception {
    return (CppCompileAction)
        getGeneratingAction(
            Iterables.find(
                getGeneratingAction(
                    Iterables.getOnlyElement(getFilesToBuild(getConfiguredTarget(label))))
                    .getInputs(),
                new Predicate<Artifact>() {
                  @Override
                  public boolean apply(Artifact artifact) {
                    return artifact.getExecPath().getBaseName().startsWith(name);
                  }
                }));
  }

  /** Returns active build variables for a compile action of given type for given target. */
  protected CcToolchainVariables getCompileBuildVariables(String label, String name)
      throws Exception {
    return getCppCompileAction(label, name).getCompileCommandLine().getVariables();
  }

  @Test
  public void testPresenceOfBasicVariables() throws Exception {
    scratch.file("x/BUILD", "cc_binary(name = 'bin', srcs = ['bin.cc'])");
    scratch.file("x/bin.cc");

    CcToolchainVariables variables = getCompileBuildVariables("//x:bin", "bin");

    assertThat(variables.getStringVariable(CompileBuildVariables.SOURCE_FILE.getVariableName()))
        .contains("x/bin.cc");
    assertThat(variables.getStringVariable(CompileBuildVariables.OUTPUT_FILE.getVariableName()))
        .contains("_objs/bin/bin");
  }

  @Test
  public void testPresenceOfConfigurationCompileFlags() throws Exception {
    useConfiguration("--copt=-foo");

    scratch.file("x/BUILD", "cc_binary(name = 'bin', srcs = ['bin.cc'], copts = ['-bar'],)");
    scratch.file("x/bin.cc");

    CcToolchainVariables variables = getCompileBuildVariables("//x:bin", "bin");

    ImmutableList<String> userCopts =
        CcToolchainVariables.toStringList(
            variables, CompileBuildVariables.USER_COMPILE_FLAGS.getVariableName());
    assertThat(userCopts).containsAllIn(ImmutableList.<String>of("-foo", "-bar")).inOrder();
  }

  @Test
  public void testPresenceOfUserCompileFlags() throws Exception {
    useConfiguration();

    scratch.file("x/BUILD", "cc_binary(name = 'bin', srcs = ['bin.cc'], copts = ['-foo'])");
    scratch.file("x/bin.cc");

    CcToolchainVariables variables = getCompileBuildVariables("//x:bin", "bin");

    ImmutableList<String> copts =
        CcToolchainVariables.toStringList(
            variables, CompileBuildVariables.USER_COMPILE_FLAGS.getVariableName());
    assertThat(copts).contains("-foo");
  }

  @Test
  public void testPerFileCoptsAreInUserCompileFlags() throws Exception {
    scratch.file("x/BUILD", "cc_binary(name = 'bin', srcs = ['bin.cc'])");
    scratch.file("x/bin.cc");
    useConfiguration("--per_file_copt=//x:bin@-foo", "--per_file_copt=//x:bar\\.cc@-bar");

    CcToolchainVariables variables = getCompileBuildVariables("//x:bin", "bin");

    ImmutableList<String> copts =
        CcToolchainVariables.toStringList(
            variables, CompileBuildVariables.USER_COMPILE_FLAGS.getVariableName());
    assertThat(copts).containsExactly("-foo").inOrder();
  }

  @Test
  public void testPresenceOfSysrootBuildVariable() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig, CcToolchainConfig.builder().withSysroot("/usr/local/custom-sysroot"));
    useConfiguration();

    scratch.file("x/BUILD", "cc_binary(name = 'bin', srcs = ['bin.cc'])");
    scratch.file("x/bin.cc");

    CcToolchainVariables variables = getCompileBuildVariables("//x:bin", "bin");

    assertThat(variables.getStringVariable(CcCommon.SYSROOT_VARIABLE_NAME))
        .isEqualTo("/usr/local/custom-sysroot");
  }

  @Test
  public void testTargetSysrootWithoutPlatforms() throws Exception {
    useConfiguration("--grte_top=//target_libc", "--host_grte_top=//host_libc");

    scratch.file("x/BUILD", "cc_binary(name = 'bin', srcs = ['bin.cc'])");
    scratch.file("x/bin.cc");
    scratch.file("target_libc/BUILD", "filegroup(name = 'everything')");
    scratch.file("host_libc/BUILD", "filegroup(name = 'everything')");

    CcToolchainVariables variables = getCompileBuildVariables("//x:bin", "bin");

    assertThat(variables.getStringVariable(CcCommon.SYSROOT_VARIABLE_NAME))
        .isEqualTo("target_libc");
  }

  @Test
  public void testTargetSysrootWithPlatforms() throws Exception {
    MockPlatformSupport.addMockK8Platform(
        mockToolsConfig, analysisMock.ccSupport().getMockCrosstoolLabel());
    useConfiguration(
        "--experimental_platforms=//mock_platform:mock-k8-platform",
        "--extra_toolchains=//mock_platform:toolchain_cc-compiler-k8",
        "--incompatible_enable_cc_toolchain_resolution",
        "--grte_top=//target_libc",
        "--host_grte_top=//host_libc");

    scratch.file("x/BUILD", "cc_binary(name = 'bin', srcs = ['bin.cc'])");
    scratch.file("x/bin.cc");
    scratch.file("target_libc/BUILD", "filegroup(name = 'everything')");
    scratch.file("host_libc/BUILD", "filegroup(name = 'everything')");

    CcToolchainVariables variables = getCompileBuildVariables("//x:bin", "bin");

    assertThat(variables.getStringVariable(CcCommon.SYSROOT_VARIABLE_NAME))
        .isEqualTo("target_libc");
  }

  @Test
  public void testPresenceOfPerObjectDebugFileBuildVariable() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder().withFeatures(CppRuleClasses.PER_OBJECT_DEBUG_INFO));
    useConfiguration("--fission=yes");

    scratch.file("x/BUILD", "cc_binary(name = 'bin', srcs = ['bin.cc'])");
    scratch.file("x/bin.cc");

    CcToolchainVariables variables = getCompileBuildVariables("//x:bin", "bin");

    assertThat(
            variables.getStringVariable(
                CompileBuildVariables.PER_OBJECT_DEBUG_INFO_FILE.getVariableName()))
        .isNotNull();
  }

  @Test
  public void testPresenceOfIsUsingFissionVariable() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder().withFeatures(CppRuleClasses.PER_OBJECT_DEBUG_INFO));
    useConfiguration("--fission=yes");

    scratch.file("x/BUILD", "cc_binary(name = 'bin', srcs = ['bin.cc'])");
    scratch.file("x/bin.cc");

    CcToolchainVariables variables = getCompileBuildVariables("//x:bin", "bin");

    assertThat(
            variables.getStringVariable(CompileBuildVariables.IS_USING_FISSION.getVariableName()))
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
                    CppRuleClasses.THIN_LTO,
                    MockCcSupport.HOST_AND_NONHOST_CONFIGURATION_FEATURES));
    useConfiguration("--fission=yes", "--features=thin_lto");

    scratch.file("x/BUILD", "cc_binary(name = 'bin', srcs = ['bin.cc'])");
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

    scratch.file("x/BUILD", "cc_binary(name = 'bin', srcs = ['bin.cc'])");
    scratch.file("x/bin.cc");

    CcToolchainVariables variables = getCompileBuildVariables("//x:bin", "bin");

    assertThat(
            variables.getStringVariable(
                CompileBuildVariables.PER_OBJECT_DEBUG_INFO_FILE.getVariableName()))
        .isNotNull();
  }

  @Test
  public void testPresenceOfMinOsVersionBuildVariable() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig, CcToolchainConfig.builder().withFeatures("min_os_version_flag"));
    useConfiguration("--minimum_os_version=6");
    scratch.file("x/BUILD", "cc_binary(name = 'bin', srcs = ['bin.cc'])");
    scratch.file("x/bin.cc");

    CcToolchainVariables variables = getCompileBuildVariables("//x:bin", "bin");
    assertThat(variables.getStringVariable(CcCommon.MINIMUM_OS_VERSION_VARIABLE_NAME))
        .isEqualTo("6");
  }
}
