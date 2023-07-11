// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.config;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.analysis.config.BuildOptions.MapBackedChecksumCache;
import com.google.devtools.build.lib.analysis.config.BuildOptions.OptionsChecksumCache;
import com.google.devtools.build.lib.analysis.config.OutputDirectories.InvalidMnemonicException;
import com.google.devtools.build.lib.analysis.util.ConfigurationTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.objc.J2ObjcConfiguration;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.common.options.Options;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link BuildConfigurationValue}. */
@RunWith(JUnit4.class)
public final class BuildConfigurationValueTest extends ConfigurationTestCase {

  @Test
  public void testBasics() throws Exception {
    if (analysisMock.isThisBazel()) {
      return;
    }

    BuildConfigurationValue config = create("--cpu=piii");
    String outputDirPrefix =
        outputBase + "/execroot/" + config.getMainRepositoryName() + "/blaze-out/.*piii-fastbuild";

    assertThat(config.getOutputDirectory(RepositoryName.MAIN).getRoot().toString())
        .matches(outputDirPrefix);
    assertThat(config.getBinDirectory(RepositoryName.MAIN).getRoot().toString())
        .matches(outputDirPrefix + "/bin");
    assertThat(config.getBuildInfoDirectory(RepositoryName.MAIN).getRoot().toString())
        .matches(outputDirPrefix + "/include");
    assertThat(config.getTestLogsDirectory(RepositoryName.MAIN).getRoot().toString())
        .matches(outputDirPrefix + "/testlogs");
  }

  @Test
  public void testPlatformSuffix() throws Exception {
    if (analysisMock.isThisBazel()) {
      return;
    }

    BuildConfigurationValue config = create("--platform_suffix=test");
    assertThat(config.getOutputDirectory(RepositoryName.MAIN).getRoot().toString())
        .matches(
            outputBase
                + "/execroot/"
                + config.getMainRepositoryName()
                + "/blaze-out/.*k8-fastbuild-test");
  }

  @Test
  public void testEnvironment() throws Exception {
    if (analysisMock.isThisBazel()) {
      return;
    }

    ImmutableMap<String, String> env = create().getLocalShellEnvironment();
    assertThat(env).containsEntry("LANG", "en_US");
    assertThat(env).containsKey("PATH");
    assertThat(env.get("PATH")).contains("/bin:/usr/bin");
  }

  @Test
  public void testHostCrosstoolTop() throws Exception {
    if (analysisMock.isThisBazel()) {
      return;
    }

    BuildConfigurationValue config = createConfiguration("--cpu=piii");
    assertThat(config.getFragment(CppConfiguration.class).getRuleProvidingCcToolchainProvider())
        .isEqualTo(Label.parseCanonicalUnchecked("//tools/cpp:toolchain"));

    BuildConfigurationValue execConfig = createExec();
    assertThat(execConfig.getFragment(CppConfiguration.class).getRuleProvidingCcToolchainProvider())
        .isEqualTo(Label.parseCanonicalUnchecked("//tools/cpp:toolchain"));
  }

  @Test
  public void testCaching() {
    CoreOptions a = Options.getDefaults(CoreOptions.class);
    CoreOptions b = Options.getDefaults(CoreOptions.class);
    // The String representations of the CoreOptions must be equal even if these are
    // different objects, if they were created with the same options (no options in this case).
    assertThat(b.toString()).isEqualTo(a.toString());
    assertThat(b.cacheKey()).isEqualTo(a.cacheKey());
  }

  @Test
  public void testTargetEnvironment() throws Exception {
    BuildConfigurationValue oneEnvConfig = create("--target_environment=//foo");
    assertThat(oneEnvConfig.getTargetEnvironments()).containsExactly(Label.parseCanonical("//foo"));

    BuildConfigurationValue twoEnvsConfig =
        create("--target_environment=//foo", "--target_environment=//bar");
    assertThat(twoEnvsConfig.getTargetEnvironments())
        .containsExactly(Label.parseCanonical("//foo"), Label.parseCanonical("//bar"));

    BuildConfigurationValue noEnvsConfig = create();
    assertThat(noEnvsConfig.getTargetEnvironments()).isEmpty();
  }

  @Test
  public void testGlobalMakeVariableOverride() throws Exception {
    assertThat(create().getMakeEnvironment()).containsEntry("COMPILATION_MODE", "fastbuild");
    BuildConfigurationValue config = create("--define", "COMPILATION_MODE=fluttershy");
    assertThat(config.getMakeEnvironment()).containsEntry("COMPILATION_MODE", "fluttershy");
  }

  @Test
  public void testGetBuildOptionDetails() throws Exception {
    // Directly defined options:
    assertThat(create("-c", "dbg").getBuildOptionDetails().getOptionValue("compilation_mode"))
        .isEqualTo(CompilationMode.DBG);
    assertThat(create("-c", "opt").getBuildOptionDetails().getOptionValue("compilation_mode"))
        .isEqualTo(CompilationMode.OPT);

    // Options defined in a fragment:
    assertThat(create("--force_pic").getBuildOptionDetails().getOptionValue("force_pic"))
        .isEqualTo(Boolean.TRUE);
    assertThat(create("--noforce_pic").getBuildOptionDetails().getOptionValue("force_pic"))
        .isEqualTo(Boolean.FALSE);

    // Legitimately null option:
    assertThat(create().getBuildOptionDetails().getOptionValue("test_filter")).isNull();
  }

  @Test
  public void testConfigFragmentsAreShareableAcrossConfigurations() throws Exception {
    // Note we can't use any fragments that load files (e.g. CROSSTOOL) because those get
    // Skyframe-invalidated between create() calls.
    BuildConfigurationValue config1 = create("--javacopt=foo");
    BuildConfigurationValue config2 = create("--javacopt=bar");
    BuildConfigurationValue config3 = create("--j2objc_translation_flags=baz");
    // Shared because all j2objc options are the same:
    assertThat(config1.getFragment(J2ObjcConfiguration.class))
        .isSameInstanceAs(config2.getFragment(J2ObjcConfiguration.class));
    // Distinct because the j2objc options differ:
    assertThat(config1.getFragment(J2ObjcConfiguration.class))
        .isNotSameInstanceAs(config3.getFragment(J2ObjcConfiguration.class));
  }

  @Test
  public void testCommandLineVariables() throws Exception {
    BuildConfigurationValue config =
        create("--define", "a=b/c:d", "--define", "b=FOO", "--define", "DEFUN=Nope");
    assertThat(config.getCommandLineBuildVariables().get("a")).isEqualTo("b/c:d");
    assertThat(config.getCommandLineBuildVariables().get("b")).isEqualTo("FOO");
    assertThat(config.getCommandLineBuildVariables().get("DEFUN")).isEqualTo("Nope");
  }

  // Regression test for bug #2518997:
  // "--define in blazerc overrides --define from command line"
  @Test
  public void testCommandLineVariablesOverride() throws Exception {
    BuildConfigurationValue config = create("--define", "a=b", "--define", "a=c");
    assertThat(config.getCommandLineBuildVariables().get("a")).isEqualTo("c");
  }

  @Test
  public void testNormalization_definesWithDifferentNames() throws Exception {
    BuildConfigurationValue config = create("--define", "a=1", "--define", "b=2");
    CoreOptions options = config.getOptions().get(CoreOptions.class);
    assertThat(ImmutableMap.copyOf(options.commandLineBuildVariables))
        .containsExactly("a", "1", "b", "2");
  }

  @Test
  public void testNormalization_definesWithSameName() throws Exception {
    BuildConfigurationValue config = create("--define", "a=1", "--define", "a=2");
    CoreOptions options = config.getOptions().get(CoreOptions.class);
    assertThat(ImmutableMap.copyOf(options.commandLineBuildVariables)).containsExactly("a", "2");
    assertThat(config).isEqualTo(create("--define", "a=2"));
  }

  // This is really a test of option parsing, not command-line variable
  // semantics.
  @Test
  public void testCommandLineVariablesWithFunnyCharacters() throws Exception {
    BuildConfigurationValue config =
        create(
            "--define", "foo=#foo",
            "--define", "comma=a,b",
            "--define", "space=foo bar",
            "--define", "thing=a \"quoted\" thing",
            "--define", "qspace=a\\ quoted\\ space",
            "--define", "#a=pounda");
    assertThat(config.getCommandLineBuildVariables().get("foo")).isEqualTo("#foo");
    assertThat(config.getCommandLineBuildVariables().get("comma")).isEqualTo("a,b");
    assertThat(config.getCommandLineBuildVariables().get("space")).isEqualTo("foo bar");
    assertThat(config.getCommandLineBuildVariables().get("thing")).isEqualTo("a \"quoted\" thing");
    assertThat(config.getCommandLineBuildVariables().get("qspace")).isEqualTo("a\\ quoted\\ space");
    assertThat(config.getCommandLineBuildVariables().get("#a")).isEqualTo("pounda");
  }

  @Test
  public void testExecDefine() throws Exception {
    BuildConfigurationValue cfg = createExec("--define=foo=bar");
    assertThat(cfg.getCommandLineBuildVariables().get("foo")).isEqualTo("bar");
  }

  @Test
  public void testHostCompilationModeDefault() throws Exception {
    BuildConfigurationValue cfg = createExec();
    assertThat(cfg.getCompilationMode()).isEqualTo(CompilationMode.OPT);
  }

  @Test
  public void testHostCompilationModeNonDefault() throws Exception {
    BuildConfigurationValue cfg = createExec("--host_compilation_mode=dbg");
    assertThat(cfg.getCompilationMode()).isEqualTo(CompilationMode.DBG);
  }

  @Test
  public void testIncompatibleMergeGenfilesDirectory() throws Exception {
    BuildConfigurationValue target = create("--incompatible_merge_genfiles_directory");
    BuildConfigurationValue exec = createExec("--incompatible_merge_genfiles_directory");
    assertThat(target.getGenfilesDirectory(RepositoryName.MAIN))
        .isEqualTo(target.getBinDirectory(RepositoryName.MAIN));
    assertThat(exec.getGenfilesDirectory(RepositoryName.MAIN))
        .isEqualTo(exec.getBinDirectory(RepositoryName.MAIN));
  }

  private ImmutableList<BuildConfigurationValue> getTestConfigurations() throws Exception {
    return ImmutableList.of(
        create(),
        create("--cpu=piii"),
        create("--javacopt=foo"),
        create("--platform_suffix=-test"),
        create("--target_environment=//foo", "--target_environment=//bar"),
        create("--incompatible_merge_genfiles_directory"),
        create(
            "--define",
            "foo=#foo",
            "--define",
            "comma=a,b",
            "--define",
            "space=foo bar",
            "--define",
            "thing=a \"quoted\" thing",
            "--define",
            "qspace=a\\ quoted\\ space",
            "--define",
            "#a=pounda"));
  }

  @Test
  public void throwsOnBadMnemonic() {
    InvalidMnemonicException e =
        assertThrows(InvalidMnemonicException.class, () -> create("--cpu=//bad/cpu"));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo("CPU name '//bad/cpu' is invalid as part of a path: must not contain /");
    e =
        assertThrows(
            InvalidMnemonicException.class, () -> create("--platform_suffix=//bad/suffix"));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo(
            "Platform suffix '//bad/suffix' is invalid as part of a path: must not contain /");
  }

  @Test
  public void testCodec() throws Exception {
    // Unnecessary ImmutableList.copyOf apparently necessary to choose non-varargs constructor.
    new SerializationTester(ImmutableList.copyOf(getTestConfigurations()))
        .addDependency(FileSystem.class, getScratch().getFileSystem())
        .addDependency(OptionsChecksumCache.class, new MapBackedChecksumCache())
        .setVerificationFunction(BuildConfigurationValueTest::verifyDeserialized)
        .runTests();
  }

  @Test
  public void testKeyCodec() throws Exception {
    new SerializationTester(
            getTestConfigurations().stream()
                .map(BuildConfigurationValue::getKey)
                .collect(ImmutableList.toImmutableList()))
        .addDependency(OptionsChecksumCache.class, new MapBackedChecksumCache())
        .runTests();
  }

  @Test
  public void testPlatformInOutputDir_defaultPlatform() throws Exception {
    BuildConfigurationValue config = create("--experimental_platform_in_output_dir", "--cpu=k8");

    assertThat(config.getOutputDirectory(RepositoryName.MAIN).getRoot().toString())
        .matches(".*/[^/]+-out/k8-fastbuild");
  }

  @Test
  public void testPlatformInOutputDir() throws Exception {
    BuildConfigurationValue config =
        create("--experimental_platform_in_output_dir", "--platforms=//platform:alpha");

    assertThat(config.getOutputDirectory(RepositoryName.MAIN).getRoot().toString())
        .matches(".*/[^/]+-out/alpha-fastbuild");
  }

  @Test
  public void testConfigurationEquality() throws Exception {
    // Note that, in practice, test_arg should not be used as a no-op argument; however,
    // these configurations are never trimmed nor even used to build targets so not an issue.
    new EqualsTester()
        .addEqualityGroup(
            createRaw(parseBuildOptions("--test_arg=1a"), "testrepo", false, ""),
            createRaw(parseBuildOptions("--test_arg=1a"), "testrepo", false, ""))
        // Different BuildOptions means non-equal
        .addEqualityGroup(createRaw(parseBuildOptions("--test_arg=1b"), "testrepo", false, ""))
        // Different --experimental_sibling_repository_layout means non-equal
        .addEqualityGroup(createRaw(parseBuildOptions("--test_arg=2"), "testrepo", true, ""))
        .addEqualityGroup(createRaw(parseBuildOptions("--test_arg=2"), "testrepo", false, ""))
        // Different repositoryName means non-equal
        .addEqualityGroup(createRaw(parseBuildOptions("--test_arg=3"), "testrepo1", false, ""))
        .addEqualityGroup(createRaw(parseBuildOptions("--test_arg=3"), "testrepo2", false, ""))
        // Different transitionDirectoryNameFragment means non-equal
        .addEqualityGroup(createRaw(parseBuildOptions("--test_arg=3"), "testrepo", false, ""))
        .addEqualityGroup(createRaw(parseBuildOptions("--test_arg=3"), "testrepo", false, "a"))
        .addEqualityGroup(createRaw(parseBuildOptions("--test_arg=3"), "testrepo", false, "b"))
        .testEquals();
  }

  /**
   * Partial verification of deserialized BuildConfigurationValue.
   *
   * <p>Direct comparison of deserialized to subject doesn't work because Fragment classes do not
   * implement equals. This runs the part of BuildConfigurationValue.equals that has equals
   * definitions.
   */
  private static void verifyDeserialized(
      BuildConfigurationValue subject, BuildConfigurationValue deserialized) {
    assertThat(deserialized.getOptions()).isEqualTo(subject.getOptions());
  }
}
