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
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.config.BuildOptions.MapBackedChecksumCache;
import com.google.devtools.build.lib.analysis.config.BuildOptions.OptionsChecksumCache;
import com.google.devtools.build.lib.analysis.config.OutputDirectories.InvalidMnemonicException;
import com.google.devtools.build.lib.analysis.util.ConfigurationTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.objc.J2ObjcConfiguration;
import com.google.devtools.build.lib.skyframe.BuildConfigurationValue;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.common.options.Options;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link BuildConfiguration}.
 */
@RunWith(JUnit4.class)
public class BuildConfigurationTest extends ConfigurationTestCase {

  @Test
  public void testBasics() throws Exception {
    if (analysisMock.isThisBazel()) {
      return;
    }

    BuildConfiguration config = create("--cpu=piii");
    String outputDirPrefix =
        outputBase + "/execroot/" + config.getMainRepositoryName() + "/blaze-out/.*piii-fastbuild";

    assertThat(config.getOutputDirectory(RepositoryName.MAIN).getRoot().toString())
        .matches(outputDirPrefix);
    assertThat(config.getBinDirectory(RepositoryName.MAIN).getRoot().toString())
        .matches(outputDirPrefix + "/bin");
    assertThat(config.getIncludeDirectory(RepositoryName.MAIN).getRoot().toString())
        .matches(outputDirPrefix + "/include");
    assertThat(config.getTestLogsDirectory(RepositoryName.MAIN).getRoot().toString())
        .matches(outputDirPrefix + "/testlogs");
  }

  @Test
  public void testPlatformSuffix() throws Exception {
    if (analysisMock.isThisBazel()) {
      return;
    }

    BuildConfiguration config = create("--platform_suffix=-test");
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

    Map<String, String> env = create().getLocalShellEnvironment();
    assertThat(env).containsEntry("LANG", "en_US");
    assertThat(env).containsKey("PATH");
    assertThat(env.get("PATH")).contains("/bin:/usr/bin");
    assertThrows(UnsupportedOperationException.class, () -> env.put("FOO", "bar"));
  }

  @Test
  public void testHostCrosstoolTop() throws Exception {
    if (analysisMock.isThisBazel()) {
      return;
    }

    BuildConfigurationCollection configs = createCollection("--cpu=piii");
    BuildConfiguration config = Iterables.getOnlyElement(configs.getTargetConfigurations());
    assertThat(config.getFragment(CppConfiguration.class).getRuleProvidingCcToolchainProvider())
        .isEqualTo(Label.parseAbsoluteUnchecked("//tools/cpp:toolchain"));

    BuildConfiguration hostConfig = configs.getHostConfiguration();
    assertThat(hostConfig.getFragment(CppConfiguration.class).getRuleProvidingCcToolchainProvider())
        .isEqualTo(Label.parseAbsoluteUnchecked("//tools/cpp:toolchain"));
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
  public void testConfigurationsHaveUniqueOutputDirectories() throws Exception {
    assertConfigurationsHaveUniqueOutputDirectories(createCollection());
    assertConfigurationsHaveUniqueOutputDirectories(createCollection("--compilation_mode=opt"));
  }

  @Test
  public void testMultiCpu() throws Exception {
    if (analysisMock.isThisBazel()) {
      return;
    }

    BuildConfigurationCollection master = createCollection("--multi_cpu=k8", "--multi_cpu=piii");
    assertThat(master.getTargetConfigurations()).hasSize(2);
    // Note: the cpus are sorted alphabetically.
    assertThat(master.getTargetConfigurations().get(0).getCpu()).isEqualTo("k8");
    assertThat(master.getTargetConfigurations().get(1).getCpu()).isEqualTo("piii");
  }

  /**
   * Check that the cpus are sorted alphabetically regardless of the order in which they are
   * specified.
   */
  @Test
  public void testMultiCpuSorting() throws Exception {
    if (analysisMock.isThisBazel()) {
      return;
    }

    for (int order = 0; order < 2; order++) {
      BuildConfigurationCollection master;
      if (order == 0) {
        master = createCollection("--multi_cpu=k8", "--multi_cpu=piii");
      } else {
        master = createCollection("--multi_cpu=piii", "--multi_cpu=k8");
      }
      assertThat(master.getTargetConfigurations()).hasSize(2);
      assertThat(master.getTargetConfigurations().get(0).getCpu()).isEqualTo("k8");
      assertThat(master.getTargetConfigurations().get(1).getCpu()).isEqualTo("piii");
    }
  }

  @Test
  public void testTargetEnvironment() throws Exception {
    BuildConfiguration oneEnvConfig = create("--target_environment=//foo");
    assertThat(oneEnvConfig.getTargetEnvironments())
        .containsExactly(Label.parseAbsolute("//foo", ImmutableMap.of()));

    BuildConfiguration twoEnvsConfig =
        create("--target_environment=//foo", "--target_environment=//bar");
    assertThat(twoEnvsConfig.getTargetEnvironments())
        .containsExactly(
            Label.parseAbsolute("//foo", ImmutableMap.of()),
            Label.parseAbsolute("//bar", ImmutableMap.of()));

    BuildConfiguration noEnvsConfig = create();
    assertThat(noEnvsConfig.getTargetEnvironments()).isEmpty();
  }

  @Test
  public void testGlobalMakeVariableOverride() throws Exception {
    assertThat(create().getMakeEnvironment()).containsEntry("COMPILATION_MODE", "fastbuild");
    BuildConfiguration config = create("--define", "COMPILATION_MODE=fluttershy");
    assertThat(config.getMakeEnvironment()).containsEntry("COMPILATION_MODE", "fluttershy");
  }

  @Test
  public void testGetTransitiveOptionDetails() throws Exception {
    // Directly defined options:
    assertThat(create("-c", "dbg").getTransitiveOptionDetails().getOptionValue("compilation_mode"))
        .isEqualTo(CompilationMode.DBG);
    assertThat(create("-c", "opt").getTransitiveOptionDetails().getOptionValue("compilation_mode"))
        .isEqualTo(CompilationMode.OPT);

    // Options defined in a fragment:
    assertThat(create("--force_pic").getTransitiveOptionDetails().getOptionValue("force_pic"))
        .isEqualTo(Boolean.TRUE);
    assertThat(create("--noforce_pic").getTransitiveOptionDetails().getOptionValue("force_pic"))
        .isEqualTo(Boolean.FALSE);

    // Legitimately null option:
    assertThat(create().getTransitiveOptionDetails().getOptionValue("test_filter")).isNull();
  }

  @Test
  public void testEqualsOrIsSupersetOf() throws Exception {
    BuildConfiguration config = create();
    BuildConfiguration trimmedConfig =
        config.clone(
            FragmentClassSet.of(
                ImmutableSortedSet.orderedBy(BuildConfiguration.lexicalFragmentSorter)
                    .add(CppConfiguration.class)
                    .build()),
            analysisMock.createRuleClassProvider());
    BuildConfiguration hostConfig = createHost();

    assertThat(config.equalsOrIsSupersetOf(trimmedConfig)).isTrue();
    assertThat(config.equalsOrIsSupersetOf(hostConfig)).isFalse();
    assertThat(trimmedConfig.equalsOrIsSupersetOf(config)).isFalse();
  }

  @Test
  public void testConfigFragmentsAreShareableAcrossConfigurations() throws Exception {
    // Note we can't use any fragments that load files (e.g. CROSSTOOL) because those get
    // Skyframe-invalidated between create() calls.
    BuildConfiguration config1 = create("--javacopt=foo");
    BuildConfiguration config2 = create("--javacopt=bar");
    BuildConfiguration config3 = create("--j2objc_translation_flags=baz");
    // Shared because all j2objc options are the same:
    assertThat(config1.getFragment(J2ObjcConfiguration.class))
        .isSameInstanceAs(config2.getFragment(J2ObjcConfiguration.class));
    // Distinct because the j2objc options differ:
    assertThat(config1.getFragment(J2ObjcConfiguration.class))
        .isNotSameInstanceAs(config3.getFragment(J2ObjcConfiguration.class));
  }

  @Test
  public void testCommandLineVariables() throws Exception {
    BuildConfiguration config = create(
        "--define", "a=b/c:d", "--define", "b=FOO", "--define", "DEFUN=Nope");
    assertThat(config.getCommandLineBuildVariables().get("a")).isEqualTo("b/c:d");
    assertThat(config.getCommandLineBuildVariables().get("b")).isEqualTo("FOO");
    assertThat(config.getCommandLineBuildVariables().get("DEFUN")).isEqualTo("Nope");
  }

  // Regression test for bug #2518997:
  // "--define in blazerc overrides --define from command line"
  @Test
  public void testCommandLineVariablesOverride() throws Exception {
    BuildConfiguration config = create("--define", "a=b", "--define", "a=c");
    assertThat(config.getCommandLineBuildVariables().get("a")).isEqualTo("c");
  }

  @Test
  public void testNormalization_definesWithSameName_collapseDuplicateDefinesDisabled()
      throws Exception {
    BuildConfiguration config =
        create("--nocollapse_duplicate_defines", "--define", "a=1", "--define", "a=2");
    CoreOptions options = config.getOptions().get(CoreOptions.class);
    assertThat(ImmutableListMultimap.copyOf(options.commandLineBuildVariables))
        .containsExactly("a", "1", "a", "2")
        .inOrder();
    assertThat(config).isNotEqualTo(create("--nocollapse_duplicate_defines", "--define", "a=2"));
  }

  @Test
  public void testNormalization_definesWithDifferentNames() throws Exception {
    BuildConfiguration config =
        create("--collapse_duplicate_defines", "--define", "a=1", "--define", "b=2");
    CoreOptions options = config.getOptions().get(CoreOptions.class);
    assertThat(ImmutableMap.copyOf(options.commandLineBuildVariables))
        .containsExactly("a", "1", "b", "2");
  }

  @Test
  public void testNormalization_definesWithSameName() throws Exception {
    BuildConfiguration config =
        create("--collapse_duplicate_defines", "--define", "a=1", "--define", "a=2");
    CoreOptions options = config.getOptions().get(CoreOptions.class);
    assertThat(ImmutableMap.copyOf(options.commandLineBuildVariables)).containsExactly("a", "2");
    assertThat(config).isEqualTo(create("--collapse_duplicate_defines", "--define", "a=2"));
  }

  // This is really a test of option parsing, not command-line variable
  // semantics.
  @Test
  public void testCommandLineVariablesWithFunnyCharacters() throws Exception {
    BuildConfiguration config = create(
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
  public void testHostDefine() throws Exception {
    BuildConfiguration cfg = createHost("--define=foo=bar");
    assertThat(cfg.getCommandLineBuildVariables().get("foo")).isEqualTo("bar");
  }

  @Test
  public void testHostCompilationModeDefault() throws Exception {
    BuildConfiguration cfg = createHost();
    assertThat(cfg.getCompilationMode()).isEqualTo(CompilationMode.OPT);
  }

  @Test
  public void testHostCompilationModeNonDefault() throws Exception {
    BuildConfiguration cfg = createHost("--host_compilation_mode=dbg");
    assertThat(cfg.getCompilationMode()).isEqualTo(CompilationMode.DBG);
  }

  @Test
  public void testIncompatibleMergeGenfilesDirectory() throws Exception {
    BuildConfiguration target = create("--incompatible_merge_genfiles_directory");
    BuildConfiguration host = createHost("--incompatible_merge_genfiles_directory");
    assertThat(target.getGenfilesDirectory(RepositoryName.MAIN))
        .isEqualTo(target.getBinDirectory(RepositoryName.MAIN));
    assertThat(host.getGenfilesDirectory(RepositoryName.MAIN))
        .isEqualTo(host.getBinDirectory(RepositoryName.MAIN));
  }

  private ImmutableList<BuildConfiguration> getTestConfigurations() throws Exception {
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
        .isEqualTo(
            "Output directory name '//bad/cpu' specified by CppConfiguration is invalid as part of"
                + " a path: must not contain /");
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
        .setVerificationFunction(BuildConfigurationTest::verifyDeserialized)
        .runTests();
  }

  @Test
  public void testKeyCodec() throws Exception {
    new SerializationTester(
            getTestConfigurations().stream()
                .map(BuildConfigurationValue::key)
                .collect(ImmutableList.toImmutableList()))
        .addDependency(OptionsChecksumCache.class, new MapBackedChecksumCache())
        .runTests();
  }

  /**
   * Partial verification of deserialized BuildConfiguration.
   *
   * <p>Direct comparison of deserialized to subject doesn't work because Fragment classes do not
   * implement equals. This runs the part of BuildConfiguration.equals that has equals definitions.
   */
  private static void verifyDeserialized(
      BuildConfiguration subject, BuildConfiguration deserialized) {
    assertThat(deserialized.getOptions()).isEqualTo(subject.getOptions());
  }
}
