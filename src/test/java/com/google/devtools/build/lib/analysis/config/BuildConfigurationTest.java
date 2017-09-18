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
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.Fragment;
import com.google.devtools.build.lib.analysis.util.ConfigurationTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;
import com.google.devtools.build.lib.rules.objc.J2ObjcConfiguration;
import com.google.devtools.common.options.Options;
import java.util.Map;
import java.util.regex.Pattern;
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

    assertThat(config.getOutputDirectory(RepositoryName.MAIN).getPath().toString())
        .matches(outputDirPrefix);
    assertThat(config.getBinDirectory(RepositoryName.MAIN).getPath().toString())
        .matches(outputDirPrefix + "/bin");
    assertThat(config.getIncludeDirectory(RepositoryName.MAIN).getPath().toString())
        .matches(outputDirPrefix + "/include");
    assertThat(config.getTestLogsDirectory(RepositoryName.MAIN).getPath().toString())
        .matches(outputDirPrefix + "/testlogs");
  }

  @Test
  public void testPlatformSuffix() throws Exception {
    if (analysisMock.isThisBazel()) {
      return;
    }

    BuildConfiguration config = create("--platform_suffix=-test");
    assertThat(config.getOutputDirectory(RepositoryName.MAIN).getPath().toString())
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
    try {
      env.put("FOO", "bar");
      fail("modifiable default environment");
    } catch (UnsupportedOperationException ignored) {
      //expected exception
    }
  }

  @Test
  public void testHostCpu() throws Exception {
    for (String cpu : new String[] { "piii", "k8" }) {
      BuildConfiguration hostConfig = createHost("--host_cpu=" + cpu);
      assertThat(hostConfig.getFragment(CppConfiguration.class).getTargetCpu()).isEqualTo(cpu);
    }
  }

  @Test
  public void testHostCrosstoolTop() throws Exception {
    if (analysisMock.isThisBazel()) {
      return;
    }

    BuildConfigurationCollection configs = createCollection("--cpu=piii");
    BuildConfiguration config = Iterables.getOnlyElement(configs.getTargetConfigurations());
    assertThat(config.getFragment(CppConfiguration.class).getCcToolchainRuleLabel())
        .isEqualTo(Label.parseAbsoluteUnchecked("//third_party/crosstool/mock:cc-compiler-piii"));

    BuildConfiguration hostConfig = configs.getHostConfiguration();
    assertThat(hostConfig.getFragment(CppConfiguration.class).getCcToolchainRuleLabel())
        .isEqualTo(Label.parseAbsoluteUnchecked("//third_party/crosstool/mock:cc-compiler-k8"));
  }

  @Test
  public void testMakeEnvFlags() throws Exception {
    BuildConfiguration config = create();
    assertThat(config.getMakeEnvironment().get("STRIP")).contains("strip");
  }

  @Test
  public void testCaching() throws Exception {
    BuildConfiguration.Options a = Options.getDefaults(BuildConfiguration.Options.class);
    BuildConfiguration.Options b = Options.getDefaults(BuildConfiguration.Options.class);
    // The String representations of the BuildConfiguration.Options must be equal even if these are
    // different objects, if they were created with the same options (no options in this case).
    assertThat(b.toString()).isEqualTo(a.toString());
    assertThat(b.cacheKey()).isEqualTo(a.cacheKey());
  }

  @Test
  public void testInvalidCpu() throws Exception {
    // TODO(ulfjack): It would be better to get the better error message also if the Jvm is enabled.
    // Currently: "No JVM target found under //tools/jdk:jdk that would work for bogus"
    try {
      create("--cpu=bogus");
      fail();
    } catch (InvalidConfigurationException e) {
      assertThat(e)
          .hasMessageThat()
          .matches(
              Pattern.compile(
                  "No toolchain found for cpu 'bogus'. Valid cpus are: \\[\n(  [\\w-]+,\n)+]"));
    }
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
    assertThat(oneEnvConfig.getTargetEnvironments()).containsExactly(Label.parseAbsolute("//foo"));

    BuildConfiguration twoEnvsConfig =
        create("--target_environment=//foo", "--target_environment=//bar");
    assertThat(twoEnvsConfig.getTargetEnvironments())
        .containsExactly(Label.parseAbsolute("//foo"), Label.parseAbsolute("//bar"));

    BuildConfiguration noEnvsConfig = create();
    assertThat(noEnvsConfig.getTargetEnvironments()).isEmpty();
  }

  @SafeVarargs
  @SuppressWarnings("unchecked")
  private final ConfigurationFragmentFactory createMockFragment(
      final Class<? extends Fragment> creates, final Class<? extends Fragment>... dependsOn) {
    return new ConfigurationFragmentFactory() {

      @Override
      public Class<? extends Fragment> creates() {
        return creates;
      }

      @Override
      public ImmutableSet<Class<? extends FragmentOptions>> requiredOptions() {
        return ImmutableSet.of();
      }

      @Override
      public Fragment create(ConfigurationEnvironment env, BuildOptions buildOptions)
          throws InvalidConfigurationException, InterruptedException {
        for (Class<? extends Fragment> fragmentType : dependsOn) {
          env.getFragment(buildOptions, fragmentType);
        }
        return new Fragment() {};
      }
    };
  }

  @Test
  public void testCycleInFragments() throws Exception {
    configurationFragmentFactories = ImmutableList.of(
        createMockFragment(CppConfiguration.class, JavaConfiguration.class),
        createMockFragment(JavaConfiguration.class, CppConfiguration.class));
    try {
      createCollection();
      fail();
    } catch (IllegalStateException e) {
      // expected
    }
  }

  @Test
  public void testMissingFragment() throws Exception {
    configurationFragmentFactories = ImmutableList.of(
        createMockFragment(CppConfiguration.class, JavaConfiguration.class));
    try {
      createCollection();
      fail();
    } catch (RuntimeException e) {
      // expected
    }
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

    // Late-bound default option:
    BuildConfiguration config = create();
    assertThat(config.getFragment(CppConfiguration.class).getCompiler())
        .isEqualTo(config.getTransitiveOptionDetails().getOptionValue("compiler"));

    // Legitimately null option:
    assertThat(create().getTransitiveOptionDetails().getOptionValue("test_filter")).isNull();
  }

  @Test
  public void testEqualsOrIsSupersetOf() throws Exception {
    BuildConfiguration config = create();
    BuildConfiguration trimmedConfig =
        config.clone(
            ImmutableSet.<Class<? extends Fragment>>of(CppConfiguration.class),
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
        .isSameAs(config2.getFragment(J2ObjcConfiguration.class));
    // Distinct because the j2objc options differ:
    assertThat(config1.getFragment(J2ObjcConfiguration.class))
        .isNotSameAs(config3.getFragment(J2ObjcConfiguration.class));
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

  /**
   * Returns a mock config fragment that loads the given label and does nothing else.
   */
  private static ConfigurationFragmentFactory createMockFragmentWithLabelDep(final String label) {
    return new ConfigurationFragmentFactory() {
      @Override
      public Fragment create(ConfigurationEnvironment env, BuildOptions buildOptions)
          throws InvalidConfigurationException, InterruptedException {
        try {
          env.getTarget(Label.parseAbsoluteUnchecked(label));
        } catch (NoSuchPackageException e) {
          fail("cannot load mock fragment's dep label " + label + ": " + e.getMessage());
        } catch (NoSuchTargetException e) {
          fail("cannot load mock fragment's dep label " + label + ": " + e.getMessage());
        }
        return new Fragment() {};
      }

      @Override
      public Class<? extends Fragment> creates() {
        return CppConfiguration.class;
      }

      @Override
      public ImmutableSet<Class<? extends FragmentOptions>> requiredOptions() {
        return ImmutableSet.<Class<? extends FragmentOptions>>of();
      }
    };
  }

  @Test
  public void depLabelCycleOnConfigurationLoading() throws Exception {
    configurationFragmentFactories = ImmutableList.of(createMockFragmentWithLabelDep("//foo"));
    getScratch().file("foo/BUILD",
        "load('//skylark:one.bzl', 'one')",
        "cc_library(name = 'foo')");
    getScratch().file("skylark/BUILD");
    getScratch().file("skylark/one.bzl",
        "load('//skylark:two.bzl', 'two')",
        "def one():",
        "  pass");
    getScratch().file("skylark/two.bzl",
        "load('//skylark:one.bzl', 'one')",
        "def two():",
        "  pass");
    checkError(String.join("\n",
        "ERROR <no location>: cycle detected in extension files: ",
        "    foo/BUILD",
        ".-> //skylark:one.bzl",
        "|   //skylark:two.bzl",
        "`-- //skylark:one.bzl"));
  }

  @Test
  public void testNoSeparateGenfilesDirectory() throws Exception {
    BuildConfiguration target = create("--noexperimental_separate_genfiles_directory");
    BuildConfiguration host = createHost("--noexperimental_separate_genfiles_directory");
    assertThat(target.getGenfilesDirectory(RepositoryName.MAIN))
        .isEqualTo(target.getBinDirectory(RepositoryName.MAIN));
    assertThat(host.getGenfilesDirectory(RepositoryName.MAIN))
        .isEqualTo(host.getBinDirectory(RepositoryName.MAIN));
  }
}
