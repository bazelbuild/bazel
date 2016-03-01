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
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.Fragment;
import com.google.devtools.build.lib.analysis.util.ConfigurationTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppOptions;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;
import com.google.devtools.build.lib.rules.objc.J2ObjcConfiguration;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.common.options.Options;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.Map;
import java.util.regex.Pattern;

/**
 * Tests for {@link BuildConfiguration}.
 */
@RunWith(JUnit4.class)
public class BuildConfigurationTest extends ConfigurationTestCase {

  @Test
  public void testBasics() throws Exception {
    if (TestConstants.THIS_IS_BAZEL) {
      return;
    }

    BuildConfiguration config = create("--cpu=piii");
    String outputDirPrefix = outputBase
        + "/workspace/blaze-out/gcc-4.4.0-glibc-2.3.6-grte-piii-fastbuild";

    assertEquals(outputDirPrefix,
                 config.getOutputDirectory().getPath().toString());
    assertEquals(outputDirPrefix + "/bin",
                 config.getBinDirectory().getPath().toString());
    assertEquals(outputDirPrefix + "/include",
                 config.getIncludeDirectory().getPath().toString());
    assertEquals(outputDirPrefix + "/genfiles",
                 config.getGenfilesDirectory().getPath().toString());
    assertEquals(outputDirPrefix + "/testlogs",
                 config.getTestLogsDirectory().getPath().toString());
  }

  @Test
  public void testPlatformSuffix() throws Exception {
    if (TestConstants.THIS_IS_BAZEL) {
      return;
    }

    BuildConfiguration config = create("--platform_suffix=-test");
    assertEquals(outputBase + "/workspace/blaze-out/gcc-4.4.0-glibc-2.3.6-grte-k8-fastbuild-test",
        config.getOutputDirectory().getPath().toString());
  }

  @Test
  public void testEnvironment() throws Exception {
    if (TestConstants.THIS_IS_BAZEL) {
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
      assertEquals(cpu, hostConfig.getFragment(CppConfiguration.class).getTargetCpu());
    }
  }

  @Test
  public void testHostCrosstoolTop() throws Exception {
    if (TestConstants.THIS_IS_BAZEL) {
      return;
    }

    BuildConfigurationCollection configs = createCollection("--cpu=piii");
    BuildConfiguration config = Iterables.getOnlyElement(configs.getTargetConfigurations());
    assertEquals(Label.parseAbsoluteUnchecked("//third_party/crosstool/mock:cc-compiler-piii"),
        config.getFragment(CppConfiguration.class).getCcToolchainRuleLabel());

    BuildConfiguration hostConfig = configs.getHostConfiguration();
    assertEquals(Label.parseAbsoluteUnchecked("//third_party/crosstool/mock:cc-compiler-k8"),
        hostConfig.getFragment(CppConfiguration.class).getCcToolchainRuleLabel());
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
    assertEquals(a.toString(), b.toString());
    assertEquals(a.cacheKey(), b.cacheKey());
  }

  private void checkInvalidCpuError(String cpuOption, Pattern messageRegex) throws Exception {
    try {
      create("--" + cpuOption + "=bogus");
      fail();
    } catch (InvalidConfigurationException e) {
      assertThat(e.getMessage()).matches(messageRegex);
    }
  }

  @Test
  public void testInvalidCpu() throws Exception {
    checkInvalidCpuError("cpu", Pattern.compile(
        "No toolchain found for cpu 'bogus'. Valid cpus are: \\[\n(  [\\w-]+,\n)+]"));
  }

  @Test
  public void testConfigurationsHaveUniqueOutputDirectories() throws Exception {
    assertConfigurationsHaveUniqueOutputDirectories(createCollection());
    assertConfigurationsHaveUniqueOutputDirectories(createCollection("--compilation_mode=opt"));
  }

  @Test
  public void testMultiCpu() throws Exception {
    if (TestConstants.THIS_IS_BAZEL) {
      return;
    }

    BuildConfigurationCollection master = createCollection("--multi_cpu=k8", "--multi_cpu=piii");
    assertThat(master.getTargetConfigurations()).hasSize(2);
    // Note: the cpus are sorted alphabetically.
    assertEquals("k8", master.getTargetConfigurations().get(0).getCpu());
    assertEquals("piii", master.getTargetConfigurations().get(1).getCpu());
  }

  /**
   * Check that the cpus are sorted alphabetically regardless of the order in which they are
   * specified.
   */
  @Test
  public void testMultiCpuSorting() throws Exception {
    if (TestConstants.THIS_IS_BAZEL) {
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
      assertEquals("k8", master.getTargetConfigurations().get(0).getCpu());
      assertEquals("piii", master.getTargetConfigurations().get(1).getCpu());
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
          throws InvalidConfigurationException {
        for (Class<? extends Fragment> fragmentType : dependsOn) {
          env.getFragment(buildOptions, fragmentType);
        }
        return new Fragment() {

        };
      }
    };
  }

  @Test
  public void testCycleInFragments() throws Exception {
    configurationFactory = new ConfigurationFactory(
        getAnalysisMock().createConfigurationCollectionFactory(),
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
    configurationFactory = new ConfigurationFactory(
        getAnalysisMock().createConfigurationCollectionFactory(),
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
  public void testGetOptionClass() throws Exception {
    BuildConfiguration config = create();
    // Directly defined option:
    assertEquals(BuildConfiguration.Options.class, config.getOptionClass("compilation_mode"));
    // Option defined in a fragment:
    assertEquals(CppOptions.class, config.getOptionClass("lipo"));
    // Unrecognized option:
    assertNull(config.getOptionClass("do_my_laundry"));
  }

  @Test
  public void testGetOptionValue() throws Exception {
    // Directly defined options:
    assertEquals(CompilationMode.DBG, create("-c", "dbg").getOptionValue("compilation_mode"));
    assertEquals(CompilationMode.OPT, create("-c", "opt").getOptionValue("compilation_mode"));

    // Options defined in a fragment:
    assertEquals(Boolean.TRUE, create("--force_pic")
        .getOptionValue("force_pic"));
    assertEquals(Boolean.FALSE, create("--noforce_pic")
        .getOptionValue("force_pic"));

    // Unrecognized option:
    assertNull(create().getOptionValue("do_my_dishes"));

    // Legitimately null option:
    assertNull(create().getOptionValue("test_filter"));
  }

  @Test
  public void testNoDistinctHostConfigurationUnsupportedWithDynamicConfigs() throws Exception {
    checkError(
        "--nodistinct_host_configuration does not currently work with dynamic configurations",
        "--nodistinct_host_configuration", "--experimental_dynamic_configs");
  }

  @Test
  public void testEqualsOrIsSupersetOf() throws Exception {
    BuildConfiguration config = create();
    BuildConfiguration trimmedConfig = config.clone(
        ImmutableSet.<Class<? extends Fragment>>of(CppConfiguration.class),
        TestRuleClassProvider.getRuleClassProvider());
    BuildConfiguration hostConfig = createHost();

    assertTrue(config.equalsOrIsSupersetOf(trimmedConfig));
    assertFalse(config.equalsOrIsSupersetOf(hostConfig));
    assertFalse(trimmedConfig.equalsOrIsSupersetOf(config));
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
}
