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
import static org.junit.Assert.assertTrue;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.config.ConfigRuleClasses.ConfigSettingRule;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.java.Jvm;
import com.google.devtools.build.lib.rules.python.PythonConfiguration;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.common.options.Option;
import java.util.Map;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link ConfigSetting}.
 */
@RunWith(JUnit4.class)
public class ConfigSettingTest extends BuildViewTestCase {

  /**
   * Test option that has its null default overridden by its fragment.
   */
  public static class LateBoundTestOptions extends FragmentOptions {
    public LateBoundTestOptions() {}

    @Option(name = "opt_with_default", defaultValue = "null")
    public String optwithDefault;
  }

  private static class LateBoundTestOptionsFragment extends BuildConfiguration.Fragment {
    @Override
    public Map<String, Object> lateBoundOptionDefaults() {
      return ImmutableMap.<String, Object>of("opt_with_default", "overridden");
    }
  }

  private static class LateBoundTestOptionsLoader implements ConfigurationFragmentFactory {
    @Override
    public BuildConfiguration.Fragment create(ConfigurationEnvironment env,
        BuildOptions buildOptions) throws InvalidConfigurationException {
      return new LateBoundTestOptionsFragment();
    }

    @Override
    public Class<? extends BuildConfiguration.Fragment> creates() {
      return LateBoundTestOptionsFragment.class;
    }

    @Override
    public ImmutableSet<Class<? extends FragmentOptions>> requiredOptions() {
      return ImmutableSet.<Class<? extends FragmentOptions>>of(LateBoundTestOptions.class);
    }
  }

  /**
   * Test option which is private.
   */
  public static class InternalTestOptions extends FragmentOptions {
    public InternalTestOptions() {}

    @Option(name = "internal_option", defaultValue = "super secret", category = "internal")
    public String optwithDefault;
  }

  private static class InternalTestOptionsFragment extends BuildConfiguration.Fragment {}

  private static class InternalTestOptionsLoader implements ConfigurationFragmentFactory {
    @Override
    public BuildConfiguration.Fragment create(ConfigurationEnvironment env,
        BuildOptions buildOptions) throws InvalidConfigurationException {
      return new InternalTestOptionsFragment();
    }

    @Override
    public Class<? extends BuildConfiguration.Fragment> creates() {
      return InternalTestOptionsFragment.class;
    }

    @Override
    public ImmutableSet<Class<? extends FragmentOptions>> requiredOptions() {
      return ImmutableSet.<Class<? extends FragmentOptions>>of(InternalTestOptions.class);
    }
  }

  @Override
  protected ConfiguredRuleClassProvider getRuleClassProvider() {
    ConfiguredRuleClassProvider.Builder builder = new ConfiguredRuleClassProvider.Builder();
    TestRuleClassProvider.addStandardRules(builder);
    builder.addConfigurationOptions(LateBoundTestOptions.class);
    builder.addConfigurationFragment(new LateBoundTestOptionsLoader());
    builder.addConfigurationOptions(InternalTestOptions.class);
    builder.addConfigurationFragment(new InternalTestOptionsLoader());
    return builder.build();
  }

  private void writeSimpleExample() throws Exception {
    scratch.file("pkg/BUILD",
        "config_setting(",
        "    name = 'foo',",
        "    values = {",
        "        'compilation_mode': 'dbg',",
        "        'stamp': '1',",
        "    })");
  }

  private ConfigMatchingProvider getConfigMatchingProvider(String label) throws Exception {
    return getConfiguredTarget(label).getProvider(ConfigMatchingProvider.class);
  }

  /**
   * Tests that a config_setting only matches build configurations where *all* of
   * its flag specifications match.
   */
  @Test
  public void matchingCriteria() throws Exception {
    writeSimpleExample();

    // First flag mismatches:
    useConfiguration("-c", "opt", "--stamp");
    assertFalse(getConfigMatchingProvider("//pkg:foo").matches());

    // Second flag mismatches:
    useConfiguration("-c", "dbg", "--nostamp");
    assertFalse(getConfigMatchingProvider("//pkg:foo").matches());

    // Both flags mismatch:
    useConfiguration("-c", "opt", "--nostamp");
    assertFalse(getConfigMatchingProvider("//pkg:foo").matches());

    // Both flags match:
    useConfiguration("-c", "dbg", "--stamp");
    assertTrue(getConfigMatchingProvider("//pkg:foo").matches());
  }

  /**
   * Tests that {@link ConfigMatchingProvider#label} is correct.
   */
  @Test
  public void labelGetter() throws Exception {
    writeSimpleExample();
    assertEquals(
        Label.parseAbsolute("//pkg:foo"),
        getConfigMatchingProvider("//pkg:foo").label());
  }

  /**
   * Tests that rule analysis fails on unknown options.
   */
  @Test
  public void unknownOption() throws Exception {
    checkError("foo", "badoption",
        "unknown option: 'not_an_option'",
        "config_setting(",
        "    name = 'badoption',",
        "    values = {'not_an_option': 'bar'})");
  }

  /**
   * Tests that rule analysis fails on internal options.
   */
  @Test
  public void internalOption() throws Exception {
    checkError("foo", "badoption",
        "unknown option: 'internal_option'",
        "config_setting(",
        "    name = 'badoption',",
        "    values = {'internal_option': 'bar'})");
  }

  /**
   * Tests that rule analysis fails on invalid option values.
   */
  @Test
  public void invalidOptionValue() throws Exception {
    checkError("foo", "badvalue",
        "Not a valid compilation mode: 'baz'",
        "config_setting(",
        "    name = 'badvalue',",
        "    values = {'compilation_mode': 'baz'})");
  }

  /**
   * Tests that when the first option is valid but the config_setting doesn't match,
   * remaining options are still validity-checked.
   */
  @Test
  public void invalidOptionFartherDown() throws Exception {
    checkError("foo", "badoption",
        "unknown option: 'not_an_option'",
        "config_setting(",
        "    name = 'badoption',",
        "    values = {",
        "        'compilation_mode': 'opt',",
        "        'not_an_option': 'bar',",
        "    })");
  }

  /**
   * Tests that *some* settings must be specified.
   */
  @Test
  public void emptySettings() throws Exception {
    checkError("foo", "empty",
        "//foo:empty: no settings specified",
        "config_setting(",
        "    name = 'empty',",
        "    values = {})");
  }

  /**
   * Tests {@link BuildConfiguration.Fragment#lateBoundOptionDefaults} options (options
   * that take alternative defaults from what's specified in {@link
   * com.google.devtools.common.options.Option#defaultValue}).
   */
  @Test
  public void lateBoundOptionDefaults() throws Exception {
    scratch.file("test/BUILD",
        "config_setting(",
        "    name = 'match',",
        "    values = { 'opt_with_default': 'overridden' }",
        ")");
    assertTrue(getConfigMatchingProvider("//test:match").matches());
  }

  /**
   * Tests matching on multi-value attributes with key=value entries (e.g. --define).
   */
  @Test
  public void multiValueDict() throws Exception {
    scratch.file("test/BUILD",
        "config_setting(",
        "    name = 'match',",
        "    values = {",
        "        'define': 'foo=bar',",
        "    })");

    useConfiguration("");
    assertFalse(getConfigMatchingProvider("//test:match").matches());
    useConfiguration("--define", "foo=bar");
    assertTrue(getConfigMatchingProvider("//test:match").matches());
    useConfiguration("--define", "foo=baz");
    assertFalse(getConfigMatchingProvider("//test:match").matches());
    useConfiguration("--define", "foo=bar", "--define", "bar=baz");
    assertTrue(getConfigMatchingProvider("//test:match").matches());
    useConfiguration("--define", "foo=bar", "--define", "bar=baz", "--define", "foo=nope");
    assertFalse(getConfigMatchingProvider("//test:match").matches());
    useConfiguration("--define", "foo=nope", "--define", "bar=baz", "--define", "foo=bar");
    assertTrue(getConfigMatchingProvider("//test:match").matches());
  }

  /**
   * Tests matching on multi-value attributes with primitive values.
   */
  @Test
  public void multiValueList() throws Exception {
    scratch.file("test/BUILD",
        "config_setting(",
        "    name = 'match',",
        "    values = {",
        "        'copt': '-Dfoo',",
        "    })");

    useConfiguration("");
    assertFalse(getConfigMatchingProvider("//test:match").matches());
    useConfiguration("--copt", "-Dfoo");
    assertTrue(getConfigMatchingProvider("//test:match").matches());
    useConfiguration("--copt", "-Dbar");
    assertFalse(getConfigMatchingProvider("//test:match").matches());
    useConfiguration("--copt", "-Dfoo", "--copt", "-Dbar");
    assertTrue(getConfigMatchingProvider("//test:match").matches());
    useConfiguration("--copt", "-Dbar", "--copt", "-Dfoo");
    assertTrue(getConfigMatchingProvider("//test:match").matches());
  }

  @Test
  public void selectForDefaultCrosstoolTop() throws Exception {
    String crosstoolTop = TestConstants.TOOLS_REPOSITORY + "//tools/cpp:toolchain";
    scratchConfiguredTarget("a", "a",
        "config_setting(name='cs', values={'crosstool_top': '" + crosstoolTop + "'})",
        "sh_library(name='a', srcs=['a.sh'], deps=select({':cs': []}))");
  }

  @Test
  public void selectForDefaultGrteTop() throws Exception {
    scratchConfiguredTarget("a", "a",
        "config_setting(name='cs', values={'grte_top': 'default'})",
        "sh_library(name='a', srcs=['a.sh'], deps=select({':cs': []}))");
  }

  @Test
  public void requiredConfigFragmentMatcher() throws Exception {
    scratch.file("test/BUILD",
        "config_setting(",
        "    name = 'match',",
        "    values = {",
        "        'copt': '-Dfoo',",
        "        'javacopt': '-Dbar'",
        "    })");

    Map<String, Class<? extends BuildConfiguration.Fragment>> map = ImmutableMap.of(
        "copt", CppConfiguration.class,
        "unused", PythonConfiguration.class,
        "javacopt", Jvm.class
    );
    assertThat(
        ConfigSettingRule.requiresConfigurationFragments((Rule) getTarget("//test:match"), map))
        .containsExactly(CppConfiguration.class, Jvm.class);
  }
}
