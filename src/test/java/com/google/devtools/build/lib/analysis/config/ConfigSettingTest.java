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
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.config.ConfigRuleClasses.ConfigSettingRule;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.java.Jvm;
import com.google.devtools.build.lib.rules.python.PythonConfiguration;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.Map;

/**
 * Tests for {@link ConfigSetting}.
 */
@RunWith(JUnit4.class)
public class ConfigSettingTest extends BuildViewTestCase {

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
   * Returns the default value of the given flag.
   */
  private Object flagDefault(String option) {
    Class<? extends OptionsBase> optionsClass = getTargetConfiguration().getOptionClass(option);
    return OptionsParser.newOptionsParser(optionsClass)
        .getOptions(optionsClass)
        .asMap()
        .get(option);
  }

  /**
   * Tests that a config_setting only matches build configurations where *all* of
   * its flag specifications match.
   */
  @Test
  public void testMatchingCriteria() throws Exception {
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
  public void testLabel() throws Exception {
    writeSimpleExample();
    assertEquals(
        Label.parseAbsolute("//pkg:foo"),
        getConfigMatchingProvider("//pkg:foo").label());
  }

  /**
   * Tests that rule analysis fails on unknown options.
   */
  @Test
  public void testUnknownOption() throws Exception {
    checkError("foo", "badoption",
        "unknown option: 'not_an_option'",
        "config_setting(",
        "    name = 'badoption',",
        "    values = {'not_an_option': 'bar'})");
  }

  /**
   * Tests that rule analysis fails on invalid option values.
   */
  @Test
  public void testInvalidOptionValue() throws Exception {
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
  public void testInvalidOptionFartherDown() throws Exception {
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
  public void testEmptySettings() throws Exception {
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
  public void testLateBoundOptionDefaults() throws Exception {
    String crosstoolCpuDefault = (String) getTargetConfiguration().getOptionValue("cpu");
    String crosstoolCompilerDefault = (String) getTargetConfiguration().getOptionValue("compiler");

    scratch.file("test/BUILD",
        "config_setting(",
        "    name = 'match',",
        "    values = {",
        "        'cpu': '" + crosstoolCpuDefault + "',",
        "        'compiler': '" + crosstoolCompilerDefault + "',", //'gcc-4.4.0',",
        "    })");

    assertTrue(getConfigMatchingProvider("//test:match").matches());
    assertNull(flagDefault("cpu"));
    assertNotNull(crosstoolCpuDefault);
    assertNull(flagDefault("compiler"));
    assertNotNull(crosstoolCompilerDefault);
  }

  /**
   * Tests matching on multi-value attributes with key=value entries (e.g. --define).
   */
  @Test
  public void testMultiValueDict() throws Exception {
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
  public void testMultiValueList() throws Exception {
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
  public void testSelectForDefaultCrosstoolTop() throws Exception {
    String crosstoolTop = TestConstants.TOOLS_REPOSITORY + "//tools/cpp:toolchain";
    scratchConfiguredTarget("a", "a",
        "config_setting(name='cs', values={'crosstool_top': '" + crosstoolTop + "'})",
        "sh_library(name='a', srcs=['a.sh'], deps=select({':cs': []}))");
  }

  @Test
  public void testRequiredConfigFragmentMatcher() throws Exception {
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
