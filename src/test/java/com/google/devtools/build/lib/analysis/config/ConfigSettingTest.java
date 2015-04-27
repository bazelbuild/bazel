// Copyright 2015 Google Inc. All rights reserved.
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

import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.syntax.Label;

/**
 * Tests for {@link ConfigSetting}.
 */
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
   * Tests that a config_setting only matches build configurations where *all* of
   * its flag specifications match.
   */
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
  public void testLabel() throws Exception {
    writeSimpleExample();
    assertEquals(
        Label.parseAbsolute("//pkg:foo"),
        getConfigMatchingProvider("//pkg:foo").label());
  }

  /**
   * Tests that rule analysis fails on unknown options.
   */
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
  public void testLateBoundOptionDefaults() throws Exception {
    scratch.file("test/BUILD",
        "config_setting(",
        "    name = 'match',",
        "    values = {",
        "        'cpu': 'k8',",
        "    })");
    useConfiguration("--cpu=k8");
    assertTrue(getConfigMatchingProvider("//test:match").matches());
  }

  /**
   * Tests matching on multi-value attributes with key=value entries (e.g. --define).
   */
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
}
