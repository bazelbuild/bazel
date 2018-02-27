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
package com.google.devtools.build.lib.skylark.cpp;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.lib.analysis.OutputGroupProvider.INTERNAL_SUFFIX;
import static org.junit.Assert.fail;

import com.google.common.base.Joiner;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.OutputGroupProvider;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.configuredtargets.FileConfiguredTarget;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesProvider;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.AttributeContainer;
import com.google.devtools.build.lib.packages.BuildFileContainsErrorsException;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.SkylarkProvider;
import com.google.devtools.build.lib.packages.SkylarkProvider.SkylarkKey;
import com.google.devtools.build.lib.skyframe.PackageFunction;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.SkylarkImportLookupFunction;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkList.MutableList;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.skyframe.InMemoryMemoizingEvaluator;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Integration tests for Skylark CPP.
 */
@RunWith(JUnit4.class)
public class SkylarkToolchainTest extends BuildViewTestCase {

  @Before
  public void createLib() throws Exception {
    Path path = FileSystems.getDefault().getPath("tools/cpp/lib_cc_configure.bzl");
    List<String> lineList = Files.readAllLines(path, StandardCharsets.UTF_8);
    String[] lines = new String[lineList.size()];
    lineList.toArray(lines);
    scratch.file(
        "tools/cpp/lib_cc_configure.bzl",
        lines);
    scratch.file(
        "tools/cpp/BUILD",
        "");
  }

  @Before
  public void createEmptyRule() throws Exception {
    scratch.file(
        "test/skylark/empty.bzl",
        "def _empty(ctx):",
        "  pass",
        "empty = rule(implementation = _empty)");
  }

  @Before
  public void createAssertions() throws Exception {
    scratch.file(
      "test/skylark/assert.bzl",
      "def assertIsNone(arg):",
      "  if arg != None:",
      "    fail('Value is %s. Expected None.' % arg)",
      "def assertListEqual(list1, list2):",
      "  eq = len(list1) == len(list2)",
      "  if eq:",
      "    for i, l in enumerate(list1):",
      "      r = list2[i]",
      "      if l != r:",
      "        eq = False",
      "        break",
      "  if eq:",
      "    return",
      "  fail('Lists do not match {} <> {}'.format(list1, list2))",
      "def assertTrue(arg):",
      "  if not bool(arg):",
      "    fail('Value is %s. Expected convertible to True.' % arg)");
  }

  @Test
  public void testEscaping() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "load('//tools/cpp:lib_cc_configure.bzl', 'escape_string', 'unescape_string')",
        "load('//test/skylark:empty.bzl', 'empty')",
        "def test(arg, **kwargs):",
        "  escaped = escape_string(arg)",
        "  result = unescape_string(escaped)",
        "  if result != arg:",
        "    fail('Could not reverse escaping {} -escape> {} -unescape> {}'.format(arg, escaped, result))",
        "  return empty(**kwargs)");

    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:extension.bzl', 'test')",
        "test('\\my%simple;%%test!%%%for?%%escaping:%works/', name = 'test_target')");

    getTarget("//test/skylark:test_target");
  }

  @Test
  public void testError() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "load('//tools/cpp:lib_cc_configure.bzl', 'get_escaping_error')",
        "load('//test/skylark:assert.bzl', 'assertIsNone', 'assertTrue')",
        "load('//test/skylark:empty.bzl', 'empty')",
        "def test(**kwargs):",
        "  error = get_escaping_error('foo')",
        "  assertIsNone(error)",
        "  error = get_escaping_error('foo;', additionals = ';')",
        "  assertTrue(error)",
        "  error = get_escaping_error('foo%a')",
        "  assertTrue(error)",
        "  error = get_escaping_error('foo%')",
        "  assertTrue(error)",
        "  error = get_escaping_error('foo%;bar', additionals = ';')",
        "  assertIsNone(error)",
        "  empty(**kwargs)");

    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:extension.bzl', 'test')",
        "test(name = 'test_target')");

    getTarget("//test/skylark:test_target");
  }

  @Test
  public void testErrorFail() throws Exception {

    scratch.file(
      "test/skylark/extension.bzl",
      "load('//tools/cpp:lib_cc_configure.bzl', 'get_escaping_error')",
      "load('//test/skylark:empty.bzl', 'empty')",
      "def test(**kwargs):",
      "  get_escaping_error('foo', additionals = '%')",
      "  empty(**kwargs)");

    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark/extension.bzl', 'test')",
        "test(name = 'test_target')");
    
    ConfiguredTarget target = getConfiguredTarget("//test/skylark:test_target");
    assertThat(target).isNull();
    assertThat(view.hasErrors(target)).isTrue();
  }

  @Test
  public void testEscapedSplit() throws Exception {
    scratch.file(
      "test/skylark/extension.bzl",
      "load('//tools/cpp:lib_cc_configure.bzl', 'split_escaped_by_sep')",
      "load('//test/skylark:assert.bzl', 'assertListEqual')",
      "load('//test/skylark:empty.bzl', 'empty')",
      "def test(**kwargs):",
      "  result = split_escaped_by_sep('test:file', sep = ':')",
      "  assertListEqual(result, ['test', 'file'])",
      "  empty(**kwargs)");

    scratch.file(
        "test/skylark/BUILD",
        "load('//test/skylark:extension.bzl', 'test')",
        "test(name = 'test_target')");

    getTarget("//test/skylark:test_target");
  }
}