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
  protected boolean keepGoing() {
    return false;
  }

  @Test
  public void testEscaping() throws Exception {
    scratch.file(
        "test/skylark/extension.bzl",
        "load(\"@bazel_tools//tools/cpp:lib_cc_configure.bzl\", \"escape_string\", \"unescape_string\")",
        "def _impl(ctx):",
        "  pass",
        "def test(arg, **kwargs):",
        "  escaped = escape_string(arg)",
        "  result = unescape_string(escaped)",
        "  if result != arg:",
        "    fail(\"Could not reverse escaping {} -escape> {} -unescape> {}\".format(arg, escaped, result))",
        "  return native.rule(implementation = _impl)(**kwargs)");

    scratch.file(
        "test/skylark/BUILD",
        "load('/test/skylark/extension', 'test')",
        "test('\\my%simple;%%test!%%%for?%%escaping:%works/', name = \"test_target\")");

    getConfiguredTarget("//test/skylark:test_target");
  }
}