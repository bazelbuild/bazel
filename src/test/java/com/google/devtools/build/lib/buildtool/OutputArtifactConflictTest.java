// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.buildtool;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertNoEvents;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.fail;

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.actions.MutableActionGraph;
import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.buildtool.util.GoogleBuildIntegrationTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for action conflicts. */
@RunWith(JUnit4.class)
public class OutputArtifactConflictTest extends GoogleBuildIntegrationTestCase {
  private void runArtifactPrefix(boolean keepGoing, boolean modifyBuildFile) throws Exception {
    if (modifyBuildFile) {
      write("x/BUILD", "cc_library(name = 'y', srcs = ['y.cc'])");
    } else {
      write("x/BUILD", "cc_binary(name = 'y', srcs = ['y.cc'], malloc = '//base:system_malloc')");
    }
    write("x/y/BUILD", "cc_library(name = 'y')");
    write("x/y.cc", "int main() { return 0; }");

    if (modifyBuildFile) {
      buildTarget("//x/y", "//x:y");
      write("x/BUILD", "cc_binary(name = 'y', srcs = ['y.cc'], malloc = '//base:system_malloc')");
    } else {
      buildTarget("//x/y");
    }
    assertNoEvents(events.errors());

    if (keepGoing) {
      runtimeWrapper.addOptions("--keep_going");
    }
    try {
      // Skyframe full should throw an error here even if we just build //x:y. However, because our
      // testing infrastructure sets up lots of symlinks, Skyframe invalidates the //x/y action, and
      // so would not find a conflict here without re-evaluating //x/y. Note that in a real client,
      // requesting the //x/y target would not be necessary to throw an exception.
      buildTarget("//x:y", "//x/y");
      fail();
    } catch (BuildFailedException | ViewCreationFailedException e) {
      // Expected.
    }
    events.assertContainsError("output path 'blaze-out/");
    // Skip over config key string ...
    events.assertContainsError(
        "/bin/x/y' (belonging to //x:y) is a prefix of output path 'blaze-out");
    if (keepGoing) {
      assertThat(Iterables.size(events.errors())).isGreaterThan(1);
    } else {
      assertThat(events.errors()).hasSize(1);
    }
  }

  @Test
  public void testArtifactPrefix_KeepGoing() throws Exception {
    runArtifactPrefix(true, false);
  }

  @Test
  public void testArtifactPrefix_NoKeepGoing() throws Exception {
    runArtifactPrefix(false, false);
  }

  @Test
  public void testArtifactPrefix_KeepGoing_ModifyBuildFile() throws Exception {
    runArtifactPrefix(true, true);
  }

  @Test
  public void testArtifactPrefix_NoKeepGoing_ModifyBuildFile() throws Exception {
    runArtifactPrefix(false, true);
  }

  @Test
  public void testInvalidatedConflict() throws Exception {
    write("x/BUILD", "cc_binary(name = 'y', srcs = ['y.cc'], malloc = '//base:system_malloc')");
    write("x/y/BUILD", "cc_library(name = 'y')");
    write("x/y.cc", "int main() { return 0; }");
    try {
      buildTarget("//x:y", "//x/y");
      fail();
    } catch (BuildFailedException | ViewCreationFailedException e) {
      // Expected.
    }
    write("x/BUILD", "# no conflict");
    events.clear();
    buildTarget("//x/y");
    events.assertNoWarningsOrErrors();
  }

  @Test
  public void testNewTargetConflict() throws Exception {
    write("x/BUILD", "cc_binary(name = 'y', srcs = ['y.cc'], malloc = '//base:system_malloc')");
    write("x/y/BUILD", "cc_library(name = 'y')");
    write("x/y.cc", "int main() { return 0; }");
    buildTarget("//x/y");
    events.assertNoWarningsOrErrors();
    try {
      buildTarget("//x:y", "//x/y");
      fail();
    } catch (BuildFailedException | ViewCreationFailedException e) {
      // Expected.
    }
  }

  @Test
  public void testTwoOverlappingBuildsHasNoConflict() throws Exception {
    write("x/BUILD", "cc_binary(name = 'y', srcs = ['y.cc'], malloc = '//base:system_malloc')");
    write("x/y/BUILD", "cc_library(name = 'y')");
    write("x/y.cc", "int main() { return 0; }");
    buildTarget("//x/y");
    events.assertNoWarningsOrErrors();
    buildTarget("//x:y");
    events.assertNoWarningsOrErrors();

    // Verify that together they fail, even though no new targets have been analyzed
    try {
      buildTarget("//x:y", "//x/y");
      fail();
    } catch (BuildFailedException | ViewCreationFailedException e) {
      // Expected.
    }
    events.clear();

    // Verify that they still don't fail individually, so no state remains
    buildTarget("//x/y");
    events.assertNoWarningsOrErrors();
    buildTarget("//x:y");
    events.assertNoWarningsOrErrors();
  }

  @Test
  public void testFailingTargetsDoNotCauseActionConflicts() throws Exception {
    write(
        "x/bad_rule.bzl",
        "def _impl(ctx):",
        "  return list().this_method_does_not_exist()",
        "bad_rule = rule(_impl, attrs = {'deps': attr.label_list()})");
    write(
        "x/BUILD",
        "load('//x:bad_rule.bzl', 'bad_rule')",
        "cc_binary(name = 'y', srcs = ['y.cc'], malloc = '//base:system_malloc')",
        "bad_rule(name = 'bad', deps = [':y'])");
    write("x/y/BUILD", "cc_library(name = 'y')");
    write("x/y.cc", "int main() { return 0; }");

    runtimeWrapper.addOptions("--keep_going");
    try {
      buildTarget("//x:y", "//x/y");
      fail();
    } catch (ViewCreationFailedException e) {
      fail("Unexpected artifact prefix conflict: " + e);
    } catch (BuildFailedException e) {
      // Expected.
    }
  }

  @Test
  public void testMultipleConflictErrors() throws Exception {
    write(
        "conflict/BUILD",
        "cc_library(name='x', srcs=['foo.cc'])",
        "cc_binary(name='_objs/x/foo.pic.o', srcs=['bar.cc'], "
            + "malloc = '//base:system_malloc')");
    write("x/BUILD", "cc_binary(name = 'y', srcs = ['y.cc'], malloc = '//base:system_malloc')");
    write("x/y.cc", "int main() { return 0; }");
    write("conflict/foo.cc", "int main() { return 0; }");
    write("conflict/bar.cc", "int main() { return 0; }");
    write("x/y/BUILD", "cc_library(name = 'y')");
    runtimeWrapper.addOptions("--keep_going");
    assertThrows(
        BuildFailedException.class,
        () -> buildTarget("//x/y", "//x:y", "//conflict:x", "//conflict:_objs/x/foo.pic.o"));
    events.assertContainsError(
        "file 'conflict/_objs/x/foo.pic.o' is generated by these conflicting actions:");
    events.assertContainsError(
        "/bin/x/y' (belonging to //x:y) is a prefix of output path 'blaze-out");
  }

  @Test
  public void repeatedConflictBuild() throws Exception {
    write(
        "foo/conflict.bzl",
        "def _conflict_impl(ctx):",
        "  conflict_output = ctx.actions.declare_file('conflict_output')",
        "  other = ctx.actions.declare_file('other' + ctx.attr.other_name)",
        "  ctx.actions.run_shell(",
        "    outputs = [conflict_output, other],",
        "    command = 'touch %s %s' % (conflict_output.path, other.path)",
        "  )",
        "  return DefaultInfo(files = depset([conflict_output, other]))",
        "",
        "my_rule = rule(",
        "   implementation=_conflict_impl,",
        "   attrs = { 'other_name': attr.string() },",
        ")");
    write(
        "foo/BUILD",
        "load('//foo:conflict.bzl', 'my_rule')",
        "my_rule(name = 'first', other_name = '1')",
        "my_rule(name = 'second', other_name = '2')");
    ViewCreationFailedException e =
        assertThrows(
            ViewCreationFailedException.class, () -> buildTarget("//foo:first", "//foo:second"));
    assertThat(e)
        .hasCauseThat()
        .hasCauseThat()
        .isInstanceOf(MutableActionGraph.ActionConflictException.class);
    e =
        assertThrows(
            ViewCreationFailedException.class, () -> buildTarget("//foo:first", "//foo:second"));
    assertThat(e)
        .hasCauseThat()
        .hasCauseThat()
        .isInstanceOf(MutableActionGraph.ActionConflictException.class);
  }
}
