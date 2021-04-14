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
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.actions.MutableActionGraph;
import com.google.devtools.build.lib.analysis.AnalysisFailureEvent;
import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId.TargetCompletedId;
import com.google.devtools.build.lib.buildtool.util.GoogleBuildIntegrationTestCase;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.util.ArrayList;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for action conflicts. */
@RunWith(TestParameterInjector.class)
public class OutputArtifactConflictTest extends GoogleBuildIntegrationTestCase {

  static class AnalysisFailureEventListener extends BlazeModule {

    private final List<TargetCompletedId> eventIds = new ArrayList<>();
    private final List<String> failedTargetNames = new ArrayList<>();

    @Override
    public void beforeCommand(CommandEnvironment env) {
      env.getEventBus().register(this);
    }

    @Subscribe
    public void onAnalysisFailure(AnalysisFailureEvent event) {
      eventIds.add(event.getEventId().getTargetCompleted());
      failedTargetNames.add(event.getFailedTarget().getLabel().toString());
    }
  }

  private final AnalysisFailureEventListener eventListener = new AnalysisFailureEventListener();

  @Override
  protected BlazeRuntime.Builder getRuntimeBuilder() throws Exception {
    return super.getRuntimeBuilder().addBlazeModule(eventListener);
  }

  @Test
  public void testArtifactPrefix(
      @TestParameter boolean keepGoing, @TestParameter boolean modifyBuildFile) throws Exception {
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
    assertThat(eventListener.failedTargetNames).isEmpty();

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
    assertThat(Iterables.size(events.errors())).isGreaterThan(1);
    if (keepGoing) {
      assertThat(eventListener.failedTargetNames).containsExactly("//x:y", "//x/y:y");
    } else {
      assertThat(eventListener.failedTargetNames).containsAnyOf("//x:y", "//x/y:y");
    }
  }

  @Test
  public void testAspectArtifactSharesPrefixWithTargetArtifact(
      @TestParameter boolean keepGoing, @TestParameter boolean modifyBuildFile) throws Exception {
    if (modifyBuildFile) {
      write("x/BUILD", "genrule(name = 'y', outs = ['y.out'], cmd = 'touch $@')");
    } else {
      write("x/BUILD", "genrule(name = 'y', outs = ['y.bad'], cmd = 'touch $@')");
    }
    write("x/y/BUILD", "cc_library(name = 'y')");
    write(
        "x/aspect.bzl",
        "def _aspect_impl(target, ctx):",
        "    if not getattr(ctx.rule.attr, 'outs', None):",
        "        return struct(output_groups = {})",
        "    conflict_outputs = list()",
        "    for out in ctx.rule.attr.outs:",
        "        if out.name[1:] == '.bad':",
        "            aspect_out = ctx.actions.declare_file(out.name[:1])",
        "            conflict_outputs.append(aspect_out)",
        "            cmd = 'echo %s > %s' % (out.name, aspect_out.path)",
        "            ctx.actions.run_shell(",
        "                outputs = [aspect_out],",
        "                command = cmd,",
        "            )",
        "    return [OutputGroupInfo(",
        "        files = depset(conflict_outputs)",
        "    )]",
        "",
        "my_aspect = aspect(implementation = _aspect_impl)");

    if (modifyBuildFile) {
      buildTarget("//x/y", "//x:y");
      write("x/BUILD", "genrule(name = 'y', outs = ['y.bad'], cmd = 'touch $@')");
    } else {
      buildTarget("//x/y");
    }
    assertNoEvents(events.errors());
    assertThat(eventListener.failedTargetNames).isEmpty();

    addOptions("--aspects=//x:aspect.bzl%my_aspect", "--output_groups=files");
    if (keepGoing) {
      addOptions("--keep_going");
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
    // When an aspect artifact's path is in aa prefix conflict with a target artifact's path, the
    // target artifact is created and only the aspect fails analysis.
    assertThat(Iterables.size(events.errors())).isGreaterThan(1);
    assertThat(eventListener.failedTargetNames).containsExactly("//x:y");
    assertThat(eventListener.eventIds.get(0).getAspect()).isEqualTo("//x:aspect.bzl%my_aspect");
  }

  @Test
  public void testAspectArtifactPrefix(
      @TestParameter boolean keepGoing, @TestParameter boolean modifyBuildFile) throws Exception {
    if (modifyBuildFile) {
      write(
          "x/BUILD",
          "genrule(name = 'y', outs = ['y.out'], cmd = 'touch $@')",
          "genrule(name = 'ydir', outs = ['y.dir'], cmd = 'touch $@')");
    } else {
      write(
          "x/BUILD",
          "genrule(name = 'y', outs = ['y.bad'], cmd = 'touch $@')",
          "genrule(name = 'ydir', outs = ['y.dir'], cmd = 'touch $@')");
    }
    write(
        "x/aspect.bzl",
        "def _aspect_impl(target, ctx):",
        "    if not getattr(ctx.rule.attr, 'outs', None):",
        "        return struct(output_groups = {})",
        "    conflict_outputs = list()",
        "    for out in ctx.rule.attr.outs:",
        "        if out.name[1:] == '.bad':",
        "            aspect_out = ctx.actions.declare_file(out.name[:1])",
        "            conflict_outputs.append(aspect_out)",
        "            cmd = 'echo %s > %s' % (out.name, aspect_out.path)",
        "            ctx.actions.run_shell(",
        "                outputs = [aspect_out],",
        "                command = cmd,",
        "            )",
        "        elif out.name[1:] == '.dir':",
        "            aspect_out = ctx.actions.declare_file(out.name[:1] + '/' + out.name)",
        "            conflict_outputs.append(aspect_out)",
        "            out_dir = aspect_out.path[:len(aspect_out.path) - len(out.name) + 1]",
        "            cmd = 'mkdir %s && echo %s > %s' % (out_dir, out.name, aspect_out.path)",
        "            ctx.actions.run_shell(",
        "                outputs = [aspect_out],",
        "                command = cmd,",
        "            )",
        "    return [OutputGroupInfo(",
        "        files = depset(conflict_outputs)",
        "    )]",
        "",
        "my_aspect = aspect(implementation = _aspect_impl)");

    if (modifyBuildFile) {
      buildTarget("//x:y", "//x:ydir");
      write(
          "x/BUILD",
          "genrule(name = 'y', outs = ['y.bad'], cmd = 'touch $@')",
          "genrule(name = 'ydir', outs = ['y.dir'], cmd = 'touch $@')");
    } else {
      buildTarget("//x:y");
    }
    assertNoEvents(events.errors());
    assertThat(eventListener.failedTargetNames).isEmpty();

    addOptions("--aspects=//x:aspect.bzl%my_aspect", "--output_groups=files");
    if (keepGoing) {
      addOptions("--keep_going");
    }
    try {
      // Skyframe full should throw an error here even if we just build //x:y. However, because our
      // testing infrastructure sets up lots of symlinks, Skyframe invalidates the //x/y action, and
      // so would not find a conflict here without re-evaluating //x/y. Note that in a real client,
      // requesting the //x/y target would not be necessary to throw an exception.
      buildTarget("//x:y", "//x:ydir");
      fail();
    } catch (BuildFailedException | ViewCreationFailedException e) {
      // Expected.
    }
    events.assertContainsError("output path 'blaze-out/");
    // Skip over config key string ...
    events.assertContainsError(
        "/bin/x/y' (belonging to //x:y) is a prefix of output path 'blaze-out");
    assertThat(events.errors()).hasSize(1);
    assertThat(eventListener.eventIds.get(0).getAspect()).isEqualTo("//x:aspect.bzl%my_aspect");
    if (keepGoing) {
      assertThat(eventListener.failedTargetNames).containsExactly("//x:y", "//x:ydir");
    } else {
      assertThat(eventListener.failedTargetNames).containsAnyOf("//x:y", "//x:ydir");
    }
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
    assertThat(eventListener.failedTargetNames).containsAnyOf("//x:y", "//x/y:y");
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
    assertThat(eventListener.failedTargetNames).containsAnyOf("//x:y", "//x/y:y");
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

  // Regression test for b/184944522.
  @Test
  public void testConflictErrorAndAnalysisError() throws Exception {
    write(
        "conflict/BUILD",
        "cc_library(name='x', srcs=['foo.cc'])",
        "cc_binary(name='_objs/x/foo.pic.o', srcs=['bar.cc'], "
            + "malloc = '//base:system_malloc')");
    write("conflict/foo.cc", "int main() { return 0; }");
    write("conflict/bar.cc", "int main() { return 0; }");
    write("x/BUILD", "sh_library(name = 'x', deps = ['//y:y'])");
    write("y/BUILD", "sh_library(name = 'y', visibility = ['//visibility:private'])");
    runtimeWrapper.addOptions("--keep_going");

    assertThrows(
        BuildFailedException.class,
        () -> buildTarget("//x:x", "//conflict:x", "//conflict:_objs/x/foo.pic.o"));
    events.assertContainsError(
        "file 'conflict/_objs/x/foo.pic.o' is generated by these conflicting actions:");
    // When two targets have conflicting artifacts, the first target named on the commandline "wins"
    // and is successfully built. All other targets fail analysis for conflicting with the first.
    assertThat(eventListener.failedTargetNames)
        .containsExactly("//x:x", "//conflict:_objs/x/foo.pic.o");
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
    // When targets have conflicting artifacts, one of them "wins" and is successfully built. All
    // other targets fail analysis for conflicting with the first.
    assertThat(eventListener.failedTargetNames).containsAtLeast("//x:y", "//x/y:y");
    assertThat(eventListener.failedTargetNames).hasSize(3);
    assertThat(eventListener.failedTargetNames)
        .containsAnyOf("//conflict:x", "//conflict:_objs/x/foo.pic.o");
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
    assertThat(eventListener.failedTargetNames).containsAnyOf("//foo:first", "//foo:second");
    eventListener.failedTargetNames.clear();

    e =
        assertThrows(
            ViewCreationFailedException.class, () -> buildTarget("//foo:first", "//foo:second"));
    assertThat(e)
        .hasCauseThat()
        .hasCauseThat()
        .isInstanceOf(MutableActionGraph.ActionConflictException.class);
    assertThat(eventListener.failedTargetNames).containsAnyOf("//foo:first", "//foo:second");
  }
}
