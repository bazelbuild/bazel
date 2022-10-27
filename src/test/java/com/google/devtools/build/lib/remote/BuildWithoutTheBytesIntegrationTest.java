// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote;

import static com.google.common.collect.Iterables.getOnlyElement;
import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.remote.util.IntegrationTestUtils.startWorker;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.dynamic.DynamicExecutionModule;
import com.google.devtools.build.lib.remote.util.IntegrationTestUtils.WorkerInstance;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.BlockWaitingModule;
import com.google.devtools.build.lib.runtime.BuildSummaryStatsModule;
import com.google.devtools.build.lib.standalone.StandaloneModule;
import com.google.devtools.build.lib.util.OS;
import java.io.IOException;
import org.junit.After;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Integration tests for Build without the Bytes. */
@RunWith(JUnit4.class)
public class BuildWithoutTheBytesIntegrationTest extends BuildWithoutTheBytesIntegrationTestBase {
  private WorkerInstance worker;

  @Override
  protected void setupOptions() throws Exception {
    super.setupOptions();

    if (worker == null) {
      worker = startWorker();
    }

    addOptions(
        "--remote_executor=grpc://localhost:" + worker.getPort(),
        "--remote_download_minimal",
        "--experimental_action_cache_store_output_metadata",
        "--dynamic_local_strategy=standalone",
        "--dynamic_remote_strategy=remote");
  }

  @Override
  protected void setDownloadToplevel() {
    addOptions("--remote_download_outputs=toplevel");
  }

  @Override
  protected void setDownloadAll() {
    addOptions("--remote_download_outputs=all");
  }

  @Override
  protected BlazeRuntime.Builder getRuntimeBuilder() throws Exception {
    return super.getRuntimeBuilder()
        .addBlazeModule(new BuildSummaryStatsModule())
        .addBlazeModule(new BlockWaitingModule());
  }

  @Override
  protected ImmutableList<BlazeModule> getSpawnModules() {
    return ImmutableList.<BlazeModule>builder()
        .addAll(super.getSpawnModules())
        .add(new StandaloneModule())
        .add(new RemoteModule())
        .add(new DynamicExecutionModule())
        .build();
  }

  @Override
  protected void assertOutputEquals(String realContent, String expectedContent) throws Exception {
    assertThat(realContent).isEqualTo(expectedContent);
  }

  @Override
  protected void assertOutputContains(String content, String contains) throws Exception {
    assertThat(content).contains(contains);
  }

  @After
  public void tearDown() throws IOException {
    if (worker != null) {
      worker.stop();
    }
  }

  @Test
  public void executeRemotely_actionFails_outputsAreAvailableLocallyForDebuggingPurpose()
      throws Exception {
    write(
        "a/BUILD",
        "genrule(",
        "  name = 'fail',",
        "  srcs = [],",
        "  outs = ['fail.txt'],",
        "  cmd = 'echo foo > $@ && exit 1',",
        ")");

    assertThrows(BuildFailedException.class, () -> buildTarget("//a:fail"));

    assertOnlyOutputContent("//a:fail", "fail.txt", "foo\n");
  }

  @Test
  public void intermediateOutputsAreInputForInternalActions_prefetchIntermediateOutputs()
      throws Exception {
    // Disable on Windows since it seems that template is not supported there.
    if (OS.getCurrent() == OS.WINDOWS) {
      return;
    }
    // Test that a remotely stored output that's an input to a internal action
    // (ctx.actions.expand_template) is staged lazily for action execution.
    write(
        "a/substitute_username.bzl",
        "def _substitute_username_impl(ctx):",
        "    ctx.actions.expand_template(",
        "        template = ctx.file.template,",
        "        output = ctx.outputs.out,",
        "        substitutions = {",
        "            '{USERNAME}': ctx.attr.username,",
        "        },",
        "    )",
        "",
        "substitute_username = rule(",
        "    implementation = _substitute_username_impl,",
        "    attrs = {",
        "        'username': attr.string(mandatory = True),",
        "        'template': attr.label(",
        "            allow_single_file = True,",
        "            mandatory = True,",
        "        ),",
        "    },",
        "    outputs = {'out': '%{name}.txt'},",
        ")");
    write(
        "a/BUILD",
        "load(':substitute_username.bzl', 'substitute_username')",
        "genrule(",
        "    name = 'generate-template',",
        "    cmd = 'echo -n \"Hello {USERNAME}!\" > $@',",
        "    outs = ['template.txt'],",
        "    srcs = [],",
        ")",
        "",
        "substitute_username(",
        "    name = 'substitute-buchgr',",
        "    username = 'buchgr',",
        "    template = ':generate-template',",
        ")");

    buildTarget("//a:substitute-buchgr");

    // The genrule //a:generate-template should run remotely and //a:substitute-buchgr should be a
    // internal action running locally.
    events.assertContainsInfo("3 processes: 2 internal, 1 remote");
    Artifact intermediateOutput = getOnlyElement(getArtifacts("//a:generate-template"));
    assertThat(intermediateOutput.getPath().exists()).isTrue();
    assertOnlyOutputContent("//a:substitute-buchgr", "substitute-buchgr.txt", "Hello buchgr!");
  }

  @Test
  public void changeOutputMode_invalidateActions() throws Exception {
    write(
        "a/BUILD",
        "genrule(",
        "  name = 'foo',",
        "  srcs = [],",
        "  outs = ['foo.txt'],",
        "  cmd = 'echo foo > $@',",
        ")",
        "",
        "genrule(",
        "  name = 'foobar',",
        "  srcs = [':foo'],",
        "  outs = ['foobar.txt'],",
        "  cmd = 'cat $(location :foo) > $@ && echo bar > $@',",
        ")");
    ActionEventCollector actionEventCollector = new ActionEventCollector();
    runtimeWrapper.registerSubscriber(actionEventCollector);
    buildTarget("//a:foobar");
    // 3 = workspace status action + //:foo + //:foobar
    assertThat(actionEventCollector.getNumActionNodesEvaluated()).isEqualTo(3);
    actionEventCollector.clear();
    events.clear();

    setDownloadAll();
    buildTarget("//a:foobar");

    // Changing output mode should invalidate SkyFrame's in-memory caching and make it re-evaluate
    // the action nodes.
    assertThat(actionEventCollector.getNumActionNodesEvaluated()).isEqualTo(3);
    events.assertContainsInfo("2 processes: 2 remote cache hit");
  }

  @Test
  public void symlinkToGeneratedFile() throws Exception {
    write(
        "a/defs.bzl",
        "def _impl(ctx):",
        "  if ctx.attr.chain_length < 1:",
        "    fail('chain_length must be > 0')",
        "",
        "  file = ctx.actions.declare_file(ctx.label.name + '.file')",
        // Use ctx.actions.run_shell instead of ctx.actions.write, so that it runs remotely.
        "  ctx.actions.run_shell(",
        "    outputs = [file],",
        "    command = 'echo hello > $1',",
        "    arguments = [file.path],",
        "  )",
        "",
        "  for i in range(ctx.attr.chain_length):",
        "    sym = ctx.actions.declare_file(ctx.label.name + '.sym' + str(i))",
        "    ctx.actions.symlink(output = sym, target_file = file)",
        "    file = sym",
        "",
        "  out = ctx.actions.declare_file(ctx.label.name + '.out')",
        "  ctx.actions.run_shell(",
        "    inputs = [sym],",
        "    outputs = [out],",
        "    command = '[[ hello == $(cat $1) ]] && touch $2',",
        "    arguments = [sym.path, out.path],",
        "    execution_requirements = {'no-remote': ''} if ctx.attr.local else {},",
        "  )",
        "",
        "  return DefaultInfo(files = depset([out]))",
        "",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'chain_length': attr.int(),",
        "    'local': attr.bool(),",
        "  },",
        ")");

    write(
        "a/BUILD",
        "load(':defs.bzl', 'my_rule')",
        "",
        "my_rule(name = 'one_local', local = True, chain_length = 1)",
        "my_rule(name = 'two_local', local = True, chain_length = 2)",
        "my_rule(name = 'one_remote', local = False, chain_length = 1)",
        "my_rule(name = 'two_remote', local = False, chain_length = 2)");

    buildTarget("//a:one_local", "//a:two_local", "//a:one_remote", "//a:two_remote");
  }

  @Test
  public void symlinkToDirectory() throws Exception {
    write(
        "a/defs.bzl",
        "def _impl(ctx):",
        "  if ctx.attr.chain_length < 1:",
        "    fail('chain_length must be > 0')",
        "",
        "  dir = ctx.actions.declare_directory(ctx.label.name + '.dir')",
        "  ctx.actions.run_shell(",
        "    outputs = [dir],",
        "    command = 'mkdir -p $1/some/path && echo hello > $1/some/path/inside.txt',",
        "    arguments = [dir.path],",
        "  )",
        "",
        "  for i in range(ctx.attr.chain_length):",
        "    sym = ctx.actions.declare_directory(ctx.label.name + '.sym' + str(i))",
        "    ctx.actions.symlink(output = sym, target_file = dir)",
        "    dir = sym",
        "",
        "  out = ctx.actions.declare_file(ctx.label.name + '.out')",
        "  ctx.actions.run_shell(",
        "    inputs = [sym],",
        "    outputs = [out],",
        "    command = '[[ hello == $(cat $1/some/path/inside.txt) ]] && touch $2',",
        "    arguments = [sym.path, out.path],",
        "    execution_requirements = {'no-remote': ''} if ctx.attr.local else {},",
        "  )",
        "",
        "  return DefaultInfo(files = depset([out]))",
        "",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'chain_length': attr.int(),",
        "    'local': attr.bool()",
        "  },",
        ")");

    write(
        "a/BUILD",
        "load(':defs.bzl', 'my_rule')",
        "",
        "my_rule(name = 'one_local', local = True, chain_length = 1)",
        "my_rule(name = 'two_local', local = True, chain_length = 2)",
        "my_rule(name = 'one_remote', local = False, chain_length = 1)",
        "my_rule(name = 'two_remote', local = False, chain_length = 2)");

    buildTarget("//a:one_local", "//a:two_local", "//a:one_remote", "//a:two_remote");
  }

  @Test
  public void symlinkToNestedFile() throws Exception {
    write(
        "a/defs.bzl",
        "def _impl(ctx):",
        "  if ctx.attr.chain_length < 1:",
        "    fail('chain_length must be > 0')",
        "",
        "  dir = ctx.actions.declare_directory(ctx.label.name + '.dir')",
        "  file = ctx.actions.declare_file(ctx.label.name + '.dir/some/path/inside.txt')",
        "  ctx.actions.run_shell(",
        "    outputs = [dir, file],",
        "    command = 'mkdir -p $1/some/path && echo hello > $1/some/path/inside.txt',",
        "    arguments = [dir.path],",
        "  )",
        "",
        "  for i in range(ctx.attr.chain_length):",
        "    sym = ctx.actions.declare_file(ctx.label.name + '.sym' + str(i))",
        "    ctx.actions.symlink(output = sym, target_file = file)",
        "    file = sym",
        "",
        "  out = ctx.actions.declare_file(ctx.label.name + '.out')",
        "  ctx.actions.run_shell(",
        "    inputs = [sym],",
        "    outputs = [out],",
        "    command = '[[ hello == $(cat $1) ]] && touch $2',",
        "    arguments = [sym.path, out.path],",
        "    execution_requirements = {'no-remote': ''} if ctx.attr.local else {},",
        "  )",
        "",
        "  return DefaultInfo(files = depset([out]))",
        "",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'chain_length': attr.int(),",
        "    'local': attr.bool(),",
        "  },",
        ")");

    write(
        "a/BUILD",
        "load(':defs.bzl', 'my_rule')",
        "",
        "my_rule(name = 'one_local', local = True, chain_length = 1)",
        "my_rule(name = 'two_local', local = True, chain_length = 2)",
        "my_rule(name = 'one_remote', local = False, chain_length = 1)",
        "my_rule(name = 'two_remote', local = False, chain_length = 2)");

    buildTarget("//a:one_local", "//a:two_local", "//a:one_remote", "//a:two_remote");
  }

  @Test
  public void symlinkToNestedDirectory() throws Exception {
    write(
        "a/defs.bzl",
        "def _impl(ctx):",
        "  if ctx.attr.chain_length < 1:",
        "    fail('chain_length must be > 0')",
        "",
        "  dir = ctx.actions.declare_directory(ctx.label.name + '.dir')",
        "  subdir = ctx.actions.declare_directory(ctx.label.name + '.dir/some/path')",
        "  ctx.actions.run_shell(",
        "    outputs = [dir, subdir],",
        "    command = 'mkdir -p $1/some/path && echo hello > $1/some/path/inside.txt',",
        "    arguments = [dir.path],",
        "  )",
        "",
        "  for i in range(ctx.attr.chain_length):",
        "    sym = ctx.actions.declare_directory(ctx.label.name + '.sym' + str(i))",
        "    ctx.actions.symlink(output = sym, target_file = subdir)",
        "    subdir = sym",
        "",
        "  out = ctx.actions.declare_file(ctx.label.name + '.out')",
        "  ctx.actions.run_shell(",
        "    inputs = [sym],",
        "    outputs = [out],",
        "    command = '[[ hello == $(cat $1/inside.txt) ]] && touch $2',",
        "    arguments = [sym.path, out.path],",
        "    execution_requirements = {'no-remote': ''} if ctx.attr.local else {},",
        "  )",
        "",
        "  return DefaultInfo(files = depset([out]))",
        "",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'chain_length': attr.int(),",
        "    'local': attr.bool(),",
        "  },",
        ")");

    write(
        "a/BUILD",
        "load(':defs.bzl', 'my_rule')",
        "",
        "my_rule(name = 'one_local', local = True, chain_length = 1)",
        "my_rule(name = 'two_local', local = True, chain_length = 2)",
        "my_rule(name = 'one_remote', local = False, chain_length = 1)",
        "my_rule(name = 'two_remote', local = False, chain_length = 2)");

    buildTarget("//a:one_local", "//a:two_local", "//a:one_remote", "//a:two_remote");
  }
}
