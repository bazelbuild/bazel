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
import static com.google.devtools.build.lib.vfs.FileSystemUtils.readContent;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;
import static org.junit.Assume.assumeFalse;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.authandtls.credentialhelper.CredentialModule;
import com.google.devtools.build.lib.dynamic.DynamicExecutionModule;
import com.google.devtools.build.lib.remote.util.IntegrationTestUtils.WorkerInstance;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.BlockWaitingModule;
import com.google.devtools.build.lib.runtime.BuildSummaryStatsModule;
import com.google.devtools.build.lib.standalone.StandaloneModule;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Symlinks;
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
        "--dynamic_local_strategy=standalone",
        "--dynamic_remote_strategy=remote");
    // (b/281655526) Skymeld is incompatible.
    addOptions("--noexperimental_merged_skyframe_analysis_execution");
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
        .addBlazeModule(new RemoteModule())
        .addBlazeModule(new BuildSummaryStatsModule())
        .addBlazeModule(new BlockWaitingModule());
  }

  @Override
  protected ImmutableList<BlazeModule> getSpawnModules() {
    return ImmutableList.<BlazeModule>builder()
        .addAll(super.getSpawnModules())
        .add(new StandaloneModule())
        .add(new CredentialModule())
        .add(new DynamicExecutionModule())
        .build();
  }

  @Override
  protected void assertOutputEquals(Path path, String expectedContent) throws Exception {
    assertThat(readContent(path, UTF_8)).isEqualTo(expectedContent);
  }

  @Override
  protected void assertOutputContains(String content, String contains) throws Exception {
    assertThat(content).contains(contains);
  }

  @Override
  protected void evictAllBlobs() throws Exception {
    worker.restart();
  }

  @Override
  protected boolean hasAccessToRemoteOutputs() {
    return true;
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
    addOptions("--noincompatible_strict_conflict_checks");

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
    addOptions("--noincompatible_strict_conflict_checks");

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

  @Test
  public void outputSymlinkHandledGracefully() throws Exception {
    // Symlinks may not be supported on Windows
    assumeFalse(OS.getCurrent() == OS.WINDOWS);
    write(
        "a/defs.bzl",
        "def _impl(ctx):",
        "  out = ctx.actions.declare_symlink(ctx.label.name)",
        "  ctx.actions.run_shell(",
        "    inputs = [],",
        "    outputs = [out],",
        "    command = 'ln -s hello $1',",
        "    arguments = [out.path],",
        "  )",
        "  return DefaultInfo(files = depset([out]))",
        "",
        "my_rule = rule(",
        "  implementation = _impl,",
        ")");

    write("a/BUILD", "load(':defs.bzl', 'my_rule')", "", "my_rule(name = 'hello')");

    buildTarget("//a:hello");

    Path outputPath = getOutputPath("a/hello");
    assertThat(outputPath.stat(Symlinks.NOFOLLOW).isSymbolicLink()).isTrue();
  }

  @Test
  public void replaceOutputDirectoryWithFile() throws Exception {
    write(
        "a/defs.bzl",
        "def _impl(ctx):",
        "  dir = ctx.actions.declare_directory(ctx.label.name + '.dir')",
        "  ctx.actions.run_shell(",
        "    outputs = [dir],",
        "    command = 'touch $1/hello',",
        "    arguments = [dir.path],",
        "  )",
        "  return DefaultInfo(files = depset([dir]))",
        "",
        "my_rule = rule(",
        "  implementation = _impl,",
        ")");
    write("a/BUILD", "load(':defs.bzl', 'my_rule')", "", "my_rule(name = 'hello')");

    setDownloadToplevel();
    buildTarget("//a:hello");

    // Replace the existing output directory of the package with a file.
    // A subsequent build should remove this file and replace it with a
    // directory.
    Path outputPath = getOutputPath("a");
    outputPath.deleteTree();
    FileSystemUtils.writeContent(outputPath, new byte[] {1, 2, 3, 4, 5});

    buildTarget("//a:hello");
  }

  @Test
  public void remoteCacheEvictBlobs_whenPrefetchingInput_exitWithCode39() throws Exception {
    // Arrange: Prepare workspace and populate remote cache
    write(
        "a/BUILD",
        "genrule(",
        "  name = 'foo',",
        "  srcs = ['foo.in'],",
        "  outs = ['foo.out'],",
        "  cmd = 'cat $(SRCS) > $@',",
        ")",
        "genrule(",
        "  name = 'bar',",
        "  srcs = ['foo.out', 'bar.in'],",
        "  outs = ['bar.out'],",
        "  cmd = 'cat $(SRCS) > $@',",
        "  tags = ['no-remote-exec'],",
        ")");
    write("a/foo.in", "foo");
    write("a/bar.in", "bar");

    // Populate remote cache
    buildTarget("//a:bar");
    var bytes = readContent(getOutputPath("a/foo.out"));
    var hashCode = getDigestHashFunction().getHashFunction().hashBytes(bytes);
    getOutputPath("a/foo.out").delete();
    getOutputPath("a/bar.out").delete();
    getOutputBase().getRelative("action_cache").deleteTreesBelow();
    restartServer();

    // Clean build, foo.out isn't downloaded
    buildTarget("//a:bar");
    assertOutputDoesNotExist("a/foo.out");

    // Act: Evict blobs from remote cache and do an incremental build
    evictAllBlobs();
    write("a/bar.in", "updated bar");
    var error = assertThrows(BuildFailedException.class, () -> buildTarget("//a:bar"));

    // Assert: Exit code is 39
    assertThat(error)
        .hasMessageThat()
        .contains("Failed to fetch blobs because they do not exist remotely");
    assertThat(error).hasMessageThat().contains(String.format("%s/%s", hashCode, bytes.length));
    assertThat(error.getDetailedExitCode().getExitCode().getNumericExitCode()).isEqualTo(39);
  }

  @Test
  public void remoteCacheEvictBlobs_whenUploadingInput_exitWithCode39() throws Exception {
    // Arrange: Prepare workspace and populate remote cache
    write(
        "a/BUILD",
        "genrule(",
        "  name = 'foo',",
        "  srcs = ['foo.in'],",
        "  outs = ['foo.out'],",
        "  cmd = 'cat $(SRCS) > $@',",
        ")",
        "genrule(",
        "  name = 'bar',",
        "  srcs = ['foo.out', 'bar.in'],",
        "  outs = ['bar.out'],",
        "  cmd = 'cat $(SRCS) > $@',",
        ")");
    write("a/foo.in", "foo");
    write("a/bar.in", "bar");

    // Populate remote cache
    setDownloadAll();
    buildTarget("//a:bar");
    waitDownloads();
    var bytes = readContent(getOutputPath("a/foo.out"));
    var hashCode = getDigestHashFunction().getHashFunction().hashBytes(bytes);
    getOutputPath("a/foo.out").delete();
    getOutputPath("a/bar.out").delete();
    getOutputBase().getRelative("action_cache").deleteTreesBelow();
    restartServer();

    // Clean build, foo.out isn't downloaded
    buildTarget("//a:bar");
    assertOutputDoesNotExist("a/foo.out");

    // Act: Evict blobs from remote cache and do an incremental build
    evictAllBlobs();
    write("a/bar.in", "updated bar");
    var error = assertThrows(BuildFailedException.class, () -> buildTarget("//a:bar"));

    // Assert: Exit code is 39
    assertThat(error).hasMessageThat().contains(String.format("%s/%s", hashCode, bytes.length));
    assertThat(error.getDetailedExitCode().getExitCode().getNumericExitCode()).isEqualTo(39);
  }

  @Test
  public void remoteCacheEvictBlobs_whenUploadingInputFile_incrementalBuildCanContinue()
      throws Exception {
    // Arrange: Prepare workspace and populate remote cache
    write(
        "a/BUILD",
        "genrule(",
        "  name = 'foo',",
        "  srcs = ['foo.in'],",
        "  outs = ['foo.out'],",
        "  cmd = 'cat $(SRCS) > $@',",
        ")",
        "genrule(",
        "  name = 'bar',",
        "  srcs = ['foo.out', 'bar.in'],",
        "  outs = ['bar.out'],",
        "  cmd = 'cat $(SRCS) > $@',",
        ")");
    write("a/foo.in", "foo");
    write("a/bar.in", "bar");

    // Populate remote cache
    buildTarget("//a:bar");
    getOutputPath("a/foo.out").delete();
    getOutputPath("a/bar.out").delete();
    getOutputBase().getRelative("action_cache").deleteTreesBelow();
    restartServer();

    // Clean build, foo.out isn't downloaded
    setDownloadToplevel();
    buildTarget("//a:bar");
    assertOutputDoesNotExist("a/foo.out");

    // Evict blobs from remote cache
    evictAllBlobs();

    // trigger build error
    write("a/bar.in", "updated bar");
    // Build failed because of remote cache eviction
    assertThrows(BuildFailedException.class, () -> buildTarget("//a:bar"));

    // Act: Do an incremental build without "clean" or "shutdown"
    buildTarget("//a:bar");
    waitDownloads();

    // Assert: target was successfully built
    assertValidOutputFile("a/bar.out", "foo" + lineSeparator() + "updated bar" + lineSeparator());
  }

  @Test
  public void remoteCacheEvictBlobs_whenUploadingInputTree_incrementalBuildCanContinue()
      throws Exception {
    // Arrange: Prepare workspace and populate remote cache
    write("BUILD");
    writeOutputDirRule();
    write(
        "a/BUILD",
        "load('//:output_dir.bzl', 'output_dir')",
        "output_dir(",
        "  name = 'foo.out',",
        "  content_map = {'file-inside': 'hello world'},",
        ")",
        "genrule(",
        "  name = 'bar',",
        "  srcs = ['foo.out', 'bar.in'],",
        "  outs = ['bar.out'],",
        "  cmd = '( ls $(location :foo.out); cat $(location :bar.in) ) > $@',",
        ")");
    write("a/bar.in", "bar");

    // Populate remote cache
    buildTarget("//a:bar");
    getOutputPath("a/foo.out").deleteTreesBelow();
    getOutputPath("a/bar.out").delete();
    getOutputBase().getRelative("action_cache").deleteTreesBelow();
    restartServer();

    // Clean build, foo.out isn't downloaded
    buildTarget("//a:bar");
    assertOutputDoesNotExist("a/foo.out/file-inside");

    // Evict blobs from remote cache
    evictAllBlobs();

    // trigger build error
    setDownloadToplevel();
    write("a/bar.in", "updated bar");
    // Build failed because of remote cache eviction
    assertThrows(BuildFailedException.class, () -> buildTarget("//a:bar"));

    // Act: Do an incremental build without "clean" or "shutdown"
    buildTarget("//a:bar");
    waitDownloads();

    // Assert: target was successfully built
    assertValidOutputFile("a/bar.out", "file-inside\nupdated bar" + lineSeparator());
  }

  @Test
  public void downloadToplevel_symlinkFile() throws Exception {
    // TODO(chiwang): Make metadata for downloaded symlink non-remote.
    assumeFalse(OS.getCurrent() == OS.WINDOWS);

    setDownloadToplevel();
    writeSymlinkRule();
    write(
        "BUILD",
        "load(':symlink.bzl', 'symlink')",
        "genrule(",
        "  name = 'foo',",
        "  srcs = [],",
        "  outs = ['out/foo.txt'],",
        "  cmd = 'echo foo > $@',",
        ")",
        "symlink(",
        "  name = 'foo-link',",
        "  target = ':foo'",
        ")");

    buildTarget("//:foo-link");

    assertValidOutputFile("foo-link", "foo\n");

    // Delete link, re-plant symlink
    getOutputPath("foo-link").delete();

    buildTarget("//:foo-link");

    assertValidOutputFile("foo-link", "foo\n");

    // Delete target, re-download it
    getOutputPath("foo").delete();

    assertValidOutputFile("foo-link", "foo\n");
  }

  @Test
  public void downloadToplevel_symlinkSourceFile() throws Exception {
    // TODO(chiwang): Make metadata for downloaded symlink non-remote.
    assumeFalse(OS.getCurrent() == OS.WINDOWS);

    setDownloadToplevel();
    writeSymlinkRule();
    write(
        "BUILD",
        "load(':symlink.bzl', 'symlink')",
        "symlink(",
        "  name = 'foo-link',",
        "  target = ':foo.txt'",
        ")");
    write("foo.txt", "foo");

    buildTarget("//:foo-link");

    assertOnlyOutputContent("//:foo-link", "foo-link", "foo" + lineSeparator());

    // Delete link, re-plant symlink
    getOutputPath("foo-link").delete();

    buildTarget("//:foo-link");

    assertOnlyOutputContent("//:foo-link", "foo-link", "foo" + lineSeparator());
  }

  @Test
  public void downloadToplevel_symlinkTree() throws Exception {
    // TODO(chiwang): Make metadata for downloaded symlink non-remote.
    assumeFalse(OS.getCurrent() == OS.WINDOWS);

    setDownloadToplevel();
    writeSymlinkRule();
    writeOutputDirRule();
    write(
        "BUILD",
        "load(':output_dir.bzl', 'output_dir')",
        "load(':symlink.bzl', 'symlink')",
        "output_dir(",
        "  name = 'foo',",
        "  content_map = {'file-1': '1', 'file-2': '2', 'file-3': '3'},",
        ")",
        "symlink(",
        "  name = 'foo-link',",
        "  target = ':foo'",
        ")");

    buildTarget("//:foo-link");

    assertValidOutputFile("foo-link/file-1", "1");
    assertValidOutputFile("foo-link/file-2", "2");
    assertValidOutputFile("foo-link/file-3", "3");

    getOutputPath("foo-link").deleteTree();

    // Delete link, re-plant symlink
    buildTarget("//:foo-link");

    assertValidOutputFile("foo-link/file-1", "1");
    assertValidOutputFile("foo-link/file-2", "2");
    assertValidOutputFile("foo-link/file-3", "3");

    // Delete target, re-download them
    getOutputPath("foo").deleteTree();

    buildTarget("//:foo-link");

    assertValidOutputFile("foo-link/file-1", "1");
    assertValidOutputFile("foo-link/file-2", "2");
    assertValidOutputFile("foo-link/file-3", "3");
  }
}
