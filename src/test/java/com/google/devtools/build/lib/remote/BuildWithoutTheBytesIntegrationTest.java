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
import static com.google.devtools.build.lib.vfs.FileSystemUtils.readContent;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;
import static org.junit.Assume.assumeFalse;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.authandtls.credentialhelper.CredentialModule;
import com.google.devtools.build.lib.dynamic.DynamicExecutionModule;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.IntegrationTestUtils;
import com.google.devtools.build.lib.remote.util.IntegrationTestUtils.WorkerInstance;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.BlockWaitingModule;
import com.google.devtools.build.lib.runtime.BuildSummaryStatsModule;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.standalone.StandaloneModule;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.io.IOException;
import org.junit.ClassRule;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Integration tests for Build without the Bytes. */
@RunWith(TestParameterInjector.class)
public class BuildWithoutTheBytesIntegrationTest extends BuildWithoutTheBytesIntegrationTestBase {
  @ClassRule @Rule public static final WorkerInstance worker = IntegrationTestUtils.createWorker();

  @Override
  protected void setupOptions() throws Exception {
    super.setupOptions();

    addOptions(
        "--remote_executor=grpc://localhost:" + worker.getPort(),
        "--remote_download_minimal",
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
  protected void enableActionRewinding() {
    addOptions(
        "--rewind_lost_inputs",
        // Disable build rewinding.
        "--experimental_remote_cache_eviction_retries=0",
        // TODO: Add support for concurrent rewinding to Bazel.
        "--jobs=1");
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
    worker.reset();
  }

  @Override
  protected boolean hasAccessToRemoteOutputs() {
    return true;
  }

  @Override
  protected void injectFile(byte[] content) {}

  @Test
  public void executeRemotely_actionFails_outputsAreAvailableLocallyForDebuggingPurpose()
      throws Exception {
    write(
        "a/BUILD",
        """
        genrule(
            name = "fail",
            srcs = [],
            outs = ["fail.txt"],
            cmd = "echo foo > $@ && exit 1",
        )
        """);

    assertThrows(BuildFailedException.class, () -> buildTarget("//a:fail"));

    assertOnlyOutputContent("//a:fail", "fail.txt", "foo\n");
  }

  @Test
  public void intermediateOutputsAreInputForInternalActions_prefetchIntermediateOutputs()
      throws Exception {
    // Test that a remotely stored output that's an input to a internal action
    // (ctx.actions.expand_template) is staged lazily for action execution.
    write(
        "a/substitute_username.bzl",
        """
        def _substitute_username_impl(ctx):
            ctx.actions.expand_template(
                template = ctx.file.template,
                output = ctx.outputs.out,
                substitutions = {
                    "{USERNAME}": ctx.attr.username,
                },
            )

        substitute_username = rule(
            implementation = _substitute_username_impl,
            attrs = {
                "username": attr.string(mandatory = True),
                "template": attr.label(
                    allow_single_file = True,
                    mandatory = True,
                ),
            },
            outputs = {"out": "%{name}.txt"},
        )
        """);
    write(
        "a/BUILD",
        """
        load(":substitute_username.bzl", "substitute_username")

        genrule(
            name = "generate-template",
            srcs = [],
            outs = ["template.txt"],
            cmd = 'echo -n "Hello {USERNAME}!" > $@',
        )

        substitute_username(
            name = "substitute-buchgr",
            template = ":generate-template",
            username = "buchgr",
        )
        """);

    buildTarget("//a:substitute-buchgr");

    // The genrule //a:generate-template should run remotely and //a:substitute-buchgr should be a
    // internal action running locally.
    events.assertContainsInfo("3 processes: 2 internal, 1 remote");
    Artifact intermediateOutput = getOnlyElement(getArtifacts("//a:generate-template"));
    assertThat(intermediateOutput.getPath().exists()).isTrue();
    assertOnlyOutputContent("//a:substitute-buchgr", "substitute-buchgr.txt", "Hello buchgr!");
  }

  @Test
  public void changeOutputMode_notInvalidateActions() throws Exception {
    write(
        "a/BUILD",
        """
        genrule(
            name = "foo",
            srcs = [],
            outs = ["foo.txt"],
            cmd = "echo foo > $@",
        )

        genrule(
            name = "foobar",
            srcs = [":foo"],
            outs = ["foobar.txt"],
            cmd = "cat $(location :foo) > $@ && echo bar > $@",
        )
        """);
    // Download all outputs with regex so in the next build with ALL mode, the actions are not
    // invalidated because of missing outputs.
    addOptions("--remote_download_regex=.*");
    ActionEventCollector actionEventCollector = new ActionEventCollector();
    runtimeWrapper.registerSubscriber(actionEventCollector);
    buildTarget("//a:foobar");
    // Add the new option here because waitDownloads below will internally create a new command
    // which will parse the new option.
    setDownloadAll();
    waitDownloads();
    // 3 = workspace status action + //:foo + //:foobar
    assertThat(actionEventCollector.getNumActionNodesEvaluated()).isEqualTo(3);
    actionEventCollector.clear();

    buildTarget("//a:foobar");

    // Changing output mode should not invalidate SkyFrame's in-memory caching.
    assertThat(actionEventCollector.getNumActionNodesEvaluated()).isEqualTo(0);
    events.assertContainsInfo("0 processes");
  }

  @Test
  public void outputSymlinkHandledGracefully() throws Exception {
    // Dangling symlink would require developer mode to be enabled in the CI environment.
    assumeFalse(OS.getCurrent() == OS.WINDOWS);

    write(
        "a/defs.bzl",
        """
        def _impl(ctx):
            out = ctx.actions.declare_symlink(ctx.label.name)
            ctx.actions.run_shell(
                inputs = [],
                outputs = [out],
                command = "ln -s hello $1",
                arguments = [out.path],
            )
            return DefaultInfo(files = depset([out]))

        my_rule = rule(
            implementation = _impl,
        )
        """);

    write(
        "a/BUILD",
        """
        load(":defs.bzl", "my_rule")

        my_rule(name = "hello")
        """);

    buildTarget("//a:hello");

    Path outputPath = getOutputPath("a/hello");
    assertThat(outputPath.stat(Symlinks.NOFOLLOW).isSymbolicLink()).isTrue();
  }

  @Test
  public void replaceOutputDirectoryWithFile() throws Exception {
    write(
        "a/defs.bzl",
        """
        def _impl(ctx):
            dir = ctx.actions.declare_directory(ctx.label.name + ".dir")
            ctx.actions.run_shell(
                outputs = [dir],
                command = "touch $1/hello",
                arguments = [dir.path],
            )
            return DefaultInfo(files = depset([dir]))

        my_rule = rule(
            implementation = _impl,
        )
        """);
    write(
        "a/BUILD",
        """
        load(":defs.bzl", "my_rule")

        my_rule(name = "hello")
        """);

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
        """
        genrule(
            name = "foo",
            srcs = ["foo.in"],
            outs = ["foo.out"],
            cmd = "cat $(SRCS) > $@",
        )

        genrule(
            name = "bar",
            srcs = [
                "foo.out",
                "bar.in",
            ],
            outs = ["bar.out"],
            cmd = "cat $(SRCS) > $@",
            tags = ["no-remote-exec"],
        )
        """);
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
    assertThat(error).hasMessageThat().contains("Lost inputs no longer available remotely");
    assertThat(error).hasMessageThat().contains("a/foo.out");
    assertThat(error).hasMessageThat().contains(String.format("%s/%s", hashCode, bytes.length));
    assertThat(error.getDetailedExitCode().getExitCode().getNumericExitCode()).isEqualTo(39);
  }

  @Test
  public void remoteCacheEvictBlobs_whenPrefetchingInput_succeedsWithActionRewinding()
      throws Exception {
    // Arrange: Prepare workspace and populate remote cache
    write(
        "a/BUILD",
        """
        genrule(
            name = "foo",
            srcs = ["foo.in"],
            outs = ["foo.out"],
            cmd = "cat $(SRCS) > $@",
        )

        genrule(
            name = "bar",
            srcs = [
                "foo.out",
                "bar.in",
            ],
            outs = ["bar.out"],
            cmd = "cat $(SRCS) > $@",
            tags = ["no-remote-exec"],
        )
        """);
    write("a/foo.in", "foo");
    write("a/bar.in", "bar");

    // Populate remote cache
    buildTarget("//a:bar");
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
    enableActionRewinding();
    buildTarget("//a:bar");

    // Assert: target was successfully built
    assertValidOutputFile("a/bar.out", "foo\nupdated bar\n");
  }

  @Test
  public void remoteCacheEvictBlobs_whenPrefetchingSymlinkedInput_exitWithCode39()
      throws Exception {
    // Arrange: Prepare workspace and populate remote cache
    writeSymlinkRule();
    write(
        "a/BUILD",
        """
        load("//:symlink.bzl", "symlink")

        genrule(
            name = "foo",
            srcs = ["foo.in"],
            outs = ["foo.out"],
            cmd = "cat $(SRCS) > $@",
        )

        symlink(
            name = "symlinked_foo",
            target_artifact = ":foo.out",
        )

        genrule(
            name = "bar",
            srcs = [
                ":symlinked_foo",
                "bar.in",
            ],
            outs = ["bar.out"],
            cmd = "cat $(SRCS) > $@",
            tags = ["no-remote-exec"],
        )
        """);
    write("a/foo.in", "foo");
    write("a/bar.in", "bar");

    // Populate remote cache
    buildTarget("//a:bar");
    var bytes = readContent(getOutputPath("a/foo.out"));
    var hashCode = getDigestHashFunction().getHashFunction().hashBytes(bytes);
    getOnlyElement(getArtifacts("//a:symlinked_foo")).getPath().delete();
    getOutputPath("a/foo.out").delete();
    getOutputPath("a/bar.out").delete();
    getOutputBase().getRelative("action_cache").deleteTreesBelow();
    restartServer();

    // Clean build, foo.out isn't downloaded
    buildTarget("//a:bar");
    assertOutputDoesNotExist("a/foo.out");
    assertOutputsDoNotExist("//a:symlinked_foo");

    // Act: Evict blobs from remote cache and do an incremental build
    evictAllBlobs();
    write("a/bar.in", "updated bar");
    var error = assertThrows(BuildFailedException.class, () -> buildTarget("//a:bar"));

    // Assert: Exit code is 39
    assertThat(error).hasMessageThat().contains("Lost inputs no longer available remotely");
    assertThat(error).hasMessageThat().contains("a/symlinked_foo");
    assertThat(error).hasMessageThat().contains(String.format("%s/%s", hashCode, bytes.length));
    assertThat(error.getDetailedExitCode().getExitCode().getNumericExitCode()).isEqualTo(39);
  }

  @Test
  public void remoteCacheEvictBlobs_whenPrefetchingSymlinkedInput_succeedsWithActionRewinding()
      throws Exception {
    writeSymlinkRule();
    write(
        "a/BUILD",
        """
        load("//:symlink.bzl", "symlink")

        genrule(
            name = "foo",
            srcs = ["foo.in"],
            outs = ["foo.out"],
            cmd = "cat $(SRCS) > $@",
        )

        symlink(
            name = "symlinked_foo",
            target_artifact = ":foo.out",
        )

        genrule(
            name = "bar",
            srcs = [
                ":symlinked_foo",
                "bar.in",
            ],
            outs = ["bar.out"],
            cmd = "cat $(SRCS) > $@",
            tags = ["no-remote-exec"],
        )
        """);
    write("a/foo.in", "foo");
    write("a/bar.in", "bar");

    // Populate remote cache
    buildTarget("//a:bar");
    getOnlyElement(getArtifacts("//a:symlinked_foo")).getPath().delete();
    getOutputPath("a/foo.out").delete();
    getOutputPath("a/bar.out").delete();
    getOutputBase().getRelative("action_cache").deleteTreesBelow();
    restartServer();

    // Clean build, foo.out isn't downloaded
    buildTarget("//a:bar");
    assertOutputDoesNotExist("a/foo.out");
    assertOutputsDoNotExist("//a:symlinked_foo");

    // Act: Evict blobs from remote cache and do an incremental build
    evictAllBlobs();
    write("a/bar.in", "updated bar");
    enableActionRewinding();
    buildTarget("//a:bar");

    // Assert: target was successfully built
    assertValidOutputFile("a/bar.out", "foo\nupdated bar\n");
  }

  @Test
  public void remoteCacheEvictBlobs_whenUploadingInput_exitWithCode39() throws Exception {
    // Arrange: Prepare workspace and populate remote cache
    write(
        "a/BUILD",
        """
        genrule(
            name = "foo",
            srcs = ["foo.in"],
            outs = ["foo.out"],
            cmd = "cat $(SRCS) > $@",
        )

        genrule(
            name = "bar",
            srcs = [
                "foo.out",
                "bar.in",
            ],
            outs = ["bar.out"],
            cmd = "cat $(SRCS) > $@",
        )
        """);
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
  public void remoteCacheEvictBlobs_whenUploadingInput_succeedsWithActionRewinding()
      throws Exception {
    // Arrange: Prepare workspace and populate remote cache
    write(
        "a/BUILD",
        """
        genrule(
            name = "foo",
            srcs = ["foo.in"],
            outs = ["foo.out"],
            cmd = "cat $(SRCS) > $@",
        )

        genrule(
            name = "bar",
            srcs = [
                "foo.out",
                "bar.in",
            ],
            outs = ["bar.out"],
            cmd = "cat $(SRCS) > $@",
        )
        """);
    write("a/foo.in", "foo");
    write("a/bar.in", "bar");

    // Populate remote cache
    setDownloadAll();
    buildTarget("//a:bar");
    waitDownloads();
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
    enableActionRewinding();
    buildTarget("//a:bar");

    // Assert: target was successfully built
    assertOutputsDoNotExist("//a:bar");
    assertOnlyOutputRemoteContent("//a:bar", "bar.out", "foo\nupdated bar\n");
  }

  @Test
  public void remoteCacheEvictBlobs_whenUploadingInputFile_incrementalBuildCanContinue()
      throws Exception {
    // Arrange: Prepare workspace and populate remote cache
    write(
        "a/BUILD",
        """
        genrule(
            name = "foo",
            srcs = ["foo.in"],
            outs = ["foo.out"],
            cmd = "cat $(SRCS) > $@",
        )

        genrule(
            name = "bar",
            srcs = [
                "foo.out",
                "bar.in",
            ],
            outs = ["bar.out"],
            cmd = "cat $(SRCS) > $@",
        )
        """);
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
    assertValidOutputFile("a/bar.out", "foo\nupdated bar\n");
  }

  @Test
  public void remoteCacheEvictBlobs_whenUploadingInputTree_incrementalBuildCanContinue()
      throws Exception {
    // Arrange: Prepare workspace and populate remote cache
    write("BUILD");
    writeOutputDirRule();
    write(
        "a/BUILD",
        """
        load("//:output_dir.bzl", "output_dir")

        output_dir(
            name = "foo.out",
            content_map = {"file-inside": "hello world"},
        )

        genrule(
            name = "bar",
            srcs = [
                "foo.out",
                "bar.in",
            ],
            outs = ["bar.out"],
            cmd = "( ls $(location :foo.out); cat $(location :bar.in) ) > $@",
        )
        """);
    write("a/bar.in", "bar");

    // Populate remote cache
    buildTarget("//a:bar");
    getOutputPath("a/foo.out").deleteTreesBelow();
    getOutputPath("a/bar.out").delete();
    getOutputBase().getRelative("action_cache").deleteTreesBelow();
    restartServer();

    // Clean build, foo.out isn't downloaded
    setDownloadToplevel();
    buildTarget("//a:bar");
    assertOutputDoesNotExist("a/foo.out/file-inside");

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
    assertValidOutputFile("a/bar.out", "file-inside\nupdated bar\n");
  }

  @Test
  public void remoteCacheEvictBlobs_whenUploadingInputTree_succeedsWithActionRewinding()
      throws Exception {
    // Arrange: Prepare workspace and populate remote cache
    write("BUILD");
    writeOutputDirRule();
    write(
        "a/BUILD",
        """
        load("//:output_dir.bzl", "output_dir")

        output_dir(
            name = "foo.out",
            content_map = {"file-inside": "hello world"},
        )

        genrule(
            name = "bar",
            srcs = [
                "foo.out",
                "bar.in",
            ],
            outs = ["bar.out"],
            cmd = "( ls $(location :foo.out); cat $(location :bar.in) ) > $@",
        )
        """);
    write("a/bar.in", "bar");

    // Populate remote cache
    buildTarget("//a:bar");
    getOutputPath("a/foo.out").deleteTreesBelow();
    getOutputPath("a/bar.out").delete();
    getOutputBase().getRelative("action_cache").deleteTreesBelow();
    restartServer();

    // Clean build, foo.out isn't downloaded
    setDownloadToplevel();
    buildTarget("//a:bar");
    assertOutputDoesNotExist("a/foo.out/file-inside");

    // Act: Do an incremental build without "clean" or "shutdown" after clearing the cache
    evictAllBlobs();
    write("a/bar.in", "updated bar");
    enableActionRewinding();
    buildTarget("//a:bar");

    // Assert: target was successfully built
    assertValidOutputFile("a/bar.out", "file-inside\nupdated bar\n");
  }

  @Test
  public void remoteCacheEvictBlobs_whenTopLevelRequested_succeedsWithActionRewinding()
      throws Exception {
    // Arrange: Prepare workspace and populate remote cache
    write("BUILD");
    writeOutputDirRule();
    write(
        "a/BUILD",
        """
        load("//:output_dir.bzl", "output_dir")

        output_dir(
            name = "foo.out",
            content_map = {"file-inside": "hello world"},
        )

        genrule(
            name = "bar",
            srcs = [
                "foo.out",
                "bar.in",
            ],
            outs = ["bar.out"],
            cmd = "( ls $(location :foo.out); cat $(location :bar.in) ) > $@",
        )
        """);
    write("a/bar.in", "bar");

    // Populate remote cache
    buildTarget("//a:bar", "//a:foo.out");
    getOutputPath("a/foo.out").deleteTreesBelow();
    getOutputPath("a/bar.out").delete();
    getOutputBase().getRelative("action_cache").deleteTreesBelow();
    restartServer();

    // Clean build, bar.out and foo.out aren't downloaded
    buildTarget("//a:bar", "//a:foo.out");
    assertOutputDoesNotExist("a/bar.out");
    assertOutputDoesNotExist("a/foo.out/file-inside");

    // Act: Do an incremental build without "clean" or "shutdown" after clearing the cache and
    // switching to download toplevel
    evictAllBlobs();
    setDownloadToplevel();
    enableActionRewinding();
    buildTarget("//a:bar", "//a:foo.out");

    // Assert: all outputs were downloaded
    assertValidOutputFile("a/bar.out", "file-inside\nbar\n");
    assertValidOutputFile("a/foo.out/file-inside", "hello world");
  }

  @Test
  public void remoteCacheEvictBlobs_whenRunfilesRequested_succeedsWithActionRewinding()
      throws Exception {
    // Arrange: Prepare workspace and populate remote cache
    write("BUILD");
    writeOutputDirRule();
    write(
        "native_binary.bzl",
        """
        def _native_binary_impl(ctx):
            runfiles = ctx.runfiles(
                transitive_files = depset(
                    transitive = [target[DefaultInfo].files for target in ctx.attr.data],
                ),
            )
            runfiles = runfiles.merge_all(
                [target[DefaultInfo].default_runfiles for target in ctx.attr.data],
            )
            executable = ctx.actions.declare_file(ctx.label.name)
            ctx.actions.symlink(
                output = executable,
                target_file = ctx.file.executable,
            )
            return [
                DefaultInfo(
                    executable = executable,
                    runfiles = runfiles,
                ),
            ]

        native_binary = rule(
            implementation = _native_binary_impl,
            attrs = {
                "executable": attr.label(allow_single_file = True),
                "data": attr.label_list(),
            },
            executable = True,
        )
        """);
    write(
        "a/BUILD",
        """
        load("//:native_binary.bzl", "native_binary")
        load("//:output_dir.bzl", "output_dir")

        output_dir(
            name = "foo.out",
            content_map = {"file-inside": "hello world"},
        )

        genrule(
            name = "bar",
            srcs = [
                "foo.out",
                "bar.in",
            ],
            outs = ["bar.out"],
            cmd = "( ls $(location :foo.out); cat $(location :bar.in) ) > $@",
        )

        native_binary(
            name = "bin",
            executable = "bin.sh",
            data = [
                ":foo.out",
                ":bar",
            ],
        )
        """);
    write("a/bar.in", "bar");
    write("a/bin.sh");

    // Populate remote cache
    buildTarget("//a:bin");
    getOutputPath("a/foo.out").deleteTreesBelow();
    getOutputPath("a/bar.out").delete();
    getOutputBase().getRelative("action_cache").deleteTreesBelow();
    restartServer();

    // Clean build, runfiles aren't downloaded
    buildTarget("//a:bin");
    assertThat(getOutputPath("a/bin.runfiles").isDirectory()).isTrue();
    assertOutputDoesNotExist("a/bar.out");
    assertOutputDoesNotExist("a/foo.out/file-inside");

    // Act: Do an incremental build without "clean" or "shutdown" after clearing the cache and
    // switching to download toplevel
    evictAllBlobs();
    setDownloadToplevel();
    enableActionRewinding();
    buildTarget("//a:bin");

    // Assert: all runfiles were downloaded
    assertValidOutputFile("a/bar.out", "file-inside\nbar\n");
    assertValidOutputFile("a/foo.out/file-inside", "hello world");
  }

  @Test
  public void leaseExtension() throws Exception {
    // Test that Bazel will extend the leases for remote output by sending FindMissingBlobs calls
    // periodically to remote server. The test assumes remote server will set mtime of referenced
    // blobs to `now`.
    write(
        "BUILD",
        "genrule(",
        "  name = 'foo',",
        "  srcs = [],",
        "  outs = ['out/foo.txt'],",
        "  cmd = 'echo -n foo > $@',",
        ")",
        "genrule(",
        "  name = 'foobar',",
        "  srcs = [':foo'],",
        "  outs = ['out/foobar.txt'],",
        // We need the action lasts more than --experimental_remote_cache_ttl so Bazel has the
        // chance to extend the lease
        "  cmd = 'sleep 2 && cat $(location :foo) > $@ && echo bar >> $@',",
        ")");
    addOptions("--experimental_remote_cache_ttl=1s", "--experimental_remote_cache_lease_extension");
    var content = "foo".getBytes(UTF_8);
    var hashCode = getFileSystem().getDigestFunction().getHashFunction().hashBytes(content);
    var digest = DigestUtil.buildDigest(hashCode.asBytes(), content.length).getHash();
    // Calculate the blob path in CAS. This is specific to the remote worker. See
    // {@link DiskCacheClient#getPath()}.
    var blobPath =
        getFileSystem()
            .getPath(worker.getCasPath())
            .getChild("cas")
            .getChild(digest.substring(0, 2))
            .getChild(digest);
    var mtimes = Sets.newConcurrentHashSet();
    // Observe the mtime of the blob in background.
    var thread =
        new Thread(
            () -> {
              while (!Thread.currentThread().isInterrupted()) {
                try {
                  mtimes.add(blobPath.getLastModifiedTime());
                } catch (IOException ignored) {
                  // Intentionally ignored
                }
              }
            });
    thread.start();

    buildTarget("//:foobar");
    waitDownloads();

    thread.interrupt();
    thread.join();
    // We should be able to observe more than 1 mtime if the server extends the lease.
    assertThat(mtimes.size()).isGreaterThan(1);
  }

  @Test
  public void downloadTopLevel_deepSymlinkToFile() throws Exception {
    setDownloadToplevel();
    write(
        "defs.bzl",
        """
        def _impl(ctx):
            file = ctx.actions.declare_file(ctx.label.name + ".file")
            ctx.actions.run_shell(
                outputs = [file],
                command = "echo -n hello > $1",
                arguments = [file.path],
            )

            shallow = ctx.actions.declare_file(ctx.label.name + ".shallow")
            ctx.actions.symlink(output = shallow, target_file = file)

            deep = ctx.actions.declare_file(ctx.label.name + ".deep")
            ctx.actions.symlink(output = deep, target_file = shallow)

            return DefaultInfo(files = depset([deep]))

        symlink = rule(_impl)
        """);
    write("BUILD", "load(':defs.bzl', 'symlink')", "symlink(name = 'foo')");

    buildTarget("//:foo");

    // Materialization skips the intermediate symlink.
    assertSymlink("foo.deep", getOutputPath("foo.file").asFragment());
    assertValidOutputFile("foo.deep", "hello");
  }

  @Test
  public void downloadTopLevel_deepSymlinkToDirectory() throws Exception {
    setDownloadToplevel();
    write(
        "defs.bzl",
        """
        def _impl(ctx):
            dir = ctx.actions.declare_directory(ctx.label.name + ".dir")
            ctx.actions.run_shell(
                outputs = [dir],
                command = "echo -n hello > $1/file.txt",
                arguments = [dir.path],
            )

            shallow = ctx.actions.declare_directory(ctx.label.name + ".shallow")
            ctx.actions.symlink(output = shallow, target_file = dir)

            deep = ctx.actions.declare_directory(ctx.label.name + ".deep")
            ctx.actions.symlink(output = deep, target_file = shallow)

            return DefaultInfo(files = depset([deep]))

        symlink = rule(_impl)
        """);
    write("BUILD", "load(':defs.bzl', 'symlink')", "symlink(name = 'foo')");

    buildTarget("//:foo");

    // Materialization skips the intermediate symlink.
    assertSymlink("foo.deep", getOutputPath("foo.dir").asFragment());
    assertValidOutputFile("foo.deep/file.txt", "hello");
  }

  @Test
  public void downloadTopLevel_genruleSymlinkToInput() throws Exception {
    setDownloadToplevel();
    write(
        "BUILD",
        "genrule(",
        "  name = 'foo',",
        "  outs = ['foo'],",
        "  cmd = 'echo hello > $@',",
        ")",
        "genrule(",
        "  name = 'gen',",
        "  srcs = ['foo'],",
        "  outs = ['foo-link'],",
        "  cmd = 'cd $(RULEDIR) && ln -s foo foo-link',",
        // In Blaze, heuristic label expansion defaults to True and will cause `foo` to be expanded
        // into `blaze-out/.../bin/foo` in the genrule command line.
        "  heuristic_label_expansion = False,",
        ")");

    buildTarget("//:gen");

    assertSymlink("foo-link", getOutputPath("foo").asFragment());
    assertValidOutputFile("foo-link", "hello\n");

    // Delete link, re-plant symlink
    getOutputPath("foo").delete();
    buildTarget("//:gen");

    assertSymlink("foo-link", getOutputPath("foo").asFragment());
    assertValidOutputFile("foo-link", "hello\n");

    // Delete target, re-download it
    getOutputPath("foo").delete();

    buildTarget("//:gen");

    assertSymlink("foo-link", getOutputPath("foo").asFragment());
    assertValidOutputFile("foo-link", "hello\n");
  }

  @Test
  public void downloadTopLevel_genruleSymlinkToOutput() throws Exception {
    setDownloadToplevel();
    write(
        "BUILD",
        "genrule(",
        "  name = 'gen',",
        "  outs = ['foo', 'foo-link'],",
        "  cmd = 'cd $(RULEDIR) && echo hello > foo && ln -s foo foo-link',",
        // In Blaze, heuristic label expansion defaults to True and will cause `foo` to be expanded
        // into `blaze-out/.../bin/foo` in the genrule command line.
        "  heuristic_label_expansion = False,",
        ")");

    buildTarget("//:gen");

    assertSymlink("foo-link", PathFragment.create("foo"));
    assertValidOutputFile("foo-link", "hello\n");

    // Delete link, re-plant symlink
    getOutputPath("foo").delete();
    buildTarget("//:gen");

    assertSymlink("foo-link", PathFragment.create("foo"));
    assertValidOutputFile("foo-link", "hello\n");

    // Delete target, re-download it
    getOutputPath("foo").delete();

    buildTarget("//:gen");

    assertSymlink("foo-link", PathFragment.create("foo"));
    assertValidOutputFile("foo-link", "hello\n");
  }

  @Test
  public void remoteAction_inputTreeWithSymlinks() throws Exception {
    setDownloadToplevel();
    write(
        "tree.bzl",
        "def _impl(ctx):",
        "  d = ctx.actions.declare_directory(ctx.label.name)",
        "  ctx.actions.run_shell(",
        "    outputs = [d],",
        "    command = 'mkdir $1/dir && touch $1/file $1/dir/file && ln -s file $1/filesym && ln"
            + " -s dir $1/dirsym',",
        "    arguments = [d.path],",
        "  )",
        "  return DefaultInfo(files = depset([d]))",
        "tree = rule(_impl)");
    write(
        "BUILD",
        "load(':tree.bzl', 'tree')",
        "tree(name = 'tree')",
        "genrule(name = 'gen', srcs = [':tree'], outs = ['out'], cmd = 'touch $@')");

    // Populate cache
    buildTarget("//:gen");

    // Delete output, replay from cache
    getOutputPath("tree").deleteTree();
    getOutputPath("out").delete();
    buildTarget("//:gen");
  }

  @Test
  public void remoteFilesExpiredBetweenBuilds_buildRewound() throws Exception {
    // Arrange: Prepare workspace and populate remote cache
    write(
        "a/BUILD",
        """
        genrule(
            name = "foo",
            srcs = ["foo.in"],
            outs = ["foo.out"],
            cmd = "cat $(SRCS) > $@",
        )

        genrule(
            name = "bar",
            srcs = [
                "foo.out",
                "bar.in",
            ],
            outs = ["bar.out"],
            cmd = "cat $(SRCS) > $@",
        )
        """);
    write("a/foo.in", "foo");
    write("a/bar.in", "bar");

    // Populate remote cache
    buildTarget("//a:bar");
    assertOutputDoesNotExist("a/foo.out");
    assertOutputDoesNotExist("a/bar.out");
    getOutputBase().getRelative("action_cache").deleteTreesBelow();
    restartServer();

    // Clean build, foo.out isn't downloaded
    setDownloadToplevel();
    addOptions("--experimental_remote_cache_ttl=0s");
    buildTarget("//a:bar");
    assertOutputDoesNotExist("a/foo.out");
    assertValidOutputFile("a/bar.out", "foo\nbar\n");

    // Evict blobs from remote cache
    evictAllBlobs();

    // Act: Do an incremental build, which is expected to fail with the exit code
    // that, in a non-integration test setup, would retry the invocation
    // automatically. Then simulate the retry.
    write("a/bar.in", "updated bar");
    addOptions("--strategy_regexp=.*bar=local");
    var e = assertThrows(BuildFailedException.class, () -> buildTarget("//a:bar"));
    assertThat(e.getDetailedExitCode().getFailureDetail().getSpawn().getCode())
        .isEqualTo(FailureDetails.Spawn.Code.REMOTE_CACHE_EVICTED);

    buildTarget("//a:bar");
    waitDownloads();

    // Assert: target was successfully built
    assertValidOutputFile("a/bar.out", "foo\nupdated bar\n");
  }

  @Test
  public void remoteTreeFilesExpiredBetweenBuilds_buildRewound() throws Exception {
    // Arrange: Prepare workspace and populate remote cache
    write("BUILD");
    writeOutputDirRule();
    write(
        "a/BUILD",
        """
        load("//:output_dir.bzl", "output_dir")

        output_dir(
            name = "foo.out",
            content_map = {"file-inside": "hello world"},
        )

        genrule(
            name = "bar",
            srcs = [
                "foo.out",
                "bar.in",
            ],
            outs = ["bar.out"],
            cmd = "( ls $(location :foo.out); cat $(location :bar.in) ) > $@",
        )
        """);
    write("a/bar.in", "bar");

    // Populate remote cache
    buildTarget("//a:bar");
    assertThat(getOutputPath("a/foo.out").getDirectoryEntries()).isEmpty();
    assertOutputDoesNotExist("a/bar.out");
    getOutputBase().getRelative("action_cache").deleteTreesBelow();
    restartServer();

    // Clean build, foo.out isn't downloaded
    setDownloadToplevel();
    addOptions("--experimental_remote_cache_ttl=0s");
    buildTarget("//a:bar");
    assertOutputDoesNotExist("a/foo.out/file-inside");
    assertValidOutputFile("a/bar.out", "file-inside\nbar\n");

    // Evict blobs from remote cache
    evictAllBlobs();

    // Act: Do an incremental build, which is expected to fail with the exit code
    // that, in a non-integration test setup, would retry the invocation
    // automatically. Then simulate the retry.
    write("a/bar.in", "updated bar");
    addOptions("--strategy_regexp=.*bar=local");
    var e = assertThrows(BuildFailedException.class, () -> buildTarget("//a:bar"));
    assertThat(e.getDetailedExitCode().getFailureDetail().getSpawn().getCode())
        .isEqualTo(FailureDetails.Spawn.Code.REMOTE_CACHE_EVICTED);

    buildTarget("//a:bar");
    waitDownloads();

    // Assert: target was successfully built
    assertValidOutputFile("a/bar.out", "file-inside\nupdated bar\n");
  }
}
