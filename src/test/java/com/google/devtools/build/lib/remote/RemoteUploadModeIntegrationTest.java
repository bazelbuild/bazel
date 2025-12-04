// Copyright 2025 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.vfs.FileSystemUtils.readContent;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.authandtls.credentialhelper.CredentialModule;
import com.google.devtools.build.lib.dynamic.DynamicExecutionModule;
import com.google.devtools.build.lib.remote.util.IntegrationTestUtils;
import com.google.devtools.build.lib.remote.util.IntegrationTestUtils.WorkerInstance;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.BlockWaitingModule;
import com.google.devtools.build.lib.runtime.BuildSummaryStatsModule;
import com.google.devtools.build.lib.standalone.StandaloneModule;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.Path;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import org.junit.ClassRule;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;

/**
 * Integration tests for {@code --experimental_remote_upload_mode}.
 *
 * <p>Tests the async upload functionality where uploads continue in the background after a build
 * completes, with the next command waiting for pending uploads.
 */
@RunWith(TestParameterInjector.class)
public class RemoteUploadModeIntegrationTest extends BuildWithoutTheBytesIntegrationTestBase {
  @ClassRule @Rule public static final WorkerInstance worker = IntegrationTestUtils.createWorker();

  @Override
  protected ImmutableList<String> getStartupOptions() {
    return OS.getCurrent() == OS.WINDOWS
        ? ImmutableList.of("--windows_enable_symlinks")
        : ImmutableList.of();
  }

  @Override
  protected void setupOptions() throws Exception {
    super.setupOptions();

    addOptions(
        "--remote_executor=grpc://localhost:" + worker.getPort(),
        "--remote_download_minimal",
        "--dynamic_local_strategy=standalone",
        "--dynamic_remote_strategy=remote");

    if (OS.getCurrent() == OS.WINDOWS) {
      addOptions("--action_env=MSYS=winsymlinks:native");
    }
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
        "--rewind_lost_inputs", "--experimental_remote_cache_eviction_retries=0", "--jobs=1");
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

  // ==================== Tests for --experimental_remote_upload_mode ====================

  @Test
  public void nowaitForUploadComplete_uploadsCompleteBeforeNextBuild() throws Exception {
    // Test that with NOWAIT_FOR_UPLOAD_COMPLETE, uploads from build N complete
    // before build N+1 executes (due to waitForPreviousInvocation).
    write(
        "BUILD",
        """
        genrule(
            name = "foo",
            srcs = [],
            outs = ["foo.txt"],
            cmd = "echo foo > $@",
        )
        genrule(
            name = "bar",
            srcs = [],
            outs = ["bar.txt"],
            cmd = "echo bar > $@",
        )
        """);
    addOptions("--experimental_remote_upload_mode=nowait_for_upload_complete");

    // First build - uploads continue in background
    buildTarget("//:foo");

    // Second build - should wait for previous uploads before starting
    // If this succeeds without errors, uploads from first build completed
    buildTarget("//:bar");

    // Both outputs should be available in the remote cache
    // Verify by doing a clean build that hits the cache
    restartServer();
    addOptions("--experimental_remote_upload_mode=nowait_for_upload_complete");

    ActionEventCollector actionEventCollector = new ActionEventCollector();
    getRuntimeWrapper().registerSubscriber(actionEventCollector);
    buildTarget("//:foo");
    waitDownloads();

    // Action should be a cache hit (not re-executed)
    assertThat(actionEventCollector.getActionExecutedEvents()).isEmpty();
  }

  @Test
  public void nowaitForUploadComplete_crossInvocationState_survivesAcrossCommands()
      throws Exception {
    // Test that pending uploads survive across command invocations within the same server.
    write(
        "BUILD",
        """
        genrule(
            name = "gen1",
            srcs = [],
            outs = ["gen1.txt"],
            cmd = "echo gen1 > $@",
        )
        genrule(
            name = "gen2",
            srcs = [],
            outs = ["gen2.txt"],
            cmd = "echo gen2 > $@",
        )
        genrule(
            name = "gen3",
            srcs = [],
            outs = ["gen3.txt"],
            cmd = "echo gen3 > $@",
        )
        """);
    addOptions("--experimental_remote_upload_mode=nowait_for_upload_complete");

    // Run multiple builds in sequence
    buildTarget("//:gen1");
    buildTarget("//:gen2");
    buildTarget("//:gen3");

    // All uploads should have completed (each build waits for the previous one)
    // Verify by restarting and checking cache hits
    restartServer();
    addOptions("--experimental_remote_upload_mode=nowait_for_upload_complete");

    ActionEventCollector collector = new ActionEventCollector();
    getRuntimeWrapper().registerSubscriber(collector);
    buildTarget("//:gen1", "//:gen2", "//:gen3");
    waitDownloads();

    // All actions should be cache hits
    assertThat(collector.getActionExecutedEvents()).isEmpty();
  }

  @Test
  public void nowaitForUploadComplete_cacheIsPopulated() throws Exception {
    // Test that async uploads actually populate the cache correctly.
    write(
        "BUILD",
        """
        genrule(
            name = "cached",
            srcs = [],
            outs = ["cached.txt"],
            cmd = "echo cached-content > $@",
        )
        """);
    addOptions("--experimental_remote_upload_mode=nowait_for_upload_complete");

    // Build with async uploads
    buildTarget("//:cached");

    // Trigger next command to wait for uploads
    waitDownloads();

    // Evict local state and rebuild - should hit cache
    restartServer();
    addOptions("--experimental_remote_upload_mode=wait_for_upload_complete");
    setDownloadToplevel();

    buildTarget("//:cached");
    waitDownloads();

    // Output should be downloaded from cache
    assertValidOutputFile("cached.txt", "cached-content\n");
  }

  @Test
  public void waitForUploadComplete_defaultBehavior_unchanged() throws Exception {
    // Test that the default mode (wait_for_upload_complete) behaves the same as before.
    write(
        "BUILD",
        """
        genrule(
            name = "default",
            srcs = [],
            outs = ["default.txt"],
            cmd = "echo default > $@",
        )
        """);
    // Explicitly set default mode (or don't set it at all)
    addOptions("--experimental_remote_upload_mode=wait_for_upload_complete");

    buildTarget("//:default");

    // Restart and verify cache is populated (uploads completed synchronously)
    restartServer();
    addOptions("--experimental_remote_upload_mode=wait_for_upload_complete");

    ActionEventCollector collector = new ActionEventCollector();
    getRuntimeWrapper().registerSubscriber(collector);
    buildTarget("//:default");
    waitDownloads();

    // Should be a cache hit
    assertThat(collector.getActionExecutedEvents()).isEmpty();
  }

  @Test
  public void nowaitForUploadComplete_incrementalBuild_works() throws Exception {
    // Test that incremental builds work correctly with async upload mode.
    write("input.txt", "original");
    write(
        "BUILD",
        """
        genrule(
            name = "incremental",
            srcs = ["input.txt"],
            outs = ["output.txt"],
            cmd = "cat $(SRCS) > $@",
        )
        """);
    addOptions("--experimental_remote_upload_mode=nowait_for_upload_complete");
    setDownloadToplevel();

    // Initial build
    buildTarget("//:incremental");
    waitDownloads();
    assertValidOutputFile("output.txt", "original\n");

    // Modify source and rebuild
    write("input.txt", "modified");
    buildTarget("//:incremental");
    waitDownloads();
    assertValidOutputFile("output.txt", "modified\n");
  }

  @Test
  public void nowaitForUploadComplete_withDiskCache_works() throws Exception {
    // Test that async upload mode works correctly with disk cache enabled.
    write(
        "BUILD",
        """
        genrule(
            name = "disk_cached",
            srcs = [],
            outs = ["disk_cached.txt"],
            cmd = "echo disk-cached > $@",
        )
        """);
    Path diskCachePath = getOutputBase().getRelative("disk-cache");
    addOptions(
        "--experimental_remote_upload_mode=nowait_for_upload_complete",
        "--disk_cache=" + diskCachePath.getPathString());

    buildTarget("//:disk_cached");

    // Next command waits for uploads
    waitDownloads();

    // Verify cache is populated
    restartServer();
    addOptions(
        "--experimental_remote_upload_mode=wait_for_upload_complete",
        "--disk_cache=" + diskCachePath.getPathString());

    ActionEventCollector collector = new ActionEventCollector();
    getRuntimeWrapper().registerSubscriber(collector);
    buildTarget("//:disk_cached");
    waitDownloads();

    assertThat(collector.getActionExecutedEvents()).isEmpty();
  }
}
