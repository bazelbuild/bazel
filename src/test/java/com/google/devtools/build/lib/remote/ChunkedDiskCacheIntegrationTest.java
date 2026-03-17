// Copyright 2026 The Bazel Authors. All rights reserved.
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
import static com.google.devtools.build.lib.testutil.TestUtils.tmpDirFile;

import com.google.common.collect.ImmutableList;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.authandtls.credentialhelper.CredentialModule;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.remote.util.IntegrationTestUtils;
import com.google.devtools.build.lib.remote.util.IntegrationTestUtils.WorkerInstance;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.BlockWaitingModule;
import com.google.devtools.build.lib.runtime.BuildSummaryStatsModule;
import com.google.devtools.build.lib.standalone.StandaloneModule;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.io.InputStream;
import org.junit.After;
import org.junit.ClassRule;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Integration tests for chunked remote cache with a combined disk + remote cache.
 *
 * <p>Verifies that chunks downloaded from the remote cache are properly captured to disk cache, and
 * that subsequent builds can serve chunks from disk cache without hitting the remote.
 */
@RunWith(JUnit4.class)
public class ChunkedDiskCacheIntegrationTest extends BuildIntegrationTestCase {
  @ClassRule @Rule public static final WorkerInstance worker = IntegrationTestUtils.createWorker();

  private static PathFragment getDiskCacheDir() {
    return PathFragment.create(tmpDirFile().getAbsolutePath()).getRelative("chunked_disk_cache");
  }

  @Override
  protected void setupOptions() throws Exception {
    super.setupOptions();
    addOptions(
        "--remote_cache=grpc://localhost:" + worker.getPort(),
        "--disk_cache=" + getDiskCacheDir(),
        "--experimental_remote_cache_chunking");
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
        .build();
  }

  @After
  public void tearDown() throws Exception {
    runtimeWrapper.newCommand();
    getWorkspace().getFileSystem().getPath(getDiskCacheDir()).deleteTree();
  }

  private Path getOutputPath(String binRelativePath) {
    return getTargetConfiguration().getBinDir().getRoot().getRelative(binRelativePath);
  }

  private void cleanAndRestartServer() throws Exception {
    getOutputBase().getRelative("action_cache").deleteTreesBelow();
    createRuntimeWrapper();
  }

  private byte[] readFileBytes(Path path) throws IOException {
    try (InputStream in = path.getInputStream()) {
      return ByteStreams.toByteArray(in);
    }
  }

  @Test
  public void largeBlob_uploadedAndDownloaded_throughDiskAndRemoteCache() throws Exception {
    write(
        "BUILD",
        """
        genrule(
            name = "large_file",
            srcs = [],
            outs = ["large.bin"],
            cmd = "dd if=/dev/zero bs=1M count=3 2>/dev/null | tr '\\\\0' 'D' > $@",
        )
        """);

    // First build: generates the file, uploads chunks to remote + disk cache.
    buildTarget("//:large_file");

    Path output = getOutputPath("large.bin");
    assertThat(output.exists()).isTrue();
    byte[] originalContent = readFileBytes(output);
    assertThat(originalContent.length).isEqualTo(3 * 1024 * 1024);

    // Second build: clean outputs + action cache, rebuild.
    // Chunks should be served from disk cache (populated during first build's download capture).
    output.delete();
    cleanAndRestartServer();

    buildTarget("//:large_file");

    assertThat(output.exists()).isTrue();
    assertThat(readFileBytes(output)).isEqualTo(originalContent);
  }

  @Test
  public void largeBlob_diskCasDeleted_rebuildFromRemote() throws Exception {
    write(
        "BUILD",
        """
        genrule(
            name = "large_file",
            srcs = [],
            outs = ["large.bin"],
            cmd = "dd if=/dev/zero bs=1M count=3 2>/dev/null | tr '\\\\0' 'E' > $@",
        )
        """);

    // First build: populates both caches.
    buildTarget("//:large_file");

    Path output = getOutputPath("large.bin");
    byte[] originalContent = readFileBytes(output);

    // Delete disk cache CAS entries (simulate cache eviction).
    Path diskCasCas = getWorkspace().getFileSystem().getPath(getDiskCacheDir().getRelative("cas"));
    if (diskCasCas.exists()) {
      diskCasCas.deleteTree();
    }

    // Clean outputs + action cache, rebuild.
    // Should fall back to remote cache since disk CAS is gone.
    output.delete();
    cleanAndRestartServer();

    buildTarget("//:large_file");

    assertThat(output.exists()).isTrue();
    assertThat(readFileBytes(output)).isEqualTo(originalContent);
  }

  @Test
  public void multipleTargets_withDiskCache_allSucceed() throws Exception {
    write(
        "BUILD",
        """
        genrule(
            name = "data_a",
            srcs = [],
            outs = ["a.bin"],
            cmd = "dd if=/dev/zero bs=1M count=3 2>/dev/null | tr '\\\\0' 'F' > $@",
        )
        genrule(
            name = "data_b",
            srcs = [],
            outs = ["b.bin"],
            cmd = "dd if=/dev/zero bs=1M count=4 2>/dev/null | tr '\\\\0' 'G' > $@",
        )
        genrule(
            name = "combined",
            srcs = [":a.bin", ":b.bin"],
            outs = ["combined.bin"],
            cmd = "cat $(SRCS) > $@",
        )
        """);

    buildTarget("//:data_a", "//:data_b", "//:combined");

    Path outputA = getOutputPath("a.bin");
    Path outputB = getOutputPath("b.bin");
    Path outputCombined = getOutputPath("combined.bin");
    byte[] contentA = readFileBytes(outputA);
    byte[] contentB = readFileBytes(outputB);
    byte[] contentCombined = readFileBytes(outputCombined);
    assertThat(contentA.length).isEqualTo(3 * 1024 * 1024);
    assertThat(contentB.length).isEqualTo(4 * 1024 * 1024);
    assertThat(contentCombined.length).isEqualTo(7 * 1024 * 1024);

    // Clean and rebuild from cache.
    outputA.delete();
    outputB.delete();
    outputCombined.delete();
    cleanAndRestartServer();

    buildTarget("//:data_a", "//:data_b", "//:combined");

    assertThat(readFileBytes(outputA)).isEqualTo(contentA);
    assertThat(readFileBytes(outputB)).isEqualTo(contentB);
    assertThat(readFileBytes(outputCombined)).isEqualTo(contentCombined);
  }
}
