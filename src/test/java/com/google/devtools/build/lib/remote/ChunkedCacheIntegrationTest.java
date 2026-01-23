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
import static com.google.devtools.build.lib.vfs.FileSystemUtils.readContent;
import static java.nio.charset.StandardCharsets.UTF_8;

import build.bazel.remote.execution.v2.ContentAddressableStorageGrpc;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.RequestMetadata;
import build.bazel.remote.execution.v2.SplitBlobRequest;
import build.bazel.remote.execution.v2.SplitBlobResponse;
import build.bazel.remote.execution.v2.ToolDetails;
import com.google.bytestream.ByteStreamGrpc;
import com.google.bytestream.ByteStreamProto.ReadRequest;
import com.google.bytestream.ByteStreamProto.ReadResponse;
import com.google.common.collect.ImmutableList;
import com.google.common.hash.HashCode;
import com.google.common.hash.Hashing;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.authandtls.credentialhelper.CredentialModule;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.remote.util.IntegrationTestUtils;
import com.google.devtools.build.lib.remote.util.IntegrationTestUtils.WorkerInstance;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.BlockWaitingModule;
import com.google.devtools.build.lib.runtime.BuildSummaryStatsModule;
import com.google.devtools.build.lib.standalone.StandaloneModule;
import com.google.devtools.build.lib.vfs.Path;
import io.grpc.ClientInterceptor;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Iterator;
import java.util.List;
import org.junit.After;
import org.junit.ClassRule;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Integration tests for chunked remote cache using SplitBlob/SpliceBlob APIs. */
@RunWith(JUnit4.class)
public class ChunkedCacheIntegrationTest extends BuildIntegrationTestCase {
  @ClassRule @Rule public static final WorkerInstance worker = IntegrationTestUtils.createWorker();

  @Override
  protected void setupOptions() throws Exception {
    super.setupOptions();
    addOptions(
        "--remote_cache=grpc://localhost:" + worker.getPort(),
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
  public void waitDownloads() throws Exception {
    runtimeWrapper.newCommand();
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

  private Digest computeDigest(byte[] data) {
    HashCode hash = Hashing.sha256().hashBytes(data);
    return Digest.newBuilder().setHash(hash.toString()).setSizeBytes(data.length).build();
  }

  @Test
  public void uploadAndDownloadLargeBlob_withChunking_succeeds() throws Exception {
    write(
        "BUILD",
        """
        genrule(
            name = "large_file",
            srcs = [],
            outs = ["large.txt"],
            cmd = "dd if=/dev/zero bs=1M count=3 2>/dev/null | tr '\\\\0' 'a' > $@",
        )
        """);

    buildTarget("//:large_file");

    Path output = getOutputPath("large.txt");
    assertThat(output.exists()).isTrue();
    byte[] originalContent = readFileBytes(output);
    assertThat(originalContent.length).isAtLeast(2 * 1024 * 1024);

    Digest blobDigest = computeDigest(originalContent);

    // Verify SplitBlob returns multiple chunks and each chunk is individually downloadable.
    RequestMetadata metadata =
        RequestMetadata.newBuilder()
            .setCorrelatedInvocationsId("test-build-id")
            .setToolInvocationId("test-command-id")
            .setActionId("test-action-id")
            .setToolDetails(ToolDetails.newBuilder().setToolName("bazel").setToolVersion("test"))
            .build();
    ClientInterceptor interceptor = TracingMetadataUtils.attachMetadataInterceptor(metadata);

    ManagedChannel channel =
        ManagedChannelBuilder.forAddress("localhost", worker.getPort())
            .usePlaintext()
            .intercept(interceptor)
            .build();
    try {
      ContentAddressableStorageGrpc.ContentAddressableStorageBlockingStub casStub =
          ContentAddressableStorageGrpc.newBlockingStub(channel);

      SplitBlobResponse splitResponse =
          casStub.splitBlob(SplitBlobRequest.newBuilder().setBlobDigest(blobDigest).build());
      List<Digest> chunkDigests = splitResponse.getChunkDigestsList();

      assertThat(chunkDigests.size()).isGreaterThan(1);
      long totalChunkSize = chunkDigests.stream().mapToLong(Digest::getSizeBytes).sum();
      assertThat(totalChunkSize).isEqualTo(originalContent.length);

      // Download each chunk individually and reassemble to verify integrity.
      ByteStreamGrpc.ByteStreamBlockingStub bsStub = ByteStreamGrpc.newBlockingStub(channel);
      ByteArrayOutputStream reassembled = new ByteArrayOutputStream();
      for (Digest chunkDigest : chunkDigests) {
        String resourceName =
            "blobs/" + chunkDigest.getHash() + "/" + chunkDigest.getSizeBytes();
        Iterator<ReadResponse> readIter =
            bsStub.read(ReadRequest.newBuilder().setResourceName(resourceName).build());
        int chunkBytesRead = 0;
        while (readIter.hasNext()) {
          byte[] data = readIter.next().getData().toByteArray();
          reassembled.write(data);
          chunkBytesRead += data.length;
        }
        assertThat(chunkBytesRead).isEqualTo((int) chunkDigest.getSizeBytes());
      }
      assertThat(reassembled.toByteArray()).isEqualTo(originalContent);
    } finally {
      channel.shutdownNow();
    }

    // Delete output and action cache, then rebuild to exercise chunked download.
    output.delete();
    assertThat(output.exists()).isFalse();
    cleanAndRestartServer();

    buildTarget("//:large_file");

    assertThat(output.exists()).isTrue();
    assertThat(readFileBytes(output)).isEqualTo(originalContent);
  }

  @Test
  public void multipleTargets_withChunking_allSucceed() throws Exception {
    // Multiple large files built in parallel, with a downstream target that depends on them.
    // Use deterministic content (filled with distinct byte patterns) so we can verify integrity.
    write(
        "BUILD",
        """
        genrule(
            name = "data_a",
            srcs = [],
            outs = ["a.bin"],
            cmd = "dd if=/dev/zero bs=1M count=3 2>/dev/null | tr '\\\\0' 'A' > $@",
        )
        genrule(
            name = "data_b",
            srcs = [],
            outs = ["b.bin"],
            cmd = "dd if=/dev/zero bs=1M count=4 2>/dev/null | tr '\\\\0' 'B' > $@",
        )
        genrule(
            name = "combined",
            srcs = [":a.bin", ":b.bin"],
            outs = ["combined.bin"],
            cmd = "cat $(SRCS) > $@",
        )
        """);

    buildTarget("//:combined");

    Path outputA = getOutputPath("a.bin");
    Path outputB = getOutputPath("b.bin");
    Path outputCombined = getOutputPath("combined.bin");
    assertThat(outputA.exists()).isTrue();
    assertThat(outputB.exists()).isTrue();
    assertThat(outputCombined.exists()).isTrue();

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

    buildTarget("//:combined");

    assertThat(readFileBytes(outputA)).isEqualTo(contentA);
    assertThat(readFileBytes(outputB)).isEqualTo(contentB);
    assertThat(readFileBytes(outputCombined)).isEqualTo(contentCombined);
  }

  @Test
  public void buildWithChunking_smallFile_succeeds() throws Exception {
    write(
        "BUILD",
        """
        genrule(
            name = "small_file",
            srcs = [],
            outs = ["small.txt"],
            cmd = "echo 'hello world' > $@",
        )
        """);

    buildTarget("//:small_file");

    Path output = getOutputPath("small.txt");
    assertThat(output.exists()).isTrue();
    assertThat(readContent(output, UTF_8)).isEqualTo("hello world\n");
  }

  @Test
  public void mixedSizes_largeAndSmallOutputs_allSucceed() throws Exception {
    write(
        "BUILD",
        """
        genrule(
            name = "large",
            srcs = [],
            outs = ["large.bin"],
            cmd = "dd if=/dev/zero bs=1M count=3 2>/dev/null | tr '\\\\0' 'X' > $@",
        )
        genrule(
            name = "small",
            srcs = [],
            outs = ["small.txt"],
            cmd = "echo 'small output' > $@",
        )
        """);

    buildTarget("//:large", "//:small");

    Path largePath = getOutputPath("large.bin");
    Path smallPath = getOutputPath("small.txt");
    byte[] largeContent = readFileBytes(largePath);
    assertThat(largeContent.length).isEqualTo(3 * 1024 * 1024);
    assertThat(readContent(smallPath, UTF_8)).isEqualTo("small output\n");

    // Clean and rebuild.
    largePath.delete();
    smallPath.delete();
    cleanAndRestartServer();

    buildTarget("//:large", "//:small");

    assertThat(readFileBytes(largePath)).isEqualTo(largeContent);
    assertThat(readContent(smallPath, UTF_8)).isEqualTo("small output\n");
  }
}
