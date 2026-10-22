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

package com.google.devtools.build.remote.worker;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;

import build.bazel.remote.execution.v2.RequestMetadata;
import com.google.bytestream.ByteStreamGrpc;
import com.google.bytestream.ByteStreamGrpc.ByteStreamBlockingStub;
import com.google.bytestream.ByteStreamProto.ReadRequest;
import com.google.bytestream.ByteStreamProto.ReadResponse;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.remote.worker.LiveOutputStreams.ActionOutputs;
import com.google.devtools.build.remote.worker.LiveOutputStreams.LiveOutput;
import com.google.devtools.common.options.Options;
import com.google.protobuf.ByteString;
import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.ServerInterceptors;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import java.io.IOException;
import java.util.Iterator;
import java.util.concurrent.TimeUnit;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link LiveOutputStreams} and the live output reads of {@link ByteStreamServer}. */
@RunWith(JUnit4.class)
public final class LiveOutputStreamsTest {
  private final LiveOutputStreams liveOutputStreams = new LiveOutputStreams();
  private Server server;
  private ManagedChannel channel;

  @Before
  public void setUp() throws IOException {
    FileSystem fs = new InMemoryFileSystem(new JavaClock(), DigestHashFunction.SHA256);
    DigestUtil digestUtil = new DigestUtil(SyscallCache.NO_CACHE, DigestHashFunction.SHA256);
    Path workPath = fs.getPath("/work");
    workPath.createDirectoryAndParents();
    OnDiskBlobStoreCache cache =
        new OnDiskBlobStoreCache(
            fs.getPath("/cas"), digestUtil, Options.getDefaults(RemoteWorkerOptions.class));
    String serverName = InProcessServerBuilder.generateName();
    server =
        InProcessServerBuilder.forName(serverName)
            .addService(
                ServerInterceptors.intercept(
                    new ByteStreamServer(cache, workPath, digestUtil, liveOutputStreams),
                    // Like the real worker, which requires request metadata on every call.
                    new TracingMetadataUtils.ServerHeadersInterceptor()))
            .build()
            .start();
    channel = InProcessChannelBuilder.forName(serverName).build();
  }

  @After
  public void tearDown() throws InterruptedException {
    channel.shutdownNow();
    server.shutdownNow();
    server.awaitTermination(5, TimeUnit.SECONDS);
  }

  private static void write(LiveOutput output, String data) throws IOException {
    output.write(data.getBytes(UTF_8));
  }

  private Iterator<ReadResponse> read(String resourceName) {
    return read(ReadRequest.newBuilder().setResourceName(resourceName).build());
  }

  private Iterator<ReadResponse> read(ReadRequest request) {
    RequestMetadata metadata =
        TracingMetadataUtils.buildMetadata("build-id", "command-id", "action-id");
    ByteStreamBlockingStub stub =
        ByteStreamGrpc.newBlockingStub(channel)
            .withInterceptors(TracingMetadataUtils.attachMetadataInterceptor(metadata));
    return stub.read(request);
  }

  private static String next(Iterator<ReadResponse> responses) {
    return responses.next().getData().toStringUtf8();
  }

  private static String readAll(Iterator<ReadResponse> responses) {
    StringBuilder data = new StringBuilder();
    while (responses.hasNext()) {
      data.append(next(responses));
    }
    return data.toString();
  }

  @Test
  public void create_registersStdoutAndStderrStreamsOfTheOperation() {
    ActionOutputs outputs = liveOutputStreams.create("operation");

    assertThat(outputs.stdout().resourceName()).isEqualTo("operation/streams/stdout");
    assertThat(outputs.stderr().resourceName()).isEqualTo("operation/streams/stderr");
    assertThat(liveOutputStreams.get("operation/streams/stdout"))
        .isSameInstanceAs(outputs.stdout());
    assertThat(liveOutputStreams.get("operation/streams/stderr"))
        .isSameInstanceAs(outputs.stderr());
    assertThat(liveOutputStreams.get("other/streams/stdout")).isNull();
  }

  @Test
  public void liveOutput_read_returnsOutputFromOffsetUpToMaxBytes() throws Exception {
    LiveOutput output = liveOutputStreams.create("operation").stdout();
    write(output, "hello world");

    assertThat(output.read(0, 5, 0)).isEqualTo(ByteString.copyFromUtf8("hello"));
    assertThat(output.read(6, 100, 0)).isEqualTo(ByteString.copyFromUtf8("world"));
  }

  @Test
  public void liveOutput_read_atEndOfOpenStream_returnsEmptyAfterTimeout() throws Exception {
    LiveOutput output = liveOutputStreams.create("operation").stdout();
    write(output, "hello");

    assertThat(output.read(5, 100, 1)).isEmpty();
  }

  @Test
  public void liveOutput_read_atEndOfClosedStream_returnsNull() throws Exception {
    LiveOutput output = liveOutputStreams.create("operation").stdout();
    write(output, "hello");
    output.close();

    assertThat(output.read(5, 100, 0)).isNull();
  }

  @Test
  public void liveOutput_read_pastEndOfClosedStream_throws() throws Exception {
    LiveOutput output = liveOutputStreams.create("operation").stdout();
    write(output, "hello");
    output.close();

    assertThrows(IllegalArgumentException.class, () -> output.read(6, 100, 0));
  }

  @Test
  public void liveOutput_write_afterClose_isIgnored() throws Exception {
    LiveOutput output = liveOutputStreams.create("operation").stdout();
    write(output, "hello");
    output.close();

    write(output, " world");

    assertThat(output.read(0, 100, 0)).isEqualTo(ByteString.copyFromUtf8("hello"));
  }

  @Test
  public void liveOutput_close_keepsStreamReadableForLateReaders() throws Exception {
    LiveOutput output = liveOutputStreams.create("operation").stdout();
    write(output, "hello");
    output.close();

    // A fast action may finish before the client's read arrives.
    assertThat(liveOutputStreams.get("operation/streams/stdout")).isSameInstanceAs(output);
    assertThat(readAll(read("operation/streams/stdout"))).isEqualTo("hello");
  }

  @Test
  public void byteStreamRead_streamsOutputAsItIsWrittenUntilTheStreamIsClosed() throws Exception {
    LiveOutput output = liveOutputStreams.create("operation").stdout();
    write(output, "hello ");

    Iterator<ReadResponse> responses = read("operation/streams/stdout");

    // Each response arrives while the stream is still open, so the client sees the output live.
    assertThat(next(responses)).isEqualTo("hello ");
    write(output, "world");
    assertThat(next(responses)).isEqualTo("world");
    output.close();
    assertThat(responses.hasNext()).isFalse();
  }

  @Test
  public void byteStreamRead_stdoutAndStderr_areSeparateStreams() throws Exception {
    ActionOutputs outputs = liveOutputStreams.create("operation");
    write(outputs.stdout(), "out");
    write(outputs.stderr(), "err");
    outputs.close();

    assertThat(readAll(read("operation/streams/stdout"))).isEqualTo("out");
    assertThat(readAll(read("operation/streams/stderr"))).isEqualTo("err");
  }

  @Test
  public void byteStreamRead_withReadOffset_resumesWhereAPreviousReadStopped() throws Exception {
    LiveOutput output = liveOutputStreams.create("operation").stdout();
    write(output, "hello world");
    output.close();

    String data =
        readAll(
            read(
                ReadRequest.newBuilder()
                    .setResourceName("operation/streams/stdout")
                    .setReadOffset(6)
                    .build()));

    assertThat(data).isEqualTo("world");
  }

  @Test
  public void byteStreamRead_withReadLimit_stopsAtTheLimit() throws Exception {
    LiveOutput output = liveOutputStreams.create("operation").stdout();
    write(output, "hello world");

    String data =
        readAll(
            read(
                ReadRequest.newBuilder()
                    .setResourceName("operation/streams/stdout")
                    .setReadOffset(2)
                    .setReadLimit(3)
                    .build()));

    // The read completes without waiting for the stream to close.
    assertThat(data).isEqualTo("llo");
  }

  @Test
  public void byteStreamRead_pastEndOfClosedStream_isOutOfRange() throws Exception {
    LiveOutput output = liveOutputStreams.create("operation").stdout();
    write(output, "hello");
    output.close();

    StatusRuntimeException e =
        assertThrows(
            StatusRuntimeException.class,
            () ->
                readAll(
                    read(
                        ReadRequest.newBuilder()
                            .setResourceName("operation/streams/stdout")
                            .setReadOffset(6)
                            .build())));

    assertThat(e.getStatus().getCode()).isEqualTo(Status.Code.OUT_OF_RANGE);
  }

  @Test
  public void byteStreamRead_negativeReadOffset_isOutOfRange() throws Exception {
    liveOutputStreams.create("operation");

    StatusRuntimeException e =
        assertThrows(
            StatusRuntimeException.class,
            () ->
                readAll(
                    read(
                        ReadRequest.newBuilder()
                            .setResourceName("operation/streams/stdout")
                            .setReadOffset(-1)
                            .build())));

    assertThat(e.getStatus().getCode()).isEqualTo(Status.Code.OUT_OF_RANGE);
  }

  @Test
  public void byteStreamRead_unknownStream_isServedAsBlob() throws Exception {
    StatusRuntimeException e =
        assertThrows(StatusRuntimeException.class, () -> readAll(read("operation/streams/stdout")));

    // Not a live stream, so the regular blob read handles (and here rejects) the resource name.
    assertThat(e.getStatus().getCode()).isEqualTo(Status.Code.INVALID_ARGUMENT);
  }
}
