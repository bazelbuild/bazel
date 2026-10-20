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
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;

import build.bazel.remote.execution.v2.ExecuteOperationMetadata;
import build.bazel.remote.execution.v2.ExecutionStage;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.remote.LiveOutputStreamer.StreamReader;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.longrunning.Operation;
import com.google.protobuf.Any;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link LiveOutputStreamer}. */
@RunWith(JUnit4.class)
public final class LiveOutputStreamerTest {

  /** A read started by the streamer, which the test feeds and completes. */
  private record Read(String resourceName, OutputStream out, SettableFuture<Void> future) {
    void write(String data) throws IOException {
      out.write(data.getBytes(UTF_8));
    }
  }

  private final List<Read> reads = new ArrayList<>();
  private final StreamReader reader =
      (resourceName, out) -> {
        Read read = new Read(resourceName, out, SettableFuture.create());
        reads.add(read);
        return read.future();
      };
  private final List<String> failedStreams = new ArrayList<>();
  private Path stdout;
  private Path stderr;
  private FileOutErr outErr;

  @Before
  public void setUp() throws IOException {
    Path dir = TestUtils.createUniqueTmpDir(null);
    stdout = dir.getChild("test.log");
    stderr = dir.getChild("test.err");
    outErr = new FileOutErr(stdout, stderr);
  }

  private LiveOutputStreamer createStreamer() {
    return LiveOutputStreamer.create(
        reader, outErr, (streamName, cause) -> failedStreams.add(streamName));
  }

  private static Operation operation(String stdoutStreamName) {
    return operation(stdoutStreamName, "");
  }

  private static Operation operation(String stdoutStreamName, String stderrStreamName) {
    return Operation.newBuilder()
        .setName("operation")
        .setMetadata(
            Any.pack(
                ExecuteOperationMetadata.newBuilder()
                    .setStage(ExecutionStage.Value.EXECUTING)
                    .setStdoutStreamName(stdoutStreamName)
                    .setStderrStreamName(stderrStreamName)
                    .build()))
        .build();
  }

  private String stdoutContent() throws IOException {
    return FileSystemUtils.readContent(stdout, UTF_8);
  }

  private String stderrContent() throws IOException {
    return FileSystemUtils.readContent(stderr, UTF_8);
  }

  @Test
  public void noStreamAdvertised_doesNotRead() throws Exception {
    LiveOutputStreamer streamer = createStreamer();

    streamer.onNext(Operation.newBuilder().setName("operation").build());
    streamer.onNext(operation(""));
    streamer.close(/* discardOutput= */ true);

    assertThat(reads).isEmpty();
    assertThat(stdout.exists()).isFalse();
    assertThat(stderr.exists()).isFalse();
  }

  @Test
  public void advertisedStream_isReadOnceIntoStdoutFile() throws Exception {
    LiveOutputStreamer streamer = createStreamer();

    streamer.onNext(operation("streams/stdout"));
    streamer.onNext(operation("streams/stdout"));
    reads.get(0).write("hello ");
    reads.get(0).write("world");

    assertThat(reads).hasSize(1);
    assertThat(reads.get(0).resourceName()).isEqualTo("streams/stdout");
    assertThat(stdoutContent()).isEqualTo("hello world");
    assertThat(stderr.exists()).isFalse();
  }

  @Test
  public void stdoutAndStderrStreams_areReadIntoTheirFiles() throws Exception {
    LiveOutputStreamer streamer = createStreamer();

    streamer.onNext(operation("streams/stdout", "streams/stderr"));
    reads.get(0).write("out");
    reads.get(1).write("err");

    assertThat(reads).hasSize(2);
    assertThat(reads.get(0).resourceName()).isEqualTo("streams/stdout");
    assertThat(reads.get(1).resourceName()).isEqualTo("streams/stderr");
    assertThat(stdoutContent()).isEqualTo("out");
    assertThat(stderrContent()).isEqualTo("err");
  }

  @Test
  public void close_discardingOutput_truncatesInPlaceSoThatTailingReadersContinue()
      throws Exception {
    LiveOutputStreamer streamer = createStreamer();
    streamer.onNext(operation("streams/stdout", "streams/stderr"));
    reads.get(0).write("hello ");
    reads.get(1).write("warning");

    try (InputStream tail = stdout.getInputStream()) {
      // Like the streamed test output, which has already shown the live output.
      assertThat(new String(tail.readNBytes(6), UTF_8)).isEqualTo("hello ");

      streamer.close(/* discardOutput= */ true);
      assertThat(stdout.getFileSize()).isEqualTo(0);
      assertThat(stderr.getFileSize()).isEqualTo(0);
      // The download of the ActionResult's stdout then writes the final output, like FileOutErr.
      try (OutputStream out = stdout.getOutputStream()) {
        out.write("hello world\n".getBytes(UTF_8));
      }

      assertThat(new String(tail.readAllBytes(), UTF_8)).isEqualTo("world\n");
    }
  }

  @Test
  public void close_keepingOutput_leavesLiveOutputInPlace() throws Exception {
    LiveOutputStreamer streamer = createStreamer();
    streamer.onNext(operation("streams/stdout", "streams/stderr"));
    reads.get(0).write("partial");
    reads.get(1).write("warning");

    // Like an interrupted build, after which there's no final output to download.
    streamer.close(/* discardOutput= */ false);

    assertThat(reads.get(0).future().isCancelled()).isTrue();
    assertThat(reads.get(1).future().isCancelled()).isTrue();
    assertThat(stdoutContent()).isEqualTo("partial");
    assertThat(stderrContent()).isEqualTo("warning");
  }

  @Test
  public void close_stopsStreaming() throws Exception {
    LiveOutputStreamer streamer = createStreamer();
    streamer.onNext(operation("streams/stdout"));
    reads.get(0).write("live");

    streamer.close(/* discardOutput= */ true);
    streamer.onNext(operation("other/stdout"));

    assertThat(reads.get(0).future().isCancelled()).isTrue();
    assertThrows(IOException.class, () -> reads.get(0).write("late"));
    assertThat(stdoutContent()).isEmpty();
    assertThat(reads).hasSize(1);
  }

  @Test
  public void newStream_appendsOutputOfNewExecution() throws Exception {
    LiveOutputStreamer streamer = createStreamer();
    streamer.onNext(operation("first/stdout"));
    reads.get(0).write("first execution, ");

    try (InputStream tail = stdout.getInputStream()) {
      // Like the streamed test output, which has already shown the first execution's output.
      assertThat(new String(tail.readNBytes(17), UTF_8)).isEqualTo("first execution, ");

      streamer.onNext(operation("second/stdout"));
      reads.get(1).write("second execution");

      assertThat(reads.get(0).future().isCancelled()).isTrue();
      assertThrows(IOException.class, () -> reads.get(0).write("stale"));
      assertThat(reads.get(1).resourceName()).isEqualTo("second/stdout");
      assertThat(new String(tail.readAllBytes(), UTF_8)).isEqualTo("second execution");
    }
  }

  @Test
  public void create_returnsNullIfStdoutFileHasContent() throws Exception {
    FileSystemUtils.writeContent(stdout, UTF_8, "existing");

    assertThat(createStreamer()).isNull();
  }

  @Test
  public void create_returnsNullIfStderrFileHasContent() throws Exception {
    FileSystemUtils.writeContent(stderr, UTF_8, "existing");

    assertThat(createStreamer()).isNull();
  }

  @Test
  public void failedRead_isReported() throws Exception {
    LiveOutputStreamer streamer = createStreamer();
    streamer.onNext(operation("streams/stdout", "streams/stderr"));

    reads.get(1).future().setException(new IOException("stream not found"));

    assertThat(failedStreams).containsExactly("streams/stderr");
  }

  @Test
  public void endedOrCancelledReads_areNotReported() throws Exception {
    LiveOutputStreamer streamer = createStreamer();
    streamer.onNext(operation("first/stdout"));
    reads.get(0).future().set(null);
    streamer.onNext(operation("second/stdout"));
    streamer.onNext(operation("third/stdout"));
    streamer.close(/* discardOutput= */ true);

    assertThat(reads.get(1).future().isCancelled()).isTrue();
    assertThat(reads.get(2).future().isCancelled()).isTrue();
    assertThat(failedStreams).isEmpty();
  }
}
