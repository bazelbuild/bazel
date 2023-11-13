// Copyright 2016 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.buildeventstream.transports;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.mockito.Mockito.when;

import com.google.devtools.build.lib.buildeventservice.BuildEventServiceOptions.BesUploadMode;
import com.google.devtools.build.lib.buildeventstream.ArtifactGroupNamer;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventProtocolOptions;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.ActionExecuted;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId.ActionCompletedId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildStarted;
import com.google.devtools.build.lib.buildeventstream.LocalFilesArtifactUploader;
import com.google.devtools.build.lib.buildeventstream.PathConverter;
import com.google.devtools.build.lib.buildeventstream.transports.JsonFormatFileTransport.UnknownAnyProtoError;
import com.google.devtools.build.lib.exec.Protos.SpawnExec;
import com.google.devtools.common.options.Options;
import com.google.gson.GsonBuilder;
import com.google.protobuf.Any;
import com.google.protobuf.util.JsonFormat;
import com.google.protobuf.util.JsonFormat.TypeRegistry;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Paths;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.ArgumentMatchers;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnit;
import org.mockito.junit.MockitoRule;
import org.mockito.quality.Strictness;

/** Tests {@link JsonFormatFileTransport}. */
@RunWith(JUnit4.class)
public class JsonFormatFileTransportTest {

  private static final TypeRegistry SPAWN_EXEC_TYPE_REGISTRY =
      TypeRegistry.newBuilder().add(SpawnExec.getDescriptor()).build();
  private final BuildEventProtocolOptions defaultOpts =
      Options.getDefaults(BuildEventProtocolOptions.class);

  @Rule public TemporaryFolder tmp = new TemporaryFolder();
  @Rule public MockitoRule mocks = MockitoJUnit.rule().strictness(Strictness.STRICT_STUBS);

  @Mock public BuildEvent buildEvent;

  @Mock public PathConverter pathConverter;
  @Mock public ArtifactGroupNamer artifactGroupNamer;

  private File output = null;
  private BufferedOutputStream outputStream = null;
  private JsonFormatFileTransport underTest = null;

  @Before
  public void setUp() throws IOException {
    output = tmp.newFile();
    outputStream =
        new BufferedOutputStream(Files.newOutputStream(Paths.get(output.getAbsolutePath())));
    underTest =
        new JsonFormatFileTransport(
            outputStream,
            defaultOpts,
            new LocalFilesArtifactUploader(),
            artifactGroupNamer,
            SPAWN_EXEC_TYPE_REGISTRY,
            BesUploadMode.WAIT_FOR_UPLOAD_COMPLETE);
  }

  @Test
  public void testCreatesFileAndWritesProtoJsonFormat() throws Exception {
    // Arrange: Prepare a simple BEP event that can be round-tripped.
    BuildEventStreamProtos.BuildEvent started =
        BuildEventStreamProtos.BuildEvent.newBuilder()
            .setStarted(BuildStarted.newBuilder().setCommand("build"))
            .build();
    when(buildEvent.asStreamProto(ArgumentMatchers.any())).thenReturn(started);

    // Act: Send the simple BuildStarted event.
    underTest.sendBuildEvent(buildEvent);
    underTest.close().get();

    // Assert: Read back the BEP event and confirm it round-tripped.
    try (Reader reader = openOutputReader()) {
      JsonFormat.Parser parser = JsonFormat.parser();
      BuildEventStreamProtos.BuildEvent.Builder builder =
          BuildEventStreamProtos.BuildEvent.newBuilder();
      parser.merge(reader, builder);
      assertThat(builder.build()).isEqualTo(started);
    }
  }

  @Test
  public void expandsKnownAnyType() throws Exception {
    // Arrange: Prepare an Any event that is recognized by the JSON formatter.
    Any spawnExecAny = Any.pack(SpawnExec.newBuilder().setExitCode(1).setMnemonic("Javac").build());
    BuildEventStreamProtos.BuildEvent action = makeActionEventWithAnyDetails(spawnExecAny);
    when(buildEvent.asStreamProto(ArgumentMatchers.any())).thenReturn(action);

    // Act: Write the event and close the transport to force flushing.
    underTest.sendBuildEvent(buildEvent);
    underTest.close().get();

    // Assert: Confirm we get the SpawnExec back.
    try (Reader reader = openOutputReader()) {
      JsonFormat.Parser parser = JsonFormat.parser().usingTypeRegistry(SPAWN_EXEC_TYPE_REGISTRY);
      BuildEventStreamProtos.BuildEvent.Builder builder =
          BuildEventStreamProtos.BuildEvent.newBuilder();
      parser.merge(reader, builder);
      assertThat(builder.build()).isEqualTo(action);
    }
  }

  @Test
  public void rejectsUnknownAnyType() throws Exception {
    // Arrange: Prepare a BuildEvent that cannot be serialized due to an unrecognized Any type.
    Any bogusAnyProto =
        Any.pack(
            BuildEventId.newBuilder()
                .setActionCompleted(ActionCompletedId.newBuilder().setLabel("//:foo"))
                .build());
    BuildEventStreamProtos.BuildEvent action = makeActionEventWithAnyDetails(bogusAnyProto);
    when(buildEvent.asStreamProto(ArgumentMatchers.any())).thenReturn(action);

    // Act: Send the event with a bogus Any value.
    underTest.sendBuildEvent(buildEvent);
    underTest.close().get();

    // Assert: Special invalid event message is written in JSON.
    try (BufferedReader reader = new BufferedReader(openOutputReader())) {
      String jsonLine = reader.readLine();
      UnknownAnyProtoError error =
          new GsonBuilder().create().fromJson(jsonLine, UnknownAnyProtoError.class);
      assertThat(error).isNotNull();
    }
  }

  @Test
  public void testFlushesStreamAfterSmallWrites() throws Exception {
    // Arrange: JsonFormatFileTransport writes to a wrapped output stream to verify flushing.
    WrappedOutputStream wrappedOutputStream = new WrappedOutputStream(outputStream);
    underTest =
        new JsonFormatFileTransport(
            wrappedOutputStream,
            defaultOpts,
            new LocalFilesArtifactUploader(),
            artifactGroupNamer,
            SPAWN_EXEC_TYPE_REGISTRY,
            BesUploadMode.WAIT_FOR_UPLOAD_COMPLETE);

    BuildEventStreamProtos.BuildEvent started =
        BuildEventStreamProtos.BuildEvent.newBuilder()
            .setStarted(BuildStarted.newBuilder().setCommand("build"))
            .build();
    when(buildEvent.asStreamProto(ArgumentMatchers.any())).thenReturn(started);

    // Act: Write an event, then wait for three flush intervals.
    underTest.sendBuildEvent(buildEvent);
    Thread.sleep(underTest.getFlushInterval().toMillis() * 3);

    // Assert: Confirm BEP events were written even though the file transport is not closed.

    // Some users, e.g. Tulsi, use JSON build event output for interactive use and expect the stream
    // to be flushed at regular short intervals.
    assertThat(wrappedOutputStream.flushCount).isGreaterThan(0);

    // We know that large writes get flushed; test is valuable only if we check small writes,
    // meaning smaller than 8192, the default buffer size used by BufferedOutputStream.
    assertThat(wrappedOutputStream.byteCount).isLessThan(8192L);
    assertThat(wrappedOutputStream.byteCount).isGreaterThan(0L);

    underTest.close().get();
  }

  private InputStreamReader openOutputReader() throws FileNotFoundException {
    return new InputStreamReader(new FileInputStream(output), UTF_8);
  }

  private static BuildEventStreamProtos.BuildEvent makeActionEventWithAnyDetails(
      Any strategyDetails) {
    return BuildEventStreamProtos.BuildEvent.newBuilder()
        .setAction(
            ActionExecuted.newBuilder().setExitCode(1).addStrategyDetails(strategyDetails).build())
        .build();
  }

  /**
   * A thin wrapper around an OutputStream that counts number of bytes written and verifies flushes.
   *
   * <p>The methods below need to be synchronized because they override methods from {@link
   * BufferedOutputStream} *not* because there is concurrent access to the stream.
   */
  private static final class WrappedOutputStream extends BufferedOutputStream {
    private long byteCount;
    private int flushCount;

    WrappedOutputStream(OutputStream out) {
      super(out);
      this.out = out;
    }

    @Override
    public synchronized void write(int b) throws IOException {
      out.write(b);
      byteCount++;
    }

    @Override
    public synchronized void write(byte[] b) throws IOException {
      out.write(b);
      byteCount += b.length;
    }

    @Override
    public synchronized void write(byte[] b, int off, int len) throws IOException {
      out.write(b, off, len);
      byteCount += len;
    }

    @Override
    public synchronized void flush() throws IOException {
      out.flush();
      flushCount++;
    }
  }
}
