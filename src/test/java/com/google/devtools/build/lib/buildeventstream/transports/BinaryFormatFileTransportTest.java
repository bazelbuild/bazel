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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.buildeventservice.BuildEventServiceOptions.BesUploadMode;
import com.google.devtools.build.lib.buildeventstream.ArtifactGroupNamer;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile.LocalFileType;
import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader;
import com.google.devtools.build.lib.buildeventstream.BuildEventContext;
import com.google.devtools.build.lib.buildeventstream.BuildEventIdUtil;
import com.google.devtools.build.lib.buildeventstream.BuildEventProtocolOptions;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildStarted;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.Progress;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.TargetComplete;
import com.google.devtools.build.lib.buildeventstream.LocalFilesArtifactUploader;
import com.google.devtools.build.lib.buildeventstream.PathConverter;
import com.google.devtools.build.lib.buildeventstream.PathConverter.FileUriPathConverter;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.Options;
import io.netty.util.AbstractReferenceCounted;
import io.netty.util.ReferenceCounted;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.Collection;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.locks.LockSupport;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.ArgumentMatchers;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

/** Tests {@link BinaryFormatFileTransport}. */
@RunWith(JUnit4.class)
public class BinaryFormatFileTransportTest {
  private final BuildEventProtocolOptions defaultOpts =
      Options.getDefaults(BuildEventProtocolOptions.class);

  @Rule public TemporaryFolder tmp = new TemporaryFolder();

  @Mock public BuildEvent buildEvent;

  @Mock public ArtifactGroupNamer artifactGroupNamer;

  @Before
  public void setUp() {
    MockitoAnnotations.initMocks(this);
  }

  @After
  public void tearDown() {
    Mockito.validateMockitoUsage();
  }

  @Test
  public void testCreatesFileAndWritesProtoBinaryFormat() throws Exception {
    File output = tmp.newFile();
    BufferedOutputStream outputStream =
        new BufferedOutputStream(Files.newOutputStream(Paths.get(output.getAbsolutePath())));

    BuildEventStreamProtos.BuildEvent started =
        BuildEventStreamProtos.BuildEvent.newBuilder()
            .setStarted(BuildStarted.newBuilder().setCommand("build"))
            .build();
    when(buildEvent.asStreamProto(ArgumentMatchers.<BuildEventContext>any())).thenReturn(started);
    BinaryFormatFileTransport transport =
        new BinaryFormatFileTransport(
            outputStream,
            defaultOpts,
            new LocalFilesArtifactUploader(),
            artifactGroupNamer,
            BesUploadMode.WAIT_FOR_UPLOAD_COMPLETE);
    transport.sendBuildEvent(buildEvent);

    BuildEventStreamProtos.BuildEvent progress =
        BuildEventStreamProtos.BuildEvent.newBuilder()
            .setProgress(Progress.getDefaultInstance())
            .build();
    when(buildEvent.asStreamProto(ArgumentMatchers.<BuildEventContext>any())).thenReturn(progress);
    transport.sendBuildEvent(buildEvent);

    BuildEventStreamProtos.BuildEvent completed =
        BuildEventStreamProtos.BuildEvent.newBuilder()
            .setCompleted(TargetComplete.newBuilder().setSuccess(true))
            .build();
    when(buildEvent.asStreamProto(ArgumentMatchers.<BuildEventContext>any())).thenReturn(completed);
    transport.sendBuildEvent(buildEvent);

    transport.close().get();
    try (InputStream in = new FileInputStream(output)) {
      assertThat(BuildEventStreamProtos.BuildEvent.parseDelimitedFrom(in)).isEqualTo(started);
      assertThat(BuildEventStreamProtos.BuildEvent.parseDelimitedFrom(in)).isEqualTo(progress);
      assertThat(BuildEventStreamProtos.BuildEvent.parseDelimitedFrom(in)).isEqualTo(completed);
      assertThat(in.available()).isEqualTo(0);
    }
  }

  @Test
  public void testCancelledUpload() throws Exception {
    Path file1 = Mockito.mock(Path.class);
    when(file1.getBaseName()).thenReturn("foo");
    BuildEvent event1 = new WithLocalFilesEvent(ImmutableList.of(file1));

    BuildEventArtifactUploader uploader =
        Mockito.spy(
            new BuildEventArtifactUploaderWithRefCounting() {
              @Override
              public ListenableFuture<PathConverter> upload(Map<Path, LocalFile> files) {
                return Futures.immediateCancelledFuture();
              }

              @Override
              public boolean mayBeSlow() {
                return false;
              }
            });

    File output = tmp.newFile();
    BufferedOutputStream outputStream =
        new BufferedOutputStream(Files.newOutputStream(Paths.get(output.getAbsolutePath())));
    BinaryFormatFileTransport transport =
        new BinaryFormatFileTransport(
            outputStream,
            defaultOpts,
            uploader,
            artifactGroupNamer,
            BesUploadMode.WAIT_FOR_UPLOAD_COMPLETE);
    transport.sendBuildEvent(event1);

    ExecutionException expected =
        assertThrows(ExecutionException.class, () -> transport.close().get());
    assertThat(expected)
        .hasMessageThat()
        .contains("Unable to write all BEP events to file due to 'Task was cancelled.'");

    try (InputStream in = new FileInputStream(output)) {
      assertThat(BuildEventStreamProtos.BuildEvent.parseDelimitedFrom(in)).isNull();
      assertThat(in.available()).isEqualTo(0);
    }
  }

  @Test
  public void testWriteWhenFileClosed() throws Exception {
    File output = tmp.newFile();
    BufferedOutputStream outputStream =
        new BufferedOutputStream(Files.newOutputStream(Paths.get(output.getAbsolutePath())));

    BuildEventStreamProtos.BuildEvent started =
        BuildEventStreamProtos.BuildEvent.newBuilder()
            .setStarted(BuildStarted.newBuilder().setCommand("build"))
            .build();
    when(buildEvent.asStreamProto(ArgumentMatchers.<BuildEventContext>any())).thenReturn(started);

    BinaryFormatFileTransport transport =
        new BinaryFormatFileTransport(
            outputStream,
            defaultOpts,
            new LocalFilesArtifactUploader(),
            artifactGroupNamer,
            BesUploadMode.WAIT_FOR_UPLOAD_COMPLETE);

    transport.close().get();

    // This should not throw an exception.
    transport.sendBuildEvent(buildEvent);
    transport.close().get();

    // Also, nothing should have been written to the file
    try (InputStream in = new FileInputStream(output)) {
      assertThat(in.available()).isEqualTo(0);
    }
  }

  @Test
  public void testWriteWhenTransportClosed() throws Exception {
    File output = tmp.newFile();
    BufferedOutputStream outputStream =
        new BufferedOutputStream(Files.newOutputStream(Paths.get(output.getAbsolutePath())));

    BuildEventStreamProtos.BuildEvent started =
        BuildEventStreamProtos.BuildEvent.newBuilder()
            .setStarted(BuildStarted.newBuilder().setCommand("build"))
            .build();
    when(buildEvent.asStreamProto(ArgumentMatchers.<BuildEventContext>any())).thenReturn(started);

    BinaryFormatFileTransport transport =
        new BinaryFormatFileTransport(
            outputStream,
            defaultOpts,
            new LocalFilesArtifactUploader(),
            artifactGroupNamer,
            BesUploadMode.WAIT_FOR_UPLOAD_COMPLETE);

    transport.sendBuildEvent(buildEvent);
    Future<Void> closeFuture = transport.close();
    closeFuture.get();
    // This should not throw an exception, but also not perform any write.
    transport.sendBuildEvent(buildEvent);

    // There should have only been one write.
    try (InputStream in = new FileInputStream(output)) {
      assertThat(BuildEventStreamProtos.BuildEvent.parseDelimitedFrom(in)).isEqualTo(started);
      assertThat(in.available()).isEqualTo(0);
    }
  }

  @Test
  public void testWritesWithUploadDelays() throws Exception {
    // Test that events are written in order if the first event
    // has to wait a bit for local file uploads to finish.

    Path file1 = Mockito.mock(Path.class);
    when(file1.getBaseName()).thenReturn("file1");
    Path file2 = Mockito.mock(Path.class);
    when(file2.getBaseName()).thenReturn("file2");
    BuildEvent event1 = new WithLocalFilesEvent(ImmutableList.of(file1));
    BuildEvent event2 = new WithLocalFilesEvent(ImmutableList.of(file2));

    BuildEventArtifactUploader uploader =
        Mockito.spy(
            new BuildEventArtifactUploaderWithRefCounting() {
              @Override
              public ListenableFuture<PathConverter> upload(Map<Path, LocalFile> files) {
                if (files.containsKey(file1)) {
                  LockSupport.parkNanos(Duration.ofMillis(200).toNanos());
                }
                return Futures.immediateFuture(new FileUriPathConverter());
              }

              @Override
              public boolean mayBeSlow() {
                return true;
              }
            });
    File output = tmp.newFile();
    BufferedOutputStream outputStream =
        new BufferedOutputStream(Files.newOutputStream(Paths.get(output.getAbsolutePath())));
    BinaryFormatFileTransport transport =
        new BinaryFormatFileTransport(
            outputStream,
            defaultOpts,
            uploader,
            artifactGroupNamer,
            BesUploadMode.WAIT_FOR_UPLOAD_COMPLETE);
    transport.sendBuildEvent(event1);
    transport.sendBuildEvent(event2);
    transport.close().get();

    try (InputStream in = new FileInputStream(output)) {
      assertThat(BuildEventStreamProtos.BuildEvent.parseDelimitedFrom(in))
          .isEqualTo(event1.asStreamProto(null));
      assertThat(BuildEventStreamProtos.BuildEvent.parseDelimitedFrom(in))
          .isEqualTo(event2.asStreamProto(null));
      assertThat(in.available()).isEqualTo(0);
    }

    verify(uploader).release();
  }

  /** Regression test for b/207287675 */
  @Test
  public void testHandlesDuplicateFiles() throws Exception {
    Path file1 = Mockito.mock(Path.class);
    when(file1.getBaseName()).thenReturn("foo");
    BuildEvent event1 = new WithLocalFilesEvent(ImmutableList.of(file1, file1));

    BuildEventArtifactUploader uploader =
        Mockito.spy(
            new BuildEventArtifactUploaderWithRefCounting() {
              @Override
              public ListenableFuture<PathConverter> upload(Map<Path, LocalFile> files) {
                return Futures.immediateFuture(new FileUriPathConverter());
              }

              @Override
              public boolean mayBeSlow() {
                return false;
              }
            });
    File output = tmp.newFile();
    BufferedOutputStream outputStream =
        new BufferedOutputStream(Files.newOutputStream(Paths.get(output.getAbsolutePath())));
    BinaryFormatFileTransport transport =
        new BinaryFormatFileTransport(
            outputStream,
            defaultOpts,
            uploader,
            artifactGroupNamer,
            BesUploadMode.WAIT_FOR_UPLOAD_COMPLETE);
    transport.sendBuildEvent(event1);
    transport.close().get();

    try (InputStream in = new FileInputStream(output)) {
      assertThat(BuildEventStreamProtos.BuildEvent.parseDelimitedFrom(in))
          .isEqualTo(event1.asStreamProto(null));
      assertThat(in.available()).isEqualTo(0);
    }
  }

  @Test
  public void testCloseWaitsForWritesToFinish() throws Exception {
    // Test that .close() waits for all writes to finish.

    Path file1 = Mockito.mock(Path.class);
    when(file1.getBaseName()).thenReturn("file1");
    BuildEvent event = new WithLocalFilesEvent(ImmutableList.of(file1));

    SettableFuture<PathConverter> upload = SettableFuture.create();
    BuildEventArtifactUploader uploader =
        Mockito.spy(
            new BuildEventArtifactUploaderWithRefCounting() {
              @Override
              public ListenableFuture<PathConverter> upload(Map<Path, LocalFile> files) {
                return upload;
              }

              @Override
              public boolean mayBeSlow() {
                return false;
              }
            });

    File output = tmp.newFile();
    BufferedOutputStream outputStream =
        new BufferedOutputStream(Files.newOutputStream(Paths.get(output.getAbsolutePath())));
    BinaryFormatFileTransport transport =
        new BinaryFormatFileTransport(
            outputStream,
            defaultOpts,
            uploader,
            artifactGroupNamer,
            BesUploadMode.WAIT_FOR_UPLOAD_COMPLETE);
    transport.sendBuildEvent(event);
    ListenableFuture<Void> closeFuture = transport.close();

    upload.set(PathConverter.NO_CONVERSION);

    closeFuture.get();

    try (InputStream in = new FileInputStream(output)) {
      assertThat(BuildEventStreamProtos.BuildEvent.parseDelimitedFrom(in))
          .isEqualTo(event.asStreamProto(null));
      assertThat(in.available()).isEqualTo(0);
    }

    verify(uploader).release();
  }

  private static class WithLocalFilesEvent implements BuildEvent {

    int id;
    ImmutableList<Path> files;

    WithLocalFilesEvent(ImmutableList<Path> files) {
      this.files = files;
    }

    @Override
    public Collection<LocalFile> referencedLocalFiles() {
      return files.stream()
          .map(f -> new LocalFile(f, LocalFileType.OUTPUT_FILE, /* artifactMetadata= */ null))
          .collect(toImmutableList());
    }

    @Override
    public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext context) {
      return BuildEventStreamProtos.BuildEvent.newBuilder()
          .setId(BuildEventIdUtil.progressId(id))
          .setProgress(
              BuildEventStreamProtos.Progress.newBuilder()
                  .setStdout(
                      "uploading: "
                          + Joiner.on(", ")
                              .join(
                                  files.stream().map(Path::getBaseName).collect(toImmutableList())))
                  .build())
          .build();
    }

    @Override
    public BuildEventId getEventId() {
      return BuildEventIdUtil.progressId(id);
    }

    @Override
    public Collection<BuildEventId> getChildrenEvents() {
      return ImmutableList.of(BuildEventIdUtil.progressId(id + 1));
    }
  }

  private abstract static class BuildEventArtifactUploaderWithRefCounting
      extends AbstractReferenceCounted implements BuildEventArtifactUploader {

    @Override
    protected void deallocate() {}

    @Override
    public ReferenceCounted touch(Object o) {
      return this;
    }
  }
}
