// Copyright 2017 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkNotNull;
import static java.lang.String.format;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableMap;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.buildeventstream.ArtifactGroupNamer;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile;
import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader;
import com.google.devtools.build.lib.buildeventstream.BuildEventContext;
import com.google.devtools.build.lib.buildeventstream.BuildEventProtocolOptions;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransport;
import com.google.devtools.build.lib.buildeventstream.PathConverter;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.io.AsynchronousFileOutputStream;
import com.google.devtools.build.lib.vfs.Path;
import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Collection;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.function.Consumer;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.annotation.concurrent.ThreadSafe;

/**
 * Non-blocking file transport.
 *
 * <p>Implementors of this class need to implement {@code #sendBuildEvent(BuildEvent)} which
 * serializes the build event and writes it to file using {@link
 * AsynchronousFileOutputStream#write}.
 */
abstract class FileTransport implements BuildEventTransport {
  private static final Logger logger = Logger.getLogger(FileTransport.class.getName());

  private final BuildEventProtocolOptions options;
  private final BuildEventArtifactUploader uploader;
  private final Consumer<AbruptExitException> exitFunc;
  @VisibleForTesting final SequentialWriter writer;

  FileTransport(
      String path,
      BuildEventProtocolOptions options,
      BuildEventArtifactUploader uploader,
      Consumer<AbruptExitException> exitFunc) {
    this.uploader = uploader;
    this.options = options;
    this.exitFunc = exitFunc;
    this.writer = new SequentialWriter(path, this::serializeEvent, exitFunc);
  }

  @ThreadSafe
  @VisibleForTesting
  static final class SequentialWriter implements Runnable {
    private static final Logger logger = Logger.getLogger(SequentialWriter.class.getName());
    private static final ListenableFuture<BuildEventStreamProtos.BuildEvent> CLOSE =
        Futures.immediateCancelledFuture();

    private final Thread writerThread;
    @VisibleForTesting OutputStream out;
    private final Function<BuildEventStreamProtos.BuildEvent, byte[]> serializeFunc;
    private final Consumer<AbruptExitException> exitFunc;

    @VisibleForTesting
    final BlockingQueue<ListenableFuture<BuildEventStreamProtos.BuildEvent>> pendingWrites =
        new LinkedBlockingDeque<>();

    private final SettableFuture<Void> closeFuture = SettableFuture.create();

    SequentialWriter(
        String path,
        Function<BuildEventStreamProtos.BuildEvent, byte[]> serializeFunc,
        Consumer<AbruptExitException> exitFunc) {
      try {
        this.out = new BufferedOutputStream(new FileOutputStream(path));
      } catch (FileNotFoundException e) {
        this.out = new ByteArrayOutputStream(0);
        closeNow();
        exitFunc.accept(
            new AbruptExitException(
                format("Couldn't open BEP file '%s' for writing.", path),
                ExitCode.PUBLISH_ERROR,
                e));
      }
      this.writerThread = new Thread(this);
      this.serializeFunc = serializeFunc;
      this.exitFunc = exitFunc;
      writerThread.start();
    }

    @Override
    public void run() {
      ListenableFuture<BuildEventStreamProtos.BuildEvent> buildEventF;
      try {
        while ((buildEventF = pendingWrites.take()) != CLOSE) {
          BuildEventStreamProtos.BuildEvent buildEvent = buildEventF.get();
          byte[] serialized = serializeFunc.apply(buildEvent);
          out.write(serialized);
        }
      } catch (Exception e) {
        exitFunc.accept(
            new AbruptExitException(
                "Failed to write BEP events to file.", ExitCode.PUBLISH_ERROR, e));
        pendingWrites.clear();
        logger.log(Level.SEVERE, "Failed to write BEP events to file.", e);
      } finally {
        try {
          closeFuture.set(null);
          out.flush();
          out.close();
        } catch (IOException e) {
          logger.log(Level.SEVERE, "Failed to close BEP file output stream.", e);
        }
      }
    }

    public void closeNow() {
      if (closeFuture.isDone()) {
        return;
      }
      try {
        pendingWrites.clear();
        pendingWrites.put(CLOSE);
      } catch (InterruptedException e) {
        logger.log(Level.SEVERE, "Failed to immediately close the sequential writer.", e);
      }
    }

    public ListenableFuture<Void> close() {
      if (closeFuture.isDone()) {
        return closeFuture;
      }
      try {
        pendingWrites.put(CLOSE);
      } catch (InterruptedException e) {
        closeNow();
        logger.log(Level.SEVERE, "Failed to close the sequential writer.", e);
        closeFuture.set(null);
      }
      return closeFuture;
    }
  }

  @Override
  public void sendBuildEvent(BuildEvent event, ArtifactGroupNamer namer) {
    if (writer.closeFuture.isDone()) {
      return;
    }
    if (!writer.pendingWrites.add(asStreamProto(event, namer))) {
      logger.log(Level.SEVERE, "Failed to add BEP event to the write queue");
    }
  }

  protected abstract byte[] serializeEvent(BuildEventStreamProtos.BuildEvent buildEvent);

  @Override
  public ListenableFuture<Void> close() {
    return writer.close();
  }

  @Override
  public synchronized void closeNow() {
    writer.closeNow();
  }

  /**
   * Converts the given event into a proto object; this may trigger uploading of referenced files as
   * a side effect. May return {@code null} if there was an interrupt. This method is not
   * thread-safe.
   */
  protected ListenableFuture<BuildEventStreamProtos.BuildEvent> asStreamProto(
      BuildEvent event, ArtifactGroupNamer namer) {
    checkNotNull(event);

    return Futures.transform(
        uploadReferencedFiles(event.referencedLocalFiles()),
        pathConverter -> {
          BuildEventContext context =
              new BuildEventContext() {
                @Override
                public PathConverter pathConverter() {
                  return pathConverter;
                }

                @Override
                public ArtifactGroupNamer artifactGroupNamer() {
                  return namer;
                }

                @Override
                public BuildEventProtocolOptions getOptions() {
                  return options;
                }
              };
          return event.asStreamProto(context);
        },
        MoreExecutors.directExecutor());
  }

  /**
   * Returns a {@link PathConverter} for the uploaded files, or {@code null} when the uploaded
   * failed.
   */
  private ListenableFuture<PathConverter> uploadReferencedFiles(Collection<LocalFile> localFiles) {
    checkNotNull(localFiles);
    ImmutableMap.Builder<Path, LocalFile> localFileMap =
        ImmutableMap.builderWithExpectedSize(localFiles.size());
    for (LocalFile localFile : localFiles) {
      localFileMap.put(localFile.path, localFile);
    }
    ListenableFuture<PathConverter> upload = uploader.upload(localFileMap.build());
    Futures.addCallback(
        upload,
        new FutureCallback<PathConverter>() {
          @Override
          public void onSuccess(PathConverter result) {
            // Intentionally left empty.
          }

          @Override
          public void onFailure(Throwable t) {
            exitFunc.accept(
                new AbruptExitException(
                    Throwables.getStackTraceAsString(t), ExitCode.PUBLISH_ERROR, t));
          }
        },
        MoreExecutors.directExecutor());
    return upload;
  }
}
