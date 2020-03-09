// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.buildeventstream;

import com.google.common.collect.Maps;
import com.google.common.io.ByteStreams;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile.LocalFileType;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Collection;
import java.util.Map;
import java.util.concurrent.ScheduledExecutorService;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/** Uploads artifacts referenced by the Build Event Protocol (BEP). */
public interface BuildEventArtifactUploader {
  /**
   * Asynchronously uploads a set of files referenced by the protobuf representation of a {@link
   * BuildEvent}. This method is expected to return quickly.
   *
   * <p>This method must not throw any exceptions.
   *
   * <p>Returns a future to a {@link PathConverter} that must provide a name for each uploaded file
   * as it should appear in the BEP.
   */
  ListenableFuture<PathConverter> upload(Map<Path, LocalFile> files);

  /** The context associated with an in-flight remote upload. */
  interface UploadContext {

    /** The {@link OutputStream} to stream the file contents to. */
    @Nullable
    OutputStream getOutputStream();

    /** The future URI of the completed upload. */
    ListenableFuture<String> uriFuture();
  }

  /**
   * Initiate a streaming upload to the remote storage.
   *
   * <p>If inputSupplier is null, the caller is expected to write to the {@link
   * UploadContext#getOutputStream()}. If inputSupplier is non-null, {@link
   * UploadContext#getOutputStream()} is null.
   */
  default UploadContext startUpload(
      LocalFileType type, @Nullable Supplier<InputStream> inputSupplier) {
    return EMPTY_UPLOAD;
  }

  UploadContext EMPTY_UPLOAD =
      new UploadContext() {
        @Override
        public OutputStream getOutputStream() {
          return ByteStreams.nullOutputStream();
        }

        @Override
        public ListenableFuture<String> uriFuture() {
          return Futures.immediateFailedFuture(new IOException("No available uploader"));
        }
      };

  /**
   * Shutdown any resources associated with the uploader.
   */
  void shutdown();

  /**
   * Return true if the upload may be "slow". Examples of slowness include writes to remote storage.
   */
  boolean mayBeSlow();

  /**
   * Returns a {@link PathConverter} for the uploaded files, or {@code null} when the uploaded
   * failed.
   */
  default ListenableFuture<PathConverter> uploadReferencedLocalFiles(
      Collection<LocalFile> localFiles) {
    Map<Path, LocalFile> localFileMap = Maps.newHashMapWithExpectedSize(localFiles.size());
    for (LocalFile localFile : localFiles) {
      // It is possible for targets to have duplicate artifacts (same path but different owners)
      // in their output groups. Since they didn't trigger an artifact conflict they are the
      // same file, so just skip either one
      localFileMap.putIfAbsent(localFile.path, localFile);
    }
    return upload(localFileMap);
  }

  /**
   * Blocks on the completion of pending remote uploads, enforcing the relevant timeout if
   * applicable.
   */
  default ListenableFuture<?> waitForRemoteUploads(
      Collection<ListenableFuture<String>> remoteUploads,
      ScheduledExecutorService timeoutExecutor) {
    return Futures.allAsList(remoteUploads);
  }
}
