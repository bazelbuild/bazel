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

package com.google.devtools.build.lib.buildeventstream;

import com.google.common.base.MoreObjects;
import com.google.common.base.Objects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.vfs.Path;
import java.util.Collection;
import javax.annotation.Nullable;

/**
 * Interface for objects that can be posted on the public event stream.
 *
 * <p>Objects posted on the build-event stream will implement this interface. This allows
 * pass-through of events, as well as proper chaining of events.
 */
public interface BuildEvent extends ChainableEvent, ExtendedEventHandler.Postable {

  /**
   * A local file that is referenced by the build event. These can be uploaded to a separate backend
   * storage.
   */
  final class LocalFile {

    /**
     * The type of the local file. This is used by uploaders to determine how long to store the
     * associated files for.
     */
    public enum LocalFileType {
      OUTPUT, /* of uncertain file type */
      OUTPUT_FILE,
      OUTPUT_DIRECTORY,
      OUTPUT_SYMLINK,
      SUCCESSFUL_TEST_OUTPUT,
      FAILED_TEST_OUTPUT,
      COVERAGE_OUTPUT,
      STDOUT,
      STDERR,
      LOG,
      PERFORMANCE_LOG;

      /** Returns whether the LocalFile is a declared action output. */
      public boolean isOutput() {
        return this == OUTPUT
            || this == OUTPUT_FILE
            || this == OUTPUT_DIRECTORY
            || this == OUTPUT_SYMLINK;
      }
    }

    /** Indicates the type of compression the local file should have. */
    public enum LocalFileCompression {
      NONE,
      GZIP,
    }

    public final Path path;
    public final LocalFileType type;
    public final LocalFileCompression compression;
    // TODO(b/199940216): use when possible to get source artifact canonical path.
    @Nullable public final Artifact artifact;
    @Nullable public final FileArtifactValue artifactMetadata;

    public LocalFile(
        Path path,
        LocalFileType type,
        @Nullable Artifact artifact,
        @Nullable FileArtifactValue artifactMetadata) {
      this(path, type, LocalFileCompression.NONE, artifact, artifactMetadata);
    }

    public LocalFile(
        Path path,
        LocalFileType type,
        LocalFileCompression compression,
        @Nullable Artifact artifact,
        @Nullable FileArtifactValue artifactMetadata) {
      this.path = Preconditions.checkNotNull(path);
      this.type = Preconditions.checkNotNull(type);
      this.compression = Preconditions.checkNotNull(compression);
      this.artifact = artifact;
      this.artifactMetadata = artifactMetadata;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (o == null || getClass() != o.getClass()) {
        return false;
      }
      LocalFile localFile = (LocalFile) o;
      return Objects.equal(path, localFile.path)
          && type == localFile.type
          && Objects.equal(artifact, localFile.artifact)
          && Objects.equal(artifactMetadata, localFile.artifactMetadata)
          && compression == localFile.compression;
    }

    @Override
    public int hashCode() {
      return Objects.hashCode(path, type, compression);
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(LocalFile.class)
          .add("path", path)
          .add("type", type)
          .add("artifact", artifact)
          .add("artifactMetadata", artifactMetadata)
          .toString();
    }
  }

  /**
   * Returns a list of files that are referenced in the protobuf representation returned by {@link
   * #asStreamProto(BuildEventContext)}.
   *
   * <p>This method is different from {@code EventReportingArtifacts#reportedArtifacts()} in that it
   * only returns files directly referenced in the protobuf returned by {@link
   * #asStreamProto(BuildEventContext)}.
   *
   * <p>Note the consistency requirement - you must not attempt to pass Path objects to the {@link
   * PathConverter} unless you have returned a corresponding {@link LocalFile} object here.
   */
  default Collection<LocalFile> referencedLocalFiles() {
    return ImmutableList.of();
  }

  /**
   * Returns a collection of URI futures corresponding to in-flight file uploads.
   *
   * <p>The files here are considered "remote" in that they may not correspond to on-disk files.
   */
  default Collection<ListenableFuture<String>> remoteUploads() {
    return ImmutableList.of();
  }

  /**
   * Provide a binary representation of the event.
   *
   * <p>Provide a presentation of the event according to the specified binary format, as appropriate
   * protocol buffer.
   */
  BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext context)
      throws InterruptedException;
}
