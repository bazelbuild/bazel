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
import com.google.devtools.build.lib.util.HashCodes;
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
   *
   * <p>Despite the name, it is possible that a {@code LocalFile} is already stored remotely. If
   * {@link #artifactMetadata} {@link FileArtifactValue#isRemote}, the upload may be skipped.
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
      QUERY_OUTPUT,
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

      /**
       * Returns the {@link LocalFileType} implied by a {@link FileArtifactValue}, or the associated
       * {@link Artifact} if metadata is not available.
       */
      public static LocalFileType forArtifact(
          Artifact artifact, @Nullable FileArtifactValue metadata) {
        if (metadata != null) {
          return switch (metadata.getType()) {
            case DIRECTORY -> LocalFileType.OUTPUT_DIRECTORY;
            case SYMLINK -> LocalFileType.OUTPUT_SYMLINK;
            default -> LocalFileType.OUTPUT_FILE;
          };
        }
        if (artifact.isDirectory()) {
          return LocalFileType.OUTPUT_DIRECTORY;
        } else if (artifact.isSymlink()) {
          return LocalFileType.OUTPUT_SYMLINK;
        }
        return LocalFileType.OUTPUT_FILE;
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
    @Nullable public final FileArtifactValue artifactMetadata;

    public LocalFile(Path path, LocalFileType type, @Nullable FileArtifactValue artifactMetadata) {
      this(path, type, LocalFileCompression.NONE, artifactMetadata);
    }

    public LocalFile(
        Path path,
        LocalFileType type,
        LocalFileCompression compression,
        @Nullable FileArtifactValue artifactMetadata) {
      this.path = Preconditions.checkNotNull(path);
      this.type = Preconditions.checkNotNull(type);
      this.compression = Preconditions.checkNotNull(compression);
      this.artifactMetadata = artifactMetadata;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof LocalFile that)) {
        return false;
      }
      return path.equals(that.path)
          && type == that.type
          && compression == that.compression
          && Objects.equal(artifactMetadata, that.artifactMetadata);
    }

    @Override
    public int hashCode() {
      return HashCodes.hashObjects(path, type, compression, artifactMetadata);
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add("path", path)
          .add("type", type)
          .add("compression", compression)
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
