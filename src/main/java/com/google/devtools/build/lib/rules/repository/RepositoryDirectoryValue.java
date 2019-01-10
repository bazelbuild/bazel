// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.repository;

import com.google.common.base.Objects;
import com.google.common.base.Preconditions;
import com.google.common.collect.Interner;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.skyframe.DirectoryListingValue;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Arrays;
import java.util.Map;
import javax.annotation.Nullable;

/** A local view of an external repository. */
public abstract class RepositoryDirectoryValue implements SkyValue {
  /**
   * Returns whether the repository exists or not. If this return false, then the methods below will
   * throw.
   */
  public abstract boolean repositoryExists();

  /**
   * Returns the path to the directory containing the repository's contents. This directory is
   * guaranteed to exist. It may contain a full Bazel repository (with a WORKSPACE file,
   * directories, and BUILD files) or simply contain a file (or set of files) for, say, a jar from
   * Maven.
   */
  public abstract Path getPath();

  public abstract boolean isFetchingDelayed();

  /** Represents a successful repository lookup. */
  public static final class SuccessfulRepositoryDirectoryValue extends RepositoryDirectoryValue {
    private final Path path;
    private final boolean fetchingDelayed;
    @Nullable private final byte[] digest;
    @Nullable private final DirectoryListingValue sourceDir;
    private final ImmutableMap<SkyKey, SkyValue> fileValues;

    private SuccessfulRepositoryDirectoryValue(
        Path path, boolean fetchingDelayed, DirectoryListingValue sourceDir, byte[] digest,
        ImmutableMap<SkyKey, SkyValue> fileValues) {
      this.path = path;
      this.fetchingDelayed = fetchingDelayed;
      this.sourceDir = sourceDir;
      this.digest = digest;
      this.fileValues = fileValues;
    }

    @Override
    public boolean repositoryExists() {
      return true;
    }

    @Override
    public Path getPath() {
      return path;
    }

    @Override
    public boolean isFetchingDelayed() {
      return fetchingDelayed;
    }

    @Override
    public boolean equals(Object other) {
      if (this == other) {
        return true;
      }

      if (other instanceof SuccessfulRepositoryDirectoryValue) {
        SuccessfulRepositoryDirectoryValue otherValue = (SuccessfulRepositoryDirectoryValue) other;
        return Objects.equal(path, otherValue.path)
            && Objects.equal(sourceDir, otherValue.sourceDir)
            && Arrays.equals(digest, otherValue.digest)
            && Objects.equal(fileValues, otherValue.fileValues);
      }
      return false;
    }

    @Override
    public int hashCode() {
      return Objects.hashCode(path, sourceDir, Arrays.hashCode(digest), fileValues);
    }

    @Override
    public String toString() {
      return path.getPathString();
    }
  }

  /** Represents an unsuccessful repository lookup. */
  public static final class NoRepositoryDirectoryValue extends RepositoryDirectoryValue {
    private NoRepositoryDirectoryValue() {}

    @Override
    public boolean repositoryExists() {
      return false;
    }

    @Override
    public Path getPath() {
      throw new IllegalStateException();
    }

    @Override
    public boolean isFetchingDelayed() {
      throw new IllegalStateException();
    }
  }

  public static final NoRepositoryDirectoryValue NO_SUCH_REPOSITORY_VALUE =
      new NoRepositoryDirectoryValue();

  /** Creates a key from the given repository name. */
  public static Key key(RepositoryName repository) {
    return Key.create(repository);
  }

  @AutoCodec.VisibleForSerialization
  @AutoCodec
  static class Key extends AbstractSkyKey<RepositoryName> {
    private static final Interner<Key> interner = BlazeInterners.newWeakInterner();

    private Key(RepositoryName arg) {
      super(arg);
    }

    @AutoCodec.VisibleForSerialization
    @AutoCodec.Instantiator
    static Key create(RepositoryName arg) {
      return interner.intern(new Key(arg));
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.REPOSITORY_DIRECTORY;
    }
  }

  public static Builder builder() {
    return new Builder();
  }

  /** A builder to create a {@link RepositoryDirectoryValue}. */
  public static class Builder {
    private Path path = null;
    private boolean fetchingDelayed = false;
    private byte[] digest = null;
    private DirectoryListingValue sourceDir = null;
    private Map<SkyKey, SkyValue> fileValues = ImmutableMap.of();

    private Builder() {}

    public Builder setPath(Path path) {
      this.path = path;
      return this;
    }

    public Builder setFetchingDelayed() {
      this.fetchingDelayed = true;
      return this;
    }

    public Builder setDigest(byte[] digest) {
      this.digest = digest;
      return this;
    }

    public Builder setSourceDir(DirectoryListingValue sourceDir) {
      this.sourceDir = sourceDir;
      return this;
    }

    public Builder setFileValues(Map<SkyKey, SkyValue> fileValues) {
      this.fileValues = fileValues;
      return this;
    }

    public SuccessfulRepositoryDirectoryValue build() {
      Preconditions.checkNotNull(path, "Repository path must be specified!");
      // Only if fetching is delayed then we are allowed to have a null digest.
      if (!this.fetchingDelayed) {
        Preconditions.checkNotNull(digest, "Repository marker digest must be specified!");
      }
      return new SuccessfulRepositoryDirectoryValue(path, fetchingDelayed, sourceDir, digest,
          ImmutableMap.copyOf(fileValues));
    }
  }
}
