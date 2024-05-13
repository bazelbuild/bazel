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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.base.Objects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.skyframe.DirectoryListingValue;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
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

  /**
   * Returns if this repo should be excluded from vendoring. The value is true for local & configure
   * repos
   */
  public abstract boolean excludeFromVendoring();

  /**
   * For an unsuccessful repository lookup, gets a detailed error message that is suitable for
   * reporting to a user.
   */
  public abstract String getErrorMsg();

  /** Represents a successful repository lookup. */
  public static final class SuccessfulRepositoryDirectoryValue extends RepositoryDirectoryValue {
    private final Path path;
    private final boolean fetchingDelayed;
    private final boolean excludeFromVendoring;
    @Nullable private final byte[] digest;
    @Nullable private final DirectoryListingValue sourceDir;
    private final ImmutableMap<SkyKey, SkyValue> fileValues;

    private SuccessfulRepositoryDirectoryValue(
        Path path,
        boolean fetchingDelayed,
        @Nullable DirectoryListingValue sourceDir,
        byte[] digest,
        ImmutableMap<SkyKey, SkyValue> fileValues,
        boolean excludeFromVendoring) {
      this.path = path;
      this.fetchingDelayed = fetchingDelayed;
      this.sourceDir = sourceDir;
      this.digest = digest;
      this.fileValues = fileValues;
      this.excludeFromVendoring = excludeFromVendoring;
    }


    @Override
    public boolean repositoryExists() {
      return true;
    }

    @Override
    public String getErrorMsg() {
      throw new IllegalStateException();
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
    public boolean excludeFromVendoring() {
      return excludeFromVendoring;
    }

    @Override
    public boolean equals(Object other) {
      if (this == other) {
        return true;
      }

      if (other instanceof SuccessfulRepositoryDirectoryValue otherValue) {
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
    private final String errorMsg;

    public NoRepositoryDirectoryValue(String errorMsg) {
      this.errorMsg = errorMsg;
    }

    @Override
    public boolean repositoryExists() {
      return false;
    }

    @Override
    public String getErrorMsg() {
      return this.errorMsg;
    }

    @Override
    public Path getPath() {
      throw new IllegalStateException();
    }

    @Override
    public boolean isFetchingDelayed() {
      throw new IllegalStateException();
    }

    @Override
    public boolean excludeFromVendoring() {
      throw new IllegalStateException();
    }
  }

  /** Creates a key from the given repository name. */
  public static Key key(RepositoryName repository) {
    return Key.create(repository);
  }

  /** The SkyKey for retrieving the local directory of an external repository. */
  @VisibleForSerialization
  @AutoCodec
  public static class Key extends AbstractSkyKey<RepositoryName> {
    private static final SkyKeyInterner<Key> interner = SkyKey.newInterner();

    private Key(RepositoryName arg) {
      super(arg);
    }

    private static Key create(RepositoryName arg) {
      return interner.intern(new Key(arg));
    }

    @VisibleForSerialization
    @AutoCodec.Interner
    static Key intern(Key key) {
      return interner.intern(key);
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.REPOSITORY_DIRECTORY;
    }

    @Override
    public SkyKeyInterner<Key> getSkyKeyInterner() {
      return interner;
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
    @Nullable private DirectoryListingValue sourceDir = null;
    private Map<SkyKey, SkyValue> fileValues = ImmutableMap.of();

    private boolean excludeFromVendoring = false;

    private Builder() {}

    @CanIgnoreReturnValue
    public Builder setPath(Path path) {
      this.path = path;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setFetchingDelayed() {
      this.fetchingDelayed = true;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setDigest(byte[] digest) {
      this.digest = digest;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setSourceDir(DirectoryListingValue sourceDir) {
      this.sourceDir = sourceDir;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setFileValues(Map<SkyKey, SkyValue> fileValues) {
      this.fileValues = fileValues;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setExcludeFromVendoring(boolean excludeFromVendoring) {
      this.excludeFromVendoring = excludeFromVendoring;
      return this;
    }

    public SuccessfulRepositoryDirectoryValue build() {
      Preconditions.checkNotNull(path, "Repository path must be specified!");
      // Only if fetching is delayed then we are allowed to have a null digest.
      if (!this.fetchingDelayed) {
        Preconditions.checkNotNull(digest, "Repository marker digest must be specified!");
      }
      return new SuccessfulRepositoryDirectoryValue(
          path,
          fetchingDelayed,
          sourceDir,
          checkNotNull(digest, "Null digest: %s %s %s", path, fetchingDelayed, sourceDir),
          ImmutableMap.copyOf(fileValues),
          excludeFromVendoring);
    }
  }
}
