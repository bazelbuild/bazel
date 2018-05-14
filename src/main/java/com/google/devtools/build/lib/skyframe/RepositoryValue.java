// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Objects;
import com.google.common.base.Preconditions;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyValue;

/** A repository's name and directory. */
public abstract class RepositoryValue implements SkyValue {
  public abstract boolean repositoryExists();

  /** Returns the path to the repository. */
  public abstract Path getPath();

  /** Successful lookup value. */
  public static final class SuccessfulRepositoryValue extends RepositoryValue {
    private final RepositoryName repositoryName;
    private final RepositoryDirectoryValue repositoryDirectory;

    /** Creates a repository with a given name in a certain directory. */
    public SuccessfulRepositoryValue(
        RepositoryName repositoryName, RepositoryDirectoryValue repository) {
      Preconditions.checkArgument(repository.repositoryExists());
      this.repositoryName = repositoryName;
      this.repositoryDirectory = repository;
    }

    @Override
    public boolean repositoryExists() {
      return true;
    }

    @Override
    public Path getPath() {
      return repositoryDirectory.getPath();
    }

    @Override
    public boolean equals(Object other) {
      if (this == other) {
        return true;
      }
      if (other == null || getClass() != other.getClass()) {
        return false;
      }

      SuccessfulRepositoryValue that = (SuccessfulRepositoryValue) other;
      return Objects.equal(repositoryName, that.repositoryName)
          && Objects.equal(repositoryDirectory, that.repositoryDirectory);
    }

    @Override
    public int hashCode() {
      return Objects.hashCode(repositoryName, repositoryDirectory);
    }
  }

  /** Repository could not be resolved. */
  public static final class NoRepositoryValue extends RepositoryValue {
    private final RepositoryName repositoryName;

    private NoRepositoryValue(RepositoryName repositoryName) {
      this.repositoryName = repositoryName;
    }

    @Override
    public boolean repositoryExists() {
      return false;
    }

    @Override
    public Path getPath() {
      throw new IllegalStateException();
    }

    @Override
    public boolean equals(Object other) {
      if (this == other) {
        return true;
      }
      if (other == null || getClass() != other.getClass()) {
        return false;
      }

      NoRepositoryValue that = (NoRepositoryValue) other;
      return Objects.equal(repositoryName, that.repositoryName);
    }

    @Override
    public int hashCode() {
      return repositoryName.hashCode();
    }
  }

  public static RepositoryValue success(
      RepositoryName repositoryName, RepositoryDirectoryValue repository) {
    return new SuccessfulRepositoryValue(repositoryName, repository);
  }

  public static RepositoryValue notFound(RepositoryName repositoryName) {
    // TODO(ulfjack): Store the cause here? The two possible causes are that the external package
    // contains errors, or that the repository with the given name does not exist.
    return new NoRepositoryValue(repositoryName);
  }

  public static Key key(RepositoryName repositoryName) {
    return Key.create(repositoryName);
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
      return SkyFunctions.REPOSITORY;
    }
  }
}
