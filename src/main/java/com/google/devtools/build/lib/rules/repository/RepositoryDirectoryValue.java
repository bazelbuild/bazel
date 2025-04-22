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

import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.NotComparableSkyValue;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;

/** A local view of an external repository. */
public sealed interface RepositoryDirectoryValue extends NotComparableSkyValue {
  /** Returns whether the repository exists or not. */
  boolean repositoryExists();

  /**
   * Represents a successful repository lookup.
   *
   * @param path Returns the path to the directory containing the repository's contents. This
   *     directory is guaranteed to exist. It may contain a full Bazel repository (with a WORKSPACE
   *     file, directories, and BUILD files) or simply contain a file (or set of files) for, say, a
   *     jar from Maven.
   * @param isFetchingDelayed
   * @param excludeFromVendoring Returns if this repo should be excluded from vendoring. The value
   *     is true for local & configure repos
   */
  record Success(Path path, boolean isFetchingDelayed, boolean excludeFromVendoring)
      implements RepositoryDirectoryValue {

    @Override
    public boolean repositoryExists() {
      return true;
    }

    public Path getPath() {
      return path;
    }
  }

  /**
   * Represents an unsuccessful repository lookup.
   *
   * @param errorMsg For an unsuccessful repository lookup, gets a detailed error message that is
   *     suitable for reporting to a user.
   */
  record Failure(String errorMsg) implements RepositoryDirectoryValue {

    @Override
    public boolean repositoryExists() {
      return false;
    }

    public String getErrorMsg() {
      return errorMsg;
    }
  }

  /** Creates a key from the given repository name. */
  static Key key(RepositoryName repository) {
    return Key.create(repository);
  }

  /** The SkyKey for retrieving the local directory of an external repository. */
  @VisibleForSerialization
  @AutoCodec
  class Key extends AbstractSkyKey<RepositoryName> {
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
}
