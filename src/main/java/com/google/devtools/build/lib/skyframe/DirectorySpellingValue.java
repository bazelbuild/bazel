// Copyright 2026 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import javax.annotation.Nullable;

/**
 * Whether every segment of a directory's root-relative path is spelled exactly like the
 * corresponding entry in its parent directory's listing.
 *
 * <p>On a case-insensitive or normalization-insensitive file system, a path such as {@code Foo/bar}
 * can refer to the directory {@code foo/bar}. Only the latter spelling is canonical.
 */
public sealed interface DirectorySpellingValue extends SkyValue {
  DirectorySpellingValue CANONICAL = Canonical.INSTANCE;

  /** The path is spelled exactly like the directory entries on disk. */
  enum Canonical implements DirectorySpellingValue {
    INSTANCE
  }

  /**
   * The path isn't spelled like the directory entries on disk.
   *
   * @param canonicalPath the root-relative path spelled like the directory entries on disk, or null
   *     if it can't be determined unambiguously
   */
  record NonCanonical(@Nullable PathFragment canonicalPath) implements DirectorySpellingValue {}

  /**
   * Returns the key for the spelling of the given directory, which is assumed to exist (e.g. via
   * {@link com.google.devtools.build.lib.actions.FileValue#isDirectory}).
   */
  static Key key(RootedPath directory) {
    return Key.create(directory);
  }

  /** {@link SkyKey} for {@link DirectorySpellingValue}. */
  @AutoCodec
  final class Key extends AbstractSkyKey<RootedPath> {
    private static final SkyKeyInterner<Key> interner = SkyKey.newInterner();

    private Key(RootedPath arg) {
      super(arg);
    }

    private static Key create(RootedPath arg) {
      return interner.intern(new Key(arg));
    }

    @VisibleForSerialization
    @AutoCodec.Interner
    static Key intern(Key key) {
      return interner.intern(key);
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.DIRECTORY_SPELLING;
    }

    @Override
    public SkyKeyInterner<Key> getSkyKeyInterner() {
      return interner;
    }
  }
}
