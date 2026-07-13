// Copyright 2024 The Bazel Authors. All rights reserved.
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

import static java.util.Objects.requireNonNull;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/**
 * Contains information about the recursive digest of a directory tree, including all transitive
 * descendant files and their contents.
 */
public record DirectoryTreeDigestValue(String hexDigest) implements SkyValue {
  public DirectoryTreeDigestValue {
    requireNonNull(hexDigest, "hexDigest");
  }

  public static DirectoryTreeDigestValue of(String hexDigest) {
    return new DirectoryTreeDigestValue(hexDigest);
  }

  public static Key key(
      RootedPath rootedPath, RootedPath globBase, ImmutableList<String> excludes) {
    return Key.of(rootedPath, globBase, excludes);
  }

  /**
   * Key type for {@link DirectoryTreeDigestValue}.
   *
   * <p>The {@code rootedPath} indicates the directory tree to compute a digest for.
   *
   * <p>To filter out files/directories, you can optionally provide a {@code globBase} and glob
   * {@code excludes} patterns. They are joined together to create the paths to be filtered out. For
   * example, if the given parameters are:
   *
   * <pre>
   * rootedPath = "/tmp"
   * globBase = "/tmp/path"
   * excludes = [".git/**", "cache/ignoreMe"]
   * </pre>
   *
   * Then the glob patterns that will be filtered/excluded from under <code>rootedPath</code> would
   * be:
   *
   * <pre>
   * /tmp/path/.git/**
   * /tmp/path/cache/ignoreMe
   * </pre>
   */
  @AutoCodec
  record Key(RootedPath rootedPath, RootedPath globBase, ImmutableList<String> excludes)
      implements SkyKey {
    Key {
      requireNonNull(rootedPath, "rootedPath");
      requireNonNull(globBase, "globBase");
      requireNonNull(excludes, "excludes");
    }

    private static final SkyKeyInterner<Key> interner = SkyKey.newInterner();

    static Key of(RootedPath rootedPath, RootedPath globBase, ImmutableList<String> excludes) {
      return create(rootedPath, globBase, excludes);
    }

    @AutoCodec.Instantiator
    static Key create(RootedPath rootedPath, RootedPath globBase, ImmutableList<String> excludes) {
      return interner.intern(new Key(rootedPath, globBase, excludes));
    }

    @Override
    public SkyKeyInterner<Key> getSkyKeyInterner() {
      return interner;
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.DIRECTORY_TREE_DIGEST;
    }
  }
}
