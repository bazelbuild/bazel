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

package com.google.devtools.build.lib.bazel.repository.decompressor;

import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Optional;

/**
 * Utility class for removing a prefix from an archive's path.
 */
@ThreadSafety.Immutable
public final class StripPrefixedPath {
  private final PathFragment pathFragment;
  private final boolean found;
  private final boolean skip;

  /**
   * If a prefix is given, it will be removed from the entry's path. This also turns absolute paths
   * into relative paths (e.g., /usr/bin/bash will become usr/bin/bash, same as unzip's default
   * behavior) and normalizes the paths (foo/../bar////baz will become bar/baz). Note that this
   * could cause collisions, if a zip file had one entry for bin/some-binary and another entry for
   * /bin/some-binary.
   *
   * <p>Note that the prefix is stripped to move the files up one level, so if you have an entry
   * "foo/../bar" and a prefix of "foo", the result will be "bar" not "../bar".
   */
  public static StripPrefixedPath maybeDeprefix(byte[] entry, Optional<String> prefix) {
    Preconditions.checkNotNull(entry);
    PathFragment entryPath = relativize(entry);
    if (prefix.isEmpty()) {
      return new StripPrefixedPath(entryPath, false, false);
    }

    // Bazel parses Starlark files, which are the ultimate source of prefixes, as Latin-1
    // (ISO-8859-1).
    PathFragment prefixPath = relativize(prefix.get().getBytes(ISO_8859_1));
    boolean found = false;
    boolean skip = false;
    if (entryPath.startsWith(prefixPath)) {
      found = true;
      entryPath = entryPath.relativeTo(prefixPath);
      if (entryPath.getPathString().isEmpty()) {
        skip = true;
      }
    } else {
      skip = true;
    }
    return new StripPrefixedPath(entryPath, found, skip);
  }

  /**
   * Normalize the path and, if it is absolute, make it relative (e.g., /foo/bar becomes foo/bar).
   */
  private static PathFragment relativize(byte[] path) {
    PathFragment entryPath = createPathFragment(path);
    if (entryPath.isAbsolute()) {
      entryPath = entryPath.toRelative();
    }
    return entryPath;
  }

  private StripPrefixedPath(PathFragment pathFragment, boolean found, boolean skip) {
    this.pathFragment = pathFragment;
    this.found = found;
    this.skip = skip;
  }

  public static PathFragment maybeDeprefixSymlink(
      byte[] rawTarget, Optional<String> prefix, Path root) {
    boolean wasAbsolute = createPathFragment(rawTarget).isAbsolute();
    // Strip the prefix from the link path if set.
    PathFragment linkPathFragment = maybeDeprefix(rawTarget, prefix).getPathFragment();
    if (wasAbsolute) {
      // Recover the path to an absolute path as maybeDeprefix() relativize the path
      // even if the prefix is not set
      return root.getRelative(linkPathFragment).asFragment();
    }
    return linkPathFragment;
  }

  public PathFragment getPathFragment() {
    return pathFragment;
  }

  public boolean foundPrefix() {
    return found;
  }

  public boolean skip() {
    return skip;
  }

  static PathFragment createPathFragment(byte[] rawBytes) {
    // Bazel internally represents paths as raw bytes by using the Latin-1 encoding, which has the
    // property that (new String(bytes, ISO_8859_1)).getBytes(ISO_8859_1)) equals bytes for every
    // byte array bytes.
    return PathFragment.create(new String(rawBytes, ISO_8859_1));
  }
}
