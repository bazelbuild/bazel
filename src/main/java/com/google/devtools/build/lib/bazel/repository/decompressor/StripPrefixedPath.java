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
import java.util.Objects;

/**
 * Utility class for removing a prefix from an archive's path.
 *
 * <p>A "prefix" can be a string of path segments or an integer representing the number of path
 * segments to remove. Given an example path <code>/tmp/path/to/file</code>, removing the string
 * prefix <code>/tmp/path</code> or stripping two components will result in the same de-prefixed
 * path: <code>to/file</code>
 */
@ThreadSafety.Immutable
public final class StripPrefixedPath {
  private final PathFragment pathFragment;
  private final boolean foundPrefix;
  private final boolean skip;

  /**
   * Removes the given prefix or the specified number of components.
   *
   * <p>If a prefix of number of components is given, it will be removed from the entry's path. This
   * also turns absolute paths into relative paths (e.g., /usr/bin/bash will become usr/bin/bash,
   * same as unzip's default behavior) and normalizes the paths (foo/../bar////baz will become
   * bar/baz). Note that this could cause collisions, if a zip file had one entry for
   * bin/some-binary and another entry for /bin/some-binary.
   *
   * <p>Note that the prefix is stripped to move the files up one level, so if you have an entry
   * "foo/../bar" and a prefix of "foo", the result will be "bar" not "../bar".
   *
   * <p>Only one of <code>prefix</code> or <code>stripComponents</code> is allowed.
   */
  public static StripPrefixedPath maybeDeprefix(byte[] entry, String prefix, int stripComponents) {
    Preconditions.checkNotNull(entry);
    Preconditions.checkArgument(
        prefix.isEmpty() || stripComponents == 0,
        "Only one of prefix or strip_components can be set.");

    PathFragment entryPath = relativize(entry);
    if (stripComponents != 0) {
      return maybeStripComponent(entryPath, stripComponents);
    } else {
      return maybeStripPrefix(entryPath, prefix);
    }
  }

  private static StripPrefixedPath maybeStripPrefix(PathFragment entryPath, String prefix) {
    if (prefix.isEmpty()) {
      return new StripPrefixedPath(entryPath, false, false);
    }

    // Bazel parses Starlark files, which are the ultimate source of prefixes, as Latin-1
    // (ISO-8859-1).
    PathFragment prefixPath = relativize(prefix.getBytes(ISO_8859_1));
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

  private static StripPrefixedPath maybeStripComponent(
      PathFragment entryPath, int stripComponents) {
    if (stripComponents == 0) {
      return new StripPrefixedPath(entryPath, /* foundPrefix= */ false, false);
    }
    PathFragment strippedPath = entryPath.stripComponents(stripComponents);
    boolean skipEntry = false;
    if (Objects.equals(strippedPath, PathFragment.EMPTY_FRAGMENT)) {
      skipEntry = true;
    }
    return new StripPrefixedPath(strippedPath, /* foundPrefix= */ false, skipEntry);
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

  private StripPrefixedPath(PathFragment pathFragment, boolean foundPrefix, boolean skip) {
    this.pathFragment = pathFragment;
    this.foundPrefix = foundPrefix;
    this.skip = skip;
  }

  public static PathFragment maybeDeprefixSymlink(
      byte[] rawTarget, String prefix, int stripComponents, Path root) {
    return maybeDeprefixSymlink(rawTarget, prefix, stripComponents, root, false);
  }

  /**
   * Normalizes and possibly deprefixes a target link.
   *
   * <p>A target link will only be deprefixed and relative to the given root if any of the
   * following:
   *
   * <ul>
   *   <li>The link is absolute
   *   <li>The link is known to be relative to the root (<code>forceExtractRootRelative</code>)
   * </ul>
   *
   * Otherwise, no deprefixing will occur.
   *
   * @param rawTarget The target path for the link.
   * @param prefix The prefix to remove. If set, <code>stripComponents</code> must be 0.
   * @param stripComponents The number of path segements to remove. If nonzero, <code>prefix</code>
   *     must be Optional.empty().
   * @param root The path for absolute or <code>forceExtractRootRelative</code> to be relative to.
   * @param forceExtractRootRelative Forces the given <code>rawTarget</code> to be relative to the
   *     <code>root</code>.
   * @return The normalized and possibly deprefixed link target.
   */
  public static PathFragment maybeDeprefixSymlink(
      byte[] rawTarget,
      String prefix,
      int stripComponents,
      Path root,
      boolean forceExtractRootRelative) {
    Preconditions.checkArgument(
        prefix.isEmpty() || stripComponents == 0,
        "Only one of prefix or strip_components can be set.");
    boolean wasAbsolute = createPathFragment(rawTarget).isAbsolute();
    if (wasAbsolute || forceExtractRootRelative) {
      // Strip the prefix from the link path if set
      PathFragment linkPathFragment =
          maybeDeprefix(rawTarget, prefix, stripComponents).getPathFragment();
      // Recover the path to an absolute path as maybeDeprefix() relativize the path
      // even if the prefix is not set
      return root.getRelative(linkPathFragment).asFragment();
    } else {
      // No deprefixing needed if not relative to extraction root.
      return relativize(rawTarget);
    }
  }

  public PathFragment getPathFragment() {
    return pathFragment;
  }

  public boolean foundPrefix() {
    return foundPrefix;
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
