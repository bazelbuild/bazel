// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.vfs;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.windows.WindowsShortPath;
import com.google.devtools.build.lib.windows.jni.WindowsFileOperations;
import java.io.IOException;
import java.util.List;
import java.util.regex.Pattern;
import javax.annotation.Nullable;

/**
 * A local file path representing a file on the host machine. You should use this when you want to
 * access local files via the file system.
 *
 * <p>Paths are either absolute or relative.
 *
 * <p>Strings are normalized with '.' and '..' removed and resolved (if possible), any multiple
 * slashes ('/') removed, and any trailing slash also removed. The current implementation does not
 * touch the incoming path string unless the string actually needs to be normalized.
 *
 * <p>There is some limited support for Windows-style paths. Most importantly, drive identifiers in
 * front of a path (c:/abc) are supported and such paths are correctly recognized as absolute, as
 * are paths with backslash separators (C:\\foo\\bar). However, advanced Windows-style features like
 * \\\\network\\paths and \\\\?\\unc\\paths are not supported. We are currently using forward
 * slashes ('/') even on Windows, so backslashes '\' get converted to forward slashes during
 * normalization.
 *
 * <p>Mac and Windows file paths are case insensitive. Case is preserved.
 *
 * <p>This class is replaces {@link Path} as the way to access the host machine's file system.
 * Developers should use this class instead of {@link Path}.
 */
public final class LocalPath implements Comparable<LocalPath> {
  private static final OsPathPolicy DEFAULT_OS = createFilePathOs();

  public static final LocalPath EMPTY = create("");

  private static final Splitter PATH_SPLITTER = Splitter.on('/').omitEmptyStrings();

  private final String path;
  private final int driveStrLength; // 0 for relative paths, 1 on Unix, 3 on Windows
  private final OsPathPolicy os;

  /** Creates a local path that is specific to the host OS. */
  public static LocalPath create(String path) {
    return createWithOs(path, DEFAULT_OS);
  }

  @VisibleForTesting
  static LocalPath createWithOs(String path, OsPathPolicy os) {
    Preconditions.checkNotNull(path);
    int normalizationLevel = os.needsToNormalize(path);
    String normalizedPath = os.normalize(path, normalizationLevel);
    int driveStrLength = os.getDriveStrLength(normalizedPath);
    return new LocalPath(normalizedPath, driveStrLength, os);
  }

  /** This method expects path to already be normalized. */
  private LocalPath(String path, int driveStrLength, OsPathPolicy os) {
    this.path = Preconditions.checkNotNull(path);
    this.driveStrLength = driveStrLength;
    this.os = Preconditions.checkNotNull(os);
  }

  public String getPathString() {
    return path;
  }

  /**
   * If called on a {@link LocalPath} instance for a mount name (eg. '/' or 'C:/'), the empty string
   * is returned.
   */
  public String getBaseName() {
    int lastSeparator = path.lastIndexOf(os.getSeparator());
    return lastSeparator < driveStrLength
        ? path.substring(driveStrLength)
        : path.substring(lastSeparator + 1);
  }

  /**
   * Returns a {@link LocalPath} instance representing the relative path between this {@link
   * LocalPath} and the given {@link LocalPath}.
   *
   * <pre>
   *   Example:
   *
   *   LocalPath.create("/foo").getRelative(LocalPath.create("bar/baz"))
   *   -> "/foo/bar/baz"
   * </pre>
   *
   * <p>If the passed path is absolute it is returned untouched. This can be useful to resolve
   * symlinks.
   */
  public LocalPath getRelative(LocalPath other) {
    Preconditions.checkNotNull(other);
    Preconditions.checkArgument(os == other.os);
    return getRelative(other.getPathString(), other.driveStrLength);
  }

  /**
   * Returns a {@link LocalPath} instance representing the relative path between this {@link
   * LocalPath} and the given path.
   *
   * <p>See {@link #getRelative(LocalPath)} for details.
   */
  public LocalPath getRelative(String other) {
    Preconditions.checkNotNull(other);
    return getRelative(other, os.getDriveStrLength(other));
  }

  private LocalPath getRelative(String other, int otherDriveStrLength) {
    if (path.isEmpty()) {
      return create(other);
    }
    if (other.isEmpty()) {
      return this;
    }
    // Note that even if other came from a LocalPath instance we still might
    // need to normalize the result if (for instance) other is a path that
    // starts with '..'
    int normalizationLevel = os.needsToNormalize(other);
    // This is an absolute path, simply return it
    if (otherDriveStrLength > 0) {
      String normalizedPath = os.normalize(other, normalizationLevel);
      return new LocalPath(normalizedPath, otherDriveStrLength, os);
    }
    String newPath;
    if (path.length() == driveStrLength) {
      newPath = path + other;
    } else {
      newPath = path + '/' + other;
    }
    newPath = os.normalize(newPath, normalizationLevel);
    return new LocalPath(newPath, driveStrLength, os);
  }

  /**
   * Returns the parent directory of this {@link LocalPath}.
   *
   * <p>If this is called on an single directory for a relative path, this returns an empty relative
   * path. If it's called on a root (like '/') or the empty string, it returns null.
   */
  @Nullable
  public LocalPath getParentDirectory() {
    int lastSeparator = path.lastIndexOf(os.getSeparator());

    // For absolute paths we need to specially handle when we hit root
    // Relative paths can't hit this path as driveStrLength == 0
    if (driveStrLength > 0) {
      if (lastSeparator < driveStrLength) {
        if (path.length() > driveStrLength) {
          String newPath = path.substring(0, driveStrLength);
          return new LocalPath(newPath, driveStrLength, os);
        } else {
          return null;
        }
      }
    } else {
      if (lastSeparator == -1) {
        if (!path.isEmpty()) {
          return EMPTY;
        } else {
          return null;
        }
      }
    }
    String newPath = path.substring(0, lastSeparator);
    return new LocalPath(newPath, driveStrLength, os);
  }

  /**
   * Returns the {@link LocalPath} relative to the base {@link LocalPath}.
   *
   * <p>For example, <code>LocalPath.create("foo/bar/wiz").relativeTo(LocalPath.create("foo"))
   * </code> returns <code>LocalPath.create("bar/wiz")</code>.
   *
   * <p>If the {@link LocalPath} is not a child of the passed {@link LocalPath} an {@link
   * IllegalArgumentException} is thrown. In particular, this will happen whenever the two {@link
   * LocalPath} instances aren't both absolute or both relative.
   */
  public LocalPath relativeTo(LocalPath base) {
    Preconditions.checkNotNull(base);
    Preconditions.checkArgument(os == base.os);
    if (isAbsolute() != base.isAbsolute()) {
      throw new IllegalArgumentException(
          "Cannot relativize an absolute and a non-absolute path pair");
    }
    String basePath = base.path;
    if (!os.startsWith(path, basePath)) {
      throw new IllegalArgumentException(
          String.format("Path '%s' is not under '%s', cannot relativize", this, base));
    }
    int bn = basePath.length();
    if (bn == 0) {
      return this;
    }
    if (path.length() == bn) {
      return EMPTY;
    }
    final int lastSlashIndex;
    if (basePath.charAt(bn - 1) == '/') {
      lastSlashIndex = bn - 1;
    } else {
      lastSlashIndex = bn;
    }
    if (path.charAt(lastSlashIndex) != '/') {
      throw new IllegalArgumentException(
          String.format("Path '%s' is not under '%s', cannot relativize", this, base));
    }
    String newPath = path.substring(lastSlashIndex + 1);
    return new LocalPath(newPath, 0 /* Always a relative path */, os);
  }

  /**
   * Splits a path into its constituent parts. The root is not included. This is an inefficient
   * operation and should be avoided.
   */
  public List<String> split() {
    List<String> segments = PATH_SPLITTER.splitToList(path);
    if (driveStrLength > 1) {
      return segments.subList(1, segments.size());
    }
    return segments;
  }

  /** Returns the drive of this local path, eg. "/" on Unix or "C:/" on Windows. */
  public LocalPath getDrive() {
    if (driveStrLength == 0) {
      throw new IllegalArgumentException("Cannot get mount of non-absolute path.");
    }
    return new LocalPath(path.substring(0, driveStrLength), driveStrLength, os);
  }

  /**
   * Returns whether this is the root of the entire file system.
   *
   * <p>Please avoid this method. On Unix, this corresponds to the '/' mount point. Windows drives
   * (C:/) do not have a parent and are not the root of the entire file system, so do not return
   * true.
   */
  public boolean isRoot() {
    return os.isRoot(this);
  }

  /**
   * Returns whether this path is an ancestor of another path.
   *
   * <p>A path is considered an ancestor of itself.
   *
   * <p>An absolute path can never be an ancestor of a relative path, and vice versa.
   */
  public boolean startsWith(LocalPath other) {
    Preconditions.checkNotNull(other);
    Preconditions.checkArgument(os == other.os);
    if (other.path.length() > path.length()) {
      return false;
    }
    if (driveStrLength != other.driveStrLength) {
      return false;
    }
    if (!os.startsWith(path, other.path)) {
      return false;
    }
    return path.length() == other.path.length()
        || other.path.length() == driveStrLength
        || path.charAt(other.path.length()) == os.getSeparator();
  }

  public boolean isAbsolute() {
    return driveStrLength > 0;
  }

  @Override
  public String toString() {
    return path;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    return os.compare(this.path, ((LocalPath) o).path) == 0;
  }

  @Override
  public int hashCode() {
    return os.hashPath(this.path);
  }

  @Override
  public int compareTo(LocalPath o) {
    return os.compare(this.path, o.path);
  }

  /**
   * An interface class representing the differences in path style between different OSs.
   *
   * <p>Eg. case sensitivity, '/' mounts vs. 'C:/', etc.
   */
  @VisibleForTesting
  interface OsPathPolicy {
    int NORMALIZED = 0; // Path is normalized
    int NEEDS_NORMALIZE = 1; // Path requires normalization

    /** Returns required normalization level, passed to {@link #normalize}. */
    int needsToNormalize(String path);

    /**
     * Normalizes the passed string according to the passed normalization level.
     *
     * @param normalizationLevel The normalizationLevel from {@link #needsToNormalize}
     */
    String normalize(String path, int normalizationLevel);

    /**
     * Returns the length of the mount, eg. 1 for unix '/', 3 for Windows 'C:/'.
     *
     * <p>If the path is relative, 0 is returned
     */
    int getDriveStrLength(String path);

    /** Compares two path strings, using the given OS case sensitivity. */
    int compare(String s1, String s2);

    /** Computes the hash code for a path string. */
    int hashPath(String s);

    /**
     * Returns whether the passed string starts with the given prefix, given the OS case
     * sensitivity.
     *
     * <p>This is a pure string operation and doesn't need to worry about matching path segments.
     */
    boolean startsWith(String path, String prefix);

    char getSeparator();

    boolean isCaseSensitive();

    boolean isRoot(LocalPath localPath);
  }

  @VisibleForTesting
  static class UnixOsPathPolicy implements OsPathPolicy {
    private static Splitter UNIX_PATH_SPLITTER =
        Splitter.on(Pattern.compile("/+")).omitEmptyStrings();

    @Override
    public int needsToNormalize(String path) {
      int n = path.length();
      int dotCount = 0;
      char prevChar = 0;
      for (int i = 0; i < n; i++) {
        char c = path.charAt(i);
        if (c == '/') {
          if (prevChar == '/') {
            return NEEDS_NORMALIZE;
          }
          if (dotCount == 1 || dotCount == 2) {
            return NEEDS_NORMALIZE;
          }
        }
        dotCount = c == '.' ? dotCount + 1 : 0;
        prevChar = c;
      }
      if ((n > 1 && prevChar == '/') || dotCount == 1 || dotCount == 2) {
        return NEEDS_NORMALIZE;
      }
      return NORMALIZED;
    }

    @Override
    public String normalize(String path, int normalizationLevel) {
      if (normalizationLevel == NORMALIZED) {
        return path;
      }
      if (path.isEmpty()) {
        return path;
      }
      boolean isAbsolute = path.charAt(0) == '/';
      String[] segments = Iterables.toArray(UNIX_PATH_SPLITTER.split(path), String.class);
      int segmentCount = removeRelativePaths(segments, 0);
      StringBuilder sb = new StringBuilder(path.length());
      if (isAbsolute) {
        sb.append('/');
      }
      for (int i = 0; i < segmentCount; ++i) {
        sb.append(segments[i]);
        sb.append('/');
      }
      if (segmentCount > 0) {
        sb.deleteCharAt(sb.length() - 1);
      }
      return sb.toString();
    }

    @Override
    public int getDriveStrLength(String path) {
      if (path.length() == 0) {
        return 0;
      }
      return (path.charAt(0) == '/') ? 1 : 0;
    }

    @Override
    public int compare(String s1, String s2) {
      return s1.compareTo(s2);
    }

    @Override
    public int hashPath(String s) {
      return s.hashCode();
    }

    @Override
    public boolean startsWith(String path, String prefix) {
      return path.startsWith(prefix);
    }

    @Override
    public char getSeparator() {
      return '/';
    }

    @Override
    public boolean isCaseSensitive() {
      return true;
    }

    @Override
    public boolean isRoot(LocalPath localPath) {
      return localPath.path.equals("/");
    }
  }

  /** Mac is a unix file system that is case insensitive. */
  @VisibleForTesting
  static class MacOsPathPolicy extends UnixOsPathPolicy {
    @Override
    public int compare(String s1, String s2) {
      return s1.compareToIgnoreCase(s2);
    }

    @Override
    public int hashPath(String s) {
      return s.toLowerCase().hashCode();
    }

    @Override
    public boolean isCaseSensitive() {
      return false;
    }
  }

  @VisibleForTesting
  static class WindowsOsPathPolicy implements OsPathPolicy {

    private static final int NEEDS_SHORT_PATH_NORMALIZATION = NEEDS_NORMALIZE + 1;

    private static Splitter WINDOWS_PATH_SPLITTER =
        Splitter.on(Pattern.compile("[\\\\/]+")).omitEmptyStrings();

    private final ShortPathResolver shortPathResolver;

    interface ShortPathResolver {
      String resolveShortPath(String path);
    }

    static class DefaultShortPathResolver implements ShortPathResolver {
      @Override
      public String resolveShortPath(String path) {
        try {
          return WindowsFileOperations.getLongPath(path);
        } catch (IOException e) {
          return path;
        }
      }
    }

    WindowsOsPathPolicy() {
      this(new DefaultShortPathResolver());
    }

    WindowsOsPathPolicy(ShortPathResolver shortPathResolver) {
      this.shortPathResolver = shortPathResolver;
    }

    @Override
    public int needsToNormalize(String path) {
      int n = path.length();
      int normalizationLevel = 0;
      int dotCount = 0;
      char prevChar = 0;
      int segmentBeginIndex = 0; // The start index of the current path index
      boolean segmentHasShortPathChar = false; // Triggers more expensive short path regex test
      for (int i = 0; i < n; i++) {
        char c = path.charAt(i);
        if (c == '/' || c == '\\') {
          if (c == '\\') {
            normalizationLevel = Math.max(normalizationLevel, NEEDS_NORMALIZE);
          }
          // No need to check for '\' here because that already causes normalization
          if (prevChar == '/') {
            normalizationLevel = Math.max(normalizationLevel, NEEDS_NORMALIZE);
          }
          if (dotCount == 1 || dotCount == 2) {
            normalizationLevel = Math.max(normalizationLevel, NEEDS_NORMALIZE);
          }
          if (segmentHasShortPathChar) {
            if (WindowsShortPath.isShortPath(path.substring(segmentBeginIndex, i))) {
              normalizationLevel = Math.max(normalizationLevel, NEEDS_SHORT_PATH_NORMALIZATION);
            }
          }
          segmentBeginIndex = i + 1;
          segmentHasShortPathChar = false;
        } else if (c == '~') {
          // This path segment might be a Windows short path segment
          segmentHasShortPathChar = true;
        }
        dotCount = c == '.' ? dotCount + 1 : 0;
        prevChar = c;
      }
      if ((n > 1 && prevChar == '/') || dotCount == 1 || dotCount == 2) {
        normalizationLevel = Math.max(normalizationLevel, NEEDS_NORMALIZE);
      }
      return normalizationLevel;
    }

    @Override
    public String normalize(String path, int normalizationLevel) {
      if (normalizationLevel == NORMALIZED) {
        return path;
      }
      if (normalizationLevel == NEEDS_SHORT_PATH_NORMALIZATION) {
        String resolvedPath = shortPathResolver.resolveShortPath(path);
        if (resolvedPath != null) {
          path = resolvedPath;
        }
      }
      String[] segments = Iterables.toArray(WINDOWS_PATH_SPLITTER.splitToList(path), String.class);
      int driveStrLength = getDriveStrLength(path);
      boolean isAbsolute = driveStrLength > 0;
      int segmentSkipCount = isAbsolute && driveStrLength > 1 ? 1 : 0;

      StringBuilder sb = new StringBuilder(path.length());
      if (isAbsolute) {
        char c = path.charAt(0);
        if (c == '/') {
          sb.append('/');
        } else {
          sb.append(Character.toUpperCase(c));
          sb.append(":/");
        }
      }
      int segmentCount = removeRelativePaths(segments, segmentSkipCount);
      for (int i = 0; i < segmentCount; ++i) {
        sb.append(segments[i]);
        sb.append('/');
      }
      if (segmentCount > 0) {
        sb.deleteCharAt(sb.length() - 1);
      }
      return sb.toString();
    }

    @Override
    public int getDriveStrLength(String path) {
      int n = path.length();
      if (n == 0) {
        return 0;
      }
      if (path.charAt(0) == '/') {
        return 1;
      }
      if (n < 3) {
        return 0;
      }
      if (isDriveLetter(path.charAt(0))
          && path.charAt(1) == ':'
          && (path.charAt(2) == '/' || path.charAt(2) == '\\')) {
        return 3;
      }
      return 0;
    }

    private static boolean isDriveLetter(char c) {
      return ((c >= 'a') && (c <= 'z')) || ((c >= 'A') && (c <= 'Z'));
    }

    @Override
    public int compare(String s1, String s2) {
      // Windows is case-insensitive
      return s1.compareToIgnoreCase(s2);
    }

    @Override
    public int hashPath(String s) {
      // Windows is case-insensitive
      return s.toLowerCase().hashCode();
    }

    @Override
    public boolean startsWith(String path, String prefix) {
      int pathn = path.length();
      int prefixn = prefix.length();
      if (pathn < prefixn) {
        return false;
      }
      for (int i = 0; i < prefixn; ++i) {
        if (Character.toLowerCase(path.charAt(i)) != Character.toLowerCase(prefix.charAt(i))) {
          return false;
        }
      }
      return true;
    }

    @Override
    public char getSeparator() {
      return '/';
    }

    @Override
    public boolean isCaseSensitive() {
      return false;
    }

    @Override
    public boolean isRoot(LocalPath localPath) {
      // Return true for Unix paths for testing
      return localPath.path.equals("/");
    }
  }

  private static OsPathPolicy createFilePathOs() {
    switch (OS.getCurrent()) {
      case LINUX:
      case FREEBSD:
      case UNKNOWN:
        return new UnixOsPathPolicy();
      case DARWIN:
        return new MacOsPathPolicy();
      case WINDOWS:
        return new WindowsOsPathPolicy();
      default:
        throw new AssertionError("Not covering all OSs");
    }
  }

  /**
   * Normalizes any '.' and '..' in-place in the segment array by shifting other segments to the
   * front. Returns the remaining number of items.
   */
  private static int removeRelativePaths(String[] segments, int starti) {
    int segmentCount = 0;
    int shift = starti;
    int n = segments.length;
    for (int i = starti; i < n; ++i) {
      String segment = segments[i];
      switch (segment) {
        case ".":
          // Just discard it
          ++shift;
          break;
        case "..":
          if (segmentCount > 0 && !segments[segmentCount - 1].equals("..")) {
            // Remove the last segment, if there is one and it is not "..". This
            // means that the resulting path can still contain ".."
            // segments at the beginning.
            segmentCount--;
            shift += 2;
            break;
          }
          // Fall through
        default:
          ++segmentCount;
          if (shift > 0) {
            segments[i - shift] = segments[i];
          }
          break;
      }
    }
    return segmentCount;
  }
}
