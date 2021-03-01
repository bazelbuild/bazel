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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.CommandLineItem;
import com.google.devtools.build.lib.skyframe.serialization.DeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.devtools.build.lib.util.FileType;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import javax.annotation.Nullable;

/**
 * A path segment representing a path fragment using the host machine's path style. That is; If you
 * are running on a Unix machine, the path style will be unix, on Windows it is the windows path
 * style.
 *
 * <p>Path fragments are either absolute or relative.
 *
 * <p>Strings are normalized with '.' and '..' removed and resolved (if possible), any multiple
 * slashes ('/') removed, and any trailing slash also removed. Windows drive letters are uppercased.
 * The current implementation does not touch the incoming path string unless the string actually
 * needs to be normalized.
 *
 * <p>There is some limited support for Windows-style paths. Most importantly, drive identifiers in
 * front of a path (c:/abc) are supported and such paths are correctly recognized as absolute, as
 * are paths with backslash separators (C:\\foo\\bar). However, advanced Windows-style features like
 * \\\\network\\paths and \\\\?\\unc\\paths are not supported. We are currently using forward
 * slashes ('/') even on Windows.
 *
 * <p>Mac and Windows path fragments are case insensitive.
 */
public final class PathFragment
    implements Comparable<PathFragment>,
        FileType.HasFileType,
        CommandLineItem {
  private static final OsPathPolicy OS = OsPathPolicy.getFilePathOs();

  @SerializationConstant public static final PathFragment EMPTY_FRAGMENT = new PathFragment("", 0);
  public static final char SEPARATOR_CHAR = '/';
  private static final char ADDITIONAL_SEPARATOR_CHAR = OS.additionalSeparator();

  private final String normalizedPath;
  private final int driveStrLength; // 0 for relative paths, 1 on Unix, 3 on Windows

  /** Creates a new normalized path fragment. */
  public static PathFragment create(String path) {
    if (path.isEmpty()) {
      return EMPTY_FRAGMENT;
    }
    int normalizationLevel = OS.needsToNormalize(path);
    String normalizedPath =
        normalizationLevel != OsPathPolicy.NORMALIZED
            ? OS.normalize(path, normalizationLevel)
            : path;
    int driveStrLength = OS.getDriveStrLength(normalizedPath);
    return new PathFragment(normalizedPath, driveStrLength);
  }

  /**
   * Creates a new path fragment, where the caller promises that the path is normalized.
   *
   * <p>WARNING! Make sure the path fragment is in fact already normalized. The rest of the code
   * assumes this is the case.
   */
  public static PathFragment createAlreadyNormalized(String normalizedPath) {
    int driveStrLength = OS.getDriveStrLength(normalizedPath);
    return createAlreadyNormalized(normalizedPath, driveStrLength);
  }

  /**
   * Creates a new path fragment, where the caller promises that the path is normalized.
   *
   * <p>Should only be used internally.
   */
  static PathFragment createAlreadyNormalized(String normalizedPath, int driveStrLength) {
    if (normalizedPath.isEmpty()) {
      return EMPTY_FRAGMENT;
    }
    return new PathFragment(normalizedPath, driveStrLength);
  }

  /** This method expects path to already be normalized. */
  private PathFragment(String normalizedPath, int driveStrLength) {
    this.normalizedPath = Preconditions.checkNotNull(normalizedPath);
    this.driveStrLength = driveStrLength;
  }

  public String getPathString() {
    return normalizedPath;
  }

  public boolean isEmpty() {
    return normalizedPath.isEmpty();
  }

  public int getDriveStrLength() {
    return driveStrLength;
  }

  /**
   * If called on a {@link PathFragment} instance for a mount name (eg. '/' or 'C:/'), the empty
   * string is returned.
   *
   * <p>This operation allocates a new string.
   */
  public String getBaseName() {
    int lastSeparator = normalizedPath.lastIndexOf(SEPARATOR_CHAR);
    return lastSeparator < driveStrLength
        ? normalizedPath.substring(driveStrLength)
        : normalizedPath.substring(lastSeparator + 1);
  }

  /**
   * Returns a {@link PathFragment} instance representing the relative path between this {@link
   * PathFragment} and the given {@link PathFragment}.
   *
   * <p>If the passed path is absolute it is returned untouched. This can be useful to resolve
   * symlinks.
   */
  public PathFragment getRelative(PathFragment other) {
    Preconditions.checkNotNull(other);
    // Fast-path: The path fragment is already normal, use cheaper normalization check
    String otherStr = other.normalizedPath;
    return getRelative(otherStr, other.driveStrLength, OS.needsToNormalizeSuffix(otherStr));
  }

  public static boolean isNormalizedRelativePath(String path) {
    int driveStrLength = OS.getDriveStrLength(path);
    int normalizationLevel = OS.needsToNormalize(path);
    return driveStrLength == 0 && normalizationLevel == OsPathPolicy.NORMALIZED;
  }

  public static boolean containsSeparator(String path) {
    return path.lastIndexOf(SEPARATOR_CHAR) != -1;
  }

  /**
   * Returns a {@link PathFragment} instance representing the relative path between this {@link
   * PathFragment} and the given path.
   *
   * <p>See {@link #getRelative(PathFragment)} for details.
   */
  public PathFragment getRelative(String other) {
    Preconditions.checkNotNull(other);
    return getRelative(other, OS.getDriveStrLength(other), OS.needsToNormalize(other));
  }

  private PathFragment getRelative(String other, int otherDriveStrLength, int normalizationLevel) {
    if (normalizedPath.isEmpty()) {
      return create(other);
    }
    if (other.isEmpty()) {
      return this;
    }
    // This is an absolute path, simply return it
    if (otherDriveStrLength > 0) {
      String normalizedPath =
          normalizationLevel != OsPathPolicy.NORMALIZED
              ? OS.normalize(other, normalizationLevel)
              : other;
      return new PathFragment(normalizedPath, otherDriveStrLength);
    }
    String newPath;
    if (normalizedPath.length() == driveStrLength) {
      newPath = normalizedPath + other;
    } else {
      newPath = normalizedPath + '/' + other;
    }
    newPath =
        normalizationLevel != OsPathPolicy.NORMALIZED
            ? OS.normalize(newPath, normalizationLevel)
            : newPath;
    return new PathFragment(newPath, driveStrLength);
  }

  public PathFragment getChild(String baseName) {
    checkBaseName(baseName);
    String newPath;
    if (normalizedPath.length() == driveStrLength) {
      newPath = normalizedPath + baseName;
    } else {
      newPath = normalizedPath + '/' + baseName;
    }
    return new PathFragment(newPath, driveStrLength);
  }

  /**
   * Returns the parent directory of this {@link PathFragment}.
   *
   * <p>If this is called on an single directory for a relative path, this returns an empty relative
   * path. If it's called on a root (like '/') or the empty string, it returns null.
   */
  @Nullable
  public PathFragment getParentDirectory() {
    int lastSeparator = normalizedPath.lastIndexOf(SEPARATOR_CHAR);

    // For absolute paths we need to specially handle when we hit root
    // Relative paths can't hit this path as driveStrLength == 0
    if (driveStrLength > 0) {
      if (lastSeparator < driveStrLength) {
        if (normalizedPath.length() > driveStrLength) {
          String newPath = normalizedPath.substring(0, driveStrLength);
          return new PathFragment(newPath, driveStrLength);
        } else {
          return null;
        }
      }
    } else {
      if (lastSeparator == -1) {
        if (!normalizedPath.isEmpty()) {
          return EMPTY_FRAGMENT;
        } else {
          return null;
        }
      }
    }
    String newPath = normalizedPath.substring(0, lastSeparator);
    return new PathFragment(newPath, driveStrLength);
  }

  /**
   * Returns the {@link PathFragment} relative to the base {@link PathFragment}.
   *
   * <p>For example, <code>
   * {@link PathFragment}.create("foo/bar/wiz").relativeTo({@link PathFragment}.create("foo"))
   * </code> returns <code>"bar/wiz"</code>.
   *
   * <p>If the {@link PathFragment} is not a child of the passed {@link PathFragment} an {@link
   * IllegalArgumentException} is thrown. In particular, this will happen whenever the two {@link
   * PathFragment} instances aren't both absolute or both relative.
   */
  public PathFragment relativeTo(PathFragment base) {
    Preconditions.checkNotNull(base);
    if (isAbsolute() != base.isAbsolute()) {
      throw new IllegalArgumentException(
          "Cannot relativize an absolute and a non-absolute path pair");
    }
    String basePath = base.normalizedPath;
    if (!OS.startsWith(normalizedPath, basePath)) {
      throw new IllegalArgumentException(
          String.format("Path '%s' is not under '%s', cannot relativize", this, base));
    }
    int bn = basePath.length();
    if (bn == 0) {
      return this;
    }
    if (normalizedPath.length() == bn) {
      return EMPTY_FRAGMENT;
    }
    final int lastSlashIndex;
    if (basePath.charAt(bn - 1) == '/') {
      lastSlashIndex = bn - 1;
    } else {
      lastSlashIndex = bn;
    }
    if (normalizedPath.charAt(lastSlashIndex) != '/') {
      throw new IllegalArgumentException(
          String.format("Path '%s' is not under '%s', cannot relativize", this, base));
    }
    String newPath = normalizedPath.substring(lastSlashIndex + 1);
    return new PathFragment(newPath, 0 /* Always a relative path */);
  }

  public PathFragment relativeTo(String base) {
    return relativeTo(PathFragment.create(base));
  }

  /**
   * Returns whether this path is an ancestor of another path.
   *
   * <p>If this == other, true is returned.
   *
   * <p>An absolute path can never be an ancestor of a relative path, and vice versa.
   */
  public boolean startsWith(PathFragment other) {
    Preconditions.checkNotNull(other);
    if (other.normalizedPath.length() > normalizedPath.length()) {
      return false;
    }
    if (driveStrLength != other.driveStrLength) {
      return false;
    }
    if (!OS.startsWith(normalizedPath, other.normalizedPath)) {
      return false;
    }
    return normalizedPath.length() == other.normalizedPath.length()
        || other.normalizedPath.length() == driveStrLength
        || normalizedPath.charAt(other.normalizedPath.length()) == SEPARATOR_CHAR;
  }

  /**
   * Returns true iff {@code suffix}, considered as a list of path segments, is relative and a
   * suffix of {@code this}, or both are absolute and equal.
   *
   * <p>This is a reflexive, transitive, anti-symmetric relation (i.e. a partial order)
   */
  public boolean endsWith(PathFragment other) {
    Preconditions.checkNotNull(other);
    if (other.normalizedPath.length() > normalizedPath.length()) {
      return false;
    }
    if (other.isAbsolute()) {
      return this.equals(other);
    }
    if (!OS.endsWith(normalizedPath, other.normalizedPath)) {
      return false;
    }
    return normalizedPath.length() == other.normalizedPath.length()
        || other.normalizedPath.isEmpty()
        || normalizedPath.charAt(normalizedPath.length() - other.normalizedPath.length() - 1)
            == SEPARATOR_CHAR;
  }

  public boolean isAbsolute() {
    return driveStrLength > 0;
  }

  public static boolean isAbsolute(String path) {
    return OS.getDriveStrLength(path) > 0;
  }

  @Override
  public String toString() {
    return normalizedPath;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    return OS.equals(this.normalizedPath, ((PathFragment) o).normalizedPath);
  }

  @Override
  public int hashCode() {
    return OS.hash(this.normalizedPath);
  }

  @Override
  public int compareTo(PathFragment o) {
    return OS.compare(this.normalizedPath, o.normalizedPath);
  }

  ////////////////////////////////////////////////////////////////////////

  /**
   * Returns the number of segments in this path.
   *
   * <p>This operation is O(N) on the length of the string.
   */
  public int segmentCount() {
    int n = normalizedPath.length();
    int segmentCount = 0;
    int i;
    for (i = driveStrLength; i < n; ++i) {
      if (normalizedPath.charAt(i) == SEPARATOR_CHAR) {
        ++segmentCount;
      }
    }
    // Add last segment if one exists.
    if (i > driveStrLength) {
      ++segmentCount;
    }
    return segmentCount;
  }

  /**
   * Returns the specified segment of this path; index must be non-negative and less than {@code
   * segmentCount()}.
   *
   * <p>This operation is O(N) on the length of the string.
   */
  public String getSegment(int index) {
    int n = normalizedPath.length();
    int segmentCount = 0;
    int i;
    for (i = driveStrLength; i < n && segmentCount < index; ++i) {
      if (normalizedPath.charAt(i) == SEPARATOR_CHAR) {
        ++segmentCount;
      }
    }
    int starti = i;
    for (; i < n; ++i) {
      if (normalizedPath.charAt(i) == SEPARATOR_CHAR) {
        break;
      }
    }
    // Add last segment if one exists.
    if (i > driveStrLength) {
      ++segmentCount;
    }
    int endi = i;
    if (index < 0 || index >= segmentCount) {
      throw new IllegalArgumentException("Illegal segment index: " + index);
    }
    return normalizedPath.substring(starti, endi);
  }

  /**
   * Returns a new path fragment that is a sub fragment of this one. The sub fragment begins at the
   * specified <code>beginIndex</code> segment and ends at the segment at index <code>endIndex - 1
   * </code>. Thus the number of segments in the new PathFragment is <code>endIndex - beginIndex
   * </code>.
   *
   * <p>This operation is O(N) on the length of the string.
   *
   * @param beginIndex the beginning index, inclusive.
   * @param endIndex the ending index, exclusive.
   * @return the specified sub fragment, never null.
   * @exception IndexOutOfBoundsException if the <code>beginIndex</code> is negative, or <code>
   *     endIndex</code> is larger than the length of this <code>String</code> object, or <code>
   *     beginIndex</code> is larger than <code>endIndex</code>.
   */
  public PathFragment subFragment(int beginIndex, int endIndex) {
    if (beginIndex < 0 || beginIndex > endIndex) {
      throw new IndexOutOfBoundsException(
          String.format("path: %s, beginIndex: %d endIndex: %d", toString(), beginIndex, endIndex));
    }
    return subFragmentImpl(beginIndex, endIndex);
  }

  public PathFragment subFragment(int beginIndex) {
    if (beginIndex < 0) {
      throw new IndexOutOfBoundsException(
          String.format("path: %s, beginIndex: %d", toString(), beginIndex));
    }
    return subFragmentImpl(beginIndex, -1);
  }

  private PathFragment subFragmentImpl(int beginIndex, int endIndex) {
    int n = normalizedPath.length();
    int segmentIndex = 0;
    int i;
    for (i = driveStrLength; i < n && segmentIndex < beginIndex; ++i) {
      if (normalizedPath.charAt(i) == SEPARATOR_CHAR) {
        ++segmentIndex;
      }
    }
    int starti = i;
    if (segmentIndex < endIndex) {
      for (; i < n; ++i) {
        if (normalizedPath.charAt(i) == SEPARATOR_CHAR) {
          ++segmentIndex;
          if (segmentIndex == endIndex) {
            break;
          }
        }
      }
    } else if (endIndex == -1) {
      i = normalizedPath.length();
    }
    int endi = i;
    // Add last segment if one exists for verification
    if (i == n && i > driveStrLength) {
      ++segmentIndex;
    }
    if (beginIndex > segmentIndex || endIndex > segmentIndex) {
      throw new IndexOutOfBoundsException(
          String.format("path: %s, beginIndex: %d endIndex: %d", toString(), beginIndex, endIndex));
    }
    // If beginIndex is 0 we include the drive. Very odd semantics.
    int driveStrLength = 0;
    if (beginIndex == 0) {
      starti = 0;
      driveStrLength = this.driveStrLength;
      endi = Math.max(endi, driveStrLength);
    }
    return new PathFragment(normalizedPath.substring(starti, endi), driveStrLength);
  }

  /**
   * Returns an {@link Iterable} that lazily yields the segments of this path.
   *
   * <p>When iterating over the segments of a path fragment, prefer this method to {@link
   * #splitToListOfSegments} as it performs a single, lazy traversal over the path string without
   * the overhead of creating a list.
   */
  public Iterable<String> segments() {
    return () -> PathSegmentIterator.create(normalizedPath, driveStrLength);
  }

  /**
   * Splits this path fragment into a list of segments.
   *
   * <p>This operation is O(N) on the length of the string. If it is not necessary to store the
   * segments in list form, consider using {@link #segments}.
   */
  public ImmutableList<String> splitToListOfSegments() {
    ImmutableList.Builder<String> segments = ImmutableList.builderWithExpectedSize(segmentCount());
    int nexti = driveStrLength;
    int n = normalizedPath.length();
    for (int i = driveStrLength; i < n; ++i) {
      if (normalizedPath.charAt(i) == SEPARATOR_CHAR) {
        segments.add(normalizedPath.substring(nexti, i));
        nexti = i + 1;
      }
    }
    // Add last segment if one exists.
    if (nexti < n) {
      segments.add(normalizedPath.substring(nexti));
    }
    return segments.build();
  }

  /** Returns the path string, or '.' if the path is empty. */
  public String getSafePathString() {
    return !normalizedPath.isEmpty() ? normalizedPath : ".";
  }

  /**
   * Returns the path string using '/' as the name-separator character, but do so in a way
   * unambiguously recognizable as path. In other words, return "." for relative and empty paths,
   * and prefix relative paths with one segment by "./".
   *
   * <p>In this way, a shell will always interpret such a string as path (absolute or relative to
   * the working directory) and not as command to be searched for in the search path.
   */
  public String getCallablePathString() {
    if (isAbsolute()) {
      return normalizedPath;
    } else if (normalizedPath.isEmpty()) {
      return ".";
    } else if (normalizedPath.indexOf(SEPARATOR_CHAR) == -1) {
      return "." + SEPARATOR_CHAR + normalizedPath;
    } else {
      return normalizedPath;
    }
  }

  /**
   * Returns the file extension of this path, excluding the period, or "" if there is no extension.
   */
  public String getFileExtension() {
    int n = normalizedPath.length();
    for (int i = n - 1; i > driveStrLength; --i) {
      char c = normalizedPath.charAt(i);
      if (c == '.') {
        return normalizedPath.substring(i + 1, n);
      } else if (c == SEPARATOR_CHAR) {
        break;
      }
    }
    return "";
  }

  /**
   * Returns a new PathFragment formed by appending {@code newName} to the parent directory. Null is
   * returned iff this method is called on a PathFragment with zero segments. If {@code newName}
   * designates an absolute path, the value of {@code this} will be ignored and a PathFragment
   * corresponding to {@code newName} will be returned. This behavior is consistent with the
   * behavior of {@link #getRelative(String)}.
   */
  public PathFragment replaceName(String newName) {
    PathFragment parent = getParentDirectory();
    return parent != null ? parent.getRelative(newName) : null;
  }

  /**
   * Returns the drive for an absolute path fragment.
   *
   * <p>On unix, this will return "/". On Windows it will return the drive letter, like "C:/".
   */
  public String getDriveStr() {
    Preconditions.checkArgument(isAbsolute());
    return normalizedPath.substring(0, driveStrLength);
  }

  /**
   * Returns a relative PathFragment created from this absolute PathFragment using the
   * same segments and drive letter.
   */
  public PathFragment toRelative() {
    Preconditions.checkArgument(isAbsolute());
    return new PathFragment(normalizedPath.substring(driveStrLength), 0);
  }

  /**
   * Returns true if this path contains uplevel references "..".
   *
   * <p>Since path fragments are normalized, this implies that the uplevel reference is at the start
   * of the path fragment.
   */
  public boolean containsUplevelReferences() {
    // Path is normalized, so any ".." would have to be the first segment.
    return normalizedPath.startsWith("..")
        && (normalizedPath.length() == 2 || normalizedPath.charAt(2) == SEPARATOR_CHAR);
  }

  /**
   * Returns true if the passed path contains uplevel references ".." or single-dot references "."
   *
   * <p>This is useful to check a string for normalization before constructing a PathFragment, since
   * these are always normalized and will throw uplevel references away.
   */
  public static boolean isNormalized(String path) {
    return isNormalizedImpl(path, /* lookForSameLevelReferences= */ true);
  }

  /**
   * Returns true if the passed path contains uplevel references "..".
   *
   * <p>This is useful to check a string for '..' segments before constructing a PathFragment, since
   * these are always normalized and will throw uplevel references away.
   */
  public static boolean containsUplevelReferences(String path) {
    return !isNormalizedImpl(path, /* lookForSameLevelReferences= */ false);
  }

  private enum NormalizedImplState {
    Base, /* No particular state, eg. an 'a' or 'L' character */
    Separator, /* We just saw a separator */
    Dot, /* We just saw a dot after a separator */
    DotDot, /* We just saw two dots after a separator */
  }

  private static boolean isNormalizedImpl(String path, boolean lookForSameLevelReferences) {
    // Starting state is equivalent to having just seen a separator
    NormalizedImplState state = NormalizedImplState.Separator;
    int n = path.length();
    for (int i = 0; i < n; ++i) {
      char c = path.charAt(i);
      boolean isSeparator = OS.isSeparator(c);
      switch (state) {
        case Base:
          if (isSeparator) {
            state = NormalizedImplState.Separator;
          } else {
            state = NormalizedImplState.Base;
          }
          break;
        case Separator:
          if (isSeparator) {
            state = NormalizedImplState.Separator;
          } else if (c == '.') {
            state = NormalizedImplState.Dot;
          } else {
            state = NormalizedImplState.Base;
          }
          break;
        case Dot:
          if (isSeparator) {
            if (lookForSameLevelReferences) {
              // "." segment found
              return false;
            }
            state = NormalizedImplState.Separator;
          } else if (c == '.') {
            state = NormalizedImplState.DotDot;
          } else {
            state = NormalizedImplState.Base;
          }
          break;
        case DotDot:
          if (isSeparator) {
            // ".." segment found
            return false;
          } else {
            state = NormalizedImplState.Base;
          }
          break;
        default:
          throw new IllegalStateException("Unhandled state: " + state);
      }
    }
    // The character just after the string is equivalent to a separator
    switch (state) {
      case Dot:
        if (lookForSameLevelReferences) {
          // "." segment found
          return false;
        }
        break;
      case DotDot:
        return false;
      default:
    }
    return true;
  }

  /**
   * Throws {@link IllegalArgumentException} if {@code paths} contains any paths that are equal to
   * {@code startingWithPath} or that are not beneath {@code startingWithPath}.
   */
  public static void checkAllPathsAreUnder(
      Iterable<PathFragment> paths, PathFragment startingWithPath) {
    for (PathFragment path : paths) {
      Preconditions.checkArgument(
          !path.equals(startingWithPath) && path.startsWith(startingWithPath),
          "%s is not beneath %s",
          path,
          startingWithPath);
    }
  }

  @Override
  public String filePathForFileTypeMatcher() {
    return normalizedPath;
  }

  @Override
  public String expandToCommandLine() {
    return normalizedPath;
  }

  private static void checkBaseName(String baseName) {
    if (baseName.isEmpty()) {
      throw new IllegalArgumentException("Child must not be empty string ('')");
    }
    if (baseName.equals(".") || baseName.equals("..")) {
      throw new IllegalArgumentException("baseName must not be '" + baseName + "'");
    }
    try {
      checkSeparators(baseName);
    } catch (InvalidBaseNameException e) {
      throw new IllegalArgumentException("baseName " + e.getMessage() + ": '" + baseName + "'", e);
    }
  }

  public static void checkSeparators(String baseName) throws InvalidBaseNameException {
    if (baseName.indexOf(SEPARATOR_CHAR) != -1) {
      throw new InvalidBaseNameException("must not contain " + SEPARATOR_CHAR);
    }
    if (ADDITIONAL_SEPARATOR_CHAR != 0) {
      if (baseName.indexOf(ADDITIONAL_SEPARATOR_CHAR) != -1) {
        throw new InvalidBaseNameException("must not contain " + ADDITIONAL_SEPARATOR_CHAR);
      }
    }
  }

  /** Indicates that a path fragment's base name had invalid characters. */
  public static final class InvalidBaseNameException extends Exception {
    private InvalidBaseNameException(String message) {
      super(message);
    }
  }

  @SuppressWarnings("unused") // found by CLASSPATH-scanning magic
  private static class Codec implements ObjectCodec<PathFragment> {
    @Override
    public Class<PathFragment> getEncodedClass() {
      return PathFragment.class;
    }

    @Override
    public void serialize(
        SerializationContext context, PathFragment obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      context.serialize(obj.normalizedPath, codedOut);
    }

    @Override
    public PathFragment deserialize(DeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      return PathFragment.createAlreadyNormalized(context.deserialize(codedIn));
    }
  }
}
