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
package com.google.devtools.build.lib.vfs;

import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrintable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.StringCanonicalizer;
import java.io.File;
import java.io.InvalidObjectException;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import java.util.Set;

/**
 * This class represents an immutable filesystem path, which may be absolute or relative. The path
 * is maintained as a simple ordered list of path segment strings.
 *
 * <p>This class is independent from other VFS classes, especially anything requiring native code.
 * It is safe to use in places that need simple segmented string path functionality.
 *
 * <p>There is some limited support for Windows-style paths. Most importantly, drive identifiers in
 * front of a path (c:/abc) are supported and such paths are correctly recognized as absolute, as
 * are paths with backslash separators (C:\\foo\\bar). However, advanced Windows-style features like
 * \\\\network\\paths and \\\\?\\unc\\paths are not supported.
 */
@Immutable
@javax.annotation.concurrent.Immutable
@ThreadSafe
public abstract class PathFragment
    implements Comparable<PathFragment>, Serializable, SkylarkPrintable {
  private static final Helper HELPER =
      OS.getCurrent() == OS.WINDOWS ? WindowsPathFragment.HELPER : UnixPathFragment.HELPER;

  public static final char SEPARATOR_CHAR = HELPER.getPrimarySeparatorChar();

  public static final int INVALID_SEGMENT = -1;

  public static final String ROOT_DIR = "/";

  /** An empty path fragment. */
  public static final PathFragment EMPTY_FRAGMENT = create("");

  /** The path fragment representing the root directory. */
  public static final PathFragment ROOT_FRAGMENT = create(ROOT_DIR);

  /**
   * A helper object for manipulating the various internal {@link PathFragment} implementations.
   *
   * <p>There will be exactly one {@link Helper} instance used to manipulate all the {@link
   * PathFragment} instances (see {@link PathFragment#HELPER}). All of the various {@link Helper}
   * and {@link PathFragment} implementations may assume this property.
   */
  protected abstract static class Helper {
    /**
     * Returns whether the two given arrays of segments have the same length and should be
     * considered have logically equal contents.
     */
    protected final boolean segmentsEqual(String[] segments1, String[] segments2) {
      return segments1.length == segments2.length
          && segmentsEqual(segments1.length, segments1, 0, segments2);
    }

    /**
     * Returns whether the {@code length} segments in {@code segments1}, starting at {@code offset1}
     * should be considered to be logically equal to the first {@code length} segments in {@code
     * segments2}.
     */
    abstract boolean segmentsEqual(int length, String[] segments1, int offset1, String[] segments2);

    /** Returns the comparison result of two {@link PathFragment} instances. */
    protected abstract int compare(PathFragment pathFragment1, PathFragment pathFragment2);

    /** Returns a fresh {@link PathFragment} instance from the given path string. */
    abstract PathFragment create(String path);
    /**
     * Returns a fresh {@link PathFragment} instance from the given information, taking ownership of
     * {@code segments} and assuming the {@link String}s within have already been interned.
     */
    abstract PathFragment createAlreadyInterned(
        char driveLetter, boolean isAbsolute, String[] segments);

    /** Returns whether {@code c} is a path separator. */
    abstract boolean isSeparator(char c);
    /** Returns the primary path separator. */
    abstract char getPrimarySeparatorChar();
    /** Return whether the given {@code path} contains a path separator. */
    abstract boolean containsSeparatorChar(String path);

    /**
     * Splits the given {@code toSegment} into path segments, starting at the given {@code offset}.
     */
    protected final String[] segment(String toSegment, int offset) {
      int length = toSegment.length();

      // We make two passes through the array of characters: count & alloc,
      // because simply using ArrayList was a bottleneck showing up during profiling.
      int seg = 0;
      int start = offset;
      for (int i = offset; i < length; i++) {
        if (isSeparator(toSegment.charAt(i))) {
          if (i > start) { // to skip repeated separators
            seg++;
          }
          start = i + 1;
        }
      }
      if (start < length) {
        seg++;
      }
      String[] result = new String[seg];
      seg = 0;
      start = offset;
      for (int i = offset; i < length; i++) {
        if (isSeparator(toSegment.charAt(i))) {
          if (i > start) { // to skip repeated separators
            result[seg] = StringCanonicalizer.intern(toSegment.substring(start, i));
            seg++;
          }
          start = i + 1;
        }
      }
      if (start < length) {
        result[seg] = StringCanonicalizer.intern(toSegment.substring(start, length));
      }
      return result;
    }
  }

  /** Lower-level API. Create a PathFragment, interning segments. */
  public static PathFragment create(char driveLetter, boolean isAbsolute, String[] segments) {
    String[] internedSegments = new String[segments.length];
    for (int i = 0; i < segments.length; i++) {
      internedSegments[i] = StringCanonicalizer.intern(segments[i]);
    }
    return createAlreadyInterned(driveLetter, isAbsolute, internedSegments);
  }

  /** Same as {@link #create(char, boolean, String[])}, except for {@link List}s of segments. */
  public static PathFragment create(char driveLetter, boolean isAbsolute, List<String> segments) {
    String[] internedSegments = new String[segments.size()];
    for (int i = 0; i < segments.size(); i++) {
      internedSegments[i] = StringCanonicalizer.intern(segments.get(i));
    }
    return createAlreadyInterned(driveLetter, isAbsolute, internedSegments);
  }

  /**
   * Construct a PathFragment from a java.io.File, which is an absolute or
   * relative UNIX path.  Does not support Windows-style Files.
   */
  public static PathFragment create(File path) {
    return HELPER.create(path.getPath());
  }

  /**
   * Construct a PathFragment from a string, which is an absolute or relative UNIX or Windows path.
   */
  public static PathFragment create(String path) {
    return HELPER.create(path);
  }

  /**
   * Constructs a PathFragment, taking ownership of {@code segments} and assuming the {@link
   * String}s within have already been interned.
   *
   * <p>Package-private because it does not perform a defensive copy of the segments array. Used
   * here in PathFragment, and by Path.asFragment() and Path.relativeTo().
   */
  static PathFragment createAlreadyInterned(
      char driveLetter, boolean isAbsolute, String[] segments) {
    return HELPER.createAlreadyInterned(driveLetter, isAbsolute, segments);
  }

  /** Returns whether the current {@code path} contains a path separator. */
  static boolean containsSeparator(String path) {
    return HELPER.containsSeparatorChar(path);
  }

  /**
   * Construct a PathFragment from a sequence of other PathFragments. The new fragment will be
   * absolute iff the first fragment was absolute.
   */
  // TODO(bazel-team): Most usages of this method are wasteful from a garbage perspective. Refactor
  // to something better.
  public static PathFragment create(PathFragment first, PathFragment second, PathFragment... more) {
    String[] segments = new String[sumLengths(first, second, more)];
    int offset = 0;
    offset += addSegmentsTo(segments, offset, first);
    offset += addSegmentsTo(segments, offset, second);
    for (PathFragment fragment : more) {
      offset += addSegmentsTo(segments, offset, fragment);
    }
    boolean isAbsolute = first.isAbsolute();
    char driveLetter = first.getDriveLetter();
    return HELPER.createAlreadyInterned(driveLetter, isAbsolute, segments);
  }

  // Medium sized builds can easily hold millions of live PathFragments, so the per-instance size of
  // PathFragment is a concern.
  //
  // We have two oop-sized fields (segments, path), and one 4-byte-sized one (hashCode).
  //
  // If Blaze is run on a jvm with -XX:+UseCompressedOops, each PathFragment instance is 24 bytes
  // and so adding any additional field will increase the per-instance size to at least 32 bytes.
  //
  // If Blaze is run on a jvm with -XX:-UseCompressedOops, each PathFragment instance is 32 bytes
  // and so adding any additional field will increase the per-instance size to at least 40 bytes.
  //
  // Therefore, do not add any additional fields unless you have considered the memory implications.

  // The individual path components.
  // Does *not* include the Windows drive letter.
  protected final String[] segments;

  // hashCode and path are lazily initialized but semantically immutable.
  private int hashCode;
  private String path;

  protected PathFragment(String[] segments) {
    this.segments = segments;
  }

  private static int addSegmentsTo(String[] segments, int offset, PathFragment fragment) {
    int count = fragment.segmentCount();
    System.arraycopy(fragment.segments, 0, segments, offset, count);
    return count;
  }

  private static int sumLengths(PathFragment first, PathFragment second, PathFragment[] more) {
    int total = first.segmentCount() + second.segmentCount();
    for (PathFragment fragment : more) {
      total += fragment.segmentCount();
    }
    return total;
  }

  protected Object writeReplace() {
    return new PathFragmentSerializationProxy(toString());
  }

  protected void readObject(ObjectInputStream stream) throws InvalidObjectException {
    throw new InvalidObjectException("Serialization is allowed only by proxy");
  }

  /**
   * Returns the path string using '/' as the name-separator character.  Returns "" if the path
   * is both relative and empty.
   */
  public String getPathString() {
    // Double-checked locking works, even without volatile, because path is a String, according to:
    // http://www.cs.umd.edu/~pugh/java/memoryModel/DoubleCheckedLocking.html
    if (path == null) {
      synchronized (this) {
        if (path == null) {
          path = StringCanonicalizer.intern(joinSegments(HELPER.getPrimarySeparatorChar()));
        }
      }
    }
    return path;
  }

  /**
   * Returns "." if the path fragment is both relative and empty, or {@link
   * #getPathString} otherwise.
   */
  // TODO(bazel-team): Change getPathString to do this - this behavior makes more sense.
  public String getSafePathString() {
    return (!isAbsolute() && (segmentCount() == 0)) ? "." : getPathString();
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
      return getPathString();
    } else if (segmentCount() == 0) {
      return ".";
    } else if (segmentCount() == 1) {
      return "." + HELPER.getPrimarySeparatorChar() + getPathString();
    } else {
      return getPathString();
    }
  }

  /**
   * Returns a sequence consisting of the {@link #getSafePathString()} return of each item in
   * {@code fragments}.
   */
  public static Iterable<String> safePathStrings(Iterable<PathFragment> fragments) {
    return Iterables.transform(fragments, PathFragment::getSafePathString);
  }

  /** Returns the subset of {@code paths} that start with {@code startingWithPath}. */
  public static ImmutableSet<PathFragment> filterPathsStartingWith(
      Set<PathFragment> paths, PathFragment startingWithPath) {
    return paths
        .stream()
        .filter(pathFragment -> pathFragment.startsWith(startingWithPath))
        .collect(toImmutableSet());
  }

  /**
  * Throws {@link IllegalArgumentException} if {@code paths} contains any paths that
  * are equal to {@code startingWithPath} or that are not beneath {@code startingWithPath}.
  */
  public static void checkAllPathsAreUnder(Iterable<PathFragment> paths,
      PathFragment startingWithPath) {
    for (PathFragment path : paths) {
      Preconditions.checkArgument(
          !path.equals(startingWithPath) && path.startsWith(startingWithPath),
              "%s is not beneath %s", path, startingWithPath);
    }
  }

  private String joinSegments(char separatorChar) {
    if (segments.length == 0 && isAbsolute()) {
      return windowsVolume() + ROOT_DIR;
    }

    // Profile driven optimization:
    // Preallocate a size determined by the number of segments, so that
    // we do not have to expand the capacity of the StringBuilder.
    // Heuristically, this estimate is right for about 99% of the time.
    int estimateSize =
        ((getDriveLetter() != '\0') ? 2 : 0)
            + ((segments.length == 0) ? 0 : (segments.length + 1) * 20);
    StringBuilder result = new StringBuilder(estimateSize);
    if (isAbsolute()) {
      // Only print the Windows volume label if the PathFragment is absolute. Do not print relative
      // Windows paths like "C:foo/bar", it would break all kinds of things, e.g. glob().
      result.append(windowsVolume());
    }
    boolean initialSegment = true;
    for (String segment : segments) {
      if (!initialSegment || isAbsolute()) {
        result.append(separatorChar);
      }
      initialSegment = false;
      result.append(segment);
    }
    return result.toString();
  }

  /**
   * Return true iff none of the segments are either "." or "..".
   */
  public boolean isNormalized() {
    for (String segment : segments) {
      if (segment.equals(".") || segment.equals("..")) {
        return false;
      }
    }
    return true;
  }

  /**
   * Normalizes the path fragment: removes "." and ".." segments if possible
   * (if there are too many ".." segments, the resulting PathFragment will still
   * start with "..").
   */
  public PathFragment normalize() {
    String[] scratchSegments = new String[segments.length];
    int segmentCount = 0;

    for (String segment : segments) {
      switch (segment) {
        case ".":
          // Just discard it
          break;
        case "..":
          if (segmentCount > 0 && !scratchSegments[segmentCount - 1].equals("..")) {
            // Remove the last segment, if there is one and it is not "..". This
            // means that the resulting PathFragment can still contain ".."
            // segments at the beginning.
            segmentCount--;
          } else {
            scratchSegments[segmentCount++] = segment;
          }
          break;
        default:
          scratchSegments[segmentCount++] = segment;
      }
    }

    if (segmentCount == segments.length) {
      // Optimization, no new PathFragment needs to be created.
      return this;
    }

    return HELPER.createAlreadyInterned(
        getDriveLetter(), isAbsolute(), subarray(scratchSegments, 0, segmentCount));
  }

  /**
   * Returns the path formed by appending the relative or absolute path fragment
   * {@code otherFragment} to this path.
   *
   * <p>If {@code otherFragment} is absolute, the current path will be ignored;
   * otherwise, they will be concatenated. This is a purely syntactic operation,
   * with no path normalization or I/O performed.
   */
  public PathFragment getRelative(PathFragment otherFragment) {
    if (otherFragment == EMPTY_FRAGMENT) {
      return this;
    }

    if (otherFragment.isAbsolute()) {
      char driveLetter = getDriveLetter();
      return driveLetter == '\0' || otherFragment.getDriveLetter() != '\0'
          ? otherFragment
          : createAlreadyInterned(driveLetter, true, otherFragment.segments);
    } else {
      return create(this, otherFragment);
    }
  }

  /**
   * Returns the path formed by appending the relative or absolute string
   * {@code path} to this path.
   *
   * <p>If the given path string is absolute, the current path will be ignored;
   * otherwise, they will be concatenated. This is a purely syntactic operation,
   * with no path normalization or I/O performed.
   */
  public PathFragment getRelative(String path) {
    return getRelative(create(path));
  }

  /**
   * Returns the path formed by appending the single non-special segment "baseName" to this path.
   *
   * <p>You should almost always use {@link #getRelative} instead, which has the same performance
   * characteristics if the given name is a valid base name, and which also works for '.', '..', and
   * strings containing '/'.
   *
   * @throws IllegalArgumentException if {@code baseName} is not a valid base name according to
   *     {@link #checkBaseName}
   */
  public PathFragment getChild(String baseName) {
    checkBaseName(baseName);
    baseName = StringCanonicalizer.intern(baseName);
    String[] newSegments = Arrays.copyOf(segments, segments.length + 1);
    newSegments[newSegments.length - 1] = baseName;
    return createAlreadyInterned(getDriveLetter(), isAbsolute(), newSegments);
  }

  /**
   * Returns the last segment of this path, or "" for the empty fragment.
   */
  public String getBaseName() {
    return (segments.length == 0) ? "" : segments[segments.length - 1];
  }

  /**
   * Returns the file extension of this path, excluding the period, or "" if there is no extension.
   */
  public String getFileExtension() {
    String baseName = getBaseName();

    int lastIndex = baseName.lastIndexOf('.');
    if (lastIndex != -1) {
      return baseName.substring(lastIndex + 1);
    }

    return "";
  }

  /**
   * Returns a relative path fragment to this path, relative to
   * {@code ancestorDirectory}.
   * <p>
   * <code>x.relativeTo(z) == y</code> implies
   * <code>z.getRelative(y) == x</code>.
   * <p>
   * For example, <code>"foo/bar/wiz".relativeTo("foo")</code>
   * returns <code>"bar/wiz"</code>.
   */
  public PathFragment relativeTo(PathFragment ancestorDirectory) {
    String[] ancestorSegments = ancestorDirectory.segments();
    int ancestorLength = ancestorSegments.length;

    if (isAbsolute() != ancestorDirectory.isAbsolute() || segments.length < ancestorLength) {
      throw new IllegalArgumentException("PathFragment " + this
          + " is not beneath " + ancestorDirectory);
    }

    if (!HELPER.segmentsEqual(ancestorLength, segments, 0, ancestorSegments)) {
      throw new IllegalArgumentException(
          "PathFragment " + this + " is not beneath " + ancestorDirectory);
    }

    int length = segments.length - ancestorLength;
    String[] resultSegments = subarray(segments, ancestorLength, length);
    return createAlreadyInterned('\0', false, resultSegments);
  }

  /**
   * Returns a relative path fragment to this path, relative to {@code path}.
   */
  public PathFragment relativeTo(String path) {
    return relativeTo(create(path));
  }

  /**
   * Returns a new PathFragment formed by appending {@code newName} to the
   * parent directory. Null is returned iff this method is called on a
   * PathFragment with zero segments.  If {@code newName} designates an absolute path,
   * the value of {@code this} will be ignored and a PathFragment corresponding to
   * {@code newName} will be returned.  This behavior is consistent with the behavior of
   * {@link #getRelative(String)}.
   */
  public PathFragment replaceName(String newName) {
    return segments.length == 0 ? null : getParentDirectory().getRelative(newName);
  }

  /**
   * Returns a path representing the parent directory of this path,
   * or null iff this Path represents the root of the filesystem.
   *
   * <p>Note: This method DOES NOT normalize ".."  and "." path segments.
   */
  public PathFragment getParentDirectory() {
    return segments.length == 0 ? null : subFragment(0, segments.length - 1);
  }

  /**
   * Returns true iff {@code prefix}, considered as a list of path segments, is
   * a prefix of {@code this}, and that they are both relative or both
   * absolute.
   *
   * <p>This is a reflexive, transitive, anti-symmetric relation (i.e. a partial
   * order)
   */
  public boolean startsWith(PathFragment prefix) {
    if (isAbsolute() != prefix.isAbsolute()
        || this.segments.length < prefix.segments.length
        || (isAbsolute() && getDriveLetter() != prefix.getDriveLetter())) {
      return false;
    }
    return HELPER.segmentsEqual(prefix.segments.length, segments, 0, prefix.segments);
  }

  /**
   * Returns true iff {@code suffix}, considered as a list of path segments, is
   * relative and a suffix of {@code this}, or both are absolute and equal.
   *
   * <p>This is a reflexive, transitive, anti-symmetric relation (i.e. a partial
   * order)
   */
  public boolean endsWith(PathFragment suffix) {
    if ((suffix.isAbsolute() && !suffix.equals(this))
        || this.segments.length < suffix.segments.length) {
      return false;
    }
    int offset = this.segments.length - suffix.segments.length;
    return HELPER.segmentsEqual(suffix.segments.length, segments, offset, suffix.segments);
  }

  private static String[] subarray(String[] array, int start, int length) {
    String[] subarray = new String[length];
    System.arraycopy(array, start, subarray, 0, length);
    return subarray;
  }

  /**
   * Returns a new path fragment that is a sub fragment of this one.
   * The sub fragment begins at the specified <code>beginIndex</code> segment
   * and ends at the segment at index <code>endIndex - 1</code>. Thus the number
   * of segments in the new PathFragment is <code>endIndex - beginIndex</code>.
   *
   * @param      beginIndex   the beginning index, inclusive.
   * @param      endIndex     the ending index, exclusive.
   * @return     the specified sub fragment, never null.
   * @exception  IndexOutOfBoundsException  if the
   *             <code>beginIndex</code> is negative, or
   *             <code>endIndex</code> is larger than the length of
   *             this <code>String</code> object, or
   *             <code>beginIndex</code> is larger than
   *             <code>endIndex</code>.
   */
  public PathFragment subFragment(int beginIndex, int endIndex) {
    int count = segments.length;
    if ((beginIndex < 0) || (beginIndex > endIndex) || (endIndex > count)) {
      throw new IndexOutOfBoundsException(String.format("path: %s, beginIndex: %d endIndex: %d",
          toString(), beginIndex, endIndex));
    }
    boolean isAbsolute = (beginIndex == 0) && isAbsolute();
    return ((beginIndex == 0) && (endIndex == count))
        ? this
        : createAlreadyInterned(
            getDriveLetter(), isAbsolute, subarray(segments, beginIndex, endIndex - beginIndex));
  }

  /**
   * Returns true iff the path represented by this object is absolute.
   *
   * <p>True both for UNIX-style absolute paths ("/foo") and Windows-style ("C:/foo"). False for a
   * Windows-style volume label ("C:") which is actually a relative path.
   */
  public abstract boolean isAbsolute();

  /**
   * Returns the segments of this path fragment. This array should not be
   * modified.
   */
  String[] segments() {
    return segments;
  }

  public ImmutableList<String> getSegments() {
    return ImmutableList.copyOf(segments);
  }

  public abstract String windowsVolume();

  /** Return the drive letter or '\0' if not applicable. */
  // TODO(bazel-team): This doesn't need to pollute the PathFragment interface (ditto for
  // windowsVolume).
  public abstract char getDriveLetter();

  /**
   * Returns the number of segments in this path.
   */
  public int segmentCount() {
    return segments.length;
  }

  /**
   * Returns the specified segment of this path; index must be positive and
   * less than numSegments().
   */
  public String getSegment(int index) {
    return segments[index];
  }

  /**
   * Returns the index of the first segment which equals one of the input values
   * or {@link PathFragment#INVALID_SEGMENT} if none of the segments match.
   */
  public int getFirstSegment(Set<String> values) {
    for (int i = 0; i < segments.length; i++) {
      if (values.contains(segments[i])) {
        return i;
      }
    }
    return INVALID_SEGMENT;
  }

  /**
   * Returns true iff this path contains uplevel references "..".
   */
  public boolean containsUplevelReferences() {
    for (String segment : segments) {
      if (segment.equals("..")) {
        return true;
      }
    }
    return false;
  }

  /**
   * Returns a relative PathFragment created from this absolute PathFragment using the
   * same segments and drive letter.
   */
  public PathFragment toRelative() {
    Preconditions.checkArgument(isAbsolute());
    return HELPER.createAlreadyInterned(getDriveLetter(), false, segments);
  }

  @Override
  public final int hashCode() {
    // We use the hash code caching strategy employed by java.lang.String. There are three subtle
    // things going on here:
    //
    // (1) We use a value of 0 to indicate that the hash code hasn't been computed and cached yet.
    // Yes, this means that if the hash code is really 0 then we will "recompute" it each time. But
    // this isn't a problem in practice since a hash code of 0 is rare.
    //
    // (2) Since we have no synchronization, multiple threads can race here thinking they are the
    // first one to compute and cache the hash code.
    //
    // (3) Moreover, since 'hashCode' is non-volatile, the cached hash code value written from one
    // thread may not be visible by another. Note that we don't need to worry about multiple
    // inefficient reads of 'hashCode' on the same thread since it's non-volatile.
    //
    // All three of these issues are benign from a correctness perspective; in the end we have no
    // overhead from synchronization, at the cost of potentially computing the hash code more than
    // once.
    if (hashCode == 0) {
      hashCode = computeHashCode();
    }
    return hashCode;
  }

  protected abstract int computeHashCode();

  @Override
  public abstract boolean equals(Object other);

  /**
   * Compares two PathFragments using the lexicographical order.
   */
  @Override
  public int compareTo(PathFragment p2) {
    return HELPER.compare(this, p2);
  }

  @Override
  public String toString() {
    return getPathString();
  }

  @Override
  public void repr(SkylarkPrinter printer) {
    printer.append(getPathString());
  }

  private static void checkBaseName(String baseName) {
    if (baseName.length() == 0) {
      throw new IllegalArgumentException("Child must not be empty string ('')");
    }
    if (baseName.equals(".") || baseName.equals("..")) {
      throw new IllegalArgumentException("baseName must not be '" + baseName + "'");
    }
    if (baseName.indexOf('/') != -1) {
      throw new IllegalArgumentException("baseName must not contain a slash: '" + baseName + "'");
    }
  }
}
