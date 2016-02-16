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

import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.common.io.ByteSink;
import com.google.common.io.ByteSource;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.Constants;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ConditionallyThreadSafe;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.util.Preconditions;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintStream;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Helper functions that implement often-used complex operations on file
 * systems.
 */
@ConditionallyThreadSafe // ThreadSafe except for deleteTree.
public class FileSystemUtils {

  static final Logger LOG = Logger.getLogger(FileSystemUtils.class.getName());
  static final boolean LOG_FINER = LOG.isLoggable(Level.FINER);

  private FileSystemUtils() {}

  /****************************************************************************
   * Path and PathFragment functions.
   */

  /**
   * Throws exceptions if {@code baseName} is not a valid base name. A valid
   * base name:
   * <ul>
   * <li>Is not null
   * <li>Is not an empty string
   * <li>Is not "." or ".."
   * <li>Does not contain a slash
   * </ul>
   */
  @ThreadSafe
  public static void checkBaseName(String baseName) {
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

  /**
   * Returns the common ancestor between two paths, or null if none (including
   * if they are on different filesystems).
   */
  public static Path commonAncestor(Path a, Path b) {
    while (a != null && !b.startsWith(a)) {
      a = a.getParentDirectory();  // returns null at root
    }
    return a;
  }

  /**
   * Returns the longest common ancestor of the two path fragments, or either "/" or "" (depending
   * on whether {@code a} is absolute or relative) if there is none.
   */
  public static PathFragment commonAncestor(PathFragment a, PathFragment b) {
    while (a != null && !b.startsWith(a)) {
      a = a.getParentDirectory();
    }

    return a;
  }
  /**
   * Returns a path fragment from a given from-dir to a given to-path. May be
   * either a short relative path "foo/bar", an up'n'over relative path
   * "../../foo/bar" or an absolute path.
   */
  public static PathFragment relativePath(Path fromDir, Path to) {
    if (to.getFileSystem() != fromDir.getFileSystem()) {
      throw new IllegalArgumentException("fromDir and to must be on the same FileSystem");
    }

    return relativePath(fromDir.asFragment(), to.asFragment());
  }

  /**
   * Returns a path fragment from a given from-dir to a given to-path.
   */
  public static PathFragment relativePath(PathFragment fromDir, PathFragment to) {
    if (to.equals(fromDir)) {
      return new PathFragment(".");  // same dir, just return '.'
    }
    if (to.startsWith(fromDir)) {
      return to.relativeTo(fromDir);  // easy case--it's a descendant
    }
    PathFragment ancestor = commonAncestor(fromDir, to);
    if (ancestor == null) {
      return to;  // no common ancestor, use 'to'
    }
    int levels = fromDir.relativeTo(ancestor).segmentCount();
    StringBuilder dotdots = new StringBuilder();
    for (int i = 0; i < levels; i++) {
      dotdots.append("../");
    }
    return new PathFragment(dotdots.toString()).getRelative(to.relativeTo(ancestor));
  }

  /**
   * Returns the longest prefix from a given set of 'prefixes' that are
   * contained in 'path'. I.e the closest ancestor directory containing path.
   * Returns null if none found.
   */
  public static PathFragment longestPathPrefix(PathFragment path, Set<PathFragment> prefixes) {
    for (int i = path.segmentCount(); i >= 1; i--) {
      PathFragment prefix = path.subFragment(0, i);
      if (prefixes.contains(prefix)) {
        return prefix;
      }
    }
    return null;
  }

  /**
   * Removes the shortest suffix beginning with '.' from the basename of the
   * filename string. If the basename contains no '.', the filename is returned
   * unchanged.
   *
   * <p>e.g. "foo/bar.x" -> "foo/bar"
   *
   * <p>Note that if the filename is composed entirely of ".", this method will return the string
   * with one fewer ".", which may have surprising effects.
   */
  @ThreadSafe
  public static String removeExtension(String filename) {
    int lastDotIndex = filename.lastIndexOf('.');
    if (lastDotIndex == -1) { return filename; }
    int lastSlashIndex = filename.lastIndexOf('/');
    if (lastSlashIndex > lastDotIndex) {
      return filename;
    }
    return filename.substring(0, lastDotIndex);
  }

  /**
   * Removes the shortest suffix beginning with '.' from the basename of the
   * PathFragment. If the basename contains no '.', the filename is returned
   * unchanged.
   *
   * <p>e.g. "foo/bar.x" -> "foo/bar"
   *
   * <p>Note that if the base filename is composed entirely of ".", this method will return the
   * filename with one fewer "." in the base filename, which may have surprising effects.
   */
  @ThreadSafe
  public static PathFragment removeExtension(PathFragment path) {
    return path.replaceName(removeExtension(path.getBaseName()));
  }

  /**
   * Removes the shortest suffix beginning with '.' from the basename of the
   * Path. If the basename contains no '.', the filename is returned
   * unchanged.
   *
   * <p>e.g. "foo/bar.x" -> "foo/bar"
   *
   * <p>Note that if the base filename is composed entirely of ".", this method will return the
   * filename with one fewer "." in the base filename, which may have surprising effects.
   */
  @ThreadSafe
  public static Path removeExtension(Path path) {
    return path.getFileSystem().getPath(removeExtension(path.asFragment()));
  }

  /**
   * Returns a new {@code PathFragment} formed by replacing the extension of the
   * last path segment of {@code path} with {@code newExtension}. Null is
   * returned iff {@code path} has zero segments.
   */
  public static PathFragment replaceExtension(PathFragment path, String newExtension) {
    return path.replaceName(removeExtension(path.getBaseName()) + newExtension);
  }

  /**
   * Returns a new {@code PathFragment} formed by replacing the extension of the
   * last path segment of {@code path} with {@code newExtension}. Null is
   * returned iff {@code path} has zero segments or it doesn't end with {@code oldExtension}.
   */
  public static PathFragment replaceExtension(PathFragment path, String newExtension,
      String oldExtension) {
    String base = path.getBaseName();
    if (!base.endsWith(oldExtension)) {
      return null;
    }
    String newBase = base.substring(0, base.length() - oldExtension.length()) + newExtension;
    return path.replaceName(newBase);
  }

  /**
   * Returns a new {@code Path} formed by replacing the extension of the
   * last path segment of {@code path} with {@code newExtension}. Null is
   * returned iff {@code path} has zero segments.
   */
  public static Path replaceExtension(Path path, String newExtension) {
    PathFragment fragment = replaceExtension(path.asFragment(), newExtension);
    return fragment == null ? null : path.getFileSystem().getPath(fragment);
  }

  /**
   * Returns a new {@code PathFragment} formed by adding the extension to the last path segment of
   * {@code path}. Null is returned if {@code path} has zero segments.
   */
  public static PathFragment appendExtension(PathFragment path, String newExtension) {
    return path.replaceName(path.getBaseName() + newExtension);
  }

  /**
   * Returns a new {@code PathFragment} formed by replacing the first, or all if
   * {@code replaceAll} is true, {@code oldSegment} of {@code path} with {@code
   * newSegment}.
   */
  public static PathFragment replaceSegments(PathFragment path,
      String oldSegment, String newSegment, boolean replaceAll) {
    int count = path.segmentCount();
    for (int i = 0; i < count; i++) {
      if (path.getSegment(i).equals(oldSegment)) {
        path = new PathFragment(path.subFragment(0, i),
                                new PathFragment(newSegment),
                                path.subFragment(i+1, count));
        if (!replaceAll) {
          return path;
        }
      }
    }
    return path;
  }

  /**
   * Returns a new {@code PathFragment} formed by appending the given string to the last path
   * segment of {@code path} without removing the extension.  Returns null if {@code path}
   * has zero segments.
   */
  public static PathFragment appendWithoutExtension(PathFragment path, String toAppend) {
    return path.replaceName(appendWithoutExtension(path.getBaseName(), toAppend));
  }

  /**
   * Given a string that represents a file with an extension separated by a '.' and a string
   * to append, return a string in which {@code toAppend} has been appended to {@code name}
   * before the last '.' character.  If {@code name} does not include a '.', appends {@code
   * toAppend} at the end.
   *
   * <p>For example,
   * ("libfoo.jar", "-src") ==> "libfoo-src.jar"
   * ("libfoo", "-src") ==> "libfoo-src"
   */
  private static String appendWithoutExtension(String name, String toAppend) {
    int dotIndex = name.lastIndexOf('.');
    if (dotIndex > 0) {
      String baseName = name.substring(0, dotIndex);
      String extension = name.substring(dotIndex);
      return baseName + toAppend + extension;
    } else {
      return name + toAppend;
    }
  }

  /****************************************************************************
   * FileSystem property functions.
   */

  /**
   * Return the current working directory as expressed by the System property
   * 'user.dir'.
   */
  public static Path getWorkingDirectory(FileSystem fs) {
    return fs.getPath(getWorkingDirectory());
  }

  /**
   * Returns the current working directory as expressed by the System property
   * 'user.dir'. This version does not require a {@link FileSystem}.
   */
  public static PathFragment getWorkingDirectory() {
    return new PathFragment(System.getProperty("user.dir", "/"));
  }

  /****************************************************************************
   * Path FileSystem mutating operations.
   */

  /**
   * "Touches" the file or directory specified by the path, following symbolic
   * links. If it does not exist, it is created as an empty file; otherwise, the
   * time of last access is updated to the current time.
   *
   * @throws IOException if there was an error while touching the file
   */
  @ThreadSafe
  public static void touchFile(Path path) throws IOException {
    if (path.exists()) {
      // -1L means "use the current time", and is ultimately implemented by
      // utime(path, null), thereby using the kernel's clock, not the JVM's.
      // (A previous implementation based on the JVM clock was found to be
      // skewy.)
      path.setLastModifiedTime(-1L);
    } else {
      createEmptyFile(path);
    }
  }

  /**
   * Creates an empty regular file with the name of the current path, following
   * symbolic links.
   *
   * @throws IOException if the file could not be created for any reason
   *         (including that there was already a file at that location)
   */
  public static void createEmptyFile(Path path) throws IOException {
    path.getOutputStream().close();
  }

  /**
   * Creates or updates a symbolic link from 'link' to 'target'. Replaces
   * existing symbolic links with target, and skips the link creation if it is
   * already present. Will also create any missing ancestor directories of the
   * link. This method is non-atomic
   *
   * <p>Note: this method will throw an IOException if there is an unequal
   * non-symlink at link.
   *
   * @throws IOException if the creation of the symbolic link was unsuccessful
   *         for any reason.
   */
  @ThreadSafe  // but not atomic
  public static void ensureSymbolicLink(Path link, Path target) throws IOException {
    ensureSymbolicLink(link, target.asFragment());
  }

  /**
   * Creates or updates a symbolic link from 'link' to 'target'. Replaces
   * existing symbolic links with target, and skips the link creation if it is
   * already present. Will also create any missing ancestor directories of the
   * link. This method is non-atomic
   *
   * <p>Note: this method will throw an IOException if there is an unequal
   * non-symlink at link.
   *
   * @throws IOException if the creation of the symbolic link was unsuccessful
   *         for any reason.
   */
  @ThreadSafe  // but not atomic
  public static void ensureSymbolicLink(Path link, String target) throws IOException {
    ensureSymbolicLink(link, new PathFragment(target));
  }

  /**
   * Creates or updates a symbolic link from 'link' to 'target'. Replaces
   * existing symbolic links with target, and skips the link creation if it is
   * already present. Will also create any missing ancestor directories of the
   * link. This method is non-atomic
   *
   * <p>Note: this method will throw an IOException if there is an unequal
   * non-symlink at link.
   *
   * @throws IOException if the creation of the symbolic link was unsuccessful
   *         for any reason.
   */
  @ThreadSafe  // but not atomic
  public static void ensureSymbolicLink(Path link, PathFragment target) throws IOException {
    // TODO(bazel-team): (2009) consider adding the logic for recovering from the case when
    // we have already created a parent directory symlink earlier.
    try {
      if (link.readSymbolicLink().equals(target)) {
        return;  // Do nothing if the link is already there.
      }
    } catch (IOException e) { // link missing or broken
      /* fallthru and do the work below */
    }
    if (link.isSymbolicLink()) {
      link.delete();  // Remove the symlink since it is pointing somewhere else.
    } else {
      createDirectoryAndParents(link.getParentDirectory());
    }
    try {
      link.createSymbolicLink(target);
    } catch (IOException e) {
      // Only pass on exceptions caused by a true link creation failure.
      if (!link.isSymbolicLink() ||
          !link.resolveSymbolicLinks().equals(link.getRelative(target))) {
        throw e;
      }
    }
  }

  public static ByteSource asByteSource(final Path path) {
    return new ByteSource() {
      @Override public InputStream openStream() throws IOException {
        return path.getInputStream();
      }
    };
  }

  public static ByteSink asByteSink(final Path path, final boolean append) {
    return new ByteSink() {
      @Override public OutputStream openStream() throws IOException {
        return path.getOutputStream(append);
      }
    };
  }

  public static ByteSink asByteSink(final Path path) {
    return asByteSink(path, false);
  }

  /**
   * Copies the file from location "from" to location "to", while overwriting a
   * potentially existing "to". File's last modified time, executable and
   * writable bits are also preserved.
   *
   * <p>If no error occurs, the method returns normally. If a parent directory does
   * not exist, a FileNotFoundException is thrown. An IOException is thrown when
   * other erroneous situations occur. (e.g. read errors)
   */
  @ThreadSafe  // but not atomic
  public static void copyFile(Path from, Path to) throws IOException {
    try {
      to.delete();
    } catch (IOException e) {
      throw new IOException("error copying file: "
          + "couldn't delete destination: " + e.getMessage());
    }
    asByteSource(from).copyTo(asByteSink(to));
    to.setLastModifiedTime(from.getLastModifiedTime()); // Preserve mtime.
    if (!from.isWritable()) {
      to.setWritable(false); // Make file read-only if original was read-only.
    }
    to.setExecutable(from.isExecutable()); // Copy executable bit.
  }

  /**
   * Copies a tool binary from one path to another, returning the target path.
   * The directory of the target path must already exist. The target copy's time
   * is set to match, as well as its read-only and executable flags. The
   * operation is skipped if the target file has the same time and size as the
   * source.
   */
  public static Path copyTool(Path source, Path target) throws IOException {
    FileStatus sourceStat = null;
    FileStatus targetStat = target.statNullable();
    if (targetStat != null) {
      // stat the source file only if we'll need the stat.
      sourceStat = source.stat(Symlinks.FOLLOW);
    }
    if (targetStat == null ||
        targetStat.getLastModifiedTime() != sourceStat.getLastModifiedTime() ||
        targetStat.getSize() != sourceStat.getSize()) {
      copyFile(source, target);
      target.setWritable(source.isWritable());
      target.setExecutable(source.isExecutable());
      target.setLastModifiedTime(source.getLastModifiedTime());
    }
    return target;
  }

  /****************************************************************************
   * Directory tree operations.
   */

  /**
   * Returns a new collection containing all of the paths below a given root
   * path, for which the given predicate is true. Symbolic links are not
   * followed, and may appear in the result.
   *
   * @throws IOException If the root does not denote a directory
   */
  @ThreadSafe
  public static Collection<Path> traverseTree(Path root, Predicate<? super Path> predicate)
      throws IOException {
    List<Path> paths = new ArrayList<>();
    traverseTree(paths, root, predicate);
    return paths;
  }

  /**
   * Populates an existing Path List, adding all of the paths below a given root
   * path for which the given predicate is true. Symbolic links are not
   * followed, and may appear in the result.
   *
   * @throws IOException If the root does not denote a directory
   */
  @ThreadSafe
  public static void traverseTree(Collection<Path> paths, Path root,
      Predicate<? super Path> predicate) throws IOException {
    for (Path p : root.getDirectoryEntries()) {
      if (predicate.apply(p)) {
        paths.add(p);
      }
      if (p.isDirectory(Symlinks.NOFOLLOW)) {
        traverseTree(paths, p, predicate);
      }
    }
  }

  /**
   * Deletes 'p', and everything recursively beneath it if it's a directory.
   * Does not follow any symbolic links.
   *
   * @throws IOException if any file could not be removed.
   */
  @ThreadSafe
  public static void deleteTree(Path p) throws IOException {
    deleteTreesBelow(p);
    p.delete();
  }

  /**
   * Deletes all dir trees recursively beneath 'dir' if it's a directory,
   * nothing otherwise. Does not follow any symbolic links.
   *
   * @throws IOException if any file could not be removed.
   */
  @ThreadSafe
  public static void deleteTreesBelow(Path dir) throws IOException {
    if (dir.isDirectory(Symlinks.NOFOLLOW)) {  // real directories (not symlinks)
      dir.setReadable(true);
      dir.setWritable(true);
      dir.setExecutable(true);
      for (Path child : dir.getDirectoryEntries()) {
        deleteTree(child);
      }
    }
  }

  /**
   * Delete all dir trees under a given 'dir' that don't start with one of a set
   * of given 'prefixes'. Does not follow any symbolic links.
   */
  @ThreadSafe
  public static void deleteTreesBelowNotPrefixed(Path dir, String[] prefixes) throws IOException {
    dirloop:
    for (Path p : dir.getDirectoryEntries()) {
      String name = p.getBaseName();
      for (int i = 0; i < prefixes.length; i++) {
        if (name.startsWith(prefixes[i])) {
          continue dirloop;
        }
      }
      deleteTree(p);
    }
  }

  /**
   * Copies all dir trees under a given 'from' dir to location 'to', while overwriting
   * all files in the potentially existing 'to'. Resolves symbolic links.
   *
   * <p>The source and the destination must be non-overlapping, otherwise an
   * IllegalArgumentException will be thrown. This method cannot be used to copy
   * a dir tree to a sub tree of itself.
   *
   * <p>If no error occurs, the method returns normally. If the given 'from' does
   * not exist, a FileNotFoundException is thrown. An IOException is thrown when
   * other erroneous situations occur. (e.g. read errors)
   */
  @ThreadSafe
  public static void copyTreesBelow(Path from , Path to) throws IOException {
    if (to.startsWith(from)) {
      throw new IllegalArgumentException(to + " is a subdirectory of " + from);
    }

    Collection<Path> entries = from.getDirectoryEntries();
    for (Path entry : entries) {
      if (entry.isFile()) {
        Path newEntry = to.getChild(entry.getBaseName());
        copyFile(entry, newEntry);
      } else {
        Path subDir = to.getChild(entry.getBaseName());
        subDir.createDirectory();
        copyTreesBelow(entry, subDir);
      }
    }
  }

  /**
   * Attempts to create a directory with the name of the given path, creating
   * ancestors as necessary.
   *
   * <p>Postcondition: completes normally iff {@code dir} denotes an existing
   * directory (not necessarily canonical); completes abruptly otherwise.
   *
   * @return true if the directory was successfully created anew, false if it
   *   already existed (including the case where {@code dir} denotes a symlink
   *   to an existing directory)
   * @throws IOException if the directory could not be created
   */
  @ThreadSafe
  public static boolean createDirectoryAndParents(Path dir) throws IOException {
    // Optimised for minimal number of I/O calls.

    // Don't attempt to create the root directory.
    if (dir.getParentDirectory() == null) { return false; }

    FileSystem filesystem = dir.getFileSystem();
    if (filesystem instanceof UnionFileSystem) {
      // If using UnionFS, make sure that we do not traverse filesystem boundaries when creating
      // parent directories by rehoming the path on the most specific filesystem.
      FileSystem delegate = ((UnionFileSystem) filesystem).getDelegate(dir);
      dir = delegate.getPath(dir.asFragment());
    }

    try {
      return dir.createDirectory();
    } catch (IOException e) {
      if (e.getMessage().endsWith(" (No such file or directory)")) { // ENOENT
        createDirectoryAndParents(dir.getParentDirectory());
        return dir.createDirectory();
      } else if (e.getMessage().endsWith(" (File exists)") && dir.isDirectory()) { // EEXIST
        return false;
      } else {
        throw e; // some other error (e.g. ENOTDIR, EACCES, etc.)
      }
    }
  }

  /**
   * Attempts to remove a relative chain of directories under a given base.
   * Returns {@code true} if the removal was successful, and returns {@code
   * false} if the removal fails because a directory was not empty. An
   * {@link IOException} is thrown for any other errors.
   */
  @ThreadSafe
  public static boolean removeDirectoryAndParents(Path base, PathFragment toRemove) {
    if (toRemove.isAbsolute()) {
      return false;
    }
    try {
      for (; toRemove.segmentCount() > 0; toRemove = toRemove.getParentDirectory()) {
        Path p = base.getRelative(toRemove);
        if (p.exists()) {
          p.delete();
        }
      }
    } catch (IOException e) {
      return false;
    }
    return true;
  }

  /**
   * Takes a map of directory fragments to root paths, and creates a symlink
   * forest under an existing linkRoot to the corresponding source dirs or
   * files. Symlink are made at the highest dir possible, linking files directly
   * only when needed with nested packages.
   */
  public static void plantLinkForest(ImmutableMap<PathFragment, Path> packageRootMap, Path linkRoot)
      throws IOException {
    Path emptyPackagePath = null;

    // Create a sorted map of all dirs (packages and their ancestors) to sets of their roots.
    // Packages come from exactly one root, but their shared ancestors may come from more.
    // The map is maintained sorted lexicographically, so parents are before their children.
    Map<PathFragment, Set<Path>> dirRootsMap = Maps.newTreeMap();
    for (Map.Entry<PathFragment, Path> entry : packageRootMap.entrySet()) {
      PathFragment pkgDir = entry.getKey();
      Path pkgRoot = entry.getValue();
      if (pkgDir.segmentCount() == 0) {
        emptyPackagePath = entry.getValue();
      }
      for (int i = 1; i <= pkgDir.segmentCount(); i++) {
        PathFragment dir = pkgDir.subFragment(0, i);
        Set<Path> roots = dirRootsMap.get(dir);
        if (roots == null) {
          roots = Sets.newHashSet();
          dirRootsMap.put(dir, roots);
        }
        roots.add(pkgRoot);
      }
    }
    // Now add in roots for all non-pkg dirs that are in between two packages, and missed above.
    for (Map.Entry<PathFragment, Set<Path>> entry : dirRootsMap.entrySet()) {
      PathFragment dir = entry.getKey();
      if (!packageRootMap.containsKey(dir)) {
        PathFragment pkgDir = longestPathPrefix(dir, packageRootMap.keySet());
        if (pkgDir != null) {
          entry.getValue().add(packageRootMap.get(pkgDir));
        }
      }
    }
    // Create output dirs for all dirs that have more than one root and need to be split.
    for (Map.Entry<PathFragment, Set<Path>> entry : dirRootsMap.entrySet()) {
      PathFragment dir = entry.getKey();
      if (entry.getValue().size() > 1) {
        if (LOG_FINER) {
          LOG.finer("mkdir " + linkRoot.getRelative(dir));
        }
        createDirectoryAndParents(linkRoot.getRelative(dir));
      }
    }
    // Make dir links for single rooted dirs.
    for (Map.Entry<PathFragment, Set<Path>> entry : dirRootsMap.entrySet()) {
      PathFragment dir = entry.getKey();
      Set<Path> roots = entry.getValue();
      // Simple case of one root for this dir.
      if (roots.size() == 1) {
        if (dir.segmentCount() > 1 && dirRootsMap.get(dir.getParentDirectory()).size() == 1) {
          continue;  // skip--an ancestor will link this one in from above
        }
        // This is the top-most dir that can be linked to a single root. Make it so.
        Path root = roots.iterator().next();  // lone root in set
        if (LOG_FINER) {
          LOG.finer("ln -s " + root.getRelative(dir) + " " + linkRoot.getRelative(dir));
        }
        linkRoot.getRelative(dir).createSymbolicLink(root.getRelative(dir));
      }
    }
    // Make links for dirs within packages, skip parent-only dirs.
    for (Map.Entry<PathFragment, Set<Path>> entry : dirRootsMap.entrySet()) {
      PathFragment dir = entry.getKey();
      if (entry.getValue().size() > 1) {
        // If this dir is at or below a package dir, link in its contents.
        PathFragment pkgDir = longestPathPrefix(dir, packageRootMap.keySet());
        if (pkgDir != null) {
          Path root = packageRootMap.get(pkgDir);
          try {
            Path absdir = root.getRelative(dir);
            if (absdir.isDirectory()) {
              if (LOG_FINER) {
                LOG.finer("ln -s " + absdir + "/* " + linkRoot.getRelative(dir) + "/");
              }
              for (Path target : absdir.getDirectoryEntries()) {
                PathFragment p = target.relativeTo(root);
                if (!dirRootsMap.containsKey(p)) {
                  //LOG.finest("ln -s " + target + " " + linkRoot.getRelative(p));
                  linkRoot.getRelative(p).createSymbolicLink(target);
                }
              }
            } else {
              LOG.fine("Symlink planting skipping dir '" + absdir + "'");
            }
          } catch (IOException e) {
            e.printStackTrace();
          }
          // Otherwise its just an otherwise empty common parent dir.
        }
      }
    }

    if (emptyPackagePath != null) {
      // For the top-level directory, generate symlinks to everything in the directory instead of
      // the directory itself.
      for (Path target : emptyPackagePath.getDirectoryEntries()) {
        String baseName = target.getBaseName();
        // Create any links that don't exist yet and don't start with bazel-.
        if (!baseName.startsWith(Constants.PRODUCT_NAME + "-")
            && !linkRoot.getRelative(baseName).exists()) {
          linkRoot.getRelative(baseName).createSymbolicLink(target);
        }
      }
    }
  }

  /****************************************************************************
   * Whole-file I/O utilities for characters and bytes. These convenience
   * methods are not efficient and should not be used for large amounts of data!
   */

  /**
   * Decodes the given byte array assumed to be encoded with ISO-8859-1 encoding (isolatin1).
   */
  public static char[] convertFromLatin1(byte[] content) {
    char[] latin1 = new char[content.length];
    for (int i = 0; i < latin1.length; i++) { // yeah, latin1 is this easy! :-)
      latin1[i] = (char) (0xff & content[i]);
    }
    return latin1;
  }

  /**
   * Writes lines to file using ISO-8859-1 encoding (isolatin1).
   */
  @ThreadSafe // but not atomic
  public static void writeIsoLatin1(Path file, String... lines) throws IOException {
    writeLinesAs(file, ISO_8859_1, lines);
  }

  /**
   * Append lines to file using ISO-8859-1 encoding (isolatin1).
   */
  @ThreadSafe // but not atomic
  public static void appendIsoLatin1(Path file, String... lines) throws IOException {
    appendLinesAs(file, ISO_8859_1, lines);
  }

  /**
   * Writes the specified String as ISO-8859-1 (latin1) encoded bytes to the
   * file. Follows symbolic links.
   *
   * @throws IOException if there was an error
   */
  public static void writeContentAsLatin1(Path outputFile, String content) throws IOException {
    writeContent(outputFile, ISO_8859_1, content);
  }

  /**
   * Writes the specified String using the specified encoding to the file.
   * Follows symbolic links.
   *
   * @throws IOException if there was an error
   */
  public static void writeContent(Path outputFile, Charset charset, String content)
      throws IOException {
    asByteSink(outputFile).asCharSink(charset).write(content);
  }

  /**
   * Writes lines to file using the given encoding, ending every line with a
   * line break '\n' character.
   */
  @ThreadSafe // but not atomic
  public static void writeLinesAs(Path file, Charset charset, String... lines)
      throws IOException {
    writeLinesAs(file, charset, Arrays.asList(lines));
  }

  /**
   * Appends lines to file using the given encoding, ending every line with a
   * line break '\n' character.
   */
  @ThreadSafe // but not atomic
  public static void appendLinesAs(Path file, Charset charset, String... lines)
      throws IOException {
    appendLinesAs(file, charset, Arrays.asList(lines));
  }

  /**
   * Writes lines to file using the given encoding, ending every line with a
   * line break '\n' character.
   */
  @ThreadSafe // but not atomic
  public static void writeLinesAs(Path file, Charset charset, Iterable<String> lines)
      throws IOException {
    createDirectoryAndParents(file.getParentDirectory());
    asByteSink(file).asCharSink(charset).writeLines(lines);
  }

  /**
   * Appends lines to file using the given encoding, ending every line with a
   * line break '\n' character.
   */
  @ThreadSafe // but not atomic
  public static void appendLinesAs(Path file, Charset charset, Iterable<String> lines)
      throws IOException {
    createDirectoryAndParents(file.getParentDirectory());
    asByteSink(file, true).asCharSink(charset).writeLines(lines);
  }

  /**
   * Writes the specified byte array to the output file. Follows symbolic links.
   *
   * @throws IOException if there was an error
   */
  public static void writeContent(Path outputFile, byte[] content) throws IOException {
    asByteSink(outputFile).write(content);
  }

  /**
   * Returns the entirety of the specified input stream and returns it as a char
   * array, decoding characters using ISO-8859-1 (Latin1).
   *
   * @throws IOException if there was an error
   */
  public static char[] readContentAsLatin1(InputStream in) throws IOException {
    return convertFromLatin1(ByteStreams.toByteArray(in));
  }

  /**
   * Returns the entirety of the specified file and returns it as a char array,
   * decoding characters using ISO-8859-1 (Latin1).
   *
   * @throws IOException if there was an error
   */
  public static char[] readContentAsLatin1(Path inputFile) throws IOException {
    return convertFromLatin1(readContent(inputFile));
  }

  /**
   * Returns an iterable that allows iterating over ISO-8859-1 (Latin1) text
   * file contents line by line. If the file ends in a line break, the iterator
   * will return an empty string as the last element.
   *
   * @throws IOException if there was an error
   */
  public static Iterable<String> iterateLinesAsLatin1(Path inputFile) throws IOException {
    return readLines(inputFile, ISO_8859_1);
  }

  /**
   * Returns an iterable that allows iterating over text file contents line by line in the given
   * {@link Charset}. If the file ends in a line break, the iterator will return an empty string
   * as the last element.
   *
   * @throws IOException if there was an error
   */
  public static Iterable<String> readLines(Path inputFile, Charset charset) throws IOException {
    return asByteSource(inputFile).asCharSource(charset).readLines();
  }
  
  /**
   * Returns the entirety of the specified file and returns it as a byte array.
   *
   * @throws IOException if there was an error
   */
  public static byte[] readContent(Path inputFile) throws IOException {
    return asByteSource(inputFile).read();
  }

  /**
   * Reads the entire file using the given charset and returns the contents as a string
   */
  public static String readContent(Path inputFile, Charset charset) throws IOException {
    return asByteSource(inputFile).asCharSource(charset).read();
  }

  /**
   * Reads at most {@code limit} bytes from {@code inputFile} and returns it as a byte array.
   *
   * @throws IOException if there was an error.
   */
  public static byte[] readContentWithLimit(Path inputFile, int limit) throws IOException {
    Preconditions.checkArgument(limit >= 0, "limit needs to be >=0, but it is %s", limit);
    ByteSource byteSource = asByteSource(inputFile);
    byte[] buffer = new byte[limit];
    try (InputStream inputStream = byteSource.openBufferedStream()) {
      int read = ByteStreams.read(inputStream, buffer, 0, limit);
      return read == limit ? buffer : Arrays.copyOf(buffer, read);
    }
  }

  /**
   * Reads the given file {@code path}, assumed to have size {@code fileSize}, and does a sanity
   * check on the number of bytes read.
   *
   * <p>Use this method when you already know the size of the file. The sanity check is intended to
   * catch issues where filesystems incorrectly truncate files.
   *
   * @throws IOException if there was an error, or if fewer than {@code fileSize} bytes were read.
   */
  public static byte[] readWithKnownFileSize(Path path, long fileSize) throws IOException {
    if (fileSize > Integer.MAX_VALUE) {
      throw new IOException("Cannot read file with size larger than 2GB");
    }
    int fileSizeInt = (int) fileSize;
    byte[] bytes = readContentWithLimit(path, fileSizeInt);
    if (fileSizeInt > bytes.length) {
      throw new IOException("Unexpected short read from file '" + path
          + "' (expected " + fileSizeInt + ", got " + bytes.length + " bytes)");
    }
    return bytes;
  }

  /**
   * Dumps diagnostic information about the specified filesystem to {@code out}.
   * This is the implementation of the filesystem part of the 'blaze dump'
   * command. It lives here, rather than in DumpCommand, because it requires
   * privileged access to members of this package.
   *
   * <p>Its results are unspecified and MUST NOT be interpreted programmatically.
   */
  public static void dump(FileSystem fs, final PrintStream out) {
    if (!(fs instanceof UnixFileSystem)) {
      out.println("  Not a UnixFileSystem.");
      return;
    }

    // Unfortunately there's no "letrec" for anonymous functions so we have to
    // (a) name the function, (b) put it in a box and (c) use List not array
    // because of the generic type.  *sigh*.
    final List<Predicate<Path>> dumpFunction = new ArrayList<>();
    dumpFunction.add(new Predicate<Path>() {
        @Override
        public boolean apply(Path child) {
          Path path = child;
          out.println("  " + path + " (" + path.toDebugString() + ")");
          path.applyToChildren(dumpFunction.get(0));
          return false;
        }
      });

    fs.getRootDirectory().applyToChildren(dumpFunction.get(0));
  }

  /**
   * Returns the type of the file system path belongs to.
   */
  public static String getFileSystem(Path path) {
    return path.getFileSystem().getFileSystemType(path);
  }

  /**
   * Returns whether the given path starts with any of the paths in the given
   * list of prefixes.
   */
  public static boolean startsWithAny(Path path, Iterable<Path> prefixes) {
    for (Path prefix : prefixes) {
      if (path.startsWith(prefix)) {
        return true;
      }
    }
    return false;
  }

  /**
   * Returns whether the given path starts with any of the paths in the given
   * list of prefixes.
   */
  public static boolean startsWithAny(PathFragment path, Iterable<PathFragment> prefixes) {
    for (PathFragment prefix : prefixes) {
      if (path.startsWith(prefix)) {
        return true;
      }
    }
    return false;
  }
}
