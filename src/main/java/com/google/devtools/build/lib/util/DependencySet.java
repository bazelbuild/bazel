// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.util;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Representation of a set of file dependencies for a given output file. There
 * are generally one input dependency and a bunch of include dependencies. The
 * files are stored as {@code PathFragment}s and may be relative or absolute.
 * <p>
 * The serialized format read and written is equivalent and compatible with the
 * ".d" file produced by the -MM for a given out (.o) file.
 * <p>
 * The file format looks like:
 *
 * <pre>
 * {outfile}:  \
 *  {infile} \
 *   {include} \
 *   ... \
 *   {include}
 * </pre>
 *
 * @see "http://gcc.gnu.org/onlinedocs/gcc-4.2.1/gcc/Preprocessor-Options.html#Preprocessor-Options"
 */
public final class DependencySet {

  /**
   * The set of dependent files that this DependencySet embodies. May be
   * relative or absolute PathFragments.  A tree set is used to ensure that we
   * write them out in a consistent order.
   */
  private final Collection<PathFragment> dependencies = new ArrayList<>();

  private final Path root;

  /**
   * Constructs a new empty DependencySet instance.
   */
  public DependencySet(Path root) {
    this.root = root;
  }

  /**
   * Gets an unmodifiable view of the set of dependencies in PathFragment form
   * from this DependencySet instance.
   */
  public Collection<PathFragment> getDependencies() {
    return Collections.unmodifiableCollection(dependencies);
  }

  /**
   * Adds a given collection of dependencies in Path form to this DependencySet
   * instance. Paths are converted to root-relative
   */
  public void addDependencies(Collection<Path> deps) {
    for (Path d : deps) {
      addDependency(d.relativeTo(root));
    }
  }

  /**
   * Adds a given dependency in PathFragment form to this DependencySet
   * instance.
   */
  public void addDependency(PathFragment dep) {
    dependencies.add(Preconditions.checkNotNull(dep));
  }

  /**
   * Returns the set of includes that are in this DependencySet's dependencies,
   * but not in another dependency set. Effectively returns the difference
   * between the sets: (this - other) => result
   *
   * @param otherDeps the other dependency set to compare this DependencySet's
   *        set against
   */
  public Set<PathFragment> getMissingDependencies(Set<Path> otherDeps) {
    Set<PathFragment> missing = new HashSet<>();
    for (PathFragment dep : dependencies) {
      // Canonicalize the path, resolve relative paths to absolute paths,
      // and expand symlinks and eliminate occurrences of "." and "..".
      Path d = root.getRelative(dep);
      if (!otherDeps.contains(d)) {
        missing.add(dep);
      }
    }
    return missing;
  }

  // regex has 3 groups: token, optional colon, optional spaces
  private static final Pattern DOTD_PATTERN =
      Pattern.compile("([^\\s:\\\\]+)(?:\\s*+)(:?)(\\s*(?:\\\\\\n)*\\s*)");

  /**
   * Reads a dotd file into this DependencySet instance.
   * <pre>
   * - read whole file into memory
   * - skip LHS of each stanza.
   * - assume sequences of ' ', '\\', '\n' are separators
   * - assume no Make-isms.
   * </pre>
   */
  public DependencySet read(Path dotdFile) throws IOException {
    return process(new String(FileSystemUtils.readContentAsLatin1(dotdFile)));
  }

  /**
   * Like read(), but accepts the contents of the dotd file.
   */
  public DependencySet process(String contents) {
    Matcher m = DOTD_PATTERN.matcher(contents);
    while (m.find()) {
      String dependency = m.group(1);
      String colon = m.group(2);
      if (colon.length() == 0) {
        dependencies.add(new PathFragment(dependency).normalize());
      }
    }
    return this;
  }

  /**
   * Writes this DependencySet object for a specified output file under the root
   * dir, and with a given suffix.
   */
  public void write(Path outFile, String suffix) throws IOException {
    Path dotdFile =
        outFile.getRelative(FileSystemUtils.replaceExtension(outFile.asFragment(), suffix));
    PrintStream out = new PrintStream(dotdFile.getOutputStream());
    try {
      out.print(outFile.relativeTo(root) + ": ");
      for (PathFragment d : dependencies) {
        out.print(" \\\n  " + d.getPathString());  // should already be root relative
      }
      out.println();
    } finally {
      out.close();
    }
  }

  @Override
  public boolean equals(Object other) {
    return other instanceof DependencySet
        && ((DependencySet) other).dependencies.equals(dependencies);
  }

  @Override
  public int hashCode() {
    return dependencies.hashCode();
  }
}
