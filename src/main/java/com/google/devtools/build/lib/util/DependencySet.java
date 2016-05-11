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

package com.google.devtools.build.lib.util;

import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;

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
   * The set of dependent files that this DependencySet embodies. They are all
   * Path with the same FileSystem  A tree set is used to ensure that we
   * write them out in a consistent order.
   */
  private final Collection<Path> dependencies = new ArrayList<>();

  private final Path root;
  private String outputFileName;

  /**
   * Get output file name for which dependencies are included in this DependencySet.
   */
  public String getOutputFileName() {
    return outputFileName;
  }

  public void setOutputFileName(String outputFileName) {
    this.outputFileName = outputFileName;
  }

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
  public Collection<Path> getDependencies() {
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
  private void addDependency(PathFragment dep) {
    Path depPath = root.getRelative(Preconditions.checkNotNull(dep));
    dependencies.add(depPath);
  }

  /**
   * Reads a dotd file into this DependencySet instance.
   */
  public DependencySet read(Path dotdFile) throws IOException {
    byte[] content = FileSystemUtils.readContent(dotdFile);
    try {
      return process(content);
    } catch (IOException e) {
      throw new IOException("Error processing " + dotdFile + ": " + e.getMessage());
    }
  }

  /**
   * Parses a .d file.
   *
   * <p>Performance-critical! In large C++ builds there are lots of .d files to read, and some of
   * them reach into hundreds of kilobytes.
   */
  public DependencySet process(byte[] content) throws IOException {
    if (content.length > 0 && content[content.length - 1] != '\n') {
      throw new IOException("File does not end in a newline");
    }
    int w = 0;
    for (int r = 0; r < content.length; ++r) {
      final byte c = content[r];
      switch (c) {
        case ' ':
        case '\n':
        case '\r':
          if (w > 0) {
            String s = new String(content, 0, w, StandardCharsets.UTF_8);
            addDependency(new PathFragment(s).normalize());
            w = 0;
          }
          break;
        case ':':
          // Normally this indicates the output file, but it might be part of a filename on Windows.
          // Peek ahead at the next character.  This is in bounds because we checked for a
          // terminating newline above.
          switch (content[r + 1]) {
            case ' ':
            case '\n':
            case '\r':
              if (w > 0) {
                outputFileName = new String(content, 0, w, StandardCharsets.UTF_8);
                w = 0;
              }
              continue;
            default:
              content[w++] = c;  // copy to filename
              continue;
          }
        case '\\':
          // Peek ahead at the next character.  This is in bounds because we checked for a
          // terminating newline above.
          switch (content[r + 1]) {
            // Backslashes are taken literally except when followed by whitespace.
            // See the Windows tests for some of the nonsense we have to tolerate.
            case ' ':
              content[w++] = ' ';  // copy a space into the filename
              ++r;  // skip the space in the input
              continue;
            case '\n':
            case '\r':
              // Let the newline act as a terminator.  Technically we could have an escaped newline
              // with no adjacent space, but compilers don't seem to generate that.
              continue;
            default:
              content[w++] = c;  // copy to filename
              continue;
          }
        default:
          content[w++] = c;
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
      for (Path d : dependencies) {
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
