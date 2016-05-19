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

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;

import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Representation of a set of file dependencies for a given output file. There
 * are generally one input dependency and a bunch of include dependencies. The
 * files are stored as {@code Path}s and may be relative or absolute.
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

  private static final Pattern DOTD_MERGED_LINE_SEPARATOR = Pattern.compile("\\\\[\n\r]+");
  private static final Pattern DOTD_LINE_SEPARATOR = Pattern.compile("[\n\r]+");
  private static final Pattern DOTD_DEP = Pattern.compile("(?:[^\\s\\\\]++|\\\\ |\\\\)+");

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
   * Gets an unmodifiable view of the set of dependencies in {@link Path} form
   * from this DependencySet instance.
   */
  public Collection<Path> getDependencies() {
    return Collections.unmodifiableCollection(dependencies);
  }

  /**
   * Adds a given collection of dependencies in Path form to this DependencySet
   * instance. Paths are converted to root-relative
   */
  @VisibleForTesting // only called from DependencySetTest
  public void addDependencies(Collection<Path> deps) {
    for (Path d : deps) {
      Preconditions.checkArgument(d.startsWith(root));
      dependencies.add(d);
    }
  }

  /**
   * Adds a given dependency to this DependencySet instance.
   */
  private void addDependency(String dep) {
    Path depPath = root.getRelative(dep);
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
    // true if there is a CR in the input.
    boolean cr = content.length > 0 && content[0] == '\r';
    // true if there is more than one line in the input, not counting \-wrapped lines.
    boolean multiline = false;

    byte prevByte = ' ';
    for (int i = 1; i < content.length; i++) {
      byte b = content[i];
      if (cr || b == '\r') {
        // CR found, abort since our little loop here does not deal with CR/LFs.
        cr = true;
        break;
      }
      if (b == '\n') {
        // Merge lines wrapped using backslashes.
        if (prevByte == '\\') {
          content[i] = ' ';
          content[i - 1] = ' ';
        } else {
          multiline = true;
        }
      }
      prevByte = b;
    }

    if (!cr && content.length > 0 && content[content.length - 1] == '\n') {
      content[content.length - 1] = ' ';
    }

    String s = new String(content, StandardCharsets.UTF_8);
    if (cr) {
      s = DOTD_MERGED_LINE_SEPARATOR.matcher(s).replaceAll(" ").trim();
      multiline = true;
    }
    return process(s, multiline);
  }

  private DependencySet process(String contents, boolean multiline) {
    String[] lines;
    if (!multiline) {
      // Microoptimization: skip the usually unnecessary expensive-ish splitting step if there is
      // only one target. This saves about 20% of CPU time.
      lines = new String[] { contents };
    } else {
      lines = DOTD_LINE_SEPARATOR.split(contents);
    }

    for (String line : lines) {
      // Split off output file name.
      int pos = line.indexOf(':');
      if (pos == -1) {
        continue;
      }
      outputFileName = line.substring(0, pos);
      
      String deps = line.substring(pos + 1);

      Matcher m = DOTD_DEP.matcher(deps);
      while (m.find()) {
        String token = m.group();
        // Process escaped spaces.
        if (token.contains("\\ ")) {
          token = token.replace("\\ ", " ");
        }
        addDependency(token);
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
