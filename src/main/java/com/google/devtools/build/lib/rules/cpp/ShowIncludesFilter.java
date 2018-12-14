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

package com.google.devtools.build.lib.rules.cpp;

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.vfs.Path;
import java.io.ByteArrayOutputStream;
import java.io.FilterOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;

/**
 * A Class for filtering the output of /showIncludes from MSVC compiler.
 *
 * <p>A discovered header file will be printed with prefix "Note: including file:", the path is
 * collected, and the line is suppressed from the actual output users can see.
 *
 * <p>Also suppress the basename of source file, which is printed unconditionally by MSVC compiler,
 * there is no way to turn it off.
 */
public class ShowIncludesFilter {

  private FilterShowIncludesOutputStream filterShowIncludesOutputStream;
  private final String sourceFileName;
  private final String workspaceName;

  public ShowIncludesFilter(String sourceFileName, String workspaceName) {
    this.sourceFileName = sourceFileName;
    this.workspaceName = workspaceName;
  }

  /**
   * Use this class to filter and collect the headers discovered by MSVC compiler, also filter out
   * the source file name printed unconditionally by the compiler.
   */
  public static class FilterShowIncludesOutputStream extends FilterOutputStream {

    private final ByteArrayOutputStream buffer = new ByteArrayOutputStream(4096);
    private final Collection<String> dependencies = new ArrayList<>();
    private static final int NEWLINE = '\n';
    private static final String SHOW_INCLUDES_PREFIX = "Note: including file:";
    private final String sourceFileName;
    private final String execRootSuffix;

    public FilterShowIncludesOutputStream(OutputStream out, String sourceFileName, String workspaceName) {
      super(out);
      this.sourceFileName = sourceFileName;
      this.execRootSuffix = "execroot\\" + workspaceName + "\\";
    }

    @Override
    public void write(int b) throws IOException {
      buffer.write(b);
      if (b == NEWLINE) {
        String line = buffer.toString(StandardCharsets.UTF_8.name());
        if (line.startsWith(SHOW_INCLUDES_PREFIX)) {
          line = line.substring(SHOW_INCLUDES_PREFIX.length()).trim();
          int index = line.indexOf(execRootSuffix);
          if (index != -1) {
            line = line.substring(index + execRootSuffix.length());
          }
          dependencies.add(line);
        } else if (!line.trim().equals(sourceFileName)) {
          buffer.writeTo(out);
        }
        buffer.reset();
      }
    }

    @Override
    public void flush() throws IOException {
      String line = buffer.toString(StandardCharsets.UTF_8.name());
      if (!line.startsWith(SHOW_INCLUDES_PREFIX)
          && !line.startsWith(sourceFileName)
          && !SHOW_INCLUDES_PREFIX.startsWith(line)
          && !sourceFileName.startsWith(line)) {
        buffer.writeTo(out);
        buffer.reset();
      }
      out.flush();
    }

    public Collection<String> getDependencies() {
      return this.dependencies;
    }
  }

  public FilterOutputStream getFilteredOutputStream(OutputStream outputStream) {
    filterShowIncludesOutputStream =
        new FilterShowIncludesOutputStream(outputStream, sourceFileName, workspaceName);
    return filterShowIncludesOutputStream;
  }

  public Collection<Path> getDependencies(Path root) {
    Collection<Path> dependenciesInPath = new ArrayList<>();
    if (filterShowIncludesOutputStream != null) {
      for (String dep : filterShowIncludesOutputStream.getDependencies()) {
        dependenciesInPath.add(root.getRelative(dep));
      }
    }
    return Collections.unmodifiableCollection(dependenciesInPath);
  }

  @VisibleForTesting
  Collection<String> getDependencies() {
    return filterShowIncludesOutputStream.getDependencies();
  }
}
