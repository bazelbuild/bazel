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
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.vfs.Path;
import java.io.ByteArrayOutputStream;
import java.io.FilterOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

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
    // "Note: including file:" in 14 languages,
    // cl.exe will print different prefix according to the locale configured for MSVC.
    private static final List<String> SHOW_INCLUDES_PREFIXES = ImmutableList.of(
        HexToUTF8String(new int[] {0x4e, 0x6f, 0x74, 0x65, 0x3a, 0x20, 0x69, 0x6e, 0x63, 0x6c, 0x75, 0x64, 0x69, 0x6e, 0x67, 0x20, 0x66, 0x69, 0x6c, 0x65, 0x3a}), // English
        HexToUTF8String(new int[] {0xe6, 0xb3, 0xa8, 0xe6, 0x84, 0x8f, 0x3a, 0x20, 0xe5, 0x8c, 0x85, 0xe5, 0x90, 0xab, 0xe6, 0xaa, 0x94, 0xe6, 0xa1, 0x88, 0x3a}), // Traditional Chinese
        HexToUTF8String(new int[] {0x50, 0x6f, 0x7a, 0x6e, 0xc3, 0xa1, 0x6d, 0x6b, 0x61, 0x3a, 0x20, 0x56, 0xc4, 0x8d, 0x65, 0x74, 0x6e, 0xc4, 0x9b, 0x20, 0x73, 0x6f, 0x75, 0x62, 0x6f, 0x72, 0x75, 0x3a}), // Czech
        HexToUTF8String(new int[] {0x48, 0x69, 0x6e, 0x77, 0x65, 0x69, 0x73, 0x3a, 0x20, 0x45, 0x69, 0x6e, 0x6c, 0x65, 0x73, 0x65, 0x6e, 0x20, 0x64, 0x65, 0x72, 0x20, 0x44, 0x61, 0x74, 0x65, 0x69, 0x3a}), // German
        HexToUTF8String(new int[] {0x52, 0x65, 0x6d, 0x61, 0x72, 0x71, 0x75, 0x65, 0xc2, 0xa0, 0x3a, 0x20, 0x69, 0x6e, 0x63, 0x6c, 0x75, 0x73, 0x69, 0x6f, 0x6e, 0x20, 0x64, 0x75, 0x20, 0x66, 0x69, 0x63, 0x68, 0x69, 0x65, 0x72, 0xc2, 0xa0, 0x3a}), // French
        HexToUTF8String(new int[] {0x4e, 0x6f, 0x74, 0x61, 0x3a, 0x20, 0x66, 0x69, 0x6c, 0x65, 0x20, 0x69, 0x6e, 0x63, 0x6c, 0x75, 0x73, 0x6f}), // Italian
        HexToUTF8String(new int[] {0xe3, 0x83, 0xa1, 0xe3, 0x83, 0xa2, 0x3a, 0x20, 0xe3, 0x82, 0xa4, 0xe3, 0x83, 0xb3, 0xe3, 0x82, 0xaf, 0xe3, 0x83, 0xab, 0xe3, 0x83, 0xbc, 0xe3, 0x83, 0x89, 0x20, 0xe3, 0x83, 0x95, 0xe3, 0x82, 0xa1, 0xe3, 0x82, 0xa4, 0xe3, 0x83, 0xab, 0x3a}), // Janpanese
        HexToUTF8String(new int[] {0xec, 0xb0, 0xb8, 0xea, 0xb3, 0xa0, 0x3a, 0x20, 0xed, 0x8f, 0xac, 0xed, 0x95, 0xa8, 0x20, 0xed, 0x8c, 0x8c, 0xec, 0x9d, 0xbc, 0x3a}), // Korean
        HexToUTF8String(new int[] {0x55, 0x77, 0x61, 0x67, 0x61, 0x3a, 0x20, 0x77, 0x20, 0x74, 0x79, 0x6d, 0x20, 0x70, 0x6c, 0x69, 0x6b, 0x75, 0x3a}), // Polish
        HexToUTF8String(new int[] {0x4f, 0x62, 0x73, 0x65, 0x72, 0x76, 0x61, 0xc3, 0xa7, 0xc3, 0xa3, 0x6f, 0x3a, 0x20, 0x69, 0x6e, 0x63, 0x6c, 0x75, 0x69, 0x6e, 0x64, 0x6f, 0x20, 0x61, 0x72, 0x71, 0x75, 0x69, 0x76, 0x6f, 0x3a}), // Portuguese
        HexToUTF8String(new int[] {0xd0, 0x9f, 0xd1, 0x80, 0xd0, 0xb8, 0xd0, 0xbc, 0xd0, 0xb5, 0xd1, 0x87, 0xd0, 0xb0, 0xd0, 0xbd, 0xd0, 0xb8, 0xd0, 0xb5, 0x3a, 0x20, 0xd0, 0xb2, 0xd0, 0xba, 0xd0, 0xbb, 0xd1, 0x8e, 0xd1, 0x87, 0xd0, 0xb5, 0xd0, 0xbd, 0xd0, 0xb8, 0xd0, 0xb5, 0x20, 0xd1, 0x84, 0xd0, 0xb0, 0xd0, 0xb9, 0xd0, 0xbb, 0xd0, 0xb0, 0x3a}), // Russian
        HexToUTF8String(new int[] {0x4e, 0x6f, 0x74, 0x3a, 0x20, 0x65, 0x6b, 0x6c, 0x65, 0x6e, 0x65, 0x6e, 0x20, 0x64, 0x6f, 0x73, 0x79, 0x61, 0x3a}), // Turkish
        HexToUTF8String(new int[] {0xe6, 0xb3, 0xa8, 0xe6, 0x84, 0x8f, 0x3a, 0x20, 0xe5, 0x8c, 0x85, 0xe5, 0x90, 0xab, 0xe6, 0x96, 0x87, 0xe4, 0xbb, 0xb6, 0x3a}), // Simplified Chinese
        HexToUTF8String(new int[] {0x4e, 0x6f, 0x74, 0x61, 0x3a, 0x20, 0x69, 0x6e, 0x63, 0x6c, 0x75, 0x73, 0x69, 0xc3, 0xb3, 0x6e, 0x20, 0x64, 0x65, 0x6c, 0x20, 0x61, 0x72, 0x63, 0x68, 0x69, 0x76, 0x6f, 0x3a}) // Spanish
    );
    private final String sourceFileName;
    private final String execRootSuffix;

    public static String HexToUTF8String(int[] hex) {
      byte[] data = new byte[hex.length];
      for (int i = 0; i < data.length; i++) {
        data[i] = (byte) hex[i];
      }
      return new String(data, StandardCharsets.UTF_8);
    }

    public FilterShowIncludesOutputStream(
        OutputStream out, String sourceFileName, String workspaceName) {
      super(out);
      this.sourceFileName = sourceFileName;
      this.execRootSuffix = "execroot\\" + workspaceName + "\\";
    }

    @Override
    public void write(int b) throws IOException {
      buffer.write(b);
      if (b == NEWLINE) {
        String line = buffer.toString(StandardCharsets.UTF_8.name());
        boolean prefixMatched = false;
        for (String prefix : SHOW_INCLUDES_PREFIXES) {
          if (line.startsWith(prefix)) {
            line = line.substring(prefix.length()).trim();
            int index = line.indexOf(execRootSuffix);
            if (index != -1) {
              line = line.substring(index + execRootSuffix.length());
            }
            dependencies.add(line);
            prefixMatched = true;
            break;
          }
        }
        // cl.exe also prints out the file name unconditionally, we need to also filter it out.
        if (!prefixMatched && !line.trim().equals(sourceFileName)) {
          buffer.writeTo(out);
        }
        buffer.reset();
      }
    }

    @Override
    public void flush() throws IOException {
      String line = buffer.toString(StandardCharsets.UTF_8.name());

      // If this line starts or could start with a prefix.
      boolean startingWithAnyPrefix = false;
      for (String prefix : SHOW_INCLUDES_PREFIXES) {
        if (line.startsWith(prefix) || prefix.startsWith(line)) {
          startingWithAnyPrefix = true;
          break;
        }
      }

      if (!startingWithAnyPrefix
          // If this line starts or could start with the source file name.
          && !line.startsWith(sourceFileName)
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
