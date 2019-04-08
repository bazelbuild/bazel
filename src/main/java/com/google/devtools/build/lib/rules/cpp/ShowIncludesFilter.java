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
import java.util.Arrays;
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
    private static final List<String> SHOW_INCLUDES_PREFIXS = Arrays.asList(
        HexToUTF8String("4E6F74653A20696E636C7564696E672066696C653A"), // English
        HexToUTF8String("E6B3A8E6848F3A20E58C85E590ABE6AA94E6A1883A"), // Traditional Chinese
        HexToUTF8String("506F7A6EC3A16D6B613A2056C48D65746EC49B20736F75626F72753A"), // Czech
        HexToUTF8String("48696E776569733A2045696E6C6573656E206465722044617465693A"), // German
        HexToUTF8String("52656D6172717565203A20696E636C7573696F6E2064752066696368696572203A"), // French
        HexToUTF8String("4E6F74613A2066696C6520696E636C75736F20"), // Italian
        HexToUTF8String("E383A1E383A23A20E382A4E383B3E382AFE383ABE383BCE3838920E38395E382A1E382A4E383AB3A20"), // Japanese
        HexToUTF8String("ECB0B8EAB3A03A20ED8FACED95A820ED8C8CEC9DBC3A"), // Korean
        HexToUTF8String("55776167613A20772074796D20706C696B753A20"), // Polish
        HexToUTF8String("4F627365727661C3A7C3A36F3A20696E636C75696E646F206172717569766F3A"), // Portuguese
        HexToUTF8String("D09FD180D0B8D0BCD0B5D187D0B0D0BDD0B8D0B53A20D0B2D0BAD0BBD18ED187D0B5D0BDD0B8D0B520D184D0B0D0B9D0BBD0B03A"), // Russian
        HexToUTF8String("4E6F743A20656B6C656E656E20646F7379613A20"), // Turkish
        HexToUTF8String("E6B3A8E6848F3A20E58C85E590ABE69687E4BBB63A20"), // Simplified Chinese
        HexToUTF8String("4E6F74613A20696E636C757369C3B36E2064656C206172636869766F3A") // Spanish
    );
    private final String sourceFileName;
    private final String execRootSuffix;

    public static String HexToUTF8String(String hex) {
      byte[] data = new byte[hex.length() / 2];
      for (int i = 0; i < data.length; i++) {
        int index = i * 2;
        int j = Integer.parseInt(hex.substring(index, index + 2), 16);
        data[i] = (byte) j;
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
        for (String prefix : SHOW_INCLUDES_PREFIXS) {
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
      for (String prefix : SHOW_INCLUDES_PREFIXS) {
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
