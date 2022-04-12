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
import java.util.regex.Matcher;
import java.util.regex.Pattern;

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

  public ShowIncludesFilter(String sourceFileName) {
    this.sourceFileName = sourceFileName;
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
    private static final ImmutableList<String> SHOW_INCLUDES_PREFIXES =
        ImmutableList.of(
            new String(
                new byte[] {
                  78, 111, 116, 101, 58, 32, 105, 110, 99, 108, 117, 100, 105, 110, 103, 32, 102,
                  105, 108, 101, 58
                },
                StandardCharsets.UTF_8), // English
            new String(
                new byte[] {
                  -26, -77, -88, -26, -124, -113, 58, 32, -27, -116, -123, -27, -112, -85, -26, -86,
                  -108, -26, -95, -120, 58
                },
                StandardCharsets.UTF_8), // Traditional Chinese
            new String(
                new byte[] {
                  80, 111, 122, 110, -61, -95, 109, 107, 97, 58, 32, 86, -60, -115, 101, 116, 110,
                  -60, -101, 32, 115, 111, 117, 98, 111, 114, 117, 58
                },
                StandardCharsets.UTF_8), // Czech
            new String(
                new byte[] {
                  72, 105, 110, 119, 101, 105, 115, 58, 32, 69, 105, 110, 108, 101, 115, 101, 110,
                  32, 100, 101, 114, 32, 68, 97, 116, 101, 105, 58
                },
                StandardCharsets.UTF_8), // German
            new String(
                new byte[] {
                  82, 101, 109, 97, 114, 113, 117, 101, -62, -96, 58, 32, 105, 110, 99, 108, 117,
                  115, 105, 111, 110, 32, 100, 117, 32, 102, 105, 99, 104, 105, 101, 114, -62, -96,
                  58
                },
                StandardCharsets.UTF_8), // French
            new String(
                new byte[] {
                  78, 111, 116, 97, 58, 32, 102, 105, 108, 101, 32, 105, 110, 99, 108, 117, 115, 111
                },
                StandardCharsets.UTF_8), // Italian
            new String(
                new byte[] {
                  -29, -125, -95, -29, -125, -94, 58, 32, -29, -126, -92, -29, -125, -77, -29, -126,
                  -81, -29, -125, -85, -29, -125, -68, -29, -125, -119, 32, -29, -125, -107, -29,
                  -126, -95, -29, -126, -92, -29, -125, -85, 58
                },
                StandardCharsets.UTF_8), // Janpanese
            new String(
                new byte[] {
                  -20, -80, -72, -22, -77, -96, 58, 32, -19, -113, -84, -19, -107, -88, 32, -19,
                  -116, -116, -20, -99, -68, 58
                },
                StandardCharsets.UTF_8), // Korean
            new String(
                new byte[] {
                  85, 119, 97, 103, 97, 58, 32, 119, 32, 116, 121, 109, 32, 112, 108, 105, 107, 117,
                  58
                },
                StandardCharsets.UTF_8), // Polish
            new String(
                new byte[] {
                  79, 98, 115, 101, 114, 118, 97, -61, -89, -61, -93, 111, 58, 32, 105, 110, 99,
                  108, 117, 105, 110, 100, 111, 32, 97, 114, 113, 117, 105, 118, 111, 58
                },
                StandardCharsets.UTF_8), // Portuguese
            new String(
                new byte[] {
                  -48, -97, -47, -128, -48, -72, -48, -68, -48, -75, -47, -121, -48, -80, -48, -67,
                  -48, -72, -48, -75, 58, 32, -48, -78, -48, -70, -48, -69, -47, -114, -47, -121,
                  -48, -75, -48, -67, -48, -72, -48, -75, 32, -47, -124, -48, -80, -48, -71, -48,
                  -69, -48, -80, 58
                },
                StandardCharsets.UTF_8), // Russian
            new String(
                new byte[] {
                  78, 111, 116, 58, 32, 101, 107, 108, 101, 110, 101, 110, 32, 100, 111, 115, 121,
                  97, 58
                },
                StandardCharsets.UTF_8), // Turkish
            new String(
                new byte[] {
                  -26, -77, -88, -26, -124, -113, 58, 32, -27, -116, -123, -27, -112, -85, -26,
                  -106, -121, -28, -69, -74, 58
                },
                StandardCharsets.UTF_8), // Simplified Chinese
            new String(
                new byte[] {
                  78, 111, 116, 97, 58, 32, 105, 110, 99, 108, 117, 115, 105, -61, -77, 110, 32,
                  100, 101, 108, 32, 97, 114, 99, 104, 105, 118, 111, 58
                },
                StandardCharsets.UTF_8) // Spanish
            );
    private final String sourceFileName;
    // Grab everything under the execroot base so that external repository header files are covered
    // in the sibling repository layout.
    private static final Pattern EXECROOT_BASE_HEADER_PATTERN =
        Pattern.compile(".*execroot\\\\(?<headerPath>.*)");

    public FilterShowIncludesOutputStream(OutputStream out, String sourceFileName) {
      super(out);
      this.sourceFileName = sourceFileName;
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
            Matcher m = EXECROOT_BASE_HEADER_PATTERN.matcher(line);
            if (m.matches()) {
              // Prefix the matched header path with "..\". This way, external repo header paths are
              // resolved to "<execroot>\..\<repo name>\<path>", and main repo file paths are
              // resolved to "<execroot>\..\<main repo>\<path>", which is nicely normalized to
              // "<execroot>\<path>".
              line = "..\\" + m.group("headerPath");
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
        new FilterShowIncludesOutputStream(outputStream, sourceFileName);
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
