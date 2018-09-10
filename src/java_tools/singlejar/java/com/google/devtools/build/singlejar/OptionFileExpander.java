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

package com.google.devtools.build.singlejar;

import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.devtools.build.lib.shell.ShellUtils;
import com.google.devtools.build.lib.shell.ShellUtils.TokenizationException;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.concurrent.Immutable;

/**
 * A utility class to parse option files and expand them.
 */
@Immutable
final class OptionFileExpander {

  /**
   * An interface that allows injecting different implementations for reading
   * files. This is mostly used for testing.
   */
  interface OptionFileProvider {

    /**
     * Opens a file for reading and returns an input stream.
     */
    InputStream getInputStream(String filename) throws IOException;
  }

  private final OptionFileProvider fileSystem;

  /**
   * Creates an instance with the given option file provider.
   */
  public OptionFileExpander(OptionFileProvider fileSystem) {
    this.fileSystem = fileSystem;
  }

  /**
   * Pre-processes an argument list, expanding options of the form &at;filename
   * to read in the content of the file and add it to the list of arguments.
   *
   * @param args the List of arguments to pre-process.
   * @return the List of pre-processed arguments.
   * @throws IOException if one of the files containing options cannot be read.
   */
  public List<String> expandArguments(List<String> args) throws IOException {
    List<String> expanded = new ArrayList<>(args.size());
    for (String arg : args) {
      expandArgument(arg, expanded);
    }
    return expanded;
  }

  /**
   * Expands a single argument, expanding options &at;filename to read in
   * the content of the file and add it to the list of processed arguments.
   *
   * @param arg the argument to pre-process.
   * @param expanded the List of pre-processed arguments.
   * @throws IOException if one of the files containing options cannot be read.
   */
  private void expandArgument(String arg, List<String> expanded) throws IOException {
    if (arg.startsWith("@")) {
      try (InputStreamReader reader =
          new InputStreamReader(fileSystem.getInputStream(arg.substring(1)), ISO_8859_1)) {
        // TODO(bazel-team): This code doesn't handle escaped newlines correctly.
        // ShellUtils doesn't support them either.
        for (String line : readAllLines(reader)) {
          List<String> parsedTokens = new ArrayList<>();
          try {
            ShellUtils.tokenize(parsedTokens, line);
          } catch (TokenizationException e) {
            throw new IOException("Could not tokenize parameter file!", e);
          }
          for (String token : parsedTokens) {
            expandArgument(token, expanded);
          }
        }
      }
    } else {
      expanded.add(arg);
    }
  }

  private List<String> readAllLines(Reader in) throws IOException {
    List<String> result = new ArrayList<>();
    BufferedReader reader = new BufferedReader(in);
    String line;
    while ((line = reader.readLine()) != null) {
      result.add(line);
    }
    return result;
  }
}
