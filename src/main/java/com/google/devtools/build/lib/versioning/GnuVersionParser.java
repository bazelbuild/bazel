// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.versioning;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.shell.Subprocess;
import com.google.devtools.build.lib.shell.SubprocessBuilder;
import com.google.devtools.build.lib.shell.SubprocessBuilder.StreamAction;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;

/**
 * Parser for version strings that comply with the GNU Coding Standards.
 *
 * <p>Programs that adhere to the GNU Coding Standards, and many more that do not, recognize a
 * single {@code --version} flag on their command line that outputs a header line in a specific
 * format.
 *
 * @param <VersionT> the type of the versions that the parser yields
 */
public class GnuVersionParser<VersionT> {

  /** The expected program name. */
  private final String programName;

  /** Parser for the expected versioning scheme. */
  private final VersionParser<VersionT> versionParser;

  /**
   * Constructs a new parser to extract version numbers of a given program.
   *
   * @param programName the name of the program expected in the version string
   * @param versionParser a parser for the versioning semantics of the program
   */
  public GnuVersionParser(String programName, VersionParser<VersionT> versionParser) {
    this.programName = programName;
    this.versionParser = versionParser;
  }

  /**
   * Extracts a version number from the given string, which should correspond to the first line of
   * the output of a program ran with {@code --version}.
   *
   * @param input the string to parse
   * @return the version number extracted from the string
   * @throws ParseException if the string does not contain the expected program name or if the
   *     version number is invalid
   */
  public VersionT parse(String input) throws ParseException {
    String prefix = programName + " ";
    if (!input.startsWith(prefix)) {
      throw new ParseException("Program name " + programName + " not found");
    }
    return versionParser.parse(input.substring(prefix.length()));
  }

  /**
   * Extracts the first line from an input stream and drains the rest.
   *
   * @param input the input stream from which to read
   * @return the first line from the input stream. Note that it's not possible to distinguish the
   *     case of an empty first line or an empty input stream.
   * @throws IOException if reading from the input fails for any reason, including encountering an
   *     invalid text encoding
   */
  private static String readFirstLineAndDiscardRest(InputStream input) throws IOException {
    ByteArrayOutputStream firstLine = new ByteArrayOutputStream();

    int b;
    while ((b = input.read()) != -1 && b != '\n') {
      firstLine.write(b);
    }
    while (input.read() != -1) {
      // Discard.
    }
    input.close();

    try {
      return firstLine.toString("UTF-8");
    } catch (IOException e) {
      throw new IOException("Program output not in UTF-8 format", e);
    }
  }

  /**
   * Extracts a version number from the given input stream, which should correspond to the output of
   * a program ran with {@code --version}. This drains the stream.
   *
   * @param input the input stream from which to read
   * @return the version number extracted from the input stream
   * @throws IOException if reading from the input fails for any reason
   * @throws ParseException if the first line from the stream does not contain a valid program name
   *     or a valid version number
   */
  public VersionT fromInputStream(InputStream input) throws IOException, ParseException {
    String firstLine = readFirstLineAndDiscardRest(input);
    if (firstLine.isEmpty()) {
      throw new ParseException("No data");
    }
    return parse(firstLine);
  }

  /**
   * Runs a program with {@code --version} and extracts its version number.
   *
   * @param program the program to execute
   * @return the version number extracted from the program
   * @throws IOException if executing the program or reading from its output fail for any reason
   * @throws ParseException if the output of the program did not follow the GNU Coding Standard
   *     conventions, or if the printed program name did not match what we expected
   */
  public VersionT fromProgram(PathFragment program) throws IOException, ParseException {
    Subprocess process =
        new SubprocessBuilder()
            .setArgv(ImmutableList.of(program.getPathString(), "--version"))
            .setStdout(StreamAction.STREAM)
            .redirectErrorStream(true)
            .start();
    boolean waited = false;
    try {
      VersionT version = fromInputStream(process.getInputStream());
      process.waitForUninterruptibly();
      waited = true;
      int exitCode = process.exitValue();
      if (exitCode != 0) {
        throw new IOException("Exited with non-zero code " + exitCode);
      }
      return version;
    } finally {
      if (!waited) {
        process.destroyAndWait();
      }
    }
  }
}
