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

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.JavaIoFileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link GnuVersionParser}. */
@RunWith(JUnit4.class)
public class GnuVersionParserTest {

  /** A version parser that always throws because it should never be called. */
  private static final VersionParser<Void> UNUSED_PARSER =
      (text) -> {
        throw new AssertionError("Not expected to be called");
      };

  @Test
  public void testParse_ok() throws Exception {
    GnuVersionParser<SemVer> parser = new GnuVersionParser<>("sandboxfs", SemVer::parse);
    assertThat(parser.parse("sandboxfs 0.1.3")).isEqualTo(SemVer.from(0, 1, 3));
  }

  /** Syntactic sugar to create a new input stream that contains a UTF-8 string. */
  private static InputStream newInputStream(String string) {
    return new ByteArrayInputStream(string.getBytes(StandardCharsets.UTF_8));
  }

  @Test
  public void testFromInputStream_ok() throws Exception {
    GnuVersionParser<SemVer> parser = new GnuVersionParser<>("GNU Emacs", SemVer::parse);
    for (String stdout :
        new String[] {
          "GNU Emacs 26.3.0",
          "GNU Emacs 26.3.0\n",
          "GNU Emacs 26.3.0\nsome other text\nthat we don't care about",
        }) {
      InputStream input = newInputStream(stdout);
      assertThat(parser.fromInputStream(input)).isEqualTo(SemVer.from(26, 3));
    }
  }

  @Test
  public void testFromInputStream_errorIfEmpty() {
    GnuVersionParser<SemVer> parser = new GnuVersionParser<>("GNU Emacs", SemVer::parse);
    InputStream input = newInputStream("");
    Exception e = assertThrows(ParseException.class, () -> parser.fromInputStream(input));
    assertThat(e).hasMessageThat().contains("No data");
  }

  @Test
  public void testFromInputStream_errorOnBadVersion() {
    GnuVersionParser<SemVer> parser = new GnuVersionParser<>("GNU Emacs", SemVer::parse);
    InputStream input = newInputStream("GNU Emacs 0.abc");
    Exception e = assertThrows(ParseException.class, () -> parser.fromInputStream(input));
    assertThat(e).hasMessageThat().contains("Invalid semver");
  }

  @Test
  public void testFromInputStream_drainsInput() throws Exception {
    GnuVersionParser<SemVer> parser = new GnuVersionParser<>("foo", SemVer::parse);
    InputStream input = newInputStream("foo 1\nsome extra text");
    parser.fromInputStream(input);
    assertThat(input.available()).isEqualTo(0);
    assertThat(input.read()).isEqualTo(-1);
  }

  /** Creates an executable shell script with the given contents and returns its path. */
  private static Path createHelper(String content) throws IOException {
    FileSystem fs = new JavaIoFileSystem(DigestHashFunction.getDefaultUnchecked());
    Path helper = fs.getPath(System.getenv("TEST_TMPDIR")).getRelative("helper.sh");
    try (PrintWriter output =
        new PrintWriter(new OutputStreamWriter(helper.getOutputStream(), StandardCharsets.UTF_8))) {
      output.println("#!/bin/sh");
      output.println(content);
    }
    helper.setExecutable(true);
    return helper;
  }

  @Test
  public void testFromProgram_ok() throws Exception {
    Path helper = createHelper("echo test 9.8; echo some more text that is ignored");
    GnuVersionParser<SemVer> parser = new GnuVersionParser<>("test", SemVer::parse);
    assertThat(parser.fromProgram(helper.asFragment())).isEqualTo(SemVer.from(9, 8));
  }

  @Test
  public void testFromProgram_badPackageName() throws Exception {
    Path helper = createHelper("echo foo 9.8; echo some more text that is ignored");
    GnuVersionParser<?> parser = new GnuVersionParser<>("test", UNUSED_PARSER);
    Exception e = assertThrows(ParseException.class, () -> parser.fromProgram(helper.asFragment()));
    assertThat(e).hasMessageThat().contains("Program name test not found");
  }

  @Test
  public void testFromProgram_badVersion() throws Exception {
    Path helper = createHelper("echo test 8.3a");
    GnuVersionParser<?> parser = new GnuVersionParser<>("test", SemVer::parse);
    Exception e = assertThrows(ParseException.class, () -> parser.fromProgram(helper.asFragment()));
    assertThat(e).hasMessageThat().contains("Invalid semver");
  }

  @Test
  public void testFromProgram_badExitCode() throws Exception {
    Path helper = createHelper("echo test 9.1; return 1");
    GnuVersionParser<?> parser = new GnuVersionParser<>("test", SemVer::parse);
    Exception e = assertThrows(IOException.class, () -> parser.fromProgram(helper.asFragment()));
    assertThat(e).hasMessageThat().contains("Exited with non-zero code");
  }

  @Test
  public void testFromProgram_execError() {
    GnuVersionParser<?> parser = new GnuVersionParser<>("test", UNUSED_PARSER);
    Exception e =
        assertThrows(
            IOException.class,
            () -> parser.fromProgram(PathFragment.create("/some/non-existent/path")));
    assertThat(e).hasMessageThat().contains("Cannot run program");
  }
}
