// Copyright 2006 The Bazel Authors. All Rights Reserved.
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
package com.google.devtools.build.lib.syntax;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.util.StringUtilities.joinLines;

import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** A test case for {@link ParserInput}. */
@RunWith(JUnit4.class)
public class ParserInputTest {

  private Scratch scratch = new Scratch();

  @Test
  public void testCreateFromFile() throws IOException {
    String content = joinLines("Line 1", "Line 2", "Line 3", "");
    Path file = scratch.file("/tmp/my/file.txt", content.getBytes(StandardCharsets.UTF_8));
    byte[] bytes = FileSystemUtils.readWithKnownFileSize(file, file.getFileSize());
    ParserInput input = ParserInput.fromLatin1(bytes, file.toString());
    assertThat(new String(input.getContent())).isEqualTo(content);
    assertThat(input.getFile()).isEqualTo("/tmp/my/file.txt");
  }

  @Test
  public void testCreateFromString() {
    String content = "Content provided as a string.";
    String pathName = "/the/name/of/the/content.txt";
    ParserInput input = ParserInput.fromString(content, pathName);
    assertThat(new String(input.getContent())).isEqualTo(content);
    assertThat(input.getFile()).isEqualTo(pathName);
  }

  @Test
  public void testCreateFromCharArray() {
    String content = "Content provided as a string.";
    String pathName = "/the/name/of/the/content.txt";
    char[] contentChars = content.toCharArray();
    ParserInput input = ParserInput.fromCharArray(contentChars, pathName);
    assertThat(new String(input.getContent())).isEqualTo(content);
    assertThat(input.getFile()).isEqualTo(pathName);
  }

  @Test
  public void testWillNotTryToReadInputFileIfContentProvidedAsString() {
    ParserInput.fromString("Content provided as string.", "/will/not/try/to/read");
  }

  @Test
  public void testWillNotTryToReadInputFileIfContentProvidedAsChars() {
    char[] content = "Content provided as char array.".toCharArray();
    ParserInput.fromCharArray(content, "/will/not/try/to/read");
  }
}
