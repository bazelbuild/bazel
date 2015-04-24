// Copyright 2006-2015 Google Inc. All Rights Reserved.
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
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.Path;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

/**
 * A test case for {@link ParserInputSource}.
 */
@RunWith(JUnit4.class)
public class ParserInputSourceTest {

  private Scratch scratch = new Scratch();

  @Test
  public void testCreateFromFile() throws IOException {
    String content = joinLines("Line 1", "Line 2", "Line 3", "");
    Path file = scratch.file("/tmp/my/file.txt", content.getBytes(StandardCharsets.UTF_8));
    ParserInputSource input = ParserInputSource.create(file);
    assertEquals(content, new String(input.getContent()));
    assertEquals("/tmp/my/file.txt", input.getPath().toString());
  }

  @Test
  public void testCreateFromString() {
    String content = "Content provided as a string.";
    String pathName = "/the/name/of/the/content.txt";
    Path path = scratch.resolve(pathName);
    ParserInputSource input = ParserInputSource.create(content, path);
    assertEquals(content, new String(input.getContent()));
    assertEquals(pathName, input.getPath().toString());
  }

  @Test
  public void testCreateFromCharArray() {
    String content = "Content provided as a string.";
    String pathName = "/the/name/of/the/content.txt";
    Path path = scratch.resolve(pathName);
    char[] contentChars = content.toCharArray();
    ParserInputSource input = ParserInputSource.create(contentChars, path);
    assertEquals(content, new String(input.getContent()));
    assertEquals(pathName, input.getPath().toString());
  }

  @Test
  public void testCreateFromInputStream() throws IOException {
    String content = "Content provided as a string.";
    byte[] bytes = content.getBytes("ISO-8859-1");
    ByteArrayInputStream in = new ByteArrayInputStream(bytes);
    String pathName = "/the/name/of/the/content.txt";
    Path path = scratch.resolve(pathName);
    ParserInputSource input = ParserInputSource.create(in, path);
    assertEquals(content, new String(input.getContent()));
    assertEquals(pathName, input.getPath().toString());
  }

  @Test
  public void testIOExceptionIfInputFileDoesNotExistForSingleArgConstructor() {
    try {
      Path path = scratch.resolve("/does/not/exist");
      ParserInputSource.create(path);
      fail();
    } catch (IOException e) {
      String expected = "/does/not/exist (No such file or directory)";
      assertThat(e).hasMessage(expected);
    }
  }

  @Test
  public void testWillNotTryToReadInputFileIfContentProvidedAsString() {
    Path path = scratch.resolve("/will/not/try/to/read");
    ParserInputSource.create("Content provided as string.", path);
  }

  @Test
  public void testWillNotTryToReadInputFileIfContentProvidedAsChars() {
    Path path = scratch.resolve("/will/not/try/to/read");
    char[] content = "Content provided as char array.".toCharArray();
    ParserInputSource.create(content, path);
  }

  @Test
  public void testWillCloseStreamWhenReadingFromInputStream() {
    final StringBuilder log = new StringBuilder();
    InputStream in = new InputStream() {
      @Override
      public int read() throws IOException {
        throw new IOException("Fault injected.");
      }
      @Override
      public void close() {
        log.append("Stream closed.");
      }
    };
    try {
      Path path = scratch.resolve("/will/not/try/to/read");
      ParserInputSource.create(in, path);
      fail();
    } catch (IOException e) {
      assertThat(e).hasMessage("Fault injected.");
    }
    assertEquals("Stream closed.", log.toString());
  }

}
