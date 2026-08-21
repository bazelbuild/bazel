// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.util.Map;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link WorkspaceStatusAction#parseValues}. */
@RunWith(JUnit4.class)
public class WorkspaceStatusActionTest {

  private Path tmpDir;

  @Before
  public void setUp() throws Exception {
    InMemoryFileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    tmpDir = fs.getPath("/tmp");
    tmpDir.createDirectory();
  }

  private Path writeFile(String name, String content) throws Exception {
    Path file = tmpDir.getRelative(name);
    FileSystemUtils.writeContentAsLatin1(file, content);
    return file;
  }

  @Test
  public void parseValues_normalKeyValue() throws Exception {
    Path file = writeFile("status.txt", "KEY value\n");
    Map<String, String> result = WorkspaceStatusAction.parseValues(file);
    assertThat(result).containsExactly("KEY", "value");
  }

  @Test
  public void parseValues_keyOnly_noSpace() throws Exception {
    Path file = writeFile("status.txt", "KEY_ONLY\n");
    Map<String, String> result = WorkspaceStatusAction.parseValues(file);
    assertThat(result).containsExactly("KEY_ONLY", "");
  }

  @Test
  public void parseValues_keyWithTrailingSpace() throws Exception {
    Path file = writeFile("status.txt", "KEY_WITH_SPACE \n");
    Map<String, String> result = WorkspaceStatusAction.parseValues(file);
    assertThat(result).containsExactly("KEY_WITH_SPACE", "");
  }

  @Test
  public void parseValues_keyValueAtEof_noTrailingNewline() throws Exception {
    Path file = writeFile("status.txt", "KEY value");
    Map<String, String> result = WorkspaceStatusAction.parseValues(file);
    assertThat(result).containsExactly("KEY", "value");
  }

  @Test
  public void parseValues_keyOnlyAtEof_noTrailingNewline() throws Exception {
    Path file = writeFile("status.txt", "KEY_ONLY");
    Map<String, String> result = WorkspaceStatusAction.parseValues(file);
    assertThat(result).containsExactly("KEY_ONLY", "");
  }

  @Test
  public void parseValues_multipleKeys() throws Exception {
    Path file = writeFile("status.txt", "FIRST_KEY first_val\nSECOND_KEY second_val\n");
    Map<String, String> result = WorkspaceStatusAction.parseValues(file);
    assertThat(result)
        .containsExactly("FIRST_KEY", "first_val", "SECOND_KEY", "second_val");
  }

  @Test
  public void parseValues_mixedKeyOnlyAndKeyValue() throws Exception {
    Path file = writeFile("status.txt", "KEY_ONLY\nKEY value\n");
    Map<String, String> result = WorkspaceStatusAction.parseValues(file);
    assertThat(result).containsExactly("KEY_ONLY", "", "KEY", "value");
  }

  @Test
  public void parseValues_mixedKeyValueAndKeyOnly() throws Exception {
    Path file = writeFile("status.txt", "KEY value\nKEY_ONLY\n");
    Map<String, String> result = WorkspaceStatusAction.parseValues(file);
    assertThat(result).containsExactly("KEY", "value", "KEY_ONLY", "");
  }

  @Test
  public void parseValues_emptyLinesSkipped() throws Exception {
    Path file = writeFile("status.txt", "KEY value\n\nOTHER other\n");
    Map<String, String> result = WorkspaceStatusAction.parseValues(file);
    assertThat(result).containsExactly("KEY", "value", "OTHER", "other");
  }

  @Test
  public void parseValues_emptyFile() throws Exception {
    Path file = writeFile("status.txt", "");
    Map<String, String> result = WorkspaceStatusAction.parseValues(file);
    assertThat(result).isEmpty();
  }

  @Test
  public void parseValues_valueWithSpaces() throws Exception {
    Path file = writeFile("status.txt", "KEY value with spaces\n");
    Map<String, String> result = WorkspaceStatusAction.parseValues(file);
    assertThat(result).containsExactly("KEY", "value with spaces");
  }
}
