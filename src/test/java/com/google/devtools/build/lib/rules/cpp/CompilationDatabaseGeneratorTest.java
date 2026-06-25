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
package com.google.devtools.build.lib.rules.cpp;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.rules.cpp.CompilationDatabaseGenerator.Entry;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link CompilationDatabaseGenerator}. */
@RunWith(JUnit4.class)
public class CompilationDatabaseGeneratorTest {

  @Test
  public void testToJson_emptyList() {
    assertThat(CompilationDatabaseGenerator.toJson(new ArrayList<>())).isEqualTo("[]");
  }

  @Test
  public void testToJson_singleEntry() {
    List<Entry> entries = new ArrayList<>();
    entries.add(
        new Entry(
            "/path/to/workspace",
            "src/main.cc",
            List.of("clang++", "-Iinclude", "-std=c++17", "-c", "src/main.cc", "-o", "main.o"),
            "bazel-out/k8-fastbuild/bin/_objs/main/main.o"));

    String json = CompilationDatabaseGenerator.toJson(entries);

    assertThat(json).contains("\"directory\":\"/path/to/workspace\"");
    assertThat(json).contains("\"file\":\"src/main.cc\"");
    assertThat(json).contains("\"arguments\"");
    // No HTML escaping: = should remain as =, not \u003d
    assertThat(json).contains("-std=c++17");
    assertThat(json).contains("\"output\":\"bazel-out/k8-fastbuild/bin/_objs/main/main.o\"");
  }

  @Test
  public void testToJson_multipleEntries() {
    List<Entry> entries = new ArrayList<>();
    entries.add(
        new Entry(
            "/path/to/workspace", "src/main.cc", List.of("clang++", "-c", "src/main.cc"), "main.o"));
    entries.add(
        new Entry(
            "/path/to/workspace", "src/util.cc", List.of("clang++", "-c", "src/util.cc"), "util.o"));

    String json = CompilationDatabaseGenerator.toJson(entries);

    assertThat(json).contains("src/main.cc");
    assertThat(json).contains("src/util.cc");
  }

  @Test
  public void testToJson_nullOutput() {
    List<Entry> entries = new ArrayList<>();
    entries.add(
        new Entry("/path/to/workspace", "src/main.cc", List.of("clang++", "-c", "src/main.cc"), null));

    String json = CompilationDatabaseGenerator.toJson(entries);

    assertThat(json).doesNotContain("\"output\"");
  }

  @Test
  public void testToJson_noHtmlEscaping() {
    List<Entry> entries = new ArrayList<>();
    entries.add(
        new Entry(
            "/ws",
            "test.cc",
            List.of("clang++", "-DFOO=<script>alert(1)</script>", "-c", "test.cc"),
            "test.o"));

    String json = CompilationDatabaseGenerator.toJson(entries);

    assertThat(json).doesNotContain("\\u003c");
    assertThat(json).doesNotContain("\\u003e");
    assertThat(json).contains("<script>");
  }

  @Test
  public void testToJsonBytes_usesUtf8() {
    List<Entry> entries = new ArrayList<>();
    entries.add(new Entry("/ws", "test.cc", List.of("clang++", "-DMSG=你好", "-c", "test.cc"), "test.o"));

    byte[] bytes = CompilationDatabaseGenerator.toJsonBytes(entries);
    String decoded = new String(bytes, StandardCharsets.UTF_8);

    assertThat(decoded).contains("你好");
  }
}
