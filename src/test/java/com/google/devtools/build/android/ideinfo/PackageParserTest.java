// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.android.ideinfo;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.protobuf.MessageLite;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.StringWriter;
import java.io.UnsupportedEncodingException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;

import javax.annotation.Nonnull;

/**
 * Unit tests for {@link PackageParser}
 */
@RunWith(JUnit4.class)
public class PackageParserTest {

  private static class MockPackageParserIoProvider extends PackageParserIoProvider {
    private final Map<Path, InputStream> sources = Maps.newHashMap();
    private StringWriter writer = new StringWriter();

    public MockPackageParserIoProvider addSource(String filePath, String javaSrc) {
      try {
        sources.put(Paths.get(filePath), new ByteArrayInputStream(javaSrc.getBytes("UTF-8")));
      } catch (UnsupportedEncodingException e) {
        fail(e.getMessage());
      }
      return this;
    }

    public void reset() {
      sources.clear();
      writer = new StringWriter();
    }

    public List<Path> getPaths() {
      return Lists.newArrayList(sources.keySet());
    }

    @Nonnull
    @Override
    public BufferedReader getReader(Path file) throws IOException {
      InputStream input = sources.get(file);
      return new BufferedReader(new InputStreamReader(input, StandardCharsets.UTF_8));
    }

    @Override
    public void writeProto(@Nonnull MessageLite message, @Nonnull Path file) throws IOException {
      writer.write(message.toString());
    }
  }

  private MockPackageParserIoProvider mockIoProvider;
  private PackageParser parser;

  @Before
  public void setUp() {
    mockIoProvider = new MockPackageParserIoProvider();
    parser = new PackageParser(mockIoProvider);
  }

  private Map<Path, String> parsePackageStrings() throws Exception {
    List<Path> paths = mockIoProvider.getPaths();
    return parser.parsePackageStrings(paths, paths);
  }

  @Test
  public void testParseCommandLineArguments() throws Exception {
    String[] args = new String[] {
        "--output_manifest",
        "/tmp/out.manifest",
        "--sources_absolute_paths",
        "/path/test1.java:/path/test2.java",
        "--sources_execution_paths",
        "/path/test1.java:/path/test2.java"
    };
    PackageParser.PackageParserOptions options = PackageParser.parseArgs(args);
    assertThat(options.outputManifest.toString()).isEqualTo("/tmp/out.manifest");
    assertThat(options.sourcesAbsolutePaths).hasSize(2);
    assertThat(options.sourcesExecutionPaths).hasSize(2);
    assertThat(options.sourcesAbsolutePaths.get(0).toString()).isEqualTo("/path/test1.java");
    assertThat(options.sourcesAbsolutePaths.get(1).toString()).isEqualTo("/path/test2.java");
  }

  @Test
  public void testReadNoSources() throws Exception {
    Map<Path, String> map = parsePackageStrings();
    assertThat(map).isEmpty();
  }

  @Test
  public void testSingleRead() throws Exception {
    mockIoProvider
        .addSource("java/com/google/Bla.java",
            "package com.test;\n public class Bla {}\"");
    Map<Path, String> map = parsePackageStrings();
    assertThat(map).hasSize(1);
    assertThat(map).containsEntry(Paths.get("java/com/google/Bla.java"), "com.test");
  }

  @Test
  public void testMultiRead() throws Exception {
    mockIoProvider
        .addSource("java/com/google/Bla.java",
            "package com.test;\n public class Bla {}\"")
        .addSource("java/com/other/Foo.java",
            "package com.other;\n public class Foo {}\"");
    Map<Path, String> map = parsePackageStrings();
    assertThat(map).hasSize(2);
    assertThat(map).containsEntry(Paths.get("java/com/google/Bla.java"), "com.test");
    assertThat(map).containsEntry(Paths.get("java/com/other/Foo.java"), "com.other");
  }

  @Test
  public void testReadSomeInvalid() throws Exception {
    mockIoProvider
        .addSource("java/com/google/Bla.java",
            "package %com.test;\n public class Bla {}\"")
        .addSource("java/com/other/Foo.java",
            "package com.other;\n public class Foo {}\"");
    Map<Path, String> map = parsePackageStrings();
    assertThat(map).hasSize(1);
    assertThat(map).containsEntry(Paths.get("java/com/other/Foo.java"), "com.other");
  }

  @Test
  public void testReadAllInvalid() throws Exception {
    mockIoProvider
        .addSource("java/com/google/Bla.java",
            "#package com.test;\n public class Bla {}\"")
        .addSource("java/com/other/Foo.java",
            "package com.other\n public class Foo {}\"");
    Map<Path, String> map = parsePackageStrings();
    assertThat(map).isEmpty();
  }

  @Test
  public void testWriteEmptyMap() throws Exception {
    parser.writeManifest(
        Maps.<Path, String> newHashMap(), Paths.get("/java/com/google/test.manifest"));
    assertThat(mockIoProvider.writer.toString()).isEmpty();
  }

  @Test
  public void testWriteMap() throws Exception {
    Map<Path, String> map = ImmutableMap.of(
        Paths.get("/java/com/google/Bla.java"), "com.google",
        Paths.get("/java/com/other/Foo.java"), "com.other"
    );
    parser.writeManifest(map, Paths.get("/java/com/google/test.manifest"));

    String writtenString = mockIoProvider.writer.toString();
    assertThat(writtenString).contains("absolute_path: \"/java/com/google/Bla.java\"");
    assertThat(writtenString).contains("package_string: \"com.google\"");
    assertThat(writtenString).contains("absolute_path: \"/java/com/other/Foo.java\"");
    assertThat(writtenString).contains("package_string: \"com.other\"");
  }

}
