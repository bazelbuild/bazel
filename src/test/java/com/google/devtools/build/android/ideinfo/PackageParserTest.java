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

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.ideinfo.androidstudio.PackageManifestOuterClass.ArtifactLocation;
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
import java.nio.file.InvalidPathException;
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
    private final List<ArtifactLocation> sourceLocations = Lists.newArrayList();
    private StringWriter writer = new StringWriter();

    public MockPackageParserIoProvider addSource(ArtifactLocation source, String javaSrc) {
      try {
        Path path = Paths.get(source.getRootExecutionPathFragment(), source.getRelativePath());
        sources.put(path, new ByteArrayInputStream(javaSrc.getBytes("UTF-8")));
        sourceLocations.add(source);

      } catch (UnsupportedEncodingException | InvalidPathException e) {
        fail(e.getMessage());
      }
      return this;
    }

    public void reset() {
      sources.clear();
      sourceLocations.clear();
      writer = new StringWriter();
    }

    public List<ArtifactLocation> getSourceLocations() {
      return Lists.newArrayList(sourceLocations);
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

  private static final ArtifactLocation DUMMY_SOURCE_ARTIFACT =
      ArtifactLocation.newBuilder()
          .setRootPath("/usr/local/google/code")
          .setRelativePath("java/com/google/Foo.java")
          .setIsSource(true)
          .build();

  private static final ArtifactLocation DUMMY_DERIVED_ARTIFACT =
      ArtifactLocation.newBuilder()
          .setRootPath("/usr/local/_tmp/code/bin")
          .setRootExecutionPathFragment("bin")
          .setRelativePath("java/com/google/Bla.java")
          .setIsSource(false)
          .build();

  private MockPackageParserIoProvider mockIoProvider;
  private PackageParser parser;

  @Before
  public void setUp() {
    mockIoProvider = new MockPackageParserIoProvider();
    parser = new PackageParser(mockIoProvider);
  }

  private Map<ArtifactLocation, String> parsePackageStrings() throws Exception {
    List<ArtifactLocation> sources = mockIoProvider.getSourceLocations();
    return parser.parsePackageStrings(sources);
  }

  @Test
  public void testParseCommandLineArguments() throws Exception {
    String[] args = new String[] {
        "--output_manifest",
        "/tmp/out.manifest",
        "--sources",
        Joiner.on(":").join(
          ",java/com/google/Foo.java,/usr/local/google/code",
          "bin,java/com/google/Bla.java,/usr/local/_tmp/code/bin"
        )};
    PackageParser.PackageParserOptions options = PackageParser.parseArgs(args);
    assertThat(options.outputManifest.toString()).isEqualTo("/tmp/out.manifest");
    assertThat(options.sources).hasSize(2);
    assertThat(options.sources.get(0)).isEqualTo(
        ArtifactLocation.newBuilder()
            .setRootPath("/usr/local/google/code")
            .setRelativePath("java/com/google/Foo.java")
            .setIsSource(true)
            .build());
    assertThat(options.sources.get(1)).isEqualTo(
        ArtifactLocation.newBuilder()
            .setRootPath("/usr/local/_tmp/code/bin")
            .setRootExecutionPathFragment("bin")
            .setRelativePath("java/com/google/Bla.java")
            .setIsSource(false)
            .build());
  }

  @Test
  public void testReadNoSources() throws Exception {
    Map<ArtifactLocation, String> map = parsePackageStrings();
    assertThat(map).isEmpty();
  }

  @Test
  public void testSingleRead() throws Exception {
    mockIoProvider
        .addSource(
            DUMMY_SOURCE_ARTIFACT,
            "package com.google;\n public class Bla {}\"");
    Map<ArtifactLocation, String> map = parsePackageStrings();
    assertThat(map).hasSize(1);
    assertThat(map).containsEntry(
        DUMMY_SOURCE_ARTIFACT,
        "com.google");
  }

  @Test
  public void testMultiRead() throws Exception {
    mockIoProvider
        .addSource(
            DUMMY_SOURCE_ARTIFACT,
            "package com.test;\n public class Bla {}\"")
        .addSource(
            DUMMY_DERIVED_ARTIFACT,
            "package com.other;\n public class Foo {}\"");
    Map<ArtifactLocation, String> map = parsePackageStrings();
    assertThat(map).hasSize(2);
    assertThat(map).containsEntry(
        DUMMY_SOURCE_ARTIFACT,
        "com.test");
    assertThat(map).containsEntry(
        DUMMY_DERIVED_ARTIFACT,
        "com.other");
  }

  @Test
  public void testReadSomeInvalid() throws Exception {
    mockIoProvider
        .addSource(
            DUMMY_SOURCE_ARTIFACT,
            "package %com.test;\n public class Bla {}\"")
        .addSource(
            DUMMY_DERIVED_ARTIFACT,
            "package com.other;\n public class Foo {}\"");
    Map<ArtifactLocation, String> map = parsePackageStrings();
    assertThat(map).hasSize(1);
    assertThat(map).containsEntry(
        DUMMY_DERIVED_ARTIFACT,
        "com.other");
  }

  @Test
  public void testReadAllInvalid() throws Exception {
    mockIoProvider
        .addSource(
            DUMMY_SOURCE_ARTIFACT,
            "#package com.test;\n public class Bla {}\"")
        .addSource(
            DUMMY_DERIVED_ARTIFACT,
            "package com.other\n public class Foo {}\"");
    Map<ArtifactLocation, String> map = parsePackageStrings();
    assertThat(map).isEmpty();
  }

  @Test
  public void testWriteEmptyMap() throws Exception {
    parser.writeManifest(
        Maps.<ArtifactLocation, String> newHashMap(),
        Paths.get("/java/com/google/test.manifest"),
        false);
    assertThat(mockIoProvider.writer.toString()).isEmpty();
  }

  @Test
  public void testWriteMap() throws Exception {
    Map<ArtifactLocation, String> map = ImmutableMap.of(
        DUMMY_SOURCE_ARTIFACT,
        "com.google",
        DUMMY_DERIVED_ARTIFACT,
        "com.other"
    );
    parser.writeManifest(map, Paths.get("/java/com/google/test.manifest"), false);

    String writtenString = mockIoProvider.writer.toString();
    assertThat(writtenString).contains(String.format(
        "root_path: \"%s\"",
        DUMMY_SOURCE_ARTIFACT.getRootPath()));
    assertThat(writtenString).contains(String.format(
        "relative_path: \"%s\"",
        DUMMY_SOURCE_ARTIFACT.getRelativePath()));
    assertThat(writtenString).contains("package_string: \"com.google\"");

    assertThat(writtenString).contains(String.format(
        "root_path: \"%s\"",
        DUMMY_DERIVED_ARTIFACT.getRootPath()));
    assertThat(writtenString).contains(String.format(
        "root_execution_path_fragment: \"%s\"",
        DUMMY_DERIVED_ARTIFACT.getRootExecutionPathFragment()));
    assertThat(writtenString).contains(String.format(
        "relative_path: \"%s\"",
        DUMMY_DERIVED_ARTIFACT.getRelativePath()));
    assertThat(writtenString).contains("package_string: \"com.other\"");
  }

  @Test
  public void testHandlesOldFormat() throws Exception {
    String[] args = new String[] {
        "--output_manifest",
        "/tmp/out.manifest",
        "--sources_absolute_paths",
        "/usr/local/code/java/com/google/Foo.java:/usr/local/code/java/com/google/Bla.java",
        "--sources_execution_paths",
        "java/com/google/Foo.java:java/com/google/Bla.java"
    };
    PackageParser.PackageParserOptions options = PackageParser.parseArgs(args);
    assertThat(options.outputManifest.toString()).isEqualTo("/tmp/out.manifest");
    assertThat(options.sourcesAbsolutePaths.get(0).toString())
        .isEqualTo("/usr/local/code/java/com/google/Foo.java");
    assertThat(options.sourcesAbsolutePaths.get(1).toString())
        .isEqualTo("/usr/local/code/java/com/google/Bla.java");

    PackageParser.convertFromOldFormat(options);
    assertThat(options.sources).hasSize(2);
    assertThat(options.sources.get(0)).isEqualTo(
        ArtifactLocation.newBuilder()
            .setRootPath("/usr/local/code/")
            .setRelativePath("java/com/google/Foo.java")
            .build());

    mockIoProvider
        .addSource(options.sources.get(0),
            "package com.test;\n public class Bla {}\"")
        .addSource(options.sources.get(1),
            "package com.other;\n public class Foo {}\"");

    Map<ArtifactLocation, String> map = parser.parsePackageStrings(options.sources);
    assertThat(map).hasSize(2);
    assertThat(map).containsEntry(options.sources.get(0), "com.test");
    assertThat(map).containsEntry(options.sources.get(1), "com.other");

    parser.writeManifest(map, options.outputManifest, true);

    String writtenString = mockIoProvider.writer.toString();
    assertThat(writtenString).doesNotContain("location");
    assertThat(writtenString).contains(
        "absolute_path: \"/usr/local/code/java/com/google/Foo.java\"");
    assertThat(writtenString).contains(
        "absolute_path: \"/usr/local/code/java/com/google/Bla.java\"");
  }

  @Test
  public void testHandlesFutureFormat() throws Exception {
    String[] args = new String[] {
        "--output_manifest",
        "/tmp/out.manifest",
        "--sources",
        ",java/com/google/Foo.java:"
            + "bin/out,java/com/google/Bla.java",
    };
    PackageParser.PackageParserOptions options = PackageParser.parseArgs(args);
    assertThat(options.outputManifest.toString()).isEqualTo("/tmp/out.manifest");
    assertThat(options.sources).hasSize(2);
    assertThat(options.sources.get(0)).isEqualTo(
        ArtifactLocation.newBuilder()
            .setRelativePath("java/com/google/Foo.java")
            .setIsSource(true)
            .build());
    assertThat(options.sources.get(1)).isEqualTo(
        ArtifactLocation.newBuilder()
            .setRootExecutionPathFragment("bin/out")
            .setRelativePath("java/com/google/Bla.java")
            .setIsSource(false)
            .build());
  }

}
