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

package com.google.devtools.build.singlejar;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.singlejar.FakeZipFile.ByteValidator;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.jar.JarFile;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for {@link SingleJar}.
 */
@RunWith(JUnit4.class)
public class SingleJarTest {

  public static final byte[] EXTRA_FOR_META_INF = new byte[] {(byte) 0xFE, (byte) 0xCA, 0x00, 0x00};

  static final Joiner LINE_JOINER = Joiner.on("\r\n");
  static final Joiner LINEFEED_JOINER = Joiner.on("\n");

  static enum EntryMode {
    DONT_CARE, EXPECT_DEFLATE, EXPECT_STORED;
  }

  public static final class BuildInfoValidator implements ByteValidator {
    private final List<String> buildInfoLines;

    public BuildInfoValidator(List<String> buildInfoLines) {
      this.buildInfoLines = buildInfoLines;
    }

    @Override
    public void validate(byte[] content) {
      String actualBuildInfo = new String(content, StandardCharsets.UTF_8);
      List<String> expectedBuildInfos = new ArrayList<>();
      for (String line : buildInfoLines) { // the character : is escaped
        expectedBuildInfos.add(line.replace(":", "\\:"));
      }
      Collections.sort(expectedBuildInfos);
      String[] actualBuildInfos = actualBuildInfo.split("\n");
      Arrays.sort(actualBuildInfos);
      assertEquals(LINEFEED_JOINER.join(expectedBuildInfos),
          LINEFEED_JOINER.join(actualBuildInfos));
    }

  }

  // Manifest file line ordering is dependent of the ordering in HashMap (Attributes class) so
  // we do a sorted comparison for Manifest.
  public static final class ManifestValidator implements ByteValidator {
    private final List<String> manifestLines;

    public ManifestValidator(List<String> manifestLines) {
      this.manifestLines = new ArrayList<>(manifestLines);
      Collections.sort(this.manifestLines);
    }

    public ManifestValidator(String... manifestLines) {
      this.manifestLines = Arrays.asList(manifestLines);
      Collections.sort(this.manifestLines);
    }

    @Override
    public void validate(byte[] content) {
      String actualManifest = new String(content, StandardCharsets.UTF_8);
      String[] actualManifestLines = actualManifest.trim().split("\r\n");
      Arrays.sort(actualManifestLines);
      assertEquals(LINEFEED_JOINER.join(manifestLines), LINEFEED_JOINER.join(actualManifestLines));
    }

  }

  private BuildInfoValidator redactedBuildData(String outputJar) {
    return new BuildInfoValidator(ImmutableList.of("build.target=" + outputJar));
  }

  private BuildInfoValidator redactedBuildData(String outputJar, String mainClass) {
    return new BuildInfoValidator(
        ImmutableList.of("build.target=" + outputJar, "main.class=" + mainClass));
  }

  static List<String> getBuildInfo() {
    return ImmutableList.of("build.build_id=11111-222-33333",
        "build.version=12659499",
        "build.location=user@machine.domain.com:/home/user/source",
        "build.target=output.jar",
        "build.time=Fri Jan 2 02:17:36 1970 (123456)",
        "build.timestamp=Fri Jan 2 02:17:36 1970 (123456)",
        "build.timestamp.as.int=123456"
                            );
  }

  private byte[] sampleZip() {
    ZipFactory factory = new ZipFactory();
    factory.addFile("hello.txt", "Hello World!");
    return factory.toByteArray();
  }

  private byte[] sampleUncompressedZip() {
    ZipFactory factory = new ZipFactory();
    factory.addFile("hello.txt", "Hello World!", false);
    return factory.toByteArray();
  }

  private byte[] sampleZipWithSF() {
    ZipFactory factory = new ZipFactory();
    factory.addFile("hello.SF", "Hello World!");
    return factory.toByteArray();
  }

  private byte[] sampleZipWithSubdirs() {
    ZipFactory factory = new ZipFactory();
    factory.addFile("dir1/file1", "contents11");
    factory.addFile("dir1/file2", "contents12");
    factory.addFile("dir2/file1", "contents21");
    factory.addFile("dir3/file1", "contents31");
    return factory.toByteArray();
  }

  private void assertStripFirstLine(String expected, String testCase) {
    byte[] result = SingleJar.stripFirstLine(testCase.getBytes(StandardCharsets.UTF_8));
    assertEquals(expected, new String(result, UTF_8));
  }

  @Test
  public void testStripFirstLine() {
    assertStripFirstLine("", "");
    assertStripFirstLine("", "no linefeed");
    assertStripFirstLine(LINEFEED_JOINER.join("toto", "titi"),
        LINEFEED_JOINER.join("# timestamp comment", "toto", "titi"));
    assertStripFirstLine(LINE_JOINER.join("toto", "titi"),
        LINE_JOINER.join("# timestamp comment", "toto", "titi"));
  }

  @Test
  public void testEmptyJar() throws IOException {
    MockSimpleFileSystem mockFs = new MockSimpleFileSystem("output.jar");
    SingleJar singleJar = new SingleJar(mockFs);
    singleJar.run(ImmutableList.of("--output", "output.jar"));
    FakeZipFile expectedResult = new FakeZipFile()
        .addEntry("META-INF/", EXTRA_FOR_META_INF)
        .addEntry(JarFile.MANIFEST_NAME, new ManifestValidator(
            "Manifest-Version: 1.0",
            "Created-By: blaze-singlejar"))
        .addEntry("build-data.properties", redactedBuildData("output.jar"));
    expectedResult.assertSame(mockFs.toByteArray());
  }

  // Test that two identical calls at different time actually returns identical results
  @Test
  public void testDeterministicJar() throws IOException, InterruptedException {
    MockSimpleFileSystem mockFs1 = new MockSimpleFileSystem("output.jar");
    SingleJar singleJar1 = new SingleJar(mockFs1);
    singleJar1.run(ImmutableList.of("--output", "output.jar", "--extra_build_info", "toto=titi",
        "--normalize"));
    Thread.sleep(1000); // ensure that we are not at the same seconds

    MockSimpleFileSystem mockFs2 = new MockSimpleFileSystem("output.jar");
    SingleJar singleJar2 = new SingleJar(mockFs2);
    singleJar2.run(ImmutableList.of("--output", "output.jar", "--extra_build_info", "toto=titi",
        "--normalize"));

    FakeZipFile.assertSame(mockFs1.toByteArray(), mockFs2.toByteArray());
  }

  @Test
  public void testExtraManifestContent() throws IOException {
    MockSimpleFileSystem mockFs = new MockSimpleFileSystem("output.jar");
    SingleJar singleJar = new SingleJar(mockFs);
    singleJar.run(ImmutableList.of("--output", "output.jar", "--deploy_manifest_lines",
        "Main-Class: SomeClass", "X-Other: Duh"));
    FakeZipFile expectedResult = new FakeZipFile()
        .addEntry("META-INF/", EXTRA_FOR_META_INF)
        .addEntry(JarFile.MANIFEST_NAME, new ManifestValidator(
            "Manifest-Version: 1.0",
            "Created-By: blaze-singlejar",
            "Main-Class: SomeClass",
            "X-Other: Duh"))
        .addEntry("build-data.properties", redactedBuildData("output.jar"));
    expectedResult.assertSame(mockFs.toByteArray());
  }

  @Test
  public void testMultipleExtraManifestContent() throws IOException {
    MockSimpleFileSystem mockFs = new MockSimpleFileSystem("output.jar");
    SingleJar singleJar = new SingleJar(mockFs);
    singleJar.run(ImmutableList.of("--deploy_manifest_lines", "X-Other: Duh",
        "--output", "output.jar",
        "--deploy_manifest_lines", "Main-Class: SomeClass"));
    FakeZipFile expectedResult = new FakeZipFile()
        .addEntry("META-INF/", EXTRA_FOR_META_INF)
        .addEntry(JarFile.MANIFEST_NAME, new ManifestValidator(
            "Manifest-Version: 1.0",
            "Created-By: blaze-singlejar",
            "Main-Class: SomeClass",
            "X-Other: Duh"))
        .addEntry("build-data.properties", redactedBuildData("output.jar"));
    expectedResult.assertSame(mockFs.toByteArray());
  }

  @Test
  public void testMainClass() throws IOException {
    MockSimpleFileSystem mockFs = new MockSimpleFileSystem("output.jar");
    SingleJar singleJar = new SingleJar(mockFs);
    singleJar.run(ImmutableList.of("--output", "output.jar", "--main_class", "SomeClass"));
    FakeZipFile expectedResult = new FakeZipFile()
        .addEntry("META-INF/", EXTRA_FOR_META_INF)
        .addEntry(JarFile.MANIFEST_NAME, new ManifestValidator(
            "Manifest-Version: 1.0",
            "Created-By: blaze-singlejar",
            "Main-Class: SomeClass"))
        .addEntry("build-data.properties", redactedBuildData("output.jar", "SomeClass"));
    expectedResult.assertSame(mockFs.toByteArray());
  }

  // These four tests test all combinations of compressed/uncompressed input and output.
  @Test
  public void testSimpleZip() throws IOException {
    MockSimpleFileSystem mockFs = new MockSimpleFileSystem("output.jar");
    mockFs.addFile("test.jar", sampleZip());
    SingleJar singleJar = new SingleJar(mockFs);
    singleJar.run(ImmutableList.of("--output", "output.jar", "--sources", "test.jar"));
    FakeZipFile expectedResult = new FakeZipFile()
        .addEntry("META-INF/", EXTRA_FOR_META_INF, false)
        .addEntry(JarFile.MANIFEST_NAME, new ManifestValidator(
            "Manifest-Version: 1.0",
            "Created-By: blaze-singlejar"), false)
        .addEntry("build-data.properties", redactedBuildData("output.jar"), false)
        .addEntry("hello.txt", "Hello World!", false);
    expectedResult.assertSame(mockFs.toByteArray());
  }

  @Test
  public void testSimpleZipExpectCompressedOutput() throws IOException {
    MockSimpleFileSystem mockFs = new MockSimpleFileSystem("output.jar");
    mockFs.addFile("test.jar", sampleZip());
    SingleJar singleJar = new SingleJar(mockFs);
    singleJar.run(ImmutableList.of("--output", "output.jar", "--sources", "test.jar",
        "--compression"));
    FakeZipFile expectedResult = new FakeZipFile()
        .addEntry("META-INF/", EXTRA_FOR_META_INF, false)
        .addEntry(JarFile.MANIFEST_NAME, new ManifestValidator(
            "Manifest-Version: 1.0",
            "Created-By: blaze-singlejar"), true)
        .addEntry("build-data.properties", redactedBuildData("output.jar"), true)
        .addEntry("hello.txt", "Hello World!", true);
    expectedResult.assertSame(mockFs.toByteArray());
  }

  @Test
  public void testSimpleUncompressedZip() throws IOException {
    MockSimpleFileSystem mockFs = new MockSimpleFileSystem("output.jar");
    mockFs.addFile("test.jar", sampleUncompressedZip());
    SingleJar singleJar = new SingleJar(mockFs);
    singleJar.run(ImmutableList.of("--output", "output.jar", "--sources", "test.jar"));
    FakeZipFile expectedResult = new FakeZipFile()
        .addEntry("META-INF/", EXTRA_FOR_META_INF, false)
        .addEntry(JarFile.MANIFEST_NAME, new ManifestValidator(ImmutableList.of(
            "Manifest-Version: 1.0",
            "Created-By: blaze-singlejar")), false)
        .addEntry("build-data.properties", redactedBuildData("output.jar"), false)
        .addEntry("hello.txt", "Hello World!", false);
    expectedResult.assertSame(mockFs.toByteArray());
  }

  @Test
  public void testSimpleUncompressedZipExpectCompressedOutput() throws IOException {
    MockSimpleFileSystem mockFs = new MockSimpleFileSystem("output.jar");
    mockFs.addFile("test.jar", sampleUncompressedZip());
    SingleJar singleJar = new SingleJar(mockFs);
    singleJar.run(ImmutableList.of("--output", "output.jar", "--sources", "test.jar",
        "--compression"));
    FakeZipFile expectedResult = new FakeZipFile()
        .addEntry("META-INF/", EXTRA_FOR_META_INF, false)
        .addEntry(JarFile.MANIFEST_NAME, new ManifestValidator(
            "Manifest-Version: 1.0",
            "Created-By: blaze-singlejar"), true)
        .addEntry("build-data.properties", redactedBuildData("output.jar"), true)
        .addEntry("hello.txt", "Hello World!", true);
    expectedResult.assertSame(mockFs.toByteArray());
  }

  // Integration test for option file expansion.
  @Test
  public void testOptionFile() throws IOException {
    MockSimpleFileSystem mockFs = new MockSimpleFileSystem("output.jar");
    mockFs.addFile("input.jar", sampleZip());
    mockFs.addFile("options", "--output output.jar --sources input.jar");
    SingleJar singleJar = new SingleJar(mockFs);
    singleJar.run(ImmutableList.of("@options"));
    FakeZipFile expectedResult = new FakeZipFile()
        .addEntry("META-INF/", EXTRA_FOR_META_INF)
        .addEntry(JarFile.MANIFEST_NAME, new ManifestValidator(
            "Manifest-Version: 1.0",
            "Created-By: blaze-singlejar"))
        .addEntry("build-data.properties", redactedBuildData("output.jar"))
        .addEntry("hello.txt", "Hello World!");
    expectedResult.assertSame(mockFs.toByteArray());
  }

  @Test
  public void testSkipsSignatureFiles() throws IOException {
    MockSimpleFileSystem mockFs = new MockSimpleFileSystem("output.jar");
    mockFs.addFile("input.jar", sampleZipWithSF());
    SingleJar singleJar = new SingleJar(mockFs);
    singleJar.run(ImmutableList.of("--output", "output.jar", "--sources", "input.jar"));
    FakeZipFile expectedResult = new FakeZipFile()
        .addEntry("META-INF/", EXTRA_FOR_META_INF)
        .addEntry(JarFile.MANIFEST_NAME, new ManifestValidator(
            "Manifest-Version: 1.0",
            "Created-By: blaze-singlejar"))
        .addEntry("build-data.properties", redactedBuildData("output.jar"));
    expectedResult.assertSame(mockFs.toByteArray());
  }

  @Test
  public void testSkipsUsingInputPrefixes() throws IOException {
    MockSimpleFileSystem mockFs = new MockSimpleFileSystem("output.jar");
    mockFs.addFile("input.jar", sampleZipWithSubdirs());
    SingleJar singleJar = new SingleJar(mockFs);
    singleJar.run(ImmutableList.of("--output", "output.jar", "--sources",
        "input.jar", "--include_prefixes", "dir1", "dir2"));

    FakeZipFile expectedResult = new FakeZipFile()
        .addEntry("META-INF/", EXTRA_FOR_META_INF)
        .addEntry(JarFile.MANIFEST_NAME, new ManifestValidator(
            "Manifest-Version: 1.0",
            "Created-By: blaze-singlejar"))
        .addEntry("build-data.properties", redactedBuildData("output.jar"))
        .addEntry("dir1/file1", "contents11")
        .addEntry("dir1/file2", "contents12")
        .addEntry("dir2/file1", "contents21");

    expectedResult.assertSame(mockFs.toByteArray());
  }

  @Test
  public void testSkipsUsingMultipleInputPrefixes() throws IOException {
    MockSimpleFileSystem mockFs = new MockSimpleFileSystem("output.jar");
    mockFs.addFile("input.jar", sampleZipWithSubdirs());
    SingleJar singleJar = new SingleJar(mockFs);
    singleJar.run(ImmutableList.of("--output", "output.jar", "--include_prefixes", "dir2",
        "--sources", "input.jar", "--include_prefixes", "dir1"));

    FakeZipFile expectedResult = new FakeZipFile()
        .addEntry("META-INF/", EXTRA_FOR_META_INF)
        .addEntry(JarFile.MANIFEST_NAME, new ManifestValidator(
            "Manifest-Version: 1.0",
            "Created-By: blaze-singlejar"))
        .addEntry("build-data.properties", redactedBuildData("output.jar"))
        .addEntry("dir1/file1", "contents11")
        .addEntry("dir1/file2", "contents12")
        .addEntry("dir2/file1", "contents21");

    expectedResult.assertSame(mockFs.toByteArray());
  }

  @Test
  public void testNormalize() throws IOException {
    MockSimpleFileSystem mockFs = new MockSimpleFileSystem("output.jar");
    mockFs.addFile("input.jar", sampleZip());
    SingleJar singleJar = new SingleJar(mockFs);
    singleJar.run(ImmutableList.of("--output", "output.jar", "--sources", "input.jar",
        "--normalize"));
    FakeZipFile expectedResult = new FakeZipFile()
        .addEntry("META-INF/", EXTRA_FOR_META_INF, false)
        .addEntry(JarFile.MANIFEST_NAME, ZipCombiner.DOS_EPOCH, new ManifestValidator(
            "Manifest-Version: 1.0", "Created-By: blaze-singlejar"), false)
        .addEntry("build-data.properties", ZipCombiner.DOS_EPOCH,
            redactedBuildData("output.jar"), false)
        .addEntry("hello.txt", ZipCombiner.DOS_EPOCH, "Hello World!", false);
    expectedResult.assertSame(mockFs.toByteArray());
  }

  @Test
  public void testNormalizeAndCompress() throws IOException {
    MockSimpleFileSystem mockFs = new MockSimpleFileSystem("output.jar");
    mockFs.addFile("input.jar", sampleZip());
    SingleJar singleJar = new SingleJar(mockFs);
    singleJar.run(ImmutableList.of("--output", "output.jar", "--sources", "input.jar",
        "--normalize", "--compression"));
    FakeZipFile expectedResult = new FakeZipFile()
        .addEntry("META-INF/", EXTRA_FOR_META_INF, false)
        .addEntry(JarFile.MANIFEST_NAME, ZipCombiner.DOS_EPOCH, new ManifestValidator(
            "Manifest-Version: 1.0", "Created-By: blaze-singlejar"), true)
        .addEntry("build-data.properties", ZipCombiner.DOS_EPOCH,
             redactedBuildData("output.jar"), true)
        .addEntry("hello.txt", ZipCombiner.DOS_EPOCH, "Hello World!", true);
    expectedResult.assertSame(mockFs.toByteArray());
  }

  @Test
  public void testAddBuildInfoProperties() throws IOException {
    List<String> buildInfo = getBuildInfo();
    FakeZipFile expectedResult = new FakeZipFile()
        .addEntry("META-INF/", EXTRA_FOR_META_INF, false)
        .addEntry(JarFile.MANIFEST_NAME, new ManifestValidator(
                "Manifest-Version: 1.0", "Created-By: blaze-singlejar"), false)
        .addEntry("build-data.properties", new BuildInfoValidator(buildInfo),
            false);

    MockSimpleFileSystem mockFs = new MockSimpleFileSystem("output.jar");
    SingleJar singleJar = new SingleJar(mockFs);
    List<String> args = new ArrayList<>();
    args.add("--output");
    args.add("output.jar");
    args.addAll(infoPropertyArguments(buildInfo));
    singleJar.run(args);
    expectedResult.assertSame(mockFs.toByteArray());
  }

  private static List<String> infoPropertyArguments(List<String> buildInfoLines) {
    List<String> args = new ArrayList<>();
    for (String s : buildInfoLines) {
      if (!s.isEmpty()) {
        args.add("--extra_build_info");
        args.add(s);
      }
    }
    return args;
  }

  @Test
  public void testAddBuildInfoPropertiesFile() throws IOException {
    MockSimpleFileSystem mockFs = new MockSimpleFileSystem("output.jar");
    SingleJar singleJar = new SingleJar(mockFs);
    doTestAddBuildInfoPropertiesFile(mockFs, "output.jar", singleJar);
  }

  public static void doTestAddBuildInfoPropertiesFile(MockSimpleFileSystem mockFs, String target,
      SingleJar singleJar) throws IOException {
    List<String> buildInfo = getBuildInfo();
    mockFs.addFile("my.properties", makePropertyFileFromBuildInfo(buildInfo));
    singleJar.run(ImmutableList.of("--output", target, "--build_info_file", "my.properties"));

    FakeZipFile expectedResult = new FakeZipFile()
        .addEntry("META-INF/", EXTRA_FOR_META_INF, false)
        .addEntry(JarFile.MANIFEST_NAME,
            new ManifestValidator("Manifest-Version: 1.0", "Created-By: blaze-singlejar"), false)
        .addEntry("build-data.properties", new BuildInfoValidator(buildInfo),
            false);
    expectedResult.assertSame(mockFs.toByteArray());
  }

  private static String makePropertyFileFromBuildInfo(List<String> buildInfo) {
    return LINEFEED_JOINER.join(buildInfo).replace(":", "\\:");
  }

  @Test
  public void testAddBuildInfoPropertiesFiles() throws IOException {
    MockSimpleFileSystem mockFs = new MockSimpleFileSystem("output.jar");
    SingleJar singleJar = new SingleJar(mockFs);
    doTestAddBuildInfoPropertiesFiles(mockFs, "output.jar", singleJar);
  }

  public static void doTestAddBuildInfoPropertiesFiles(MockSimpleFileSystem mockFs, String target,
      SingleJar singleJar) throws IOException {
    List<String> buildInfo = getBuildInfo();

    mockFs.addFile("my1.properties", makePropertyFileFromBuildInfo(buildInfo.subList(0, 4)));
    mockFs.addFile("my2.properties",
        makePropertyFileFromBuildInfo(buildInfo.subList(4, buildInfo.size())));
    singleJar.run(ImmutableList.of("--output", target,
        "--build_info_file", "my1.properties",
        "--build_info_file", "my2.properties"));

    FakeZipFile expectedResult = new FakeZipFile()
        .addEntry("META-INF/", EXTRA_FOR_META_INF, false)
        .addEntry(JarFile.MANIFEST_NAME,
            new ManifestValidator("Manifest-Version: 1.0", "Created-By: blaze-singlejar"), false)
        .addEntry("build-data.properties", new BuildInfoValidator(buildInfo),
            false);
    expectedResult.assertSame(mockFs.toByteArray());
  }

  @Test
  public void testAddBuildInfoPropertiesAndFiles() throws IOException {
    MockSimpleFileSystem mockFs = new MockSimpleFileSystem("output.jar");
    SingleJar singleJar = new SingleJar(mockFs);
    doTestAddBuildInfoPropertiesAndFiles(mockFs, "output.jar", singleJar);
  }

  public static void doTestAddBuildInfoPropertiesAndFiles(MockSimpleFileSystem mockFs,
      String target, SingleJar singleJar) throws IOException {
    List<String> buildInfo = getBuildInfo();

    mockFs.addFile("my1.properties", makePropertyFileFromBuildInfo(buildInfo.subList(0, 4)));
    mockFs.addFile("my2.properties", makePropertyFileFromBuildInfo(
        buildInfo.subList(4, buildInfo.size())));
    List<String> args = ImmutableList.<String>builder()
        .add("--output").add(target)
        .add("--build_info_file").add("my1.properties")
        .add("--build_info_file").add("my2.properties")
        .addAll(infoPropertyArguments(buildInfo.subList(4, buildInfo.size())))
        .build();

    singleJar.run(args);
    FakeZipFile expectedResult = new FakeZipFile()
        .addEntry("META-INF/", EXTRA_FOR_META_INF, false)
        .addEntry(JarFile.MANIFEST_NAME,
            new ManifestValidator("Manifest-Version: 1.0", "Created-By: blaze-singlejar"), false)
        .addEntry("build-data.properties", new BuildInfoValidator(buildInfo),
            false);
    expectedResult.assertSame(mockFs.toByteArray());
  }


  @Test
  public void testExcludeBuildData() throws IOException {
    MockSimpleFileSystem mockFs = new MockSimpleFileSystem("output.jar");
    SingleJar singleJar = new SingleJar(mockFs);
    doTestExcludeBuildData(mockFs, "output.jar", singleJar);
  }

  public static void doTestExcludeBuildData(MockSimpleFileSystem mockFs, String target,
      SingleJar singleJar) throws IOException {
    singleJar.run(ImmutableList.of("--output", target, "--exclude_build_data"));
    FakeZipFile expectedResult = new FakeZipFile()
        .addEntry("META-INF/", EXTRA_FOR_META_INF)
        .addEntry(JarFile.MANIFEST_NAME, new ManifestValidator(
            "Manifest-Version: 1.0",
            "Created-By: blaze-singlejar"));
    expectedResult.assertSame(mockFs.toByteArray());
  }

  @Test
  public void testResourceMapping() throws IOException {
    MockSimpleFileSystem mockFs = new MockSimpleFileSystem("output.jar");
    mockFs.addFile("a/b/c", "Test");
    SingleJar singleJar = new SingleJar(mockFs);
    singleJar.run(ImmutableList.of("--output", "output.jar", "--exclude_build_data",
        "--resources", "a/b/c:c/b/a"));
    FakeZipFile expectedResult = new FakeZipFile()
        .addEntry("META-INF/", EXTRA_FOR_META_INF)
        .addEntry(JarFile.MANIFEST_NAME, new ManifestValidator(
            "Manifest-Version: 1.0",
            "Created-By: blaze-singlejar"))
        .addEntry("c/", (String) null)
        .addEntry("c/b/", (String) null)
        .addEntry("c/b/a", "Test");
    expectedResult.assertSame(mockFs.toByteArray());
  }

  @Test
  public void testResourceMappingIdentity() throws IOException {
    MockSimpleFileSystem mockFs = new MockSimpleFileSystem("output.jar");
    mockFs.addFile("a/b/c", "Test");
    SingleJar singleJar = new SingleJar(mockFs);
    singleJar.run(ImmutableList.of("--output", "output.jar", "--exclude_build_data",
        "--resources", "a/b/c"));
    FakeZipFile expectedResult =
        new FakeZipFile()
            .addEntry("META-INF/", EXTRA_FOR_META_INF)
            .addEntry(
                JarFile.MANIFEST_NAME,
                new ManifestValidator("Manifest-Version: 1.0", "Created-By: blaze-singlejar"))
            .addEntry("a/", (String) null)
            .addEntry("a/b/", (String) null)
            .addEntry("a/b/c", "Test");
    expectedResult.assertSame(mockFs.toByteArray());
  }

  @Test
  public void testResourceMappingDuplicateError() throws IOException {
    MockSimpleFileSystem mockFs = new MockSimpleFileSystem("output.jar");
    mockFs.addFile("a/b/c", "Test");
    SingleJar singleJar = new SingleJar(mockFs);
    try {
      singleJar.run(ImmutableList.of("--output", "output.jar", "--exclude_build_data",
          "--resources", "a/b/c", "a/b/c"));
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e.getMessage()).contains("already contains a file named 'a/b/c'.");
    }
  }

  @Test
  public void testResourceMappingDuplicateWarning() throws IOException {
    MockSimpleFileSystem mockFs = new MockSimpleFileSystem("output.jar");
    mockFs.addFile("a/b/c", "Test");
    SingleJar singleJar = new SingleJar(mockFs);
    singleJar.run(ImmutableList.of("--output", "output.jar", "--exclude_build_data",
        "--warn_duplicate_resources", "--resources", "a/b/c", "a/b/c"));
    FakeZipFile expectedResult =
        new FakeZipFile()
            .addEntry("META-INF/", EXTRA_FOR_META_INF)
            .addEntry(
                JarFile.MANIFEST_NAME,
                new ManifestValidator("Manifest-Version: 1.0", "Created-By: blaze-singlejar"))
            .addEntry("a/", (String) null)
            .addEntry("a/b/", (String) null)
            .addEntry("a/b/c", "Test");
    expectedResult.assertSame(mockFs.toByteArray());
  }

  @Test
  public void testCanAddPreamble() throws IOException {
    MockSimpleFileSystem mockFs = new MockSimpleFileSystem("output.jar");
    String preamble = "WeThePeople";
    mockFs.addFile(preamble, preamble.getBytes(UTF_8));
    SingleJar singleJar = new SingleJar(mockFs);
    singleJar.run(ImmutableList.of("--output", "output.jar",
        "--java_launcher", preamble,
        "--main_class", "SomeClass"));
    FakeZipFile expectedResult =
        new FakeZipFile()
            .addPreamble(preamble.getBytes(UTF_8))
            .addEntry("META-INF/", EXTRA_FOR_META_INF)
            .addEntry(
                JarFile.MANIFEST_NAME,
                new ManifestValidator("Manifest-Version: 1.0", "Created-By: blaze-singlejar",
                    "Main-Class: SomeClass"))
            .addEntry("build-data.properties", redactedBuildData("output.jar", "SomeClass"));
    expectedResult.assertSame(mockFs.toByteArray());
  }
}
