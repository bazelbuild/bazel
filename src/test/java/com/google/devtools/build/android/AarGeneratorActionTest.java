// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.hamcrest.CoreMatchers.containsString;
import static org.junit.Assert.assertNotNull;

import com.android.builder.core.VariantType;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.android.AarGeneratorAction.AarGeneratorOptions;
import com.google.devtools.build.zip.ZipReader;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link AarGeneratorAction}. */
@RunWith(JUnit4.class)
public class AarGeneratorActionTest {

  private static class AarData {
    /** Templates for resource files generation. */
    enum ResourceType {
      VALUE {
        @Override public String create(String... lines) {
          return String.format(
              "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n<resources>%s</resources>",
              Joiner.on("\n").join(lines));
        }
      },
      LAYOUT {
        @Override public String create(String... lines) {
          return String.format("<?xml version=\"1.0\" encoding=\"utf-8\"?>"
              + "<LinearLayout xmlns:android=\"http://schemas.android.com/apk/res/android\""
              + " android:layout_width=\"fill_parent\""
              + " android:layout_height=\"fill_parent\">%s</LinearLayout>",
              Joiner.on("\n").join(lines));
        }
      },
      UNFORMATTED {
        @Override public String create(String... lines) {
          return String.format(Joiner.on("\n").join(lines));
        }
      };

      public abstract String create(String... lines);
    }

    private static class Builder {

      private final Path root;
      private final Path assetDir;
      private final Path resourceDir;
      private Path manifest;
      private Path rtxt;
      private Path classes;
      private Map<Path, String> filesToWrite = new HashMap<>();
      private Map<String, String> classesToWrite = new HashMap<>();
      private ImmutableList.Builder<Path> proguardSpecs = ImmutableList.builder();
      private boolean withEmptyRes = false;
      private boolean withEmptyAssets = false;

      public Builder(Path root) {
        this(root, "res", "assets");
      }

      public Builder(Path root, String resourceRoot, String assetRoot) {
        this.root = root;
        assetDir = root.resolve(assetRoot);
        resourceDir = root.resolve(resourceRoot);
        manifest = root.resolve("fake-manifest-path");
        rtxt = root.resolve("fake-rtxt-path");
        classes = root.resolve("fake-classes-path");
      }

      public Builder addResource(String path, ResourceType template, String... lines) {
        filesToWrite.put(resourceDir.resolve(path), template.create(lines));
        return this;
      }

      public Builder withEmptyResources(boolean isEmpty) {
        this.withEmptyRes = isEmpty;
        return this;
      }

      public Builder addAsset(String path, String... lines) {
        filesToWrite.put(assetDir.resolve(path), Joiner.on("\n").join(lines));
        return this;
      }

      public Builder withEmptyAssets(boolean isEmpty) {
        this.withEmptyAssets = isEmpty;
        return this;
      }

      public Builder createManifest(String path, String manifestPackage, String... lines) {
        this.manifest = root.resolve(path);
        filesToWrite.put(manifest, String.format("<?xml version=\"1.0\" encoding=\"utf-8\"?>"
            + "<manifest xmlns:android='http://schemas.android.com/apk/res/android' package='%s'>"
            + "%s</manifest>", manifestPackage, Joiner.on("\n").join(lines)));
        return this;
      }

      public Builder createRtxt(String path, String... lines) {
        this.rtxt = root.resolve(path);
        filesToWrite.put(rtxt, String.format("%s", Joiner.on("\n").join(lines)));
        return this;
      }

      public Builder createClassesJar(String path) {
        this.classes = root.resolve(path);
        classesToWrite.put("META-INF/MANIFEST.MF", "Manifest-Version: 1.0\n");
        return this;
      }

      public Builder addClassesFile(String filePackage, String filename, String... lines) {
        classesToWrite.put(filePackage.replace(".", "/") + "/" + filename,
            String.format("%s", Joiner.on("\n").join(lines)));
        return this;
      }

      public Builder addProguardSpec(String path, String... lines) {
        Path proguardSpecPath = root.resolve(path);
        proguardSpecs.add(proguardSpecPath);
        filesToWrite.put(proguardSpecPath, String.format("%s", Joiner.on("\n").join(lines)));
        return this;
      }

      public AarData build() throws IOException {
        writeFiles();
        return new AarData(buildMerged(), manifest, rtxt, classes, proguardSpecs.build());
      }

      private MergedAndroidData buildMerged() {
        return new MergedAndroidData(
            resourceDir,
            assetDir,
            manifest);
      }

      private void writeFiles() throws IOException {
        assertNotNull("A manifest is required.", manifest);
        assertNotNull("A resource file is required.", rtxt);
        assertNotNull("A classes jar is required.", classes);
        if (withEmptyRes) {
          Files.createDirectories(resourceDir);
        }
        if (withEmptyAssets) {
          Files.createDirectories(assetDir);
        }
        for (Map.Entry<Path, String> entry : filesToWrite.entrySet()) {
          Path file = entry.getKey();
          // only write files in assets if assets has not been set to empty and same for resources
          if (!((file.startsWith(assetDir) && withEmptyAssets)
              || (file.startsWith(resourceDir) && withEmptyRes))) {
            Files.createDirectories(file.getParent());
            Files.write(file, entry.getValue().getBytes(StandardCharsets.UTF_8));
            assertThat(Files.exists(file)).isTrue();
          }
        }
        if (!classesToWrite.isEmpty()) {
          writeClassesJar();
        }
      }

      private void writeClassesJar() throws IOException {
        try (ZipOutputStream zout = new ZipOutputStream(Files.newOutputStream(classes))) {
          for (Map.Entry<String, String> file : classesToWrite.entrySet()) {
            ZipEntry entry = new ZipEntry(file.getKey());
            zout.putNextEntry(entry);
            zout.write(file.getValue().getBytes(UTF_8));
            zout.closeEntry();
          }
        }

        classes.toFile().setLastModified(AarGeneratorAction.DEFAULT_TIMESTAMP);
      }
    }

    final MergedAndroidData data;
    final Path manifest;
    final Path rtxt;
    final Path classes;
    final ImmutableList<Path> proguardSpecs;

    private AarData(
        MergedAndroidData data,
        Path manifest,
        Path rtxt,
        Path classes,
        ImmutableList<Path> proguardSpecs) {
      this.data = data;
      this.manifest = manifest;
      this.rtxt = rtxt;
      this.classes = classes;
      this.proguardSpecs = proguardSpecs;
    }
  }

  /**
   * Operation to perform on a file.
   */
  private interface FileOperation {
    /**
     * Performs the operation on a file, given its name, modificationTime and contents.
     */
    void perform(String name, long modificationTime, String contents);
  }

  /**
   * Runs a {@link FileOperation} on every entry in a zip file.
   *
   * @param zip {@link Path} of the zip file to traverse.
   * @param operation {@link FileOperation} to be run on every entry of the zip file.
   * @throws IOException if there is an error reading the zip file.
   */
  private void traverseZipFile(Path zip, FileOperation operation) throws IOException {
    ZipInputStream zis = new ZipInputStream(Files.newInputStream(zip));
    ZipEntry z = zis.getNextEntry();
    while (z != null) {
      ByteArrayOutputStream baos = new ByteArrayOutputStream();
      byte[] buffer = new byte[1024];
      for (int count = 0; count != -1; count = zis.read(buffer)) {
        baos.write(buffer);
      }
      // Replace Windows path separators so that test cases are consistent across platforms.
      String name = z.getName().replace('\\', '/');
      operation.perform(
          name, z.getTime(), new String(baos.toByteArray(), StandardCharsets.UTF_8));
      z = zis.getNextEntry();
    }
  }

  private Set<String> getZipEntries(Path zip) throws IOException {
    final Set<String> zipEntries = new HashSet<>();
    traverseZipFile(zip, new FileOperation() {
      @Override public void perform(String name, long modificationTime, String contents) {
        zipEntries.add(name);
      }
    });
    return zipEntries;
  }

  private Set<Long> getZipEntryTimestamps(Path zip) throws IOException {
    final Set<Long> timestamps = new HashSet<>();
    traverseZipFile(zip, new FileOperation() {
      @Override public void perform(String name, long modificationTime, String contents) {
        timestamps.add(modificationTime);
      }
    });
    return timestamps;
  }

  private Path tempDir;

  @Rule public ExpectedException thrown = ExpectedException.none();

  @Before public void setUp() throws IOException {
    tempDir = Files.createTempDirectory(toString());
    tempDir.toFile().deleteOnExit();

  }

  @Test public void testCheckFlags() throws IOException, OptionsParsingException {
    Path manifest = tempDir.resolve("AndroidManifest.xml");
    Files.createFile(manifest);
    Path rtxt = tempDir.resolve("R.txt");
    Files.createFile(rtxt);
    Path classes = tempDir.resolve("classes.jar");
    Files.createFile(classes);

    String[] args = new String[] {"--manifest", manifest.toString(), "--rtxt", rtxt.toString(),
        "--classes", classes.toString()};
    OptionsParser optionsParser = OptionsParser.newOptionsParser(AarGeneratorOptions.class);
    optionsParser.parse(args);
    AarGeneratorOptions options = optionsParser.getOptions(AarGeneratorOptions.class);
    AarGeneratorAction.checkFlags(options);
  }

  @Test public void testCheckFlags_MissingClasses() throws IOException, OptionsParsingException {
    Path manifest = tempDir.resolve("AndroidManifest.xml");
    Files.createFile(manifest);
    Path rtxt = tempDir.resolve("R.txt");
    Files.createFile(rtxt);

    String[] args = new String[] {"--manifest", manifest.toString(), "--rtxt", rtxt.toString()};
    OptionsParser optionsParser = OptionsParser.newOptionsParser(AarGeneratorOptions.class);
    optionsParser.parse(args);
    AarGeneratorOptions options = optionsParser.getOptions(AarGeneratorOptions.class);
    thrown.expect(IllegalArgumentException.class);
    thrown.expectMessage("classes must be specified. Building an .aar without"
          + " classes is unsupported.");
    AarGeneratorAction.checkFlags(options);
  }

  @Test public void testCheckFlags_MissingMultiple() throws IOException, OptionsParsingException {
    Path manifest = tempDir.resolve("AndroidManifest.xml");
    Files.createFile(manifest);
    String[] args = new String[] {"--manifest", manifest.toString()};
    OptionsParser optionsParser = OptionsParser.newOptionsParser(AarGeneratorOptions.class);
    optionsParser.parse(args);
    AarGeneratorOptions options = optionsParser.getOptions(AarGeneratorOptions.class);
    thrown.expect(IllegalArgumentException.class);
    thrown.expectMessage("rtxt, classes must be specified. Building an .aar without"
          + " rtxt, classes is unsupported.");
    AarGeneratorAction.checkFlags(options);
  }

  @Test public void testWriteAar() throws Exception {
    Path aar = tempDir.resolve("foo.aar");
    AarData aarData = new AarData.Builder(tempDir.resolve("data"))
        .createManifest("AndroidManifest.xml", "com.google.android.apps.foo.d1", "")
        .createRtxt("R.txt",
            "int string app_name 0x7f050001",
            "int string hello_world 0x7f050002")
        .addResource("values/ids.xml",
            AarData.ResourceType.VALUE,
            "<item name=\"id_name\" type=\"id\"/>")
        .addAsset("some/other/ft/data.txt", "bar")
        .createClassesJar("classes.jar")
        .addClassesFile("com.google.android.apps.foo", "Test.class", "test file contents")
        .build();

    AarGeneratorAction.writeAar(aar,
        aarData.data,
        aarData.manifest,
        aarData.rtxt,
        aarData.classes,
        aarData.proguardSpecs);
  }

  @Test public void testWriteAar_DefaultTimestamps() throws Exception {
    Path aar = tempDir.resolve("foo.aar");
    AarData aarData = new AarData.Builder(tempDir.resolve("data"))
        .createManifest("AndroidManifest.xml", "com.google.android.apps.foo.d1", "")
        .createRtxt("R.txt",
            "int string app_name 0x7f050001",
            "int string hello_world 0x7f050002")
        .addResource("values/ids.xml",
            AarData.ResourceType.VALUE,
            "<item name=\"id_name\" type=\"id\"/>")
        .addAsset("some/other/ft/data.txt", "bar")
        .createClassesJar("classes.jar")
        .addClassesFile("com.google.android.apps.foo", "Test.class", "test file contents")
        .build();

    AarGeneratorAction.writeAar(aar,
        aarData.data,
        aarData.manifest,
        aarData.rtxt,
        aarData.classes,
        aarData.proguardSpecs);

    assertThat(getZipEntryTimestamps(aar)).containsExactly(AarGeneratorAction.DEFAULT_TIMESTAMP);
    assertThat(aar.toFile().lastModified()).isEqualTo(AarGeneratorAction.DEFAULT_TIMESTAMP);
  }

  @Test public void testAssetResourceSubdirs() throws Exception {
    Path aar = tempDir.resolve("foo.aar");
    AarData aarData = new AarData.Builder(tempDir.resolve("data"), "xyz", "assets")
        .createManifest("AndroidManifest.xml", "com.google.android.apps.foo.d1", "")
        .createRtxt("R.txt",
            "int string app_name 0x7f050001",
            "int string hello_world 0x7f050002")
        .addResource("values/ids.xml",
            AarData.ResourceType.VALUE,
            "<item name=\"id_name\" type=\"id\"/>")
        .addAsset("some/other/ft/data.txt", "bar")
        .createClassesJar("classes.jar")
        .addClassesFile("com.google.android.apps.foo", "Test.class", "test file contents")
        .build();

    AarGeneratorAction.writeAar(aar,
        aarData.data,
        aarData.manifest,
        aarData.rtxt,
        aarData.classes,
        aarData.proguardSpecs);

    // verify aar archive
    Set<String> zipEntries = getZipEntries(aar);
    assertThat(zipEntries).contains("res/");
    assertThat(zipEntries).contains("assets/");
  }

  @Test public void testMissingManifest() throws Exception {
    Path aar = tempDir.resolve("foo.aar");
    AarData aarData = new AarData.Builder(tempDir.resolve("data"))
        .createRtxt("R.txt",
            "int string app_name 0x7f050001",
            "int string hello_world 0x7f050002")
        .addResource("values/ids.xml",
            AarData.ResourceType.VALUE,
            "<item name=\"id_name\" type=\"id\"/>")
        .addAsset("some/other/ft/data.txt", "bar")
        .createClassesJar("classes.jar")
        .addClassesFile("com.google.android.apps.foo", "Test.class", "test file contents")
        .build();

    thrown.expect(IOException.class);
    thrown.expectMessage(containsString("fake-manifest-path"));
    AarGeneratorAction.writeAar(aar,
        aarData.data,
        aarData.manifest,
        aarData.rtxt,
        aarData.classes,
        aarData.proguardSpecs);
  }

  @Test public void testMissingRtxt() throws Exception {
    Path aar = tempDir.resolve("foo.aar");
    AarData aarData = new AarData.Builder(tempDir.resolve("data"))
        .createManifest("AndroidManifest.xml", "com.google.android.apps.foo.d1", "")
        .addResource("values/ids.xml",
            AarData.ResourceType.VALUE,
            "<item name=\"id_name\" type=\"id\"/>")
        .addAsset("some/other/ft/data.txt", "bar")
        .createClassesJar("classes.jar")
        .addClassesFile("com.google.android.apps.foo", "Test.class", "test file contents")
        .build();

    thrown.expect(IOException.class);
    thrown.expectMessage(containsString("fake-rtxt-path"));
    AarGeneratorAction.writeAar(aar,
        aarData.data,
        aarData.manifest,
        aarData.rtxt,
        aarData.classes,
        aarData.proguardSpecs);
  }

  @Test public void testMissingClasses() throws Exception {
    Path aar = tempDir.resolve("foo.aar");
    AarData aarData = new AarData.Builder(tempDir.resolve("data"))
        .createManifest("AndroidManifest.xml", "com.google.android.apps.foo.d1", "")
        .createRtxt("R.txt",
            "int string app_name 0x7f050001",
            "int string hello_world 0x7f050002")
        .addResource("values/ids.xml",
            AarData.ResourceType.VALUE,
            "<item name=\"id_name\" type=\"id\"/>")
        .addAsset("some/other/ft/data.txt", "bar")
        .build();

    thrown.expect(IOException.class);
    thrown.expectMessage(containsString("fake-classes-path"));
    AarGeneratorAction.writeAar(aar,
        aarData.data,
        aarData.manifest,
        aarData.rtxt,
        aarData.classes,
        aarData.proguardSpecs);
  }

  @Test public void testMissingResources() throws Exception {
    Path aar = tempDir.resolve("foo.aar");
    AarData aarData = new AarData.Builder(tempDir.resolve("data"))
        .createManifest("AndroidManifest.xml", "com.google.android.apps.foo.d1", "")
        .createRtxt("R.txt",
            "int string app_name 0x7f050001",
            "int string hello_world 0x7f050002")
        .addAsset("some/other/ft/data.txt", "bar")
        .createClassesJar("classes.jar")
        .addClassesFile("com.google.android.apps.foo", "Test.class", "test file contents")
        .build();

    thrown.expect(IOException.class);
    thrown.expectMessage(containsString("res"));
    AarGeneratorAction.writeAar(aar,
        aarData.data,
        aarData.manifest,
        aarData.rtxt,
        aarData.classes,
        aarData.proguardSpecs);
  }

  @Test public void testEmptyResources() throws Exception {
    Path aar = tempDir.resolve("foo.aar");
    AarData aarData = new AarData.Builder(tempDir.resolve("data"))
        .createManifest("AndroidManifest.xml", "com.google.android.apps.foo.d1", "")
        .createRtxt("R.txt",
            "int string app_name 0x7f050001",
            "int string hello_world 0x7f050002")
        .withEmptyResources(true)
        .addResource("values/ids.xml",
            AarData.ResourceType.VALUE,
            "<item name=\"id_name\" type=\"id\"/>")
        .addAsset("some/other/ft/data.txt", "bar")
        .createClassesJar("classes.jar")
        .addClassesFile("com.google.android.apps.foo", "Test.class", "test file contents")
        .build();

    AarGeneratorAction.writeAar(aar,
        aarData.data,
        aarData.manifest,
        aarData.rtxt,
        aarData.classes,
        aarData.proguardSpecs);
  }

  @Test public void testMissingAssets() throws Exception {
    Path aar = tempDir.resolve("foo.aar");
    AarData aarData = new AarData.Builder(tempDir.resolve("data"))
        .createManifest("AndroidManifest.xml", "com.google.android.apps.foo.d1", "")
        .createRtxt("R.txt",
            "int string app_name 0x7f050001",
            "int string hello_world 0x7f050002")
        .addResource("values/ids.xml",
            AarData.ResourceType.VALUE,
            "<item name=\"id_name\" type=\"id\"/>")
        .createClassesJar("classes.jar")
        .addClassesFile("com.google.android.apps.foo", "Test.class", "test file contents")
        .build();

    AarGeneratorAction.writeAar(aar,
        aarData.data,
        aarData.manifest,
        aarData.rtxt,
        aarData.classes,
        aarData.proguardSpecs);
  }

  @Test public void testEmptyAssets() throws Exception {
    Path aar = tempDir.resolve("foo.aar");
    AarData aarData = new AarData.Builder(tempDir.resolve("data"))
        .createManifest("AndroidManifest.xml", "com.google.android.apps.foo.d1", "")
        .createRtxt("R.txt",
            "int string app_name 0x7f050001",
            "int string hello_world 0x7f050002")
        .addResource("values/ids.xml",
            AarData.ResourceType.VALUE,
            "<item name=\"id_name\" type=\"id\"/>")
        .withEmptyAssets(true)
        .createClassesJar("classes.jar")
        .addClassesFile("com.google.android.apps.foo", "Test.class", "test file contents")
        .build();

    AarGeneratorAction.writeAar(aar,
        aarData.data,
        aarData.manifest,
        aarData.rtxt,
        aarData.classes,
        aarData.proguardSpecs);
  }

  @Test public void testFullIntegration() throws Exception {
    Path aar = tempDir.resolve("foo.aar");
    AarData aarData = new AarData.Builder(tempDir.resolve("data"))
        .createManifest("AndroidManifest.xml", "com.google.android.apps.foo", "")
        .createRtxt("R.txt",
            "int string app_name 0x7f050001",
            "int string hello_world 0x7f050002")
        .addResource("values/ids.xml",
            AarData.ResourceType.VALUE,
            "<item name=\"id\" type=\"id\"/>")
        .addResource("layout/layout.xml",
            AarData.ResourceType.LAYOUT,
            "<TextView android:id=\"@+id/text2\""
                + " android:layout_width=\"wrap_content\""
                + " android:layout_height=\"wrap_content\""
                + " android:text=\"Hello, I am a TextView\" />")
        .addAsset("some/other/ft/data.txt",
            "foo")
        .createClassesJar("classes.jar")
        .addClassesFile("com.google.android.apps.foo", "Test.class", "test file contents")
        .build();

    MergedAndroidData md1 = new AarData.Builder(tempDir.resolve("d1"))
        .addResource("values/ids.xml",
            AarData.ResourceType.VALUE,
            "<item name=\"id\" type=\"id\"/>")
        .addResource("layout/foo.xml",
            AarData.ResourceType.LAYOUT,
            "<TextView android:id=\"@+id/text\""
                + " android:layout_width=\"wrap_content\""
                + " android:layout_height=\"wrap_content\""
                + " android:text=\"Hello, I am a TextView\" />")
        .addAsset("some/other/ft/data1.txt",
            "bar")
        .createManifest("AndroidManifest.xml", "com.google.android.apps.foo.d1", "")
        .build().data;

    MergedAndroidData md2 = new AarData.Builder(tempDir.resolve("d2"))
        .addResource("values/ids.xml",
            AarData.ResourceType.VALUE,
            "<item name=\"id2\" type=\"id\"/>")
        .addResource("layout/bar.xml",
            AarData.ResourceType.LAYOUT,
            "<TextView android:id=\"@+id/textbar\""
                + " android:layout_width=\"wrap_content\""
                + " android:layout_height=\"wrap_content\""
                + " android:text=\"Hello, I am a TextView\" />")
        .addResource("drawable-mdpi/icon.png",
            AarData.ResourceType.UNFORMATTED,
            "Thttpt.")
        .addResource("drawable-xxxhdpi/icon.png",
            AarData.ResourceType.UNFORMATTED,
            "Double Thttpt.")
        .addAsset("some/other/ft/data2.txt",
            "foo")
        .createManifest("AndroidManifest.xml", "com.google.android.apps.foo.d2", "")
        .build().data;

    UnvalidatedAndroidData primary = new UnvalidatedAndroidData(
        ImmutableList.of(aarData.data.getResourceDir()),
        ImmutableList.of(aarData.data.getAssetDir()),
        aarData.data.getManifest());

    DependencyAndroidData d1 =
        new DependencyAndroidData(
            ImmutableList.of(md1.getResourceDir()),
            ImmutableList.of(md1.getAssetDir()),
            md1.getManifest(),
            null,
            null,
            null);

    DependencyAndroidData d2 =
        new DependencyAndroidData(
            ImmutableList.of(md2.getResourceDir()),
            ImmutableList.of(md2.getAssetDir()),
            md2.getManifest(),
            null,
            null,
            null);

    Path working = tempDir;

    Path resourcesOut = working.resolve("resources");
    Path assetsOut = working.resolve("assets");

    MergedAndroidData mergedData =
        AndroidResourceMerger.mergeDataAndWrite(
            primary,
            ImmutableList.of(d1, d2),
            ImmutableList.<DependencyAndroidData>of(),
            resourcesOut,
            assetsOut,
            null,
            VariantType.LIBRARY,
            null,
            /* filteredResources= */ ImmutableList.of(),
            true);

    AarGeneratorAction.writeAar(
        aar, mergedData, aarData.manifest, aarData.rtxt, aarData.classes, aarData.proguardSpecs);

    // verify aar archive
    Set<String> zipEntries = getZipEntries(aar);
    assertThat(zipEntries).containsExactly(
        "AndroidManifest.xml",
        "R.txt",
        "classes.jar",
        "res/",
        "res/values/",
        "res/values/values.xml",
        "res/layout/",
        "res/layout/layout.xml",
        "res/layout/foo.xml",
        "res/layout/bar.xml",
        "res/drawable-mdpi-v4/",
        "res/drawable-mdpi-v4/icon.png",
        "res/drawable-xxxhdpi-v4/",
        "res/drawable-xxxhdpi-v4/icon.png",
        "assets/",
        "assets/some/",
        "assets/some/other/",
        "assets/some/other/ft/",
        "assets/some/other/ft/data.txt",
        "assets/some/other/ft/data1.txt",
        "assets/some/other/ft/data2.txt");
  }

  @Test public void testProguardSpecs() throws Exception {
    Path aar = tempDir.resolve("foo.aar");
    AarData aarData =
        new AarData.Builder(tempDir.resolve("data"))
            .createManifest("AndroidManifest.xml", "com.google.android.apps.foo.d1", "")
            .createRtxt("R.txt", "")
            .withEmptyResources(true)
            .withEmptyAssets(true)
            .createClassesJar("classes.jar")
            .addProguardSpec("spec1", "foo", "bar")
            .addProguardSpec("spec2", "baz")
            .build();

    AarGeneratorAction.writeAar(
        aar,
        aarData.data,
        aarData.manifest,
        aarData.rtxt,
        aarData.classes,
        aarData.proguardSpecs);
    Set<String> zipEntries = getZipEntries(aar);
    assertThat(zipEntries).contains("proguard.txt");
    List<String> proguardTxtContents = null;
    try (ZipReader aarReader = new ZipReader(aar.toFile())) {
      try (BufferedReader entryReader =
          new BufferedReader(
              new InputStreamReader(
                  aarReader.getInputStream(aarReader.getEntry("proguard.txt")),
                  StandardCharsets.UTF_8))) {
        proguardTxtContents = entryReader.lines().collect(Collectors.toList());
      }
    }
    assertThat(proguardTxtContents).containsExactly("foo", "bar", "baz").inOrder();
  }
}
