// Copyright 2017 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth.assertAbout;
import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.jimfs.Jimfs;
import com.google.common.truth.Subject;
import com.google.devtools.build.android.resources.JavaIdentifierValidator.InvalidJavaIdentifier;
import java.io.IOException;
import java.nio.file.FileSystem;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link AndroidResourceClassWriter}. */
@RunWith(JUnit4.class)
public class AndroidResourceClassWriterTest {

  private FileSystem fs;

  @Rule public final ExpectedException thrown = ExpectedException.none();

  private static final AndroidFrameworkAttrIdProvider mockAndroidFrameworkIds =
      new MockAndroidFrameworkAttrIdProvider(ImmutableMap.<String, Integer>of());

  @Before
  public void createCleanEnvironment() {
    fs = Jimfs.newFileSystem();
  }

  @Test
  public void simpleIdFromLayout() throws Exception {
    Path target = fs.getPath("target");
    Path source = fs.getPath("source");
    AndroidResourceClassWriter resourceClassWriter =
        AndroidResourceClassWriter.of(mockAndroidFrameworkIds, target, "com.carroll.lewis");
    ParsedAndroidData direct =
        AndroidDataBuilder.of(source)
            .addResource(
                "layout/some_layout.xml",
                AndroidDataBuilder.ResourceType.LAYOUT,
                "<TextView android:id=\"@+id/HelloView\"",
                "          android:text=\"Hello World!\"",
                "          android:layout_width=\"wrap_content\"",
                "          android:layout_height=\"wrap_content\" />",
                "<Button android:id=\"@+id/AdiosButton\"",
                "        android:text=\"Adios!\"",
                "        android:layout_width=\"wrap_content\"",
                "        android:layout_height=\"wrap_content\" />")
            .createManifest("AndroidManifest.xml", "com.carroll.lewis", "")
            .buildParsed();
    UnwrittenMergedAndroidData unwrittenMergedAndroidData =
        UnwrittenMergedAndroidData.of(
            source.resolve("AndroidManifest.xml"), direct, ParsedAndroidDataBuilder.empty());
    unwrittenMergedAndroidData.writeResourceClass(resourceClassWriter);

    assertAbout(paths)
        .that(target.resolve("com/carroll/lewis/R.java"))
        .javaContentsIsEqualTo(
            "package com.carroll.lewis;",
            "public final class R {",
            "public static final class id {",
            "public static int AdiosButton = 0x7f030000;",
            "public static int HelloView = 0x7f030001;",
            "}",
            "public static final class layout {",
            "public static int some_layout = 0x7f020000;",
            "}",
            "}");
    assertAbout(paths)
        .that(target)
        .withClass("com.carroll.lewis.R$id")
        .classContentsIsEqualTo(
            ImmutableMap.of(
                "AdiosButton", 0x7f030000,
                "HelloView", 0x7f030001),
            ImmutableMap.<String, List<Integer>>of(),
            false);
    assertAbout(paths)
        .that(target)
        .withClass("com.carroll.lewis.R$layout")
        .classContentsIsEqualTo(
            ImmutableMap.of("some_layout", 0x7f020000),
            ImmutableMap.<String, List<Integer>>of(),
            false);
  }

  @Test
  public void ninePatchFieldNames() throws Exception {
    Path target = fs.getPath("target");
    Path source = fs.getPath("source");
    String drawable = "drawable/light.png";
    String ninePatch = "drawable/patchface.9.png";
    AndroidResourceClassWriter resourceClassWriter =
        AndroidResourceClassWriter.of(mockAndroidFrameworkIds, target, "com.boop");
    ParsedAndroidData direct =
        AndroidDataBuilder.of(source)
            .addResourceBinary(drawable, Files.createFile(fs.getPath("lightbringer.png")))
            .addResourceBinary(ninePatch, Files.createFile(fs.getPath("patchface.9.png")))
            .createManifest("AndroidManifest.xml", "com.boop", "")
            .buildParsed();
    UnwrittenMergedAndroidData unwrittenMergedAndroidData =
        UnwrittenMergedAndroidData.of(
            source.resolve("AndroidManifest.xml"), direct, ParsedAndroidDataBuilder.empty());
    unwrittenMergedAndroidData.writeResourceClass(resourceClassWriter);
    assertAbout(paths)
        .that(target.resolve("com/boop/R.java"))
        .javaContentsIsEqualTo(
            "package com.boop;",
            "public final class R {",
            "public static final class drawable {",
            "public static int light = 0x7f020000;",
            "public static int patchface = 0x7f020001;",
            "}",
            "}");
    assertAbout(paths)
        .that(target)
        .withClass("com.boop.R$drawable")
        .classContentsIsEqualTo(
            ImmutableMap.of(
                "light", 0x7f020000,
                "patchface", 0x7f020001),
            ImmutableMap.<String, List<Integer>>of(),
            false);
  }

  @Test
  public void unionOfResourcesInConfigurations() throws Exception {
    // See what happens if there are some configuration specific resources
    // (selection guarded by checks at runtime?).
    Path target = fs.getPath("target");
    Path source = fs.getPath("source");
    String drawable = "drawable/light.png";
    String drawableV18 = "drawable-v18/light18.png";
    String drawableV19 = "drawable-xxhdpi-v19/light19.png";
    String drawableV20 = "drawable-ldltr-v20/light20.png";
    Path stubImage = Files.createFile(fs.getPath("stub.png"));
    AndroidResourceClassWriter resourceClassWriter =
        AndroidResourceClassWriter.of(mockAndroidFrameworkIds, target, "com.boop");
    ParsedAndroidData direct =
        AndroidDataBuilder.of(source)
            .addResourceBinary(drawable, stubImage)
            .addResourceBinary(drawableV18, stubImage)
            .addResourceBinary(drawableV19, stubImage)
            .addResourceBinary(drawableV20, stubImage)
            .createManifest("AndroidManifest.xml", "com.boop", "")
            .buildParsed();
    UnwrittenMergedAndroidData unwrittenMergedAndroidData =
        UnwrittenMergedAndroidData.of(
            source.resolve("AndroidManifest.xml"), direct, ParsedAndroidDataBuilder.empty());
    unwrittenMergedAndroidData.writeResourceClass(resourceClassWriter);
    assertAbout(paths)
        .that(target.resolve("com/boop/R.java"))
        .javaContentsIsEqualTo(
            "package com.boop;",
            "public final class R {",
            "public static final class drawable {",
            "public static int light = 0x7f020000;",
            "public static int light18 = 0x7f020001;",
            "public static int light19 = 0x7f020002;",
            "public static int light20 = 0x7f020003;",
            "}",
            "}");
    assertAbout(paths)
        .that(target)
        .withClass("com.boop.R$drawable")
        .classContentsIsEqualTo(
            ImmutableMap.of(
                "light", 0x7f020000,
                "light18", 0x7f020001,
                "light19", 0x7f020002,
                "light20", 0x7f020003),
            ImmutableMap.<String, List<Integer>>of(),
            false);
  }

  @Test
  public void normalizeStyleAndStyleableNames() throws Exception {
    // Style and Styleables can have dots in the name. In order for it to be a legal Java
    // identifier, the dots are converted to underscore.
    Path target = fs.getPath("target");
    Path source = fs.getPath("source");
    Path transitive = fs.getPath("transitive");

    AndroidResourceClassWriter resourceClassWriter =
        AndroidResourceClassWriter.of(mockAndroidFrameworkIds, target, "com.carroll.lewis");
    ParsedAndroidData direct =
        AndroidDataBuilder.of(source)
            .addResource(
                "values/attr.xml",
                AndroidDataBuilder.ResourceType.VALUE,
                "<attr name=\"y_color\" format=\"color\" />",
                "<attr name=\"z_color\" format=\"color\" />")
            .addResource(
                "values/style.xml",
                AndroidDataBuilder.ResourceType.VALUE,
                "<style name=\"YStyle\">",
                "  <item name=\"y_color\">#FF00FF00</item>",
                "</style>",
                "<style name=\"ZStyle.ABC\" parent=\"YStyle\">",
                "  <item name=\"z_color\">#00FFFF00</item>",
                "</style>")
            .addResource(
                "values/styleable.xml",
                AndroidDataBuilder.ResourceType.VALUE,
                "<declare-styleable name=\"com.google.android.Dots\">",
                "  <attr name=\"y_color\"/>",
                "  <attr name=\"z_color\"/>",
                "  <attr name=\"x_color\"/>",
                "</declare-styleable>")
            .createManifest("AndroidManifest.xml", "com.carroll.lewis", "")
            .buildParsed();

    ParsedAndroidData transitiveDep =
        AndroidDataBuilder.of(transitive)
            .addResource(
                "values/attr.xml",
                AndroidDataBuilder.ResourceType.VALUE,
                "<attr name=\"x_color\" format=\"color\" />")
            .addResource(
                "values/styleable.xml",
                AndroidDataBuilder.ResourceType.VALUE,
                "<declare-styleable name=\"com.google.android.Swirls.Fancy\">",
                "  <attr name=\"z_color\"/>",
                "  <attr name=\"x_color\"/>",
                "  <attr name=\"y_color\"/>",
                "</declare-styleable>")
            .createManifest("AndroidManifest.xml", "com.library", "")
            .buildParsed();

    UnwrittenMergedAndroidData unwrittenMergedAndroidData =
        UnwrittenMergedAndroidData.of(source.resolve("AndroidManifest.xml"), direct, transitiveDep);
    unwrittenMergedAndroidData.writeResourceClass(resourceClassWriter);

    assertAbout(paths)
        .that(target.resolve("com/carroll/lewis/R.java"))
        .javaContentsIsEqualTo(
            "package com.carroll.lewis;",
            "public final class R {",
            "public static final class attr {",
            "public static int x_color = 0x7f010000;",
            "public static int y_color = 0x7f010001;",
            "public static int z_color = 0x7f010002;",
            "}",
            "public static final class style {",
            "public static int YStyle = 0x7f020000;",
            "public static int ZStyle_ABC = 0x7f020001;",
            "}",
            "public static final class styleable {",
            "public static int[] com_google_android_Dots = { 0x7f010000, 0x7f010001, 0x7f010002 };",
            "public static int com_google_android_Dots_x_color = 0x0;",
            "public static int com_google_android_Dots_y_color = 0x1;",
            "public static int com_google_android_Dots_z_color = 0x2;",
            "public static int[] com_google_android_Swirls_Fancy ="
                + " { 0x7f010000, 0x7f010001, 0x7f010002 };",
            "public static int com_google_android_Swirls_Fancy_x_color = 0x0;",
            "public static int com_google_android_Swirls_Fancy_y_color = 0x1;",
            "public static int com_google_android_Swirls_Fancy_z_color = 0x2;",
            "}",
            "}");
    assertAbout(paths)
        .that(target)
        .withClass("com.carroll.lewis.R$attr")
        .classContentsIsEqualTo(
            ImmutableMap.of(
                "x_color", 0x7f010000,
                "y_color", 0x7f010001,
                "z_color", 0x7f010002),
            ImmutableMap.<String, List<Integer>>of(),
            false);
    assertAbout(paths)
        .that(target)
        .withClass("com.carroll.lewis.R$style")
        .classContentsIsEqualTo(
            ImmutableMap.of(
                "YStyle", 0x7f020000,
                "ZStyle_ABC", 0x7f020001),
            ImmutableMap.<String, List<Integer>>of(),
            false);
    assertAbout(paths)
        .that(target)
        .withClass("com.carroll.lewis.R$styleable")
        .classContentsIsEqualTo(
            ImmutableMap.<String, Integer>builder()
                .put("com_google_android_Dots_x_color", 0)
                .put("com_google_android_Dots_y_color", 1)
                .put("com_google_android_Dots_z_color", 2)
                .put("com_google_android_Swirls_Fancy_x_color", 0)
                .put("com_google_android_Swirls_Fancy_y_color", 1)
                .put("com_google_android_Swirls_Fancy_z_color", 2)
                .build(),
            ImmutableMap.<String, List<Integer>>of(
                "com_google_android_Dots",
                ImmutableList.of(0x7f010000, 0x7f010001, 0x7f010002),
                "com_google_android_Swirls_Fancy",
                ImmutableList.of(0x7f010000, 0x7f010001, 0x7f010002)),
            false);
  }

  @Test
  public void handleAndroidFrameworkAttributes() throws Exception {
    // Attributes in the styleable array need to be sorted by integer ID value, so android
    // framework attributes need to come before application attributes.
    Path target = fs.getPath("target");
    Path source = fs.getPath("source");
    AndroidResourceClassWriter resourceClassWriter =
        AndroidResourceClassWriter.of(
            new MockAndroidFrameworkAttrIdProvider(
                ImmutableMap.of(
                    "textColor", 0x01000000,
                    "textColorSecondary", 0x01000006,
                    "textSize", 0x01000010)),
            target,
            "com.carroll.lewis");
    ParsedAndroidData direct =
        AndroidDataBuilder.of(source)
            .addResource(
                "values/attr.xml",
                AndroidDataBuilder.ResourceType.VALUE,
                "<attr name=\"aaa\" format=\"boolean\" />",
                "<attr name=\"zzz\" format=\"boolean\" />")
            .addResource(
                "values/style.xml",
                AndroidDataBuilder.ResourceType.VALUE,
                "<style name=\"YStyle\">",
                "  <item name=\"android:textSize\">15sp</item>",
                "  <item name=\"android:textColor\">#ffffff</item>",
                "  <item name=\"android:textColorSecondary\">#ffffff</item>",
                "</style>")
            .addResource(
                "values/styleable.xml",
                AndroidDataBuilder.ResourceType.VALUE,
                "<declare-styleable name=\"com.google.android.Dots\">",
                "  <attr name=\"aaa\"/>",
                "  <attr name=\"zzz\"/>",
                // the android framework attr should be sorted first, even if it's alphabetically
                // after the "aaa" attribute.
                "  <attr name=\"android:textSize\"/>",
                "  <attr name=\"android:textColor\"/>",
                "</declare-styleable>")
            .createManifest("AndroidManifest.xml", "com.carroll.lewis", "")
            .buildParsed();
    UnwrittenMergedAndroidData unwrittenMergedAndroidData =
        UnwrittenMergedAndroidData.of(
            source.resolve("AndroidManifest.xml"), direct, ParsedAndroidDataBuilder.empty());
    unwrittenMergedAndroidData.writeResourceClass(resourceClassWriter);

    assertAbout(paths)
        .that(target.resolve("com/carroll/lewis/R.java"))
        .javaContentsIsEqualTo(
            "package com.carroll.lewis;",
            "public final class R {",
            "public static final class attr {",
            "public static int aaa = 0x7f010000;",
            "public static int zzz = 0x7f010001;",
            "}",
            "public static final class style {",
            "public static int YStyle = 0x7f020000;",
            "}",
            "public static final class styleable {",
            "public static int[] com_google_android_Dots = "
                + "{ 0x1000000, 0x1000010, 0x7f010000, 0x7f010001 };",
            "public static int com_google_android_Dots_android_textColor = 0x0;",
            "public static int com_google_android_Dots_android_textSize = 0x1;",
            "public static int com_google_android_Dots_aaa = 0x2;",
            "public static int com_google_android_Dots_zzz = 0x3;",
            "}",
            "}");
    assertAbout(paths)
        .that(target)
        .withClass("com.carroll.lewis.R$attr")
        .classContentsIsEqualTo(
            ImmutableMap.of(
                "aaa", 0x7f010000,
                "zzz", 0x7f010001),
            ImmutableMap.<String, List<Integer>>of(),
            false);
    assertAbout(paths)
        .that(target)
        .withClass("com.carroll.lewis.R$style")
        .classContentsIsEqualTo(
            ImmutableMap.of("YStyle", 0x7f020000), ImmutableMap.<String, List<Integer>>of(), false);
    assertAbout(paths)
        .that(target)
        .withClass("com.carroll.lewis.R$styleable")
        .classContentsIsEqualTo(
            ImmutableMap.of(
                "com_google_android_Dots_android_textColor", 0,
                "com_google_android_Dots_android_textSize", 1,
                "com_google_android_Dots_aaa", 2,
                "com_google_android_Dots_zzz", 3),
            ImmutableMap.<String, List<Integer>>of(
                "com_google_android_Dots",
                ImmutableList.of(0x01000000, 0x01000010, 0x7f010000, 0x7f010001)),
            false);
  }

  @Test
  public void missingFrameworkAttribute() throws Exception {
    Path target = fs.getPath("target");
    Path source = fs.getPath("source");
    AndroidResourceClassWriter resourceClassWriter =
        AndroidResourceClassWriter.of(
            new MockAndroidFrameworkAttrIdProvider(ImmutableMap.<String, Integer>of()),
            target,
            "com.carroll.lewis");
    ParsedAndroidData direct =
        AndroidDataBuilder.of(source)
            .addResource(
                "values/attr.xml",
                AndroidDataBuilder.ResourceType.VALUE,
                "<attr name=\"aaazzz\" format=\"boolean\" />")
            .addResource(
                "values/styleable.xml",
                AndroidDataBuilder.ResourceType.VALUE,
                "<declare-styleable name=\"com.google.android.Dots\">",
                "  <attr name=\"aaazzz\"/>",
                "  <attr name=\"android:aaazzz\"/>",
                "</declare-styleable>")
            .createManifest("AndroidManifest.xml", "com.carroll.lewis", "")
            .buildParsed();
    UnwrittenMergedAndroidData unwrittenMergedAndroidData =
        UnwrittenMergedAndroidData.of(
            source.resolve("AndroidManifest.xml"), direct, ParsedAndroidDataBuilder.empty());
    thrown.expect(IOException.class);
    thrown.expectMessage("Android attribute not found: aaazzz");
    unwrittenMergedAndroidData.writeResourceClass(resourceClassWriter);
  }

  @Test
  public void missingAppAttribute() throws Exception {
    Path target = fs.getPath("target");
    Path source = fs.getPath("source");
    AndroidResourceClassWriter resourceClassWriter =
        AndroidResourceClassWriter.of(
            new MockAndroidFrameworkAttrIdProvider(ImmutableMap.<String, Integer>of()),
            target,
            "com.carroll.lewis");
    ParsedAndroidData direct =
        AndroidDataBuilder.of(source)
            .addResource(
                "values/styleable.xml",
                AndroidDataBuilder.ResourceType.VALUE,
                "<declare-styleable name=\"com.google.android.Dots\">",
                "  <attr name=\"aaazzz\"/>",
                "</declare-styleable>")
            .createManifest("AndroidManifest.xml", "com.carroll.lewis", "")
            .buildParsed();
    UnwrittenMergedAndroidData unwrittenMergedAndroidData =
        UnwrittenMergedAndroidData.of(
            source.resolve("AndroidManifest.xml"), direct, ParsedAndroidDataBuilder.empty());
    thrown.expect(IOException.class);
    thrown.expectMessage("App attribute not found: aaazzz");
    unwrittenMergedAndroidData.writeResourceClass(resourceClassWriter);
  }

  /**
   * Test what happens if we try to create a field name that is not a valid Java identifier. Here,
   * we start the field name with a number, which is not legal according to {@link
   * Character#isJavaIdentifierStart}.
   *
   * <p>See: {@link com.android.ide.common.res2.FileResourceNameValidator}, and {@link
   * com.android.ide.common.res2.ValueResourceNameValidator}.
   *
   * <p>AAPT seems to miss out on checking this case (it only checks for [a-z0-9_.], but isn't
   * position-sensitive).
   */
  @Test
  public void illegalFileResFieldNamesStart() throws Exception {
    Path target = fs.getPath("target");
    Path source = fs.getPath("source");
    String drawable = "drawable/1.png";
    assertThat(Character.isJavaIdentifierStart('1')).isFalse();
    AndroidResourceClassWriter resourceClassWriter =
        AndroidResourceClassWriter.of(mockAndroidFrameworkIds, target, "com.boop");
    ParsedAndroidData direct =
        AndroidDataBuilder.of(source)
            .addResourceBinary(drawable, Files.createFile(fs.getPath("1.png")))
            .createManifest("AndroidManifest.xml", "com.boop", "")
            .buildParsed();
    UnwrittenMergedAndroidData unwrittenMergedAndroidData =
        UnwrittenMergedAndroidData.of(
            source.resolve("AndroidManifest.xml"), direct, ParsedAndroidDataBuilder.empty());
    assertThrows(
        InvalidJavaIdentifier.class,
        () -> unwrittenMergedAndroidData.writeResourceClass(resourceClassWriter));
  }

  interface CheckedRunnable {
    void run() throws Throwable;
  }

  static void assertThrows(Class<? extends Throwable> expectedException, CheckedRunnable test) {
    try {
      test.run();
    } catch (Throwable e) {
      assertThat(e.getClass()).isAssignableTo(expectedException);
    }
  }

  /**
   * Test embedding a character that doesn't satisfy Character#isJavaIdentifierPart. Do so in a file
   * resource. In this case, AAPT will actually complain, so we may not need to do earlier
   * validation.
   */
  @Test
  public void illegalFileResFieldNamesCharacters() throws Exception {
    Path target = fs.getPath("target");
    Path source = fs.getPath("source");
    String drawable = "drawable/c++.png";
    assertThat(Character.isJavaIdentifierStart('c')).isTrue();
    assertThat(Character.isJavaIdentifierPart('+')).isFalse();
    AndroidResourceClassWriter resourceClassWriter =
        AndroidResourceClassWriter.of(mockAndroidFrameworkIds, target, "com.boop");
    ParsedAndroidData direct =
        AndroidDataBuilder.of(source)
            .addResourceBinary(drawable, Files.createFile(fs.getPath("phone#.png")))
            .createManifest("AndroidManifest.xml", "com.boop", "")
            .buildParsed();
    UnwrittenMergedAndroidData unwrittenMergedAndroidData =
        UnwrittenMergedAndroidData.of(
            source.resolve("AndroidManifest.xml"), direct, ParsedAndroidDataBuilder.empty());
    assertThrows(
        InvalidJavaIdentifier.class,
        () -> unwrittenMergedAndroidData.writeResourceClass(resourceClassWriter));
  }

  /**
   * Test embedding a character that doesn't satisfy Character#isJavaIdentifierPart. Do so in a
   * value resource. This is a case that AAPT doesn't validate, so it may pass through to the java
   * compiler.
   */
  @Test
  public void illegalValueResFieldNamesCharacters() throws Exception {
    Path target = fs.getPath("target");
    Path source = fs.getPath("source");
    assertThat(Character.isJavaIdentifierStart('c')).isTrue();
    assertThat(Character.isJavaIdentifierPart('+')).isFalse();
    AndroidResourceClassWriter resourceClassWriter =
        AndroidResourceClassWriter.of(mockAndroidFrameworkIds, target, "com.boop");
    ParsedAndroidData direct =
        AndroidDataBuilder.of(source)
            .addResource(
                "values/integers.xml",
                AndroidDataBuilder.ResourceType.VALUE,
                "<integer name=\"c++\">0xd</integer>")
            .createManifest("AndroidManifest.xml", "com.boop", "")
            .buildParsed();
    UnwrittenMergedAndroidData unwrittenMergedAndroidData =
        UnwrittenMergedAndroidData.of(
            source.resolve("AndroidManifest.xml"), direct, ParsedAndroidDataBuilder.empty());

    assertThrows(
        InvalidJavaIdentifier.class,
        () -> unwrittenMergedAndroidData.writeResourceClass(resourceClassWriter));
  }

  private static class MockAndroidFrameworkAttrIdProvider
      implements AndroidFrameworkAttrIdProvider {

    private final Map<String, Integer> mapToUse;

    MockAndroidFrameworkAttrIdProvider(Map<String, Integer> mapToUse) {
      this.mapToUse = mapToUse;
    }

    @Override
    public int getAttrId(String fieldName) throws AttrLookupException {
      if (mapToUse.containsKey(fieldName)) {
        return mapToUse.get(fieldName);
      }
      throw new AttrLookupException("Android attribute not found: " + fieldName);
    }
  }

  private static final Subject.Factory<ClassPathsSubject, Path> paths = ClassPathsSubject::new;
}
