// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.resources;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;

import com.android.SdkConstants;
import com.google.common.base.Function;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.android.resources.JavaIdentifierValidator.InvalidJavaIdentifier;
import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.stream.IntStream;
import org.hamcrest.BaseMatcher;
import org.hamcrest.Description;
import org.hamcrest.Matcher;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RClassGenerator}. */
@RunWith(JUnit4.class)
public class RClassGeneratorTest {

  private Path temp;
  @Rule public final ExpectedException thrown = ExpectedException.none();

  @Before
  public void setUp() throws Exception {
    temp = Files.createTempDirectory(toString());
  }

  @Test
  public void plainInts() throws Exception {
    checkSimpleInts(true);
  }

  @Test
  public void nonFinalFields() throws Exception {
    checkSimpleInts(false);
  }

  @Test
  public void clinitOverflowIntFields() throws Exception {
    // Ensures that RClassGenerator avoids MethodTooLargeException when there are too many fields
    // to initialize int the <clinit> alone.
    int startResourceId = 0x7f010000;
    int numResourceFields = 15000;
    List<String> resourceFields = new ArrayList<>(numResourceFields);
    Map<String, Integer> resourceFieldValues = Maps.newHashMapWithExpectedSize(numResourceFields);
    for (int i = 0; i < numResourceFields; ++i) {
      String fieldName = "field" + i;
      int fieldValue = (startResourceId + i);
      resourceFields.add("int string " + fieldName + " " + fieldValue);
      resourceFieldValues.put(fieldName, fieldValue);
    }
    ResourceSymbols symbolValues = createSymbolFile("R.txt", resourceFields.toArray(new String[0]));
    Path out = temp.resolve("classes");
    Files.createDirectories(out);
    RClassGenerator writer =
        RClassGenerator.with(out, symbolValues.asInitializers(), /*finalFields=*/ false);

    writer.write("com.bar", symbolValues.asInitializers());

    Class<?> outerClass = checkTopLevelClass(out, "com.bar.R", "com.bar.R$string");
    checkInnerClass(
        out,
        "com.bar.R$string",
        outerClass,
        ImmutableMap.copyOf(resourceFieldValues),
        ImmutableMap.<String, List<Integer>>of(),
        /*areFieldsFinal=*/ false);
  }

  @Test
  public void clinitOverflowIntArrayFields() throws Exception {
    // Ensures that RClassGenerator avoids MethodTooLargeException when there are too many array
    // fields to initialize int the <clinit> alone.
    int numResourceFields = 15000;
    List<String> resourceFields = new ArrayList<>(numResourceFields);
    Map<String, List<Integer>> resourceFieldValues =
        Maps.newHashMapWithExpectedSize(numResourceFields);
    for (int i = 0; i < numResourceFields; ++i) {
      String fieldName = "ActionMenu" + i;
      resourceFields.add("int[] styleable " + fieldName + " { 1, 2, 3, 4 }");
      resourceFieldValues.put(fieldName, ImmutableList.of(1, 2, 3, 4));
    }
    ResourceSymbols symbolValues = createSymbolFile("R.txt", resourceFields.toArray(new String[0]));
    Path out = temp.resolve("classes");
    Files.createDirectories(out);
    RClassGenerator writer =
        RClassGenerator.with(out, symbolValues.asInitializers(), /*finalFields=*/ false);

    writer.write("com.bar", symbolValues.asInitializers());

    Class<?> outerClass = checkTopLevelClass(out, "com.bar.R", "com.bar.R$styleable");
    checkInnerClass(
        out,
        "com.bar.R$styleable",
        outerClass,
        ImmutableMap.<String, Integer>of(),
        ImmutableMap.copyOf(resourceFieldValues),
        /*areFieldsFinal=*/ false);
  }

  @Test
  public void arrayFieldTooBigToWrite() throws Exception {
    ImmutableList<String> arrayValue =
        IntStream.range(0, 10000).boxed().map(Object::toString).collect(toImmutableList());
    String fieldName = "ActionMenu";
    String resourceField =
        "int[] styleable " + fieldName + " { " + String.join(", ", arrayValue) + " }";
    ResourceSymbols symbolValues = createSymbolFile("R.txt", resourceField);
    Path out = temp.resolve("classes");
    Files.createDirectories(out);
    RClassGenerator writer =
        RClassGenerator.with(out, symbolValues.asInitializers(), /*finalFields=*/ false);

    thrown.expect(IllegalStateException.class);

    writer.write("com.bar", symbolValues.asInitializers());
  }

  private void checkSimpleInts(boolean finalFields) throws Exception {
    // R.txt with the real IDs after linking together libraries.
    ResourceSymbols symbolValues =
        createSymbolFile(
            "R.txt",
            "int attr agility 0x7f010000",
            "int attr dexterity 0x7f010001",
            "int drawable heart 0x7f020000",
            "int id someTextView 0x7f080000",
            "int integer maxNotifications 0x7f090000",
            "int string alphabet 0x7f100000",
            "int string ok 0x7f100001",
            // aapt2 link --package-id 0x80 produces IDs that are out of range of a java integer.
            "int string largePackageId 0x80001000");
    // R.txt for the library, where the values are not the final ones (so ignore them). We only use
    // this to keep the # of inner classes small (exactly the set needed by the library).
    ResourceSymbols symbolsInLibrary =
        createSymbolFile(
            "lib.R.txt",
            "int attr agility 0x1",
            "int id someTextView 0x1",
            "int string ok 0x1",
            "int string largePackageId 0x1");
    Path out = temp.resolve("classes");
    Files.createDirectories(out);
    RClassGenerator writer = RClassGenerator.with(out, symbolValues.asInitializers(), finalFields);
    writer.write("com.bar", symbolsInLibrary.asInitializers());

    Path packageDir = out.resolve("com/bar");
    checkFilesInPackage(packageDir, "R.class", "R$attr.class", "R$id.class", "R$string.class");
    Class<?> outerClass =
        checkTopLevelClass(out, "com.bar.R", "com.bar.R$attr", "com.bar.R$id", "com.bar.R$string");
    checkInnerClass(
        out,
        "com.bar.R$attr",
        outerClass,
        ImmutableMap.of("agility", 0x7f010000),
        ImmutableMap.<String, List<Integer>>of(),
        finalFields);
    checkInnerClass(
        out,
        "com.bar.R$id",
        outerClass,
        ImmutableMap.of("someTextView", 0x7f080000),
        ImmutableMap.<String, List<Integer>>of(),
        finalFields);
    checkInnerClass(
        out,
        "com.bar.R$string",
        outerClass,
        ImmutableMap.of("ok", 0x7f100001, "largePackageId", 0x80001000),
        ImmutableMap.<String, List<Integer>>of(),
        finalFields);
  }

  @Test
  public void checkFileWriteThrowsOnExisting() throws Exception {
    checkFileWriteThrowsOnExisting(SdkConstants.FN_COMPILED_RESOURCE_CLASS);
  }

  private void checkFileWriteThrowsOnExisting(String existingFile) throws Exception {
    ResourceSymbols symbolValues = createSymbolFile("R.txt", "int string ok 0x7f100001");
    ResourceSymbols symbolsInLibrary = createSymbolFile("lib.R.txt", "int string ok 0x1");

    Path out = temp.resolve("classes");
    String packageName = "com";
    Path packageFolder = out.resolve(packageName);
    Files.createDirectories(packageFolder);

    RClassGenerator writer = RClassGenerator.with(out, symbolValues.asInitializers(), false);
    Files.write(packageFolder.resolve(existingFile), new byte[0]);

    try {
      writer.write(packageName, symbolsInLibrary.asInitializers());
    } catch (FileAlreadyExistsException e) {
      return;
    }
    throw new Exception("Expected to throw a FileAlreadyExistsException");
  }

  @Test
  public void checkInnerFileWriteThrowsOnExisting() throws Exception {
    checkFileWriteThrowsOnExisting("R$string.class");
  }

  @Test
  public void emptyIntArrays() throws Exception {
    boolean finalFields = true;
    // Make sure we parse an empty array the way the R.txt writes it.
    ResourceSymbols symbolValues = createSymbolFile("R.txt", "int[] styleable ActionMenuView { }");
    ResourceSymbols symbolsInLibrary = symbolValues;
    Path out = temp.resolve("classes");
    Files.createDirectories(out);
    RClassGenerator writer = RClassGenerator.with(out, symbolValues.asInitializers(), finalFields);
    writer.write("com.testEmptyIntArray", symbolsInLibrary.asInitializers());

    Path packageDir = out.resolve("com/testEmptyIntArray");
    checkFilesInPackage(packageDir, "R.class", "R$styleable.class");
    Class<?> outerClass =
        checkTopLevelClass(out, "com.testEmptyIntArray.R", "com.testEmptyIntArray.R$styleable");
    checkInnerClass(
        out,
        "com.testEmptyIntArray.R$styleable",
        outerClass,
        ImmutableMap.<String, Integer>of(),
        ImmutableMap.<String, List<Integer>>of("ActionMenuView", ImmutableList.<Integer>of()),
        finalFields);
  }

  static final Matcher<Throwable> NUMBER_FORMAT_EXCEPTION =
      new BaseMatcher<Throwable>() {
        @Override
        public boolean matches(Object item) {
          if (item instanceof NumberFormatException) {
            return true;
          }
          return false;
        }

        @Override
        public void describeTo(Description description) {
          description.appendText(NumberFormatException.class.toString());
        }
      };

  static final Matcher<Throwable> INVALID_JAVA_IDENTIFIER =
      new BaseMatcher<Throwable>() {
        @Override
        public boolean matches(Object item) {
          return item instanceof InvalidJavaIdentifier;
        }

        @Override
        public void describeTo(Description description) {
          description.appendText(InvalidJavaIdentifier.class.getName());
        }
      };

  @Test
  public void corruptIntArraysTrailingComma() throws Exception {
    // Test a few cases of what happens if the R.txt is corrupted. It shouldn't happen unless there
    // is a bug in aapt, or R.txt is manually written the wrong way.
    Path path = createFile("R.txt", new String[] {"int[] styleable ActionMenuView { 1, }"});
    thrown.expectCause(NUMBER_FORMAT_EXCEPTION);
    ResourceSymbols.load(path, MoreExecutors.newDirectExecutorService()).get();
  }

  @Test
  public void corruptIntArraysOmittedMiddle() throws Exception {
    Path path = createFile("R.txt", "int[] styleable ActionMenuView { 1, , 2 }");
    thrown.expectCause(NUMBER_FORMAT_EXCEPTION);
    ResourceSymbols.load(path, MoreExecutors.newDirectExecutorService()).get();
  }

  @Test
  public void invalidJavaIdentifierNumber() throws Exception {
    Path path = createFile("R.txt", "int id 42ActionMenuView 0x7f020000");
    final ResourceSymbols resourceSymbols =
        ResourceSymbols.load(path, MoreExecutors.newDirectExecutorService()).get();
    Path out = Files.createDirectories(temp.resolve("classes"));
    thrown.expect(INVALID_JAVA_IDENTIFIER);
    RClassGenerator.with(out, resourceSymbols.asInitializers(), true).write("somepackage");
  }

  @Test
  public void invalidJavaIdentifierColon() throws Exception {
    Path path = createFile("R.txt", "int id Action:MenuView 0x7f020000");
    final ResourceSymbols resourceSymbols =
        ResourceSymbols.load(path, MoreExecutors.newDirectExecutorService()).get();
    Path out = Files.createDirectories(temp.resolve("classes"));
    thrown.expect(INVALID_JAVA_IDENTIFIER);
    RClassGenerator.with(out, resourceSymbols.asInitializers(), true).write("somepackage");
  }

  @Test
  public void reservedJavaIdentifier() throws Exception {
    Path path = createFile("R.txt", "int id package 0x7f020000");
    final ResourceSymbols resourceSymbols =
        ResourceSymbols.load(path, MoreExecutors.newDirectExecutorService()).get();
    Path out = Files.createDirectories(temp.resolve("classes"));
    thrown.expect(INVALID_JAVA_IDENTIFIER);
    RClassGenerator.with(out, resourceSymbols.asInitializers(), true).write("somepackage");
  }

  @Test
  public void binaryDropsLibraryFields() throws Exception {
    boolean finalFields = true;
    // Test what happens if the binary R.txt is not a strict superset of the
    // library R.txt (overrides that drop elements).
    ResourceSymbols symbolValues =
        createSymbolFile("R.txt", "int layout stubbable_activity 0x7f020000");
    ResourceSymbols symbolsInLibrary =
        createSymbolFile(
            "lib.R.txt",
            "int id debug_text_field 0x1",
            "int id debug_text_field2 0x1",
            "int layout stubbable_activity 0x1");
    Path out = temp.resolve("classes");
    Files.createDirectories(out);
    RClassGenerator writer = RClassGenerator.with(out, symbolValues.asInitializers(), finalFields);
    writer.write("com.foo", symbolsInLibrary.asInitializers());

    Path packageDir = out.resolve("com/foo");
    checkFilesInPackage(packageDir, "R.class", "R$layout.class");
    Class<?> outerClass = checkTopLevelClass(out, "com.foo.R", "com.foo.R$layout");
    checkInnerClass(
        out,
        "com.foo.R$layout",
        outerClass,
        ImmutableMap.of("stubbable_activity", 0x7f020000),
        ImmutableMap.<String, List<Integer>>of(),
        finalFields);
  }

  @Test
  public void writeNothingWithNoResources() throws Exception {
    boolean finalFields = true;
    // Test what happens if the library R.txt has no elements.
    ResourceSymbols symbolValues =
        createSymbolFile("R.txt", "int layout stubbable_activity 0x7f020000");
    ResourceSymbols symbolsInLibrary = createSymbolFile("lib.R.txt");
    Path out = temp.resolve("classes");
    Files.createDirectories(out);
    RClassGenerator writer = RClassGenerator.with(out, symbolValues.asInitializers(), finalFields);
    writer.write("com.foo", symbolsInLibrary.asInitializers());

    Path packageDir = out.resolve("com/foo");

    checkFilesInPackage(packageDir);
  }

  @Test
  public void intArraysFinal() throws Exception {
    checkIntArrays(true);
  }

  @Test
  public void intArraysNonFinal() throws Exception {
    checkIntArrays(false);
  }

  public void checkIntArrays(boolean finalFields) throws Exception {
    ResourceSymbols symbolValues =
        createSymbolFile(
            "R.txt",
            "int attr android_layout 0x010100f2",
            "int attr bar 0x7f010001",
            "int attr baz 0x7f010002",
            "int attr fox 0x7f010003",
            "int attr attr 0x7f010004",
            "int attr another_attr 0x7f010005",
            "int attr zoo 0x7f010006",
            // Test several > 5 elements, clinit must use bytecodes other than iconst_0 to 5.
            "int[] styleable ActionButton { 0x010100f2, "
                + "com.google.devtools.build.android.resources.android.R.Attr.staged, "
                + "com.google.devtools.build.android.resources.android.R$Attr.stagedOther, "
                + "0x7f010001, 0x7f010002, 0x7f010003, 0x7f010004, 0x7f010005, 0x7f010006 }",
            // The array indices of each attribute.
            "int styleable ActionButton_android_layout 0",
            "int styleable ActionButton_android_staged 1",
            "int styleable ActionButton_android_stagedOther 2",
            "int styleable ActionButton_another_attr 7",
            "int styleable ActionButton_attr 6",
            "int styleable ActionButton_bar 3",
            "int styleable ActionButton_baz 4",
            "int styleable ActionButton_fox 5",
            "int styleable ActionButton_zoo 8");
    ResourceSymbols symbolsInLibrary = symbolValues;
    Path out = temp.resolve("classes");
    Files.createDirectories(out);
    RClassGenerator writer = RClassGenerator.with(out, symbolValues.asInitializers(), finalFields);
    writer.write("com.intArray", symbolsInLibrary.asInitializers());

    Path packageDir = out.resolve("com/intArray");
    checkFilesInPackage(packageDir, "R.class", "R$attr.class", "R$styleable.class");
    Class<?> outerClass =
        checkTopLevelClass(
            out, "com.intArray.R", "com.intArray.R$attr", "com.intArray.R$styleable");
    checkInnerClass(
        out,
        "com.intArray.R$attr",
        outerClass,
        ImmutableMap.<String, Integer>builder()
            .put("android_layout", 0x010100f2)
            .put("bar", 0x7f010001)
            .put("baz", 0x7f010002)
            .put("fox", 0x7f010003)
            .put("attr", 0x7f010004)
            .put("another_attr", 0x7f010005)
            .put("zoo", 0x7f010006)
            .build(),
        ImmutableMap.<String, List<Integer>>of(),
        finalFields);
    checkInnerClass(
        out,
        "com.intArray.R$styleable",
        outerClass,
        ImmutableMap.<String, Integer>builder()
            .put("ActionButton_android_layout", 0)
            .put("ActionButton_android_staged", 1)
            .put("ActionButton_android_stagedOther", 2)
            .put("ActionButton_bar", 3)
            .put("ActionButton_baz", 4)
            .put("ActionButton_fox", 5)
            .put("ActionButton_attr", 6)
            .put("ActionButton_another_attr", 7)
            .put("ActionButton_zoo", 8)
            .build(),
        ImmutableMap.<String, List<Integer>>of(
            "ActionButton",
            ImmutableList.of(
                0x010100f2,
                0x0101ff00,
                0x0101ff01,
                0x7f010001,
                0x7f010002,
                0x7f010003,
                0x7f010004,
                0x7f010005,
                0x7f010006)),
        finalFields);
  }

  @Test
  public void emptyPackage() throws Exception {
    boolean finalFields = true;
    // Make sure we handle an empty package string.
    ResourceSymbols symbolValues = createSymbolFile("R.txt", "int string some_string 0x7f200000");
    ResourceSymbols symbolsInLibrary = symbolValues;
    Path out = temp.resolve("classes");
    Files.createDirectories(out);
    RClassGenerator writer = RClassGenerator.with(out, symbolValues.asInitializers(), finalFields);
    writer.write("", symbolsInLibrary.asInitializers());

    Path packageDir = out.resolve("");
    checkFilesInPackage(packageDir, "R.class", "R$string.class");
    Class<?> outerClass = checkTopLevelClass(out, "R", "R$string");
    checkInnerClass(
        out,
        "R$string",
        outerClass,
        ImmutableMap.of("some_string", 0x7f200000),
        ImmutableMap.<String, List<Integer>>of(),
        finalFields);
  }

  // Test utilities

  private Path createFile(String name, String... contents) throws IOException {
    Path path = temp.resolve(name);
    Files.createDirectories(path.getParent());
    Files.newOutputStream(path)
        .write(Joiner.on("\n").join(contents).getBytes(StandardCharsets.UTF_8));
    return path;
  }

  private ResourceSymbols createSymbolFile(String name, String... contents)
      throws IOException, InterruptedException, ExecutionException {
    Path path = createFile(name, contents);
    ListeningExecutorService executorService = MoreExecutors.newDirectExecutorService();
    ResourceSymbols symbolFile = ResourceSymbols.load(path, executorService).get();
    return symbolFile;
  }

  private static void checkFilesInPackage(Path packageDir, String... expectedFiles)
      throws IOException {
    try (DirectoryStream<Path> stream = Files.newDirectoryStream(packageDir)) {
      ImmutableList<String> filesInPackage =
          ImmutableList.copyOf(
              Iterables.transform(
                  stream,
                  new Function<Path, String>() {
                    @Override
                    public String apply(Path path) {
                      return path.getFileName().toString();
                    }
                  }));
      assertThat(filesInPackage).containsExactly((Object[]) expectedFiles);
    }
  }

  private static Class<?> checkTopLevelClass(
      Path baseDir, String expectedClassName, String... expectedInnerClasses) throws Exception {
    try (URLClassLoader urlClassLoader = new URLClassLoader(new URL[] {baseDir.toUri().toURL()})) {
      Class<?> toplevelClass = urlClassLoader.loadClass(expectedClassName);
      assertThat(toplevelClass.getSuperclass()).isEqualTo(Object.class);
      int outerModifiers = toplevelClass.getModifiers();
      assertThat(Modifier.isFinal(outerModifiers)).isTrue();
      assertThat(Modifier.isPublic(outerModifiers)).isTrue();
      ImmutableList.Builder<String> actualClasses = ImmutableList.builder();
      for (Class<?> innerClass : toplevelClass.getClasses()) {
        assertThat(innerClass.getDeclaredClasses()).isEmpty();
        int modifiers = innerClass.getModifiers();
        assertThat(Modifier.isFinal(modifiers)).isTrue();
        assertThat(Modifier.isPublic(modifiers)).isTrue();
        assertThat(Modifier.isStatic(modifiers)).isTrue();
        actualClasses.add(innerClass.getName());
      }
      assertThat(actualClasses.build()).containsExactly((Object[]) expectedInnerClasses);
      return toplevelClass;
    }
  }

  private void checkInnerClass(
      Path baseDir,
      String expectedClassName,
      Class<?> outerClass,
      ImmutableMap<String, Integer> intFields,
      ImmutableMap<String, List<Integer>> intArrayFields,
      boolean areFieldsFinal)
      throws Exception {
    try (URLClassLoader urlClassLoader =
        new URLClassLoader(new URL[] {baseDir.toUri().toURL()}, getClass().getClassLoader())) {
      Class<?> innerClass = urlClassLoader.loadClass(expectedClassName);
      assertThat(innerClass.getSuperclass()).isEqualTo(Object.class);
      assertThat(innerClass.getEnclosingClass().toString()).isEqualTo(outerClass.toString());
      ImmutableMap.Builder<String, Integer> actualIntFields = ImmutableMap.builder();
      ImmutableMap.Builder<String, List<Integer>> actualIntArrayFields = ImmutableMap.builder();
      for (Field f : innerClass.getFields()) {
        int fieldModifiers = f.getModifiers();
        assertThat(Modifier.isFinal(fieldModifiers)).isEqualTo(areFieldsFinal);
        assertThat(Modifier.isPublic(fieldModifiers)).isTrue();
        assertThat(Modifier.isStatic(fieldModifiers)).isTrue();

        Class<?> fieldType = f.getType();
        if (fieldType.isPrimitive()) {
          assertThat(fieldType).isEqualTo(Integer.TYPE);
          actualIntFields.put(f.getName(), (Integer) f.get(null));
        } else {
          assertThat(fieldType.isArray()).isTrue();
          int[] asArray = (int[]) f.get(null);
          ImmutableList.Builder<Integer> list = ImmutableList.builder();
          for (int i : asArray) {
            list.add(i);
          }
          actualIntArrayFields.put(f.getName(), list.build());
        }
      }
      assertThat(actualIntFields.build()).containsExactlyEntriesIn(intFields);
      assertThat(actualIntArrayFields.build()).containsExactlyEntriesIn(intArrayFields);
    }
  }
}
