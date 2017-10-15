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

package com.google.devtools.build.benchmark.codegenerator;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.base.Joiner;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Scanner;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for {@link JavaCodeGeneratorHelper}. */
@RunWith(JUnit4.class)
public class JavaCodeGeneratorHelperTest {

  private static final String RANDOM_CLASS_CONTENT = joinLines(
      "package com.package.name;",
      "",
      "import java.lang.System;",
      "import java.util.Random;",
      "",
      "public final class ClassName {",
      "  public static void PrintSth() {",
      "    Random rand = new Random();",
      "    int n = rand.nextInt(100);",
      "    System.out.format(\"This is method(%s) with random number(%d)\\n\", \"PrintSth\", n);",
      "  }",
      "}");

  private static final String RANDOM_CLASS_EXTRA_CONTENT = joinLines(
      "package com.package.name;",
      "",
      "import java.lang.System;",
      "import java.util.Random;",
      "",
      "public final class ClassNameExtra {",
      "  public static void PrintSth() {",
      "    Random rand = new Random();",
      "    int n = rand.nextInt(100);",
      "    System.out.format(\"This is method(%s) with random number(%d)\\n\", \"PrintSth\", n);",
      "  }",
      "",
      "  public static void PrintSthElse() {",
      "    Random rand = new Random();",
      "    int n = rand.nextInt(100);",
      "    System.out.format(\"This is method(%s) with random number(%d)\\n\","
          + " \"PrintSthElse\", n);",
      "  }",
      "}");

  private static final String MAIN_CLASS_CONTENT = joinLines(
      "package com.package.name;",
      "",
      "import java.lang.String;",
      "",
      "public class Main {",
      "  public static void main(String[] args) {",
      "  }",
      "}");

  private static final String DEPS_BUILD_FILE_CONTENT = joinLines(
      "java_library(",
      "    name = 'Deps42',",
      "    srcs = glob([ 'com/example/deps42/*.java' ]),",
      "<this is deps>",
      "    visibility = [ '//visibility:public' ],",
      ")");

  private static final String TARGET_BUILD_FILE_CONTENT = joinLines(
      "java_binary(",
      "    name = 'Target',",
      "    srcs = glob([ 'com/example/generated/*.java' ]),",
      "    main_class = 'com.example.generated.Main',",
      "<this is deps>",
      ")");

  private static final String DEPS_CLASS_CONTENT = joinLines(
      "package com.example.deps42;",
      "",
      "import com.example.deps43.Deps43;",
      "import java.lang.System;",
      "import java.util.Random;",
      "",
      "public final class Deps42 {",
      "  public static void PrintSth() {",
      "    Random rand = new Random();",
      "    int n = rand.nextInt(100);",
      "    System.out.format(\"This is method(%s) with random number(%d)\\n\", \"PrintSth\", n);",
      "  }",
      "",
      "  public static void CallNext() {",
      "    Deps43.PrintSth();",
      "  }",
      "}");

  private static final String DEPS_CLASS_EXTRA_CONTENT = joinLines(
      "package com.example.deps42;",
      "",
      "import java.lang.System;",
      "import java.util.Random;",
      "",
      "public final class Deps42 {",
      "  public static void PrintSth() {",
      "    Random rand = new Random();",
      "    int n = rand.nextInt(100);",
      "    System.out.format(\"This is method(%s) with random number(%d)\\n\", \"PrintSth\", n);",
      "  }",
      "",
      "  public static void PrintSthElse() {",
      "    Random rand = new Random();",
      "    int n = rand.nextInt(100);",
      "    System.out.format(\"This is method(%s) with random number(%d)\\n\","
          + " \"PrintSthElse\", n);",
      "  }",
      "}");

  private static final String MAIN_CLASS_WITH_DEPS_CONTENT = joinLines(
      "package com.example.generated;",
      "",
      "import com.example.deps1.Deps1;",
      "import com.example.deps2.Deps2;",
      "import com.example.deps3.Deps3;",
      "import java.lang.String;",
      "",
      "public final class Main {",
      "  public static void main(String[] args) {",
      "    Deps1.PrintSth();",
      "    Deps2.PrintSth();",
      "    Deps3.PrintSth();",
      "  }",
      "}");

  @Rule public TemporaryFolder folder = new TemporaryFolder();

  @Test
  public void testWriteRandomClassToDir() throws IOException {
    Path dir = folder.newFolder("WriteRandomClassToDir").toPath();
    JavaCodeGeneratorHelper.writeRandomClassToDir(false, "ClassName", "com.package.name", dir);

    Path javaFile = dir.resolve("com/package/name/ClassName.java");
    assertThat(javaFile.toFile().exists()).isTrue();

    String content = new Scanner(javaFile).useDelimiter("\\Z").next();
    assertThat(content).isEqualTo(RANDOM_CLASS_CONTENT);
  }

  @Test
  public void testWriteRandomClassToDirExtraMethod() throws IOException {
    Path dir = folder.newFolder("WriteRandomClassToDirExtraMethod").toPath();
    JavaCodeGeneratorHelper.writeRandomClassToDir(true, "ClassNameExtra", "com.package.name", dir);

    Path javaFile = dir.resolve("com/package/name/ClassNameExtra.java");
    assertThat(javaFile.toFile().exists()).isTrue();

    String content = new Scanner(javaFile).useDelimiter("\\Z").next();
    assertThat(content).isEqualTo(RANDOM_CLASS_EXTRA_CONTENT);
  }

  @Test
  public void testWriteMainClassToDir() throws IOException {
    Path dir = folder.newFolder("WriteMainClassToDir").toPath();
    JavaCodeGeneratorHelper.writeMainClassToDir("com.package.name", dir);

    Path javaFile = dir.resolve("com/package/name/Main.java");
    assertThat(javaFile.toFile().exists()).isTrue();

    String content = new Scanner(javaFile).useDelimiter("\\Z").next();
    assertThat(content).isEqualTo(MAIN_CLASS_CONTENT);
  }

  @Test
  public void testBuildFileWithNextDeps() throws IOException {
    Path dir = folder.newFolder("BuildFileWithNextDeps").toPath();
    Files.createDirectories(dir);
    JavaCodeGeneratorHelper.buildFileWithNextDeps(42, "<this is deps>", dir);

    Path buildFile = dir.resolve("BUILD");
    assertThat(buildFile.toFile().exists()).isTrue();

    String content = new Scanner(buildFile).useDelimiter("\\Z").next();
    assertThat(content).isEqualTo(DEPS_BUILD_FILE_CONTENT);
  }

  @Test
  public void testBuildFileWithMainClass() throws IOException {
    Path dir = folder.newFolder("BuildFileWithMainClass").toPath();
    Files.createDirectories(dir);
    JavaCodeGeneratorHelper.buildFileWithMainClass("Target", "<this is deps>", dir);

    Path buildFile = dir.resolve("BUILD");
    assertThat(buildFile.toFile().exists()).isTrue();

    String content = new Scanner(buildFile).useDelimiter("\\Z").next();
    assertThat(content).isEqualTo(TARGET_BUILD_FILE_CONTENT);
  }

  @Test
  public void testTargetWithNextHelper() throws IOException {
    Path dir = folder.newFolder("TargetWithNextHelper").toPath();
    JavaCodeGeneratorHelper.targetWithNextHelper(42, true, dir);

    Path javaFile = dir.resolve("com/example/deps42/Deps42.java");
    assertThat(javaFile.toFile().exists()).isTrue();

    String content = new Scanner(javaFile).useDelimiter("\\Z").next();
    assertThat(content).isEqualTo(DEPS_CLASS_CONTENT);
  }

  @Test
  public void testTargetWithNextExtraHelper() throws IOException {
    Path dir = folder.newFolder("TargetWithNextHelperExtra").toPath();
    JavaCodeGeneratorHelper.targetWithNextExtraHelper(42, false, dir);

    Path javaFile = dir.resolve("com/example/deps42/Deps42.java");
    assertThat(javaFile.toFile().exists()).isTrue();

    String content = new Scanner(javaFile).useDelimiter("\\Z").next();
    assertThat(content).isEqualTo(DEPS_CLASS_EXTRA_CONTENT);
  }

  @Test
  public void testParallelDepsMainClassHelper() throws IOException {
    Path dir = folder.newFolder("ParallelDepsMainClassHelper").toPath();
    JavaCodeGeneratorHelper.parallelDepsMainClassHelper(4, dir);

    Path javaFile = dir.resolve("com/example/generated/Main.java");
    assertThat(javaFile.toFile().exists()).isTrue();

    String content = new Scanner(javaFile).useDelimiter("\\Z").next();
    assertThat(content).isEqualTo(MAIN_CLASS_WITH_DEPS_CONTENT);
  }

  private static String joinLines(String... lines) {
    return Joiner.on("\n").join(lines);
  }
}
