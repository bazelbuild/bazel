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
import java.nio.file.Path;
import java.util.Scanner;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for {@link CppCodeGeneratorHelper}. */
@RunWith(JUnit4.class)
public class CppCodeGeneratorHelperTest {

  private static final String RANDOM_CLASS_HEADER_CONTENT = joinLines(
      "class ClassName {",
      "public:",
      "  static void printSth();",
      "};");

  private static final String RANDOM_CLASS_CONTENT = joinLines(
      "#include <random>",
      "#include <iostream>",
      "#include <ctime>",
      "using namespace std;",
      "",
      "class ClassName {",
      "public:",
      "  static void printSth() {",
      "    srand(time(NULL));",
      "    int n = rand();",
      "    cout << \"This is method(printSth) with random number(\" << n << \")\\n\";",
      "  }",
      "};");

  private static final String RANDOM_CLASS_HEADER_EXTRA_CONTENT = joinLines(
      "class ClassName {",
      "public:",
      "  static void printSth();",
      "  static void printSthElse();",
      "};");

  private static final String RANDOM_CLASS_EXTRA_CONTENT = joinLines(
      "#include <random>",
      "#include <iostream>",
      "#include <ctime>",
      "using namespace std;",
      "",
      "class ClassName {",
      "public:",
      "  static void printSth() {",
      "    srand(time(NULL));",
      "    int n = rand();",
      "    cout << \"This is method(printSth) with random number(\" << n << \")\\n\";",
      "  }",
      "  static void printSthElse() {",
      "    srand(time(NULL));",
      "    int n = rand();",
      "    cout << \"This is method(printSthElse) with random number(\" << n << \")\\n\";",
      "  }",
      "};");

  private static final String ALL_FILES_BUILD_FILE_CONTENT = joinLines(
      "cc_library(",
      "    name = 'target',",
      "    srcs = glob([ '*.cc', '*.h' ]),",
      ")");

  private static final String RANDOM_CLASS_HEADER_NEXT_CONTENT = joinLines(
      "class Deps42 {",
      "public:",
      "  static void printSth();",
      "  static void callNext();",
      "};");

  private static final String RANDOM_CLASS_NEXT_CONTENT = joinLines(
      "#include <random>",
      "#include <iostream>",
      "#include <ctime>",
      "#include \"Deps43.h\"",
      "using namespace std;",
      "",
      "class Deps42 {",
      "public:",
      "  static void printSth() {",
      "    srand(time(NULL));",
      "    int n = rand();",
      "    cout << \"This is method(printSth) with random number(\" << n << \")\\n\";",
      "  }",
      "  static void callNext() {",
      "    Deps43::printSth();",
      "  }",
      "};");

  private static final String BUILD_FILE_NEXT_CONTENT = joinLines(
      "cc_library(",
      "    name = 'Deps42',",
      "    srcs = [ 'Deps42.cc', 'Deps42.h' ],",
      "    deps = [ ':Deps43' ],",
      ")");

  private static final String RANDOM_CLASS_HEADER_NEXT_EXTRA_CONTENT = joinLines(
      "class Deps42 {",
      "public:",
      "  static void printSth();",
      "  static void printSthElse();",
      "  static void callNext();",
      "};");

  private static final String RANDOM_CLASS_NEXT_EXTRA_CONTENT = joinLines(
      "#include <random>",
      "#include <iostream>",
      "#include <ctime>",
      "#include \"Deps43.h\"",
      "using namespace std;",
      "",
      "class Deps42 {",
      "public:",
      "  static void printSth() {",
      "    srand(time(NULL));",
      "    int n = rand();",
      "    cout << \"This is method(printSth) with random number(\" << n << \")\\n\";",
      "  }",
      "  static void printSthElse() {",
      "    srand(time(NULL));",
      "    int n = rand();",
      "    cout << \"This is method(printSthElse) with random number(\" << n << \")\\n\";",
      "  }",
      "  static void callNext() {",
      "    Deps43::printSth();",
      "  }",
      "};");

  private static final String BUILD_FILE_CONTENT = joinLines(
      "cc_library(",
      "    name = 'target',",
      "    srcs = [ 'target.cc', 'target.h' ],",
      ")");

  private static final String MAIN_CLASS_CONTENT = joinLines(
      "int main() {",
      "  return 0;",
      "}");

  private static final String MAIN_CLASS_BUILD_FILE_CONTENT = joinLines(
      "cc_binary(",
      "    name = 'target',",
      "    srcs = [ 'Main.cc' ],",
      "This is deps",
      ")");

  @Rule
  public TemporaryFolder folder = new TemporaryFolder();

  @Test
  public void testCreateRandomClass() throws IOException {
    Path dir = folder.newFolder("CreateRandomClass").toPath();
    CppCodeGeneratorHelper.createRandomClass("ClassName", dir);

    Path cppFile = dir.resolve("ClassName.h");
    assertThat(cppFile.toFile().exists()).isTrue();
    String content = new Scanner(cppFile).useDelimiter("\\Z").next();
    assertThat(content).isEqualTo(RANDOM_CLASS_HEADER_CONTENT);

    cppFile = dir.resolve("ClassName.cc");
    assertThat(cppFile.toFile().exists()).isTrue();
    content = new Scanner(cppFile).useDelimiter("\\Z").next();
    assertThat(content).isEqualTo(RANDOM_CLASS_CONTENT);
  }

  @Test
  public void testCreateRandomClassExtra() throws IOException {
    Path dir = folder.newFolder("CreateRandomClassExtra").toPath();
    CppCodeGeneratorHelper.createRandomClassExtra("ClassName", dir);

    Path cppFile = dir.resolve("ClassName.h");
    assertThat(cppFile.toFile().exists()).isTrue();
    String content = new Scanner(cppFile).useDelimiter("\\Z").next();
    assertThat(content).isEqualTo(RANDOM_CLASS_HEADER_EXTRA_CONTENT);

    cppFile = dir.resolve("ClassName.cc");
    assertThat(cppFile.toFile().exists()).isTrue();
    content = new Scanner(cppFile).useDelimiter("\\Z").next();
    assertThat(content).isEqualTo(RANDOM_CLASS_EXTRA_CONTENT);
  }

  @Test
  public void testWriteBuildFileWithAllFilesToDir() throws IOException {
    Path dir = folder.newFolder("WriteBuildFileWithAllFilesToDir").toPath();
    CppCodeGeneratorHelper.writeBuildFileWithAllFilesToDir("target", dir);

    Path buildFile = dir.resolve("BUILD");
    assertThat(buildFile.toFile().exists()).isTrue();
    String content = new Scanner(buildFile).useDelimiter("\\Z").next();
    assertThat(content).isEqualTo(ALL_FILES_BUILD_FILE_CONTENT);
  }

  @Test
  public void testCreateClassAndBuildFileWithDepsNext() throws IOException {
    Path dir = folder.newFolder("CreateClassAndBuildFileWithDepsNext").toPath();
    CppCodeGeneratorHelper.createClassAndBuildFileWithDepsNext(42, dir);

    Path file = dir.resolve("Deps42.h");
    assertThat(file.toFile().exists()).isTrue();
    String content = new Scanner(file).useDelimiter("\\Z").next();
    assertThat(content).isEqualTo(RANDOM_CLASS_HEADER_NEXT_CONTENT);

    file = dir.resolve("Deps42.cc");
    assertThat(file.toFile().exists()).isTrue();
    content = new Scanner(file).useDelimiter("\\Z").next();
    assertThat(content).isEqualTo(RANDOM_CLASS_NEXT_CONTENT);

    file = dir.resolve("BUILD");
    assertThat(file.toFile().exists()).isTrue();
    content = new Scanner(file).useDelimiter("\\Z").next();
    assertThat(content).isEqualTo(BUILD_FILE_NEXT_CONTENT);
  }

  @Test
  public void testCreateClassWithDepsNextExtra() throws IOException {
    Path dir = folder.newFolder("CreateClassWithDepsNextExtra").toPath();
    CppCodeGeneratorHelper.createClassWithDepsNextExtra(42, dir);

    Path file = dir.resolve("Deps42.h");
    assertThat(file.toFile().exists()).isTrue();
    String content = new Scanner(file).useDelimiter("\\Z").next();
    assertThat(content).isEqualTo(RANDOM_CLASS_HEADER_NEXT_EXTRA_CONTENT);

    file = dir.resolve("Deps42.cc");
    assertThat(file.toFile().exists()).isTrue();
    content = new Scanner(file).useDelimiter("\\Z").next();
    assertThat(content).isEqualTo(RANDOM_CLASS_NEXT_EXTRA_CONTENT);
  }

  @Test
  public void testAppendTargetToBuildFile() throws IOException {
    Path dir = folder.newFolder("AppendTargetToBuildFile").toPath();
    CppCodeGeneratorHelper.appendTargetToBuildFile("target", dir);

    Path file = dir.resolve("BUILD");
    assertThat(file.toFile().exists()).isTrue();
    String content = new Scanner(file).useDelimiter("\\Z").next();
    assertThat(content).isEqualTo(BUILD_FILE_CONTENT);
  }

  @Test
  public void testCreateMainClassAndBuildFileWithDeps() throws IOException {
    Path dir = folder.newFolder("CreateMainClassAndBuildFileWithDeps").toPath();
    CppCodeGeneratorHelper.createMainClassAndBuildFileWithDeps("target", "This is deps", dir);

    Path file = dir.resolve("Main.cc");
    assertThat(file.toFile().exists()).isTrue();
    String content = new Scanner(file).useDelimiter("\\Z").next();
    assertThat(content).isEqualTo(MAIN_CLASS_CONTENT);

    file = dir.resolve("BUILD");
    assertThat(file.toFile().exists()).isTrue();
    content = new Scanner(file).useDelimiter("\\Z").next();
    assertThat(content).isEqualTo(MAIN_CLASS_BUILD_FILE_CONTENT);
  }

  private static String joinLines(String... lines) {
    return Joiner.on("\n").join(lines);
  }
}
