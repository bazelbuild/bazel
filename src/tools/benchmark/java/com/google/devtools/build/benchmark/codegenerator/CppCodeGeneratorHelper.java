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

import static java.nio.charset.StandardCharsets.UTF_8;
import static java.nio.file.StandardOpenOption.APPEND;
import static java.nio.file.StandardOpenOption.CREATE;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/** Helper class of {@link CppCodeGenerator} */
class CppCodeGeneratorHelper {

  private static final String CPP_FILE_SUFFIX = ".cc";
  private static final String CPP_HEADER_FILE_SUFFIX = ".h";
  private static final String BUILD_FILE_NAME = "BUILD";

  static void createRandomClass(String className, Path dir) throws IOException {
    writeLinesToFile(dir.resolve(className + CPP_FILE_SUFFIX),
        "#include <random>",
        "#include <iostream>",
        "#include <ctime>",
        "using namespace std;",
        "",
        "class " + className + " {",
        "public:",
        "  static void printSth() {",
        "    srand(time(NULL));",
        "    int n = rand();",
        "    cout << \"This is method(printSth) with random number(\" << n << \")\\n\";",
        "  }",
        "};");
    writeLinesToFile(dir.resolve(className + CPP_HEADER_FILE_SUFFIX),
        "class " + className + " {",
        "public:",
        "  static void printSth();",
        "};");
  }

  static void createRandomClassExtra(String className, Path dir) throws IOException {
    writeLinesToFile(dir.resolve(className + CPP_FILE_SUFFIX),
        "#include <random>",
        "#include <iostream>",
        "#include <ctime>",
        "using namespace std;",
        "",
        "class " + className + " {",
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
    writeLinesToFile(dir.resolve(className + CPP_HEADER_FILE_SUFFIX),
        "class " + className + " {",
        "public:",
        "  static void printSth();",
        "  static void printSthElse();",
        "};");
  }

  static void writeBuildFileWithAllFilesToDir(String targetName, Path dir) throws IOException {
    writeLinesToFile(dir.resolve(BUILD_FILE_NAME),
        "cc_library(",
        "    name = '" + targetName + "',",
        "    srcs = glob([ '*" + CPP_FILE_SUFFIX + "', '*" + CPP_HEADER_FILE_SUFFIX + "' ]),",
        ")");
  }

  static void createClassAndBuildFileWithDepsNext(int index, Path dir) throws IOException {
    writeLinesToFile(dir.resolve("Deps" + index + CPP_HEADER_FILE_SUFFIX),
        "class Deps" + index + " {",
        "public:",
        "  static void printSth();",
        "  static void callNext();",
        "};");
    writeLinesToFile(dir.resolve("Deps" + index + CPP_FILE_SUFFIX),
        "#include <random>",
        "#include <iostream>",
        "#include <ctime>",
        "#include \"Deps" + (index + 1) + ".h\"",
        "using namespace std;",
        "",
        "class Deps" + index + " {",
        "public:",
        "  static void printSth() {",
        "    srand(time(NULL));",
        "    int n = rand();",
        "    cout << \"This is method(printSth) with random number(\" << n << \")\\n\";",
        "  }",
        "  static void callNext() {",
        "    Deps" + (index + 1) + "::printSth();",
        "  }",
        "};");
    appendLinesToFile(dir.resolve(BUILD_FILE_NAME),
        "cc_library(",
        "    name = 'Deps" + index + "',",
        "    srcs = [ 'Deps" + index + ".cc', 'Deps" + index + ".h' ],",
        "    deps = [ ':Deps" + (index + 1) + "' ],",
        ")");
  }

  static void createClassWithDepsNextExtra(int index, Path dir) throws IOException {
    writeLinesToFile(dir.resolve("Deps" + index + CPP_HEADER_FILE_SUFFIX),
        "class Deps" + index + " {",
        "public:",
        "  static void printSth();",
        "  static void printSthElse();",
        "  static void callNext();",
        "};");
    writeLinesToFile(dir.resolve("Deps" + index + CPP_FILE_SUFFIX),
        "#include <random>",
        "#include <iostream>",
        "#include <ctime>",
        "#include \"Deps" + (index + 1) + ".h\"",
        "using namespace std;",
        "",
        "class Deps" + index + " {",
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
        "    Deps" + (index + 1) + "::printSth();",
        "  }",
        "};");
  }

  static void appendTargetToBuildFile(String targetName, Path dir) throws IOException {
    appendLinesToFile(dir.resolve(BUILD_FILE_NAME),
        "cc_library(",
        "    name = '" + targetName + "',",
        "    srcs = [ '" + targetName + ".cc', '" + targetName + ".h' ],",
        ")");
  }

  static void createMainClassAndBuildFileWithDeps(String targetName, String deps, Path dir)
      throws IOException {
    writeLinesToFile(dir.resolve("Main" + CPP_FILE_SUFFIX),
        "int main() {",
        "  return 0;",
        "}");
    appendLinesToFile(dir.resolve(BUILD_FILE_NAME),
        "cc_binary(",
        "    name = '" + targetName + "',",
        "    srcs = [ 'Main.cc' ],",
        deps,
        ")");
  }

  private static void appendLinesToFile(Path filePath, String... lines) throws IOException {
    writeOrAppendLinesToFile(true, filePath, lines);
  }

  private static void writeLinesToFile(Path filePath, String... lines) throws IOException {
    writeOrAppendLinesToFile(false, filePath, lines);
  }

  private static void writeOrAppendLinesToFile(boolean append, Path filePath, String... lines)
      throws IOException {
    File file = filePath.toFile();
    if (!file.exists() && !file.createNewFile()) {
      return;
    }

    PrintWriter printWriter =
        new PrintWriter(
            Files.newBufferedWriter(
                file.toPath(),
                UTF_8,
                append
                    ? new StandardOpenOption[] {CREATE, APPEND}
                    : new StandardOpenOption[] {CREATE}));
    for (String line : lines) {
      printWriter.println(line);
    }
    printWriter.close();
  }
}
