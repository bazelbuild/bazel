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

import com.google.common.base.Joiner;
import com.squareup.javapoet.ClassName;
import com.squareup.javapoet.JavaFile;
import com.squareup.javapoet.MethodSpec;
import com.squareup.javapoet.TypeSpec;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Random;
import javax.lang.model.element.Modifier;

/** Helper class of {@link JavaCodeGenerator} */
class JavaCodeGeneratorHelper {

  private static final Joiner JOINER = Joiner.on("\n");
  private static final MethodSpec randomMethod = genRandomMethod("PrintSth");
  private static final MethodSpec somethingElseMethod = genRandomMethod("PrintSthElse");

  /**
   * Writes a class file {@code Deps(index).java} to the directory
   * {@code projectPath/com/example/deps(index)}
   *
   * @param callNext if we should call the method from {@code Deps(index+1).java}
   */
  static void targetWithNextHelper(int index, boolean callNext, Path projectPath)
      throws IOException {
    ClassName nextClass = ClassName.get("com.example.deps" + (index + 1), "Deps" + (index + 1));

    MethodSpec callNextMethod =
        MethodSpec.methodBuilder("CallNext")
            .addModifiers(Modifier.PUBLIC, Modifier.STATIC)
            .returns(void.class)
            .addStatement("$T.PrintSth()", nextClass)
            .build();

    TypeSpec.Builder klassBuilder =
        TypeSpec.classBuilder("Deps" + index)
            .addModifiers(Modifier.PUBLIC, Modifier.FINAL)
            .addMethod(randomMethod);
    if (callNext) {
      klassBuilder.addMethod(callNextMethod);
    }
    TypeSpec klass = klassBuilder.build();

    writeClassToDir(klass, "com.example.deps" + index, projectPath);
  }

  /**
   * Writes a class file {@code Deps(index).java} with extra method {@code printSthElse()}
   * to the directory {@code projectPath/com/example/deps(index)}
   *
   * @param callNext if we should call the method from {@code Deps(index+1).java}
   */
  static void targetWithNextExtraHelper(int index, boolean callNext, Path projectPath)
      throws IOException {
    ClassName nextClass = ClassName.get("com.example.deps" + (index + 1), "Deps" + (index + 1));

    MethodSpec callNextMethod =
        MethodSpec.methodBuilder("CallNext")
            .addModifiers(Modifier.PUBLIC, Modifier.STATIC)
            .returns(void.class)
            .addStatement("$T.PrintSth()", nextClass)
            .build();

    TypeSpec.Builder klassBuilder =
        TypeSpec.classBuilder("Deps" + index)
            .addModifiers(Modifier.PUBLIC, Modifier.FINAL)
            .addMethod(randomMethod)
            .addMethod(somethingElseMethod);
    if (callNext) {
      klassBuilder.addMethod(callNextMethod);
    }
    TypeSpec klass = klassBuilder.build();

    writeClassToDir(klass, "com.example.deps" + index, projectPath);
  }

  /**
   * Writes {@code count-1} class files to the directory {@code projectPath/com/example/deps(index)}
   * and one main class.
   */
  static void parallelDepsMainClassHelper(int count, Path projectPath) throws IOException {
    MethodSpec.Builder callDepsBuilder =
        MethodSpec.methodBuilder("main")
            .addModifiers(Modifier.PUBLIC, Modifier.STATIC)
            .addParameter(String[].class, "args")
            .returns(void.class);
    for (int i = 1; i < count; ++i) {
      ClassName callingClass = ClassName.get("com.example.deps" + i, "Deps" + i);
      callDepsBuilder.addStatement("$T.PrintSth()", callingClass);
    }
    MethodSpec callDeps = callDepsBuilder.build();
    TypeSpec klass =
        TypeSpec.classBuilder("Main")
            .addModifiers(Modifier.PUBLIC, Modifier.FINAL)
            .addMethod(callDeps)
            .build();
    writeClassToDir(klass, "com.example.generated", projectPath);
  }

  static void writeRandomClassToDir(
      boolean addExtraMethod, String className, String packageName, Path projectPath)
      throws IOException {
    TypeSpec klass = genRandomClass(addExtraMethod, className);
    writeClassToDir(klass, packageName, projectPath);
  }

  static void writeMainClassToDir(String packageName, Path projectPath) throws IOException {
    TypeSpec main = genMainClass();
    writeClassToDir(main, packageName, projectPath);
  }

  static void buildFileWithNextDeps(int index, String deps, Path projectPath) throws IOException {
    Path buildFilePath = projectPath.resolve("BUILD");

    String buildFileContent =
        String.format(
            JOINER
                .join(
                    "java_library(",
                    "    name = 'Deps%d',",
                    "    srcs = glob([ 'com/example/deps%d/*.java' ]),",
                    "%s",
                    "    visibility = [ '//visibility:public' ],",
                    ")",
                    ""),
            index,
            index,
            deps);

    createAndAppendFile(buildFilePath, buildFileContent);
  }

  static void buildFileWithMainClass(String targetName, String deps, Path projectPath)
      throws IOException {
    Path buildFilePath = projectPath.resolve("BUILD");

    String buildFileContent =
        String.format(
            JOINER
                .join(
                    "java_binary(",
                    "    name = '%s',",
                    "    srcs = glob([ 'com/example/generated/*.java' ]),",
                    "    main_class = 'com.example.generated.Main',",
                    "%s",
                    ")",
                    ""),
            targetName,
            deps);

    createAndAppendFile(buildFilePath, buildFileContent);
  }

  private static MethodSpec genRandomMethod(String methodName) {
    return MethodSpec.methodBuilder(methodName)
        .addModifiers(Modifier.PUBLIC, Modifier.STATIC)
        .returns(void.class)
        .addStatement("$T rand = new Random()", Random.class)
        .addStatement("int n = rand.nextInt(100)")
        .addStatement(
            "$T.out.format($S, $S, $L)",
            System.class,
            "This is method(%s) with random number(%d)\n",
            methodName,
            "n")
        .build();
  }

  private static TypeSpec genRandomClass(boolean addExtraMethod, String className) {

    TypeSpec.Builder klassBuilder =
        TypeSpec.classBuilder(className)
            .addModifiers(Modifier.PUBLIC, Modifier.FINAL)
            .addMethod(randomMethod);

    if (addExtraMethod) {
      klassBuilder.addMethod(somethingElseMethod);
    }

    return klassBuilder.build();
  }

  private static TypeSpec genMainClass() {
    MethodSpec method =
        MethodSpec.methodBuilder("main")
            .addModifiers(Modifier.PUBLIC, Modifier.STATIC)
            .returns(void.class)
            .addParameter(String[].class, "args")
            .build();

    return TypeSpec.classBuilder("Main").addModifiers(Modifier.PUBLIC).addMethod(method).build();
  }

  private static void writeClassToDir(TypeSpec klass, String packageName, Path path)
      throws IOException {
    JavaFile javaFile = JavaFile.builder(packageName, klass).build();
    javaFile.writeTo(path);
  }

  private static void createAndAppendFile(Path path, String content) throws IOException {
    if (!Files.exists(path)) {
      Files.createFile(path);
    }
    Files.write(path, content.getBytes(UTF_8), StandardOpenOption.APPEND);
  }

}
