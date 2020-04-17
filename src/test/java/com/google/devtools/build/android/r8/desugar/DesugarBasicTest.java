// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.r8.desugar;

import static com.google.common.truth.Truth.assertThat;
import static org.objectweb.asm.Opcodes.V1_7;

import com.google.devtools.build.android.r8.FileUtils;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.function.Consumer;
import java.util.jar.JarEntry;
import java.util.jar.JarInputStream;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.objectweb.asm.ClassReader;
import org.objectweb.asm.ClassVisitor;

/** Basic test of D8 desugar */
@RunWith(JUnit4.class)
public class DesugarBasicTest {
  private Path basic;
  private Path desugared;

  @Before
  public void setup() {
    // Jar file with the compiled Java code in the sub-package basic before desugaring.
    basic = Paths.get(System.getProperty("DesugarBasicTest.testdata_basic"));
    // Jar file with the compiled Java code in the sub-package basic after desugaring.
    desugared = Paths.get(System.getProperty("DesugarBasicTest.testdata_basic_desugared"));
  }

  @Test
  public void checkBeforeDesugar() throws Exception {
    DesugarInfoCollector desugarInfoCollector = new DesugarInfoCollector();
    forAllClasses(basic, desugarInfoCollector);
    assertThat(desugarInfoCollector.getLargestMajorClassFileVersion()).isGreaterThan(V1_7);
    assertThat(desugarInfoCollector.getNumberOfInvokeDynamic()).isGreaterThan(0);
    assertThat(desugarInfoCollector.getNumberOfDefaultMethods()).isGreaterThan(0);
    assertThat(desugarInfoCollector.getNumberOfDesugaredLambdas()).isEqualTo(0);
    assertThat(desugarInfoCollector.getNumberOfCompanionClasses()).isEqualTo(0);
  }

  @Test
  public void checkAfterDesugar() throws Exception {
    DesugarInfoCollector desugarInfoCollector = new DesugarInfoCollector();
    forAllClasses(desugared, desugarInfoCollector);
    // TODO(b/153971249): The class file version of desugared class files should be Java 7.
    // assertThat(lambdaUse.getMajorCfVersion()).isEqualTo(V1_7);
    assertThat(desugarInfoCollector.getNumberOfInvokeDynamic()).isEqualTo(0);
    assertThat(desugarInfoCollector.getNumberOfDefaultMethods()).isEqualTo(0);
    assertThat(desugarInfoCollector.getNumberOfDesugaredLambdas()).isEqualTo(1);
    assertThat(desugarInfoCollector.getNumberOfCompanionClasses()).isEqualTo(1);
  }

  private static void forAllClasses(Path jar, ClassVisitor classVisitor) throws Exception {
    forAllClasses(
        jar,
        classReader ->
            classReader.accept(classVisitor, ClassReader.SKIP_DEBUG | ClassReader.SKIP_FRAMES));
  }

  private static void forAllClasses(Path jar, Consumer<ClassReader> classReader) throws Exception {

    try (JarInputStream jarInputStream =
        new JarInputStream(Files.newInputStream(jar, StandardOpenOption.READ))) {
      JarEntry entry;
      while ((entry = jarInputStream.getNextJarEntry()) != null) {
        String entryName = entry.getName();
        if (FileUtils.isClassFile(entryName)) {
          classReader.accept(new ClassReader(jarInputStream));
        }
      }
    }
  }
}
