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

package com.google.devtools.build.lib.skylarkinterface.processor;

import static com.google.common.truth.Truth.assertAbout;
import static com.google.testing.compile.JavaSourceSubjectFactory.javaSource;

import com.google.common.io.Resources;
import com.google.testing.compile.JavaFileObjects;
import javax.tools.JavaFileObject;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for SkylarkCallableProcessor.
 */
@RunWith(JUnit4.class)
public final class SkylarkCallableProcessorTest {

  private static JavaFileObject getFile(String pathToFile) {
    return JavaFileObjects.forResource(Resources.getResource(
        SkylarkCallableProcessorTest.class, "testsources/" + pathToFile));
  }

  @Test
  public void testGoldenCase() throws Exception {
    assertAbout(javaSource())
        .that(getFile("GoldenCase.java"))
        .processedWith(new SkylarkCallableProcessor())
        .compilesWithoutError();
  }

  @Test
  public void testPrivateMethod() throws Exception {
    assertAbout(javaSource())
        .that(getFile("PrivateMethod.java"))
        .processedWith(new SkylarkCallableProcessor())
        .failsToCompile()
        .withErrorContaining("@SkylarkCallable annotated methods must be public.");
  }

  @Test
  public void testStructFieldWithArguments() throws Exception {
    assertAbout(javaSource())
        .that(getFile("StructFieldWithArguments.java"))
        .processedWith(new SkylarkCallableProcessor())
        .failsToCompile()
        .withErrorContaining(
            "@SkylarkCallable annotated methods with structField=true must have zero arguments.");
  }

  @Test
  public void testArgumentMissing() throws Exception {
    assertAbout(javaSource())
        .that(getFile("ArgumentMissing.java"))
        .processedWith(new SkylarkCallableProcessor())
        .failsToCompile()
        .withErrorContaining(
            "@SkylarkCallable annotated method has 0 parameters, but annotation declared 1.");
  }

  @Test
  public void testTooManyArguments() throws Exception {
    assertAbout(javaSource())
        .that(getFile("TooManyArguments.java"))
        .processedWith(new SkylarkCallableProcessor())
        .failsToCompile()
        .withErrorContaining(
            "@SkylarkCallable annotated method has 2 parameters, but annotation declared 1.");
  }
}
