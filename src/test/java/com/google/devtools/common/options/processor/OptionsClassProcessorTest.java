// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.common.options.processor;

import static com.google.common.truth.Truth.assertAbout;
import static com.google.testing.compile.JavaSourceSubjectFactory.javaSource;
import static com.google.testing.compile.JavaSourcesSubjectFactory.javaSources;

import com.google.common.io.Resources;
import com.google.testing.compile.JavaFileObjects;
import java.util.Arrays;
import javax.tools.JavaFileObject;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link OptionsClassProcessor}. */
@RunWith(JUnit4.class)
public class OptionsClassProcessorTest {
  private static JavaFileObject getFile(String pathToFile) {
    return JavaFileObjects.forResource(
        Resources.getResource(
            "com/google/devtools/common/options/processor/optiontestsources/" + pathToFile));
  }

  @Test
  public void nonPublicOptionMethodIsRejected() {
    assertAbout(javaSource())
        .that(getFile("NonPublicOptionMethod.java"))
        .processedWith(new OptionsClassProcessor())
        .failsToCompile()
        .withErrorContaining("@Option method must be public");
  }

  @Test
  public void nonAbstractOptionMethodIsRejected() {
    assertAbout(javaSource())
        .that(getFile("NonAbstractGetter.java"))
        .processedWith(new OptionsClassProcessor())
        .failsToCompile()
        .withErrorContaining("@Option method must be abstract");
  }

  @Test
  public void nonAbstractSetterIsRejected() {
    assertAbout(javaSource())
        .that(getFile("NonAbstractSetter.java"))
        .processedWith(new OptionsClassProcessor())
        .failsToCompile()
        .withErrorContaining("Setter must be abstract");
  }

  @Test
  public void setterWithMultipleArgumentsIsRejected() {
    assertAbout(javaSource())
        .that(getFile("SetterWithMultipleArguments.java"))
        .processedWith(new OptionsClassProcessor())
        .failsToCompile()
        .withErrorContaining("Setter must have exactly one argument");
  }

  @Test
  public void setterWithIncorrectArgumentTypeIsRejected() {
    assertAbout(javaSource())
        .that(getFile("SetterWithIncorrectArgumentType.java"))
        .processedWith(new OptionsClassProcessor())
        .failsToCompile()
        .withErrorContaining("Setter argument type must be same as getter return type (int)");
  }

  @Test
  public void invalidOptionMethodNameIsRejected() {
    assertAbout(javaSource())
        .that(getFile("InvalidOptionMethodName.java"))
        .processedWith(new OptionsClassProcessor())
        .failsToCompile()
        .withErrorContaining(
            "Annotated method name must start with 'get' followed by an uppercase letter");
  }

  @Test
  public void validOptionsClassCompiles() {
    assertAbout(javaSource())
        .that(getFile("ValidOptions.java"))
        .processedWith(new OptionsClassProcessor())
        .compilesWithoutError();
  }

  @Test
  public void nestedOptionsClassesWithSameSimpleNameCompile() {
    assertAbout(javaSources())
        .that(Arrays.asList(getFile("Outer1.java"), getFile("Outer2.java")))
        .processedWith(new OptionsClassProcessor())
        .compilesWithoutError();
  }
}
