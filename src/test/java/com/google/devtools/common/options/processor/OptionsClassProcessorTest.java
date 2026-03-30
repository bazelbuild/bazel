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

import com.google.common.io.Resources;
import com.google.testing.compile.JavaFileObjects;
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
  public void invalidClassNameIsRejected() {
    assertAbout(javaSource())
        .that(getFile("InvalidClassName.java"))
        .processedWith(new OptionsClassProcessor())
        .failsToCompile()
        .withErrorContaining(
            "Class InvalidClassName is annotated with @OptionsClass but its name does not end in"
                + " 'Fields'");
  }

  @Test
  public void nonPublicOptionMethodIsRejected() {
    assertAbout(javaSource())
        .that(getFile("NonPublicOptionMethodFields.java"))
        .processedWith(new OptionsClassProcessor())
        .failsToCompile()
        .withErrorContaining("@Option method must be public");
  }

  @Test
  public void invalidOptionMethodNameIsRejected() {
    assertAbout(javaSource())
        .that(getFile("InvalidOptionMethodNameFields.java"))
        .processedWith(new OptionsClassProcessor())
        .failsToCompile()
        .withErrorContaining(
            "Annotated method name must start with 'get' followed by an uppercase letter");
  }

  @Test
  public void validOptionsClassCompiles() {
    assertAbout(javaSource())
        .that(getFile("ValidOptionsFields.java"))
        .processedWith(new OptionsClassProcessor())
        .compilesWithoutError();
  }
}
