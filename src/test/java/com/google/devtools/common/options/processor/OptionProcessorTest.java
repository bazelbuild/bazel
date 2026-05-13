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
package com.google.devtools.common.options.processor;

import static com.google.common.truth.Truth.assertAbout;
import static com.google.testing.compile.JavaSourceSubjectFactory.javaSource;

import com.google.common.io.Resources;
import com.google.testing.compile.JavaFileObjects;
import javax.tools.JavaFileObject;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for the compile-time checks in {@link OptionProcessor}. */
@RunWith(JUnit4.class)
public class OptionProcessorTest {
  private static JavaFileObject getFile(String pathToFile) {
    return JavaFileObjects.forResource(
        Resources.getResource(
            "com/google/devtools/common/options/processor/optiontestsources/" + pathToFile));
  }

  @Test
  public void fieldOptionsAreRejected() {
    assertAbout(javaSource())
        .that(getFile("FieldOption.java"))
        .processedWith(new OptionProcessor())
        .failsToCompile()
        .withErrorContaining("Field options not supported anymore. Use method options instead.");
  }
}
