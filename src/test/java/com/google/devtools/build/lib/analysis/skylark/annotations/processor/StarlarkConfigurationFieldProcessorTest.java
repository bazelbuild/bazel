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

package com.google.devtools.build.lib.analysis.skylark.annotations.processor;

import static com.google.common.truth.Truth.assertAbout;
import static com.google.testing.compile.JavaSourceSubjectFactory.javaSource;

import com.google.common.io.Resources;
import com.google.testing.compile.JavaFileObjects;
import javax.tools.JavaFileObject;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for StarlarkConfigurationFieldProcessor.
 */
@RunWith(JUnit4.class)
public final class StarlarkConfigurationFieldProcessorTest {

  private static JavaFileObject getFile(String pathToFile) {
    return JavaFileObjects.forResource(Resources.getResource(
        StarlarkConfigurationFieldProcessorTest.class, "optiontestsources/" + pathToFile));
  }

  @Test
  public void testGoldenConfigurationField() throws Exception {
    assertAbout(javaSource())
        .that(getFile("GoldenConfigurationField.java"))
        .processedWith(new StarlarkConfigurationFieldProcessor())
        .compilesWithoutError();
  }

  @Test
  public void testGoldenConfigurationFieldThroughApi() throws Exception {
    assertAbout(javaSource())
        .that(getFile("GoldenConfigurationFieldThroughApi.java"))
        .processedWith(new StarlarkConfigurationFieldProcessor())
        .compilesWithoutError();
  }

  @Test
  public void testHasMethodParameters() throws Exception {
    assertAbout(javaSource())
        .that(getFile("HasMethodParameters.java"))
        .processedWith(new StarlarkConfigurationFieldProcessor())
        .failsToCompile()
        .withErrorContaining(
            "@StarlarkConfigurationField annotated methods must have zero arguments.");
  }

  @Test
  public void testMethodIsPrivate() throws Exception {
    assertAbout(javaSource())
        .that(getFile("MethodIsPrivate.java"))
        .processedWith(new StarlarkConfigurationFieldProcessor())
        .failsToCompile()
        .withErrorContaining("@StarlarkConfigurationField annotated methods must be public.");
  }

  @Test
  public void testMethodThrowsException() throws Exception {
    assertAbout(javaSource())
        .that(getFile("MethodThrowsException.java"))
        .processedWith(new StarlarkConfigurationFieldProcessor())
        .failsToCompile()
        .withErrorContaining("@StarlarkConfigurationField annotated must not throw exceptions.");
  }

  @Test
  public void testNonConfigurationFragment() throws Exception {
    assertAbout(javaSource())
        .that(getFile("NonConfigurationFragment.java"))
        .processedWith(new StarlarkConfigurationFieldProcessor())
        .failsToCompile()
        .withErrorContaining("@StarlarkConfigurationField annotated methods must be methods "
            + "of configuration fragments.");
  }

  @Test
  public void testNonExposedConfigurationFragment() throws Exception {
    assertAbout(javaSource())
        .that(getFile("NonExposedConfigurationFragment.java"))
        .processedWith(new StarlarkConfigurationFieldProcessor())
        .compilesWithoutError();
  }

  @Test
  public void testReturnsOtherType() throws Exception {
    assertAbout(javaSource())
        .that(getFile("ReturnsOtherType.java"))
        .processedWith(new StarlarkConfigurationFieldProcessor())
        .failsToCompile()
        .withErrorContaining("@StarlarkConfigurationField annotated methods must return Label.");
  }
}
