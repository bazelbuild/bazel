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
import static org.junit.Assume.assumeTrue;

import com.google.common.io.Resources;
import com.google.devtools.build.lib.util.OS;
import com.google.testing.compile.JavaFileObjects;
import javax.tools.JavaFileObject;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for SkylarkConfigurationFieldProcessor.
 */
@RunWith(JUnit4.class)
public final class SkylarkConfigurationFieldProcessorTest {

  private static JavaFileObject getFile(String pathToFile) {
    return JavaFileObjects.forResource(Resources.getResource(
        SkylarkConfigurationFieldProcessorTest.class, "optiontestsources/" + pathToFile));
  }

  @Test
  public void testGoldenConfigurationField() throws Exception {
    // TODO(b/71644521): Compile-testing is not fully functional on Windows; test sources are
    // unable to resolve cross-package dependencies.
    assumeTrue(OS.getCurrent() != OS.WINDOWS);

    assertAbout(javaSource())
        .that(getFile("GoldenConfigurationField.java"))
        .processedWith(new SkylarkConfigurationFieldProcessor())
        .compilesWithoutError();
  }

  @Test
  public void testGoldenConfigurationFieldThroughApi() throws Exception {
    // TODO(b/71644521): Compile-testing is not fully functional on Windows; test sources are
    // unable to resolve cross-package dependencies.
    assumeTrue(OS.getCurrent() != OS.WINDOWS);

    assertAbout(javaSource())
        .that(getFile("GoldenConfigurationFieldThroughApi.java"))
        .processedWith(new SkylarkConfigurationFieldProcessor())
        .compilesWithoutError();
  }

  @Test
  public void testHasMethodParameters() throws Exception {
    // TODO(b/71644521): Compile-testing is not fully functional on Windows; test sources are
    // unable to resolve cross-package dependencies.
    assumeTrue(OS.getCurrent() != OS.WINDOWS);

    assertAbout(javaSource())
        .that(getFile("HasMethodParameters.java"))
        .processedWith(new SkylarkConfigurationFieldProcessor())
        .failsToCompile()
        .withErrorContaining(
            "@SkylarkConfigurationField annotated methods must have zero arguments.");
  }

  @Test
  public void testMethodIsPrivate() throws Exception {
    // TODO(b/71644521): Compile-testing is not fully functional on Windows; test sources are
    // unable to resolve cross-package dependencies.
    assumeTrue(OS.getCurrent() != OS.WINDOWS);

    assertAbout(javaSource())
        .that(getFile("MethodIsPrivate.java"))
        .processedWith(new SkylarkConfigurationFieldProcessor())
        .failsToCompile()
        .withErrorContaining("@SkylarkConfigurationField annotated methods must be public.");
  }

  @Test
  public void testMethodThrowsException() throws Exception {
    // TODO(b/71644521): Compile-testing is not fully functional on Windows; test sources are
    // unable to resolve cross-package dependencies.
    assumeTrue(OS.getCurrent() != OS.WINDOWS);

    assertAbout(javaSource())
        .that(getFile("MethodThrowsException.java"))
        .processedWith(new SkylarkConfigurationFieldProcessor())
        .failsToCompile()
        .withErrorContaining("@SkylarkConfigurationField annotated must not throw exceptions.");
  }

  @Test
  public void testNonConfigurationFragment() throws Exception {
    // TODO(b/71644521): Compile-testing is not fully functional on Windows; test sources are
    // unable to resolve cross-package dependencies.
    assumeTrue(OS.getCurrent() != OS.WINDOWS);

    assertAbout(javaSource())
        .that(getFile("NonConfigurationFragment.java"))
        .processedWith(new SkylarkConfigurationFieldProcessor())
        .failsToCompile()
        .withErrorContaining("@SkylarkConfigurationField annotated methods must be methods "
            + "of configuration fragments.");
  }

  @Test
  public void testNonExposedConfigurationFragment() throws Exception {
    // TODO(b/71644521): Compile-testing is not fully functional on Windows; test sources are
    // unable to resolve cross-package dependencies.
    assumeTrue(OS.getCurrent() != OS.WINDOWS);

    assertAbout(javaSource())
        .that(getFile("NonExposedConfigurationFragment.java"))
        .processedWith(new SkylarkConfigurationFieldProcessor())
        .compilesWithoutError();
  }

  @Test
  public void testReturnsOtherType() throws Exception {
    // TODO(b/71644521): Compile-testing is not fully functional on Windows; test sources are
    // unable to resolve cross-package dependencies.
    assumeTrue(OS.getCurrent() != OS.WINDOWS);

    assertAbout(javaSource())
        .that(getFile("ReturnsOtherType.java"))
        .processedWith(new SkylarkConfigurationFieldProcessor())
        .failsToCompile()
        .withErrorContaining("@SkylarkConfigurationField annotated methods must return Label.");
  }
}
