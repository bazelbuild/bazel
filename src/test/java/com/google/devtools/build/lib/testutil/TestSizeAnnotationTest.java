// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.testutil;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.Suite.getSize;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests {@link com.google.devtools.build.lib.testutil.Suite#getSize(Class)}.
 */
@RunWith(JUnit4.class)
public class TestSizeAnnotationTest {

  private static class HasNoTestSpecAnnotation {

  }

  @TestSpec
  private static class HasNoSizeAnnotationElement {}

  @TestSpec(size = Suite.SMALL_TESTS)
  private static class IsAnnotatedWithSmallSize {

  }

  @TestSpec(size = Suite.MEDIUM_TESTS)
  private static class IsAnnotatedWithMediumSize {

  }

  @TestSpec(size = Suite.LARGE_TESTS)
  private static class IsAnnotatedWithLargeSize {

  }

  private static class SuperclassHasAnnotationButNoSizeElement
      extends HasNoSizeAnnotationElement {

  }

  @TestSpec(size = Suite.LARGE_TESTS)
  private static class HasSizeElementAndSuperclassHasAnnotationButNoSizeElement
      extends HasNoSizeAnnotationElement {

  }

  private static class SuperclassHasAnnotationWithSizeElement
      extends IsAnnotatedWithSmallSize {

  }

  @TestSpec(size = Suite.LARGE_TESTS)
  private static class HasSizeElementAndSuperclassHasAnnotationWithSizeElement
      extends IsAnnotatedWithSmallSize {

  }

  @Test
  public void testHasNoTestSpecAnnotationIsSmall() {
    assertThat(getSize(HasNoTestSpecAnnotation.class)).isEqualTo(Suite.SMALL_TESTS);
  }

  @Test
  public void testHasNoSizeAnnotationElementIsSmall() {
    assertThat(getSize(HasNoSizeAnnotationElement.class)).isEqualTo(Suite.SMALL_TESTS);
  }

  @Test
  public void testIsAnnotatedWithSmallSizeIsSmall() {
    assertThat(getSize(IsAnnotatedWithSmallSize.class)).isEqualTo(Suite.SMALL_TESTS);
  }

  @Test
  public void testIsAnnotatedWithMediumSizeIsMedium() {
    assertThat(getSize(IsAnnotatedWithMediumSize.class)).isEqualTo(Suite.MEDIUM_TESTS);
  }

  @Test
  public void testIsAnnotatedWithLargeSizeIsLarge() {
    assertThat(getSize(IsAnnotatedWithLargeSize.class)).isEqualTo(Suite.LARGE_TESTS);
  }

  @Test
  public void testSuperclassHasAnnotationButNoSizeElement() {
    assertThat(getSize(SuperclassHasAnnotationButNoSizeElement.class)).isEqualTo(Suite.SMALL_TESTS);
  }

  @Test
  public void testHasSizeElementAndSuperclassHasAnnotationButNoSizeElement() {
    assertThat(getSize(HasSizeElementAndSuperclassHasAnnotationButNoSizeElement.class))
        .isEqualTo(Suite.LARGE_TESTS);
  }

  @Test
  public void testSuperclassHasAnnotationWithSizeElement() {
    assertThat(getSize(SuperclassHasAnnotationWithSizeElement.class)).isEqualTo(Suite.SMALL_TESTS);
  }

  @Test
  public void testHasSizeElementAndSuperclassHasAnnotationWithSizeElement() {
    assertThat(getSize(HasSizeElementAndSuperclassHasAnnotationWithSizeElement.class))
        .isEqualTo(Suite.LARGE_TESTS);
  }
}
