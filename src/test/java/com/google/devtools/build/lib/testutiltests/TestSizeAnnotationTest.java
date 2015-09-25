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
package com.google.devtools.build.lib.testutiltests;

import static com.google.devtools.build.lib.testutil.Suite.getSize;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;

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

  @TestSpec(flaky = true)
  private static class FlakyTestSpecAnnotation {

  }

  @TestSpec(suite = "foo")
  private static class HasNoSizeAnnotationElement {

  }

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
    assertEquals(Suite.SMALL_TESTS, getSize(HasNoTestSpecAnnotation.class));
  }

  @Test
  public void testHasNoSizeAnnotationElementIsSmall() {
    assertEquals(Suite.SMALL_TESTS, getSize(HasNoSizeAnnotationElement.class));
  }

  @Test
  public void testIsAnnotatedWithSmallSizeIsSmall() {
    assertEquals(Suite.SMALL_TESTS, getSize(IsAnnotatedWithSmallSize.class));
  }

  @Test
  public void testIsAnnotatedWithMediumSizeIsMedium() {
    assertEquals(Suite.MEDIUM_TESTS, getSize(IsAnnotatedWithMediumSize.class));
  }

  @Test
  public void testIsAnnotatedWithLargeSizeIsLarge() {
    assertEquals(Suite.LARGE_TESTS, getSize(IsAnnotatedWithLargeSize.class));
  }

  @Test
  public void testSuperclassHasAnnotationButNoSizeElement() {
    assertEquals(Suite.SMALL_TESTS, getSize(SuperclassHasAnnotationButNoSizeElement.class));
  }

  @Test
  public void testHasSizeElementAndSuperclassHasAnnotationButNoSizeElement() {
    assertEquals(Suite.LARGE_TESTS,
        getSize(HasSizeElementAndSuperclassHasAnnotationButNoSizeElement.class));
  }

  @Test
  public void testSuperclassHasAnnotationWithSizeElement() {
    assertEquals(Suite.SMALL_TESTS, getSize(SuperclassHasAnnotationWithSizeElement.class));
  }

  @Test
  public void testHasSizeElementAndSuperclassHasAnnotationWithSizeElement() {
    assertEquals(Suite.LARGE_TESTS,
        getSize(HasSizeElementAndSuperclassHasAnnotationWithSizeElement.class));
  }

  @Test
  public void testIsNotFlaky() {
    assertFalse(Suite.isFlaky(HasNoTestSpecAnnotation.class));
  }
  
  @Test
  public void testIsFlaky() {
    assertTrue(Suite.isFlaky(FlakyTestSpecAnnotation.class));
  }
}
