// Copyright 2010 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.junit4.runner;

import static com.google.common.truth.Truth.assertThat;

import java.util.regex.Pattern;
import org.junit.Test;
import org.junit.runner.Description;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link RegExTestCaseFilter}.
 */
@RunWith(JUnit4.class)
public class RegExTestCaseFilterTest {

  @Test
  public void testIncludesSuites() {
    RegExTestCaseFilter filter = RegExTestCaseFilter.include("doNotMatch");
    assertThat(filter.shouldRun(createSuiteDescription("suite"))).isTrue();
  }

  private Description createSuiteDescription(String name) {
    Description suite = Description.createSuiteDescription(name);
    suite.addChild(Description.createTestDescription(Object.class, "child"));
    return suite;
  }

  @Test
  public void testIncludesMatchingTestByFullNameQuotedRegex() {
    RegExTestCaseFilter filter = RegExTestCaseFilter.include(
        Pattern.quote("java.lang.Object#nameToMatch"));
    assertThat(filter.shouldRun(Description.createTestDescription(Object.class, "nameToMatch")))
        .isTrue();
  }

  @Test
  public void testIncludesMatchingTestByFullNameRegex() {
    RegExTestCaseFilter filter = RegExTestCaseFilter.include("^java.lang.Object#nameToMatch$");
    assertThat(filter.shouldRun(Description.createTestDescription(Object.class, "nameToMatch")))
        .isTrue();
  }

  @Test
  public void testIncludesMatchingTestBySimpleClassNameAndMethodName() {
    RegExTestCaseFilter filter = RegExTestCaseFilter.include("Object#nameToMatch");
    assertThat(filter.shouldRun(Description.createTestDescription(Object.class, "nameToMatch")))
        .isTrue();
  }

  @Test
  public void testIncludesMatchingTestWithNullMethodName() {
    RegExTestCaseFilter filter = RegExTestCaseFilter.include("java.lang.Object$");
    assertThat(filter.shouldRun(Description.createSuiteDescription(Object.class))).isTrue();
  }

  @Test
  public void testIncludesMatchingTestWithUnexpectedNameFormat() {
    RegExTestCaseFilter filter = RegExTestCaseFilter.include(
        Pattern.quote("java.lang.Object.hashCode()"));
    assertThat(filter.shouldRun(Description.createSuiteDescription("java.lang.Object.hashCode()")))
        .isTrue();
  }

  @Test
  public void testIncludesMatchingTestByTestMethodName() {
    RegExTestCaseFilter filter = RegExTestCaseFilter.include("nameToMatch");
    assertThat(filter.shouldRun(Description.createTestDescription(Object.class, "nameToMatch")))
        .isTrue();
  }

  @Test
  public void testExcludesNonmatchingTest() {
    RegExTestCaseFilter filter = RegExTestCaseFilter.include("doNotMatch");
    assertThat(filter.shouldRun(Description.createTestDescription(Object.class, "nameToMatch")))
        .isFalse();
  }

  @Test
  public void testFilterExcludeNonmatchingTest() {
    RegExTestCaseFilter filter = RegExTestCaseFilter.exclude("nameToMatch");
    assertThat(filter.shouldRun(Description.createTestDescription(Object.class, "nameToMatch")))
        .isFalse();
  }

  @Test
  public void testIncludesEmptyString() {
    RegExTestCaseFilter filter = RegExTestCaseFilter.include("");
    assertThat(filter.shouldRun(Description.createTestDescription(Object.class, "nameToMatch")))
        .isTrue();
  }

  @Test
  public void testIncludesMatchingByCaseRegex() {
    RegExTestCaseFilter filter = RegExTestCaseFilter.include("[Nn]ameToMatch");
    assertThat(filter.shouldRun(Description.createTestDescription(Object.class, "nameToMatch")))
        .isTrue();
  }

  @Test
  public void testIncludesMatchingByEscapedRegex() {
    RegExTestCaseFilter filter = RegExTestCaseFilter.include("java\\.lang\\.Object#nameToMatch");
    assertThat(filter.shouldRun(Description.createTestDescription(Object.class, "nameToMatch")))
        .isTrue();
  }

  @Test
  public void testIncludesMatchingByIncorrectCase() {
    RegExTestCaseFilter filter = RegExTestCaseFilter.include("NAMETOMATCH");
    assertThat(filter.shouldRun(Description.createTestDescription(Object.class, "nameToMatch")))
        .isFalse();
  }
}
