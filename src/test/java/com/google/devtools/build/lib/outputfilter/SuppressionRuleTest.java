// Copyright 2024 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.outputfilter;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link SuppressionRule}. */
@RunWith(JUnit4.class)
public class SuppressionRuleTest {

  @Test
  public void testSimpleRule() {
    SuppressionRule rule = SuppressionRule.create("some pattern");
    assertThat(rule.getPattern().pattern()).isEqualTo("some pattern");
    assertThat(rule.getExpectedCount()).isEqualTo(-1);
  }

  @Test
  public void testRuleWithCount() {
    SuppressionRule rule = SuppressionRule.create("count:5 some pattern");
    assertThat(rule.getPattern().pattern()).isEqualTo("some pattern");
    assertThat(rule.getExpectedCount()).isEqualTo(5);
  }

  @Test
  public void testRuleWithPackage() {
    SuppressionRule rule = SuppressionRule.create("package:@@foo//bar some pattern");
    assertThat(rule.hasKeyword("package")).isTrue();
    assertThat(rule.getKeywordValue("package")).isEqualTo("@@foo//bar");
    assertThat(rule.getPattern().pattern()).isEqualTo("some pattern");
  }

  @Test
  public void testRuleWithPath() {
    SuppressionRule rule = SuppressionRule.create("path:.*/foo/bar/.* some pattern");
    assertThat(rule.hasKeyword("path")).isTrue();
    assertThat(rule.getKeywordValue("path")).isEqualTo(".*/foo/bar/.*");
    assertThat(rule.getPattern().pattern()).isEqualTo("some pattern");
  }

  @Test
  public void testRuleWithMultipleKeywords() {
    SuppressionRule rule = SuppressionRule.create("count:3 path:.*/foo/.* package:@@bar some pattern");
    assertThat(rule.getExpectedCount()).isEqualTo(3);
    assertThat(rule.hasKeyword("path")).isTrue();
    assertThat(rule.getKeywordValue("path")).isEqualTo(".*/foo/.*");
    assertThat(rule.hasKeyword("package")).isTrue();
    assertThat(rule.getKeywordValue("package")).isEqualTo("@@bar");
    assertThat(rule.getPattern().pattern()).isEqualTo("some pattern");
  }

  @Test
  public void testInvalidRule_noPattern() {
    assertThrows(IllegalArgumentException.class, () -> SuppressionRule.create("count:1"));
  }
}
