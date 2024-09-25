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

package com.google.devtools.build.lib.packages;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RuleVisibility}. */
@RunWith(JUnit4.class)
public final class RuleVisibilityTest {

  private static Label label(String labelString) {
    return Label.parseCanonicalUnchecked(labelString);
  }

  private static RuleVisibility ruleVisibility(String... labelStrings) {
    ImmutableList.Builder<Label> labels = ImmutableList.builder();
    for (String labelString : labelStrings) {
      labels.add(label(labelString));
    }
    return RuleVisibility.parseUnchecked(labels.build());
  }

  // Needed because RuleVisibility has no equals() override.
  private static void assertEqual(RuleVisibility vis1, RuleVisibility vis2) {
    assertThat(vis1.getDependencyLabels()).isEqualTo(vis2.getDependencyLabels());
    assertThat(vis1.getDeclaredLabels()).isEqualTo(vis2.getDeclaredLabels());
  }

  @Test
  public void concatenation() throws Exception {
    RuleVisibility normalVis = ruleVisibility("//a:__pkg__", "//b:__pkg__");
    RuleVisibility publicVis = RuleVisibility.PUBLIC;
    RuleVisibility privateVis = RuleVisibility.PRIVATE;
    // Technically the empty visibility is a distinct object from private visibility, even though
    // it has the same semantics.
    RuleVisibility emptyVis = ruleVisibility();

    assertEqual(
        RuleVisibility.concatWithElement(normalVis, label("//c:__pkg__")),
        ruleVisibility("//a:__pkg__", "//b:__pkg__", "//c:__pkg__"));
    assertEqual(
        RuleVisibility.concatWithElement(normalVis, label("//visibility:public")), publicVis);
    assertEqual(
        RuleVisibility.concatWithElement(normalVis, label("//visibility:private")), normalVis);

    assertEqual(RuleVisibility.concatWithElement(publicVis, label("//c:__pkg__")), publicVis);
    assertEqual(
        RuleVisibility.concatWithElement(publicVis, label("//visibility:public")), publicVis);
    assertEqual(
        RuleVisibility.concatWithElement(publicVis, label("//visibility:private")), publicVis);

    assertEqual(
        RuleVisibility.concatWithElement(privateVis, label("//c:__pkg__")),
        ruleVisibility("//c:__pkg__"));
    assertEqual(
        RuleVisibility.concatWithElement(privateVis, label("//visibility:public")), publicVis);
    assertEqual(
        RuleVisibility.concatWithElement(privateVis, label("//visibility:private")), privateVis);

    assertEqual(
        RuleVisibility.concatWithElement(emptyVis, label("//c:__pkg__")),
        ruleVisibility("//c:__pkg__"));
    assertEqual(
        RuleVisibility.concatWithElement(emptyVis, label("//visibility:public")), publicVis);
    assertEqual(
        RuleVisibility.concatWithElement(emptyVis, label("//visibility:private")), emptyVis);
  }
}
