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
import static com.google.devtools.build.lib.packages.RuleVisibility.concatWithElement;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label;
import net.starlark.java.eval.EvalException;
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

  private static final String A = "//a:__pkg__";
  // Package group labels are represented differently than __pkg__ labels, so cover both cases.
  private static final String B = "//b:pkggroup";
  private static final String C = "//c:__pkg__";
  private static final String PUBLIC = "//visibility:public";
  private static final String PRIVATE = "//visibility:private";
  private static final RuleVisibility PUBLIC_VIS = RuleVisibility.PUBLIC;
  private static final RuleVisibility PRIVATE_VIS = RuleVisibility.PRIVATE;

  @Test
  public void validateAndSimplify_validates() throws Exception {
    EvalException e1 =
        assertThrows(
            EvalException.class,
            () ->
                RuleVisibility.validateAndSimplify(
                    ImmutableList.of(label("//visibility:pirvate"))));
    assertThat(e1)
        .hasMessageThat()
        .contains(
            "Invalid visibility label '//visibility:pirvate'; did you mean //visibility:public or"
                + " //visibility:private?");

    EvalException e2 =
        assertThrows(
            EvalException.class,
            () ->
                RuleVisibility.validateAndSimplify(
                    ImmutableList.of(label(PUBLIC), label("//visibility:pbulic"))));
    assertThat(e2)
        .hasMessageThat()
        .contains(
            "Invalid visibility label '//visibility:pbulic'; did you mean //visibility:public or"
                + " //visibility:private?");
  }

  @Test
  public void validateAndSimplify_simplifiesPublic() throws Exception {
    assertThat(RuleVisibility.validateAndSimplify(ImmutableList.of(label(A), label(PUBLIC))))
        .containsExactly(label(PUBLIC));
    assertThat(RuleVisibility.validateAndSimplify(ImmutableList.of(label(PUBLIC), label(B))))
        .containsExactly(label(PUBLIC));
    assertThat(RuleVisibility.validateAndSimplify(ImmutableList.of(label(PUBLIC), label(PRIVATE))))
        .containsExactly(label(PUBLIC));
    assertThat(RuleVisibility.validateAndSimplify(ImmutableList.of(label(PUBLIC), label(PUBLIC))))
        .containsExactly(label(PUBLIC));
  }

  @Test
  public void validateAndSimplify_simplifiesPrivate() throws Exception {
    assertThat(RuleVisibility.validateAndSimplify(ImmutableList.of(label(A), label(PRIVATE))))
        .containsExactly(label(A));
    assertThat(RuleVisibility.validateAndSimplify(ImmutableList.of(label(PRIVATE), label(B))))
        .containsExactly(label(B));
    assertThat(RuleVisibility.validateAndSimplify(ImmutableList.of(label(PRIVATE), label(PRIVATE))))
        .containsExactly(label(PRIVATE));
  }

  @Test
  public void emptyListCanonicalizedToPrivate() throws Exception {
    assertThat(RuleVisibility.validateAndSimplify(ImmutableList.of()))
        .containsExactly(label(PRIVATE));
    IllegalArgumentException e =
        assertThrows(
            IllegalArgumentException.class,
            () -> RuleVisibility.parseUnchecked(ImmutableList.of()));
    assertThat(e).hasMessageThat().contains("must not be empty");
  }

  // TODO(arostovtsev): we ought to uniquify the labels, but that would be an incompatible change
  // (affects query output).
  @Test
  public void validateAndSimplify_doesNotUniquify() throws Exception {
    assertThat(RuleVisibility.validateAndSimplify(ImmutableList.of(label(A), label(A))))
        .containsExactly(label(A), label(A));
  }

  @Test
  public void packageGroupsRuleVisibility_create_requiresValidatedSimplifiedNonConstantLabels()
      throws Exception {
    IllegalArgumentException e1 =
        assertThrows(
            IllegalArgumentException.class,
            () -> PackageGroupsRuleVisibility.create(ImmutableList.of()));
    assertThat(e1).hasMessageThat().contains("must not be empty");
    IllegalArgumentException e2 =
        assertThrows(
            IllegalArgumentException.class,
            () -> PackageGroupsRuleVisibility.create(ImmutableList.of(label(PUBLIC), label(A))));
    assertThat(e2).hasMessageThat().contains("must be validated and simplified");
    IllegalArgumentException e3 =
        assertThrows(
            IllegalArgumentException.class,
            () -> PackageGroupsRuleVisibility.create(ImmutableList.of(label(A), label(PRIVATE))));
    assertThat(e3).hasMessageThat().contains("must be validated and simplified");
    IllegalArgumentException e4 =
        assertThrows(
            IllegalArgumentException.class,
            () -> PackageGroupsRuleVisibility.create(ImmutableList.of(label(PUBLIC))));
    assertThat(e4)
        .hasMessageThat()
        .contains("must not equal [\"//visibility:public\"] or [\"//visibility:private\"]");
    IllegalArgumentException e5 =
        assertThrows(
            IllegalArgumentException.class,
            () -> PackageGroupsRuleVisibility.create(ImmutableList.of(label(PRIVATE))));
    assertThat(e5)
        .hasMessageThat()
        .contains("must not equal [\"//visibility:public\"] or [\"//visibility:private\"]");
  }

  @Test
  public void concatenation() throws Exception {
    assertEqual(concatWithElement(ruleVisibility(A, B), label(C)), ruleVisibility(A, B, C));
    assertEqual(concatWithElement(ruleVisibility(A, B), label(PUBLIC)), PUBLIC_VIS);
    assertEqual(concatWithElement(ruleVisibility(A, B), label(PRIVATE)), ruleVisibility(A, B));

    assertEqual(concatWithElement(PUBLIC_VIS, label(C)), PUBLIC_VIS);
    assertEqual(concatWithElement(PUBLIC_VIS, label(PUBLIC)), PUBLIC_VIS);
    assertEqual(concatWithElement(PUBLIC_VIS, label(PRIVATE)), PUBLIC_VIS);

    assertEqual(concatWithElement(PRIVATE_VIS, label(C)), ruleVisibility(C));
    assertEqual(concatWithElement(PRIVATE_VIS, label(PUBLIC)), PUBLIC_VIS);
    assertEqual(concatWithElement(PRIVATE_VIS, label(PRIVATE)), PRIVATE_VIS);

    // Duplicates are not added, though they are preserved.
    assertEqual(concatWithElement(ruleVisibility(A, B), label(A)), ruleVisibility(A, B));
    assertEqual(
        concatWithElement(ruleVisibility(A, B, B, A), label(A)), ruleVisibility(A, B, B, A));
    assertEqual(
        concatWithElement(ruleVisibility(A, B, B, A), label(C)), ruleVisibility(A, B, B, A, C));
  }
}
