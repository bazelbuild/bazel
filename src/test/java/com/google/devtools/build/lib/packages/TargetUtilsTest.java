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
package com.google.devtools.build.lib.packages;

import com.google.common.base.Predicate;
import com.google.devtools.build.lib.packages.util.PackageLoadingTestCase;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.Starlark;
import net.starlark.java.syntax.Expression;
import net.starlark.java.syntax.ParserInput;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.Map;

import static com.google.common.truth.Truth.assertThat;

/**
 * Test for {@link TargetUtils}
 */
@RunWith(JUnit4.class)
public class TargetUtilsTest extends PackageLoadingTestCase {

  @Test
  public void getRuleLanguage() {
    assertThat(TargetUtils.getRuleLanguage("java_binary")).isEqualTo("java");
    assertThat(TargetUtils.getRuleLanguage("foobar")).isEqualTo("foobar");
    assertThat(TargetUtils.getRuleLanguage("")).isEmpty();
  }

  @Test
  public void testFilterWithNotExpression() throws Exception {
    scratch.file(
        "tests/BUILD",
        "sh_binary(name = 'tag1', srcs=['sh.sh'], tags=['tag1'])",
        "sh_binary(name = 'tag2', srcs=['sh.sh'], tags=['tag2'])",
        "sh_binary(name = 'tag1_and_tag2', srcs=['sh.sh'], tags=['tag1', 'tag2'])");

    Target tag1 = getTarget("//tests:tag1");
    Target tag2 = getTarget("//tests:tag2");
    Target tag1AndTag2 = getTarget("//tests:tag1_and_tag2");

    Predicate<Target> tagFilter = TargetUtils.tagFilter(Expression.parse(ParserInput.fromLines("not tag1")));
    assertEvaluatesToTrue(tagFilter, tag2);
    assertEvaluatesToFalse(tagFilter, tag1, tag1AndTag2);
  }

  @Test
  public void testFilterWithOrExpression() throws Exception {
    scratch.file(
        "tests/BUILD",
        "sh_binary(name = 'tag1', srcs=['sh.sh'], tags=['tag1'])",
        "sh_binary(name = 'tag2', srcs=['sh.sh'], tags=['tag2'])",
        "sh_binary(name = 'tag3', srcs=['sh.sh'], tags=['tag3'])",
        "sh_binary(name = 'tag1_and_tag2', srcs=['sh.sh'], tags=['tag1', 'tag2'])",
        "sh_binary(name = 'tag1_and_tag3', srcs=['sh.sh'], tags=['tag1', 'tag3'])",
        "sh_binary(name = 'tag2_and_tag3', srcs=['sh.sh'], tags=['tag2', 'tag3'])",
        "sh_binary(name = 'all_tags', srcs=['sh.sh'], tags=['tag1', 'tag2', 'tag3'])");

    Target tag1 = getTarget("//tests:tag1");
    Target tag2 = getTarget("//tests:tag2");
    Target tag3 = getTarget("//tests:tag3");
    Target tag1AndTag2 = getTarget("//tests:tag1_and_tag2");
    Target tag1AndTag3 = getTarget("//tests:tag1_and_tag3");
    Target tag2AndTag3 = getTarget("//tests:tag2_and_tag3");
    Target allTags = getTarget("//tests:all_tags");

    Predicate<Target> tagFilter = TargetUtils.tagFilter(Expression.parse(ParserInput.fromLines("tag1 or tag2")));
    assertEvaluatesToTrue(tagFilter, tag1, tag2, tag1AndTag2, tag1AndTag3, tag2AndTag3, allTags);
    assertEvaluatesToFalse(tagFilter, tag3);
  }

  @Test
  public void testFilterWithAndExpression() throws Exception {
    scratch.file(
        "tests/BUILD",
        "sh_binary(name = 'tag1', srcs=['sh.sh'], tags=['tag1'])",
        "sh_binary(name = 'tag2', srcs=['sh.sh'], tags=['tag2'])",
        "sh_binary(name = 'tag3', srcs=['sh.sh'], tags=['tag3'])",
        "sh_binary(name = 'tag1_and_tag2', srcs=['sh.sh'], tags=['tag1', 'tag2'])",
        "sh_binary(name = 'tag1_and_tag3', srcs=['sh.sh'], tags=['tag1', 'tag3'])",
        "sh_binary(name = 'tag2_and_tag3', srcs=['sh.sh'], tags=['tag2', 'tag3'])",
        "sh_binary(name = 'all_tags', srcs=['sh.sh'], tags=['tag1', 'tag2', 'tag3'])");

    Target tag1 = getTarget("//tests:tag1");
    Target tag2 = getTarget("//tests:tag2");
    Target tag3 = getTarget("//tests:tag3");
    Target tag1AndTag2 = getTarget("//tests:tag1_and_tag2");
    Target tag1AndTag3 = getTarget("//tests:tag1_and_tag3");
    Target tag2AndTag3 = getTarget("//tests:tag2_and_tag3");
    Target allTags = getTarget("//tests:all_tags");

    Predicate<Target> tagFilter = TargetUtils.tagFilter(Expression.parse(ParserInput.fromLines("tag1 and tag2")));
    assertEvaluatesToTrue(tagFilter, tag1AndTag2, allTags);
    assertEvaluatesToFalse(tagFilter, tag1, tag2, tag3, tag1AndTag3, tag2AndTag3);
  }

  @Test
  public void testFilterWithComplexExpression() throws Exception {
    scratch.file(
        "tests/BUILD",
        "sh_binary(name = 'tag1', srcs=['sh.sh'], tags=['tag1'])",
        "sh_binary(name = 'tag2', srcs=['sh.sh'], tags=['tag2'])",
        "sh_binary(name = 'tag3', srcs=['sh.sh'], tags=['tag3'])",
        "sh_binary(name = 'tag1_and_tag2', srcs=['sh.sh'], tags=['tag1', 'tag2'])",
        "sh_binary(name = 'tag1_and_tag3', srcs=['sh.sh'], tags=['tag1', 'tag3'])",
        "sh_binary(name = 'tag2_and_tag3', srcs=['sh.sh'], tags=['tag2', 'tag3'])",
        "sh_binary(name = 'all_tags', srcs=['sh.sh'], tags=['tag1', 'tag2', 'tag3'])");

    Target tag1 = getTarget("//tests:tag1");
    Target tag2 = getTarget("//tests:tag2");
    Target tag3 = getTarget("//tests:tag3");
    Target tag1AndTag2 = getTarget("//tests:tag1_and_tag2");
    Target tag1AndTag3 = getTarget("//tests:tag1_and_tag3");
    Target tag2AndTag3 = getTarget("//tests:tag2_and_tag3");
    Target allTags = getTarget("//tests:all_tags");

    Predicate<Target> tagFilter = TargetUtils.tagFilter(Expression.parse(ParserInput.fromLines("tag1 and tag2 or tag3")));
    assertEvaluatesToTrue(tagFilter, tag3, tag1AndTag2, tag1AndTag3, tag2AndTag3, allTags);
    assertEvaluatesToFalse(tagFilter, tag1, tag2);

    tagFilter = TargetUtils.tagFilter(Expression.parse(ParserInput.fromLines("tag1 and not tag2 or tag3")));
    assertEvaluatesToTrue(tagFilter, tag1, tag3, tag1AndTag3, tag2AndTag3, allTags);
    assertEvaluatesToFalse(tagFilter, tag2, tag1AndTag2);

    tagFilter = TargetUtils.tagFilter(Expression.parse(ParserInput.fromLines("tag1 and (tag2 or tag3)")));
    assertEvaluatesToTrue(tagFilter, tag1AndTag2, tag1AndTag3, allTags);
    assertEvaluatesToFalse(tagFilter, tag1, tag2, tag3, tag2AndTag3);

    tagFilter = TargetUtils.tagFilter(Expression.parse(ParserInput.fromLines("tag1 and (tag2 or not tag3)")));
    assertEvaluatesToTrue(tagFilter, tag1, tag1AndTag2, allTags);
    assertEvaluatesToFalse(tagFilter, tag2, tag3, tag1AndTag3, tag2AndTag3);

    tagFilter = TargetUtils.tagFilter(Expression.parse(ParserInput.fromLines("tag1 and not (tag2 or tag3)")));
    assertEvaluatesToTrue(tagFilter, tag1);
    assertEvaluatesToFalse(tagFilter, tag2, tag3, tag1AndTag2, tag1AndTag3, tag2AndTag3, allTags);

    tagFilter = TargetUtils.tagFilter(Expression.parse(ParserInput.fromLines("not tag1 and (not tag2 or tag3)")));
    assertEvaluatesToTrue(tagFilter, tag3, tag2AndTag3);
    assertEvaluatesToFalse(tagFilter, tag1, tag2, tag1AndTag2, tag1AndTag3, allTags);
  }

  @Test(expected = IllegalStateException.class)
  public void testForbiddenOperatorInFilter_FilterEvaluationExceptionIsThrown() throws Exception {
    scratch.file(
        "tests/BUILD",
        "sh_binary(name = 'tag1', srcs=['sh.sh'], tags=['tag1'])");

    Target tag1 = getTarget("//tests:tag1");

    Predicate<Target> tagFilter = TargetUtils.tagFilter(Expression.parse(ParserInput.fromLines("tag1 > tag2")));
    tagFilter.apply(tag1);
  }

  @Test
  public void testExecutionInfo() throws Exception {
    scratch.file(
        "tests/BUILD",
        "sh_binary(name = 'tag1', srcs=['sh.sh'], tags=['supports-workers', 'no-cache'])",
        "sh_binary(name = 'tag2', srcs=['sh.sh'], tags=['disable-local-prefetch'])",
        "sh_binary(name = 'tag1b', srcs=['sh.sh'], tags=['local', 'cpu:4'])");

    Rule tag1 = (Rule) getTarget("//tests:tag1");
    Rule tag2 = (Rule) getTarget("//tests:tag2");
    Rule tag1b = (Rule) getTarget("//tests:tag1b");

    Map<String, String> execInfo = TargetUtils.getExecutionInfo(tag1);
    assertThat(execInfo).containsExactly("supports-workers", "", "no-cache", "");
    execInfo = TargetUtils.getExecutionInfo(tag2);
    assertThat(execInfo).containsExactly("disable-local-prefetch", "");
    execInfo = TargetUtils.getExecutionInfo(tag1b);
    assertThat(execInfo).containsExactly("local", "", "cpu:4", "");
  }

  @Test
  public void testExecutionInfo_withPrefixSupports() throws Exception {
    scratch.file(
        "tests/BUILD",
        "sh_binary(name = 'with-prefix-supports', srcs=['sh.sh'], tags=['supports-workers',"
            + " 'supports-whatever', 'my-tag'])");

    Rule withSupportsPrefix = (Rule) getTarget("//tests:with-prefix-supports");

    Map<String, String> execInfo = TargetUtils.getExecutionInfo(withSupportsPrefix);
    assertThat(execInfo).containsExactly("supports-whatever", "", "supports-workers", "");
  }

  @Test
  public void testExecutionInfo_withPrefixDisable() throws Exception {
    scratch.file(
        "tests/BUILD",
        "sh_binary(name = 'with-prefix-disable', srcs=['sh.sh'], tags=['disable-local-prefetch',"
            + " 'disable-something-else', 'another-tag'])");

    Rule withDisablePrefix = (Rule) getTarget("//tests:with-prefix-disable");

    Map<String, String> execInfo = TargetUtils.getExecutionInfo(withDisablePrefix);
    assertThat(execInfo)
        .containsExactly("disable-local-prefetch", "", "disable-something-else", "");
  }

  @Test
  public void testExecutionInfo_withPrefixNo() throws Exception {
    scratch.file(
        "tests/BUILD",
        "sh_binary(name = 'with-prefix-no', srcs=['sh.sh'], tags=['no-remote-imaginary-flag',"
            + " 'no-sandbox', 'unknown'])");

    Rule withNoPrefix = (Rule) getTarget("//tests:with-prefix-no");

    Map<String, String> execInfo = TargetUtils.getExecutionInfo(withNoPrefix);
    assertThat(execInfo).containsExactly("no-remote-imaginary-flag", "", "no-sandbox", "");
  }

  @Test
  public void testExecutionInfo_withPrefixRequires() throws Exception {
    scratch.file(
        "tests/BUILD",
        "sh_binary(name = 'with-prefix-requires', srcs=['sh.sh'], tags=['requires-network',"
            + " 'requires-sunlight', 'test-only'])");

    Rule withRequiresPrefix = (Rule) getTarget("//tests:with-prefix-requires");

    Map<String, String> execInfo = TargetUtils.getExecutionInfo(withRequiresPrefix);
    assertThat(execInfo).containsExactly("requires-network", "", "requires-sunlight", "");
  }

  @Test
  public void testExecutionInfo_withPrefixBlock() throws Exception {
    scratch.file(
        "tests/BUILD",
        "sh_binary(name = 'with-prefix-block', srcs=['sh.sh'], tags=['block-some-feature',"
            + " 'block-network', 'wrong-tag'])");

    Rule withBlockPrefix = (Rule) getTarget("//tests:with-prefix-block");

    Map<String, String> execInfo = TargetUtils.getExecutionInfo(withBlockPrefix);
    assertThat(execInfo).containsExactly("block-network", "", "block-some-feature", "");
  }

  @Test
  public void testExecutionInfo_withPrefixCpu() throws Exception {
    scratch.file(
        "tests/BUILD",
        "sh_binary(name = 'with-prefix-cpu', srcs=['sh.sh'], tags=['cpu:123', 'wrong-tag'])");

    Rule withCpuPrefix = (Rule) getTarget("//tests:with-prefix-cpu");

    Map<String, String> execInfo = TargetUtils.getExecutionInfo(withCpuPrefix);
    assertThat(execInfo).containsExactly("cpu:123", "");
  }

  @Test
  public void testExecutionInfo_withLocalTag() throws Exception {
    scratch.file(
        "tests/BUILD",
        "sh_binary(name = 'with-local-tag', srcs=['sh.sh'], tags=['local', 'some-tag'])");

    Rule withLocal = (Rule) getTarget("//tests:with-local-tag");

    Map<String, String> execInfo = TargetUtils.getExecutionInfo(withLocal);
    assertThat(execInfo).containsExactly("local", "");
  }

  @Test
  public void testFilteredExecutionInfo_fromUncheckedExecRequirements() throws Exception {
    scratch.file("tests/BUILD", "sh_binary(name = 'no-tag', srcs=['sh.sh'])");

    Rule noTag = (Rule) getTarget("//tests:no-tag");

    Map<String, String> execInfo =
        TargetUtils.getFilteredExecutionInfo(
            Dict.<String, String>builder().put("supports-worker", "1").buildImmutable(),
            noTag, /* allowTagsPropagation */
            true);
    assertThat(execInfo).containsExactly("supports-worker", "1");

    execInfo =
        TargetUtils.getFilteredExecutionInfo(
            Dict.<String, String>builder()
                .put("some-custom-tag", "1")
                .put("no-cache", "1")
                .buildImmutable(),
            noTag,
            /* allowTagsPropagation */ true);
    assertThat(execInfo).containsExactly("no-cache", "1");
  }

  @Test
  public void testFilteredExecutionInfo_fromUncheckedExecRequirements_withWorkerKeyMnemonic()
      throws Exception {
    scratch.file("tests/BUILD", "sh_binary(name = 'no-tag', srcs=['sh.sh'])");

    Rule noTag = (Rule) getTarget("//tests:no-tag");

    Map<String, String> execInfo =
        TargetUtils.getFilteredExecutionInfo(
            Dict.<String, String>builder()
                .put("supports-workers", "1")
                .put("worker-key-mnemonic", "MyMnemonic")
                .buildImmutable(),
            noTag, /* allowTagsPropagation */
            true);
    assertThat(execInfo)
        .containsExactly("supports-workers", "1", "worker-key-mnemonic", "MyMnemonic");
  }

  @Test
  public void testFilteredExecutionInfo() throws Exception {
    scratch.file(
        "tests/BUILD",
        "sh_binary(name = 'tag1', srcs=['sh.sh'], tags=['supports-workers', 'no-cache'])");
    Rule tag1 = (Rule) getTarget("//tests:tag1");
    Dict<String, String> executionRequirementsUnchecked =
        Dict.<String, String>builder().put("no-remote", "1").buildImmutable();

    Map<String, String> execInfo =
        TargetUtils.getFilteredExecutionInfo(
            executionRequirementsUnchecked, tag1, /* allowTagsPropagation */ true);

    assertThat(execInfo).containsExactly("no-cache", "", "supports-workers", "", "no-remote", "1");
  }

  @Test
  public void testFilteredExecutionInfo_withDuplicateTags() throws Exception {
    scratch.file(
        "tests/BUILD",
        "sh_binary(name = 'tag1', srcs=['sh.sh'], tags=['supports-workers', 'no-cache'])");
    Rule tag1 = (Rule) getTarget("//tests:tag1");
    Dict<String, String> executionRequirementsUnchecked =
        Dict.<String, String>builder().put("no-cache", "1").buildImmutable();

    Map<String, String> execInfo =
        TargetUtils.getFilteredExecutionInfo(
            executionRequirementsUnchecked, tag1, /* allowTagsPropagation */ true);

    assertThat(execInfo).containsExactly("no-cache", "1", "supports-workers", "");
  }

  @Test
  public void testFilteredExecutionInfo_withNullUncheckedExecRequirements() throws Exception {
    scratch.file(
        "tests/BUILD",
        "sh_binary(name = 'tag1', srcs=['sh.sh'], tags=['supports-workers', 'no-cache'])");
    Rule tag1 = (Rule) getTarget("//tests:tag1");

    Map<String, String> execInfo =
        TargetUtils.getFilteredExecutionInfo(null, tag1, /* allowTagsPropagation */ true);
    assertThat(execInfo).containsExactly("no-cache", "", "supports-workers", "");

    execInfo =
        TargetUtils.getFilteredExecutionInfo(Starlark.NONE, tag1, /* allowTagsPropagation */ true);
    assertThat(execInfo).containsExactly("no-cache", "", "supports-workers", "");
  }

  @Test
  public void testFilteredExecutionInfo_whenIncompatibleFlagDisabled() throws Exception {
    // when --incompatible_allow_tags_propagation=false
    scratch.file(
        "tests/BUILD",
        "sh_binary(name = 'tag1', srcs=['sh.sh'], tags=['supports-workers', 'no-cache'])");
    Rule tag1 = (Rule) getTarget("//tests:tag1");
    Dict<String, String> executionRequirementsUnchecked =
        Dict.<String, String>builder().put("no-remote", "1").buildImmutable();

    Map<String, String> execInfo =
        TargetUtils.getFilteredExecutionInfo(
            executionRequirementsUnchecked, tag1, /* allowTagsPropagation */ false);

    assertThat(execInfo).containsExactly("no-remote", "1");
  }

  private void assertEvaluatesToTrue(Predicate<Target> tagFilter, Target ... targets) {
    for (Target target : targets) {
      assertThat(tagFilter.apply(target)).isTrue();
    }
  }

  private void assertEvaluatesToFalse(Predicate<Target> tagFilter, Target ... targets) {
    for (Target target : targets) {
      assertThat(tagFilter.apply(target)).isFalse();
    }
  }
}
