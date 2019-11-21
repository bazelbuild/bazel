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

import static com.google.common.truth.Truth.assertThat;

import com.google.common.base.Predicate;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.packages.util.PackageLoadingTestCase;
import com.google.devtools.build.lib.syntax.Dict;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.Starlark;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

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
  public void testFilterByTag() throws Exception {
    scratch.file(
        "tests/BUILD",
        "sh_binary(name = 'tag1', srcs=['sh.sh'], tags=['tag1'])",
        "sh_binary(name = 'tag2', srcs=['sh.sh'], tags=['tag2'])",
        "sh_binary(name = 'tag1b', srcs=['sh.sh'], tags=['tag1'])");

    Target tag1 = getTarget("//tests:tag1");
    Target tag2 = getTarget("//tests:tag2");
    Target  tag1b = getTarget("//tests:tag1b");

    Predicate<Target> tagFilter = TargetUtils.tagFilter(Lists.<String>newArrayList());
    assertThat(tagFilter.apply(tag1)).isTrue();
    assertThat(tagFilter.apply(tag2)).isTrue();
    assertThat(tagFilter.apply(tag1b)).isTrue();
    tagFilter = TargetUtils.tagFilter(Lists.newArrayList("tag1", "tag2"));
    assertThat(tagFilter.apply(tag1)).isTrue();
    assertThat(tagFilter.apply(tag2)).isTrue();
    assertThat(tagFilter.apply(tag1b)).isTrue();
    tagFilter = TargetUtils.tagFilter(Lists.newArrayList("tag1"));
    assertThat(tagFilter.apply(tag1)).isTrue();
    assertThat(tagFilter.apply(tag2)).isFalse();
    assertThat(tagFilter.apply(tag1b)).isTrue();
    tagFilter = TargetUtils.tagFilter(Lists.newArrayList("-tag2"));
    assertThat(tagFilter.apply(tag1)).isTrue();
    assertThat(tagFilter.apply(tag2)).isFalse();
    assertThat(tagFilter.apply(tag1b)).isTrue();
    // Applying same tag as positive and negative filter produces an empty
    // result because the negative filter is applied first and positive filter will
    // not match anything.
    tagFilter = TargetUtils.tagFilter(Lists.newArrayList("tag2", "-tag2"));
    assertThat(tagFilter.apply(tag1)).isFalse();
    assertThat(tagFilter.apply(tag2)).isFalse();
    assertThat(tagFilter.apply(tag1b)).isFalse();
    tagFilter = TargetUtils.tagFilter(Lists.newArrayList("tag2", "-tag1"));
    assertThat(tagFilter.apply(tag1)).isFalse();
    assertThat(tagFilter.apply(tag2)).isTrue();
    assertThat(tagFilter.apply(tag1b)).isFalse();
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
  public void testFilteredExecutionInfo_FromUncheckedExecRequirements() throws Exception {
    scratch.file("tests/BUILD", "sh_binary(name = 'no-tag', srcs=['sh.sh'])");

    Rule noTag = (Rule) getTarget("//tests:no-tag");

    Map<String, String> execInfo =
        TargetUtils.getFilteredExecutionInfo(
            Dict.of((Mutability) null, "supports-worker", "1"),
            noTag, /* allowTagsPropagation */
            true);
    assertThat(execInfo).containsExactly("supports-worker", "1");

    execInfo =
        TargetUtils.getFilteredExecutionInfo(
            Dict.of((Mutability) null, "some-custom-tag", "1", "no-cache", "1"),
            noTag,
            /* allowTagsPropagation */ true);
    assertThat(execInfo).containsExactly("no-cache", "1");
  }

  @Test
  public void testFilteredExecutionInfo() throws Exception {
    scratch.file(
        "tests/BUILD",
        "sh_binary(name = 'tag1', srcs=['sh.sh'], tags=['supports-workers', 'no-cache'])");
    Rule tag1 = (Rule) getTarget("//tests:tag1");
    Dict<String, String> executionRequirementsUnchecked =
        Dict.of((Mutability) null, "no-remote", "1");

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
        Dict.of((Mutability) null, "no-cache", "1");

    Map<String, String> execInfo =
        TargetUtils.getFilteredExecutionInfo(
            executionRequirementsUnchecked, tag1, /* allowTagsPropagation */ true);

    assertThat(execInfo).containsExactly("no-cache", "1", "supports-workers", "");
  }

  @Test
  public void testFilteredExecutionInfo_WithNullUncheckedExecRequirements() throws Exception {
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
  public void testFilteredExecutionInfoWhenIncompatibleFlagDisabled() throws Exception {
    // when --incompatible_allow_tags_propagation=false
    scratch.file(
        "tests/BUILD",
        "sh_binary(name = 'tag1', srcs=['sh.sh'], tags=['supports-workers', 'no-cache'])");
    Rule tag1 = (Rule) getTarget("//tests:tag1");
    Dict<String, String> executionRequirementsUnchecked =
        Dict.of((Mutability) null, "no-remote", "1");

    Map<String, String> execInfo =
        TargetUtils.getFilteredExecutionInfo(
            executionRequirementsUnchecked, tag1, /* allowTagsPropagation */ false);

    assertThat(execInfo).containsExactly("no-remote", "1");
  }
}
