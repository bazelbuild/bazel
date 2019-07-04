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
}
