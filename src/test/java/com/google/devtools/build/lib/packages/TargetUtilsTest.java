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
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.google.common.base.Predicate;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.packages.util.PackageLoadingTestCase;
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
    assertEquals("java", TargetUtils.getRuleLanguage("java_binary"));
    assertEquals("foobar", TargetUtils.getRuleLanguage("foobar"));
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
    assertTrue(tagFilter.apply(tag1));
    assertTrue(tagFilter.apply(tag2));
    assertTrue(tagFilter.apply(tag1b));
    tagFilter = TargetUtils.tagFilter(Lists.newArrayList("tag1", "tag2"));
    assertTrue(tagFilter.apply(tag1));
    assertTrue(tagFilter.apply(tag2));
    assertTrue(tagFilter.apply(tag1b));
    tagFilter = TargetUtils.tagFilter(Lists.newArrayList("tag1"));
    assertTrue(tagFilter.apply(tag1));
    assertFalse(tagFilter.apply(tag2));
    assertTrue(tagFilter.apply(tag1b));
    tagFilter = TargetUtils.tagFilter(Lists.newArrayList("-tag2"));
    assertTrue(tagFilter.apply(tag1));
    assertFalse(tagFilter.apply(tag2));
    assertTrue(tagFilter.apply(tag1b));
    // Applying same tag as positive and negative filter produces an empty
    // result because the negative filter is applied first and positive filter will
    // not match anything.
    tagFilter = TargetUtils.tagFilter(Lists.newArrayList("tag2", "-tag2"));
    assertFalse(tagFilter.apply(tag1));
    assertFalse(tagFilter.apply(tag2));
    assertFalse(tagFilter.apply(tag1b));
    tagFilter = TargetUtils.tagFilter(Lists.newArrayList("tag2", "-tag1"));
    assertFalse(tagFilter.apply(tag1));
    assertTrue(tagFilter.apply(tag2));
    assertFalse(tagFilter.apply(tag1b));
  }
}
