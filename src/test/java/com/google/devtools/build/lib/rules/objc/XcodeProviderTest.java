// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.rules.objc.XcodeProvider.xcodeTargetName;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.vfs.PathFragment;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for XcodeProvider.
 *
 * <p>We test {@code rootEach} individually. Because it is used as test utility code, any wrong
 * behavior may go undetected otherwise.
 */
@RunWith(JUnit4.class)
public class XcodeProviderTest {
  @Test
  public void testXcodeTargetName() throws Exception {
    assertThat(xcodeTargetName(Label.parseAbsolute("//foo:bar"))).isEqualTo("bar_foo");
    assertThat(xcodeTargetName(Label.parseAbsolute("//foo/bar:baz"))).isEqualTo("baz_bar_foo");
  }

  @Test
  public void testExternalXcodeTargetName() throws Exception {
    assertThat(xcodeTargetName(Label.parseAbsolute("@repo_name//foo:bar")))
        .isEqualTo("bar_external_repo_name_foo");
  }

  private static Iterable<PathFragment> fragments(String... paths) {
    return Iterables.transform(ImmutableList.copyOf(paths), PathFragment.TO_PATH_FRAGMENT);
  }

  @Test
  public void testRootEach_nonEmptySequence() {
    assertThat(XcodeProvider.rootEach("$(prefix)", fragments("a", "b/c")))
        .containsExactly("$(prefix)/a", "$(prefix)/b/c")
        .inOrder();
  }

  @Test
  public void testRootEach_emptyFragment() {
    assertThat(XcodeProvider.rootEach("$(foo)", fragments("", "bar", ".")))
        .containsExactly("$(foo)", "$(foo)/bar", "$(foo)")
        .inOrder();
  }

  @Test
  public void testRootEach_noElements() {
    assertThat(XcodeProvider.rootEach("$(prefix)", fragments()))
        .isEmpty();
  }

  @Test
  public void testRootEach_errorForTrailingSlash() {
    try {
      XcodeProvider.rootEach("$(prefix)/", fragments("a"));
      fail("should have thrown");
    } catch (IllegalArgumentException expected) {
    }
  }
}
