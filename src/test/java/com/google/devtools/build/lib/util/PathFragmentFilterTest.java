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
package com.google.devtools.build.lib.util;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * A test for {@link PathFragmentFilter}.
 */
@RunWith(JUnit4.class)
public class PathFragmentFilterTest {
  protected PathFragmentFilter filter = null;

  protected void createFilter(String filterString) {
    filter = new PathFragmentFilter.PathFragmentFilterConverter().convert(filterString);
  }

  protected void assertIncluded(String path) {
    assertThat(filter.isIncluded(PathFragment.create(path))).isTrue();
  }

  protected void assertExcluded(String path) {
    assertThat(filter.isIncluded(PathFragment.create(path))).isFalse();
  }

  @Test
  public void emptyFilter() {
    createFilter("");
    assertIncluded("a/b/c");
    assertIncluded("d");
  }

  @Test
  public void inclusions() {
    createFilter("a/b,c");
    assertIncluded("a/b");
    assertIncluded("a/b/c");
    assertIncluded("c");
    assertIncluded("c/d");
    assertExcluded("a");
    assertExcluded("a/c");
    assertExcluded("d");
    assertExcluded("e/f/g");
  }

  @Test
  public void exclusions() {
    createFilter("-a/b,-c");
    assertExcluded("a/b");
    assertExcluded("a/b/c");
    assertExcluded("c");
    assertExcluded("c/d");
    assertIncluded("a");
    assertIncluded("a/c");
    assertIncluded("d");
    assertIncluded("e/f/g");
  }

  @Test
  public void inclusionsAndExclusions() {
    createFilter("a,-c,,d,a/b/c,-a/b,a/b/d");
    assertIncluded("a");
    assertIncluded("a/c");
    assertExcluded("a/b");
    assertExcluded("a/b/c"); // Exclusions take precedence over inclusions. Order is not important.
    assertExcluded("a/b/d"); // Exclusions take precedence over inclusions. Order is not important.
    assertExcluded("c");
    assertExcluded("c/d");
    assertIncluded("d/e");
    assertExcluded("e");
    // When converted back to string, inclusion entries will be put first, followed by exclusion
    // entries.
    assertThat(filter.toString()).isEqualTo("a,d,a/b/c,a/b/d,-c,-a/b");
  }

}
