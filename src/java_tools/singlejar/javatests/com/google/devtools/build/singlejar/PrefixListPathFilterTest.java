// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.singlejar;

import static com.google.common.truth.Truth.assertWithMessage;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.singlejar.DefaultJarEntryFilter.PathFilter;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests {@link PrefixListPathFilter}.
 */
@RunWith(JUnit4.class)
public class PrefixListPathFilterTest {
  private PathFilter filter;

  @Test
  public void testPrefixList() {
    filter = new PrefixListPathFilter(ImmutableList.of("dir1", "dir/subdir"));
    assertIncluded("dir1/file1");
    assertExcluded("dir2/file1");
    assertIncluded("dir/subdir/file1");
    assertExcluded("dir2/subdir/file1");
    assertExcluded("dir/othersub/file1");
    assertExcluded("dir3/file1");
  }

  private void assertExcluded(String path) {
    assertWithMessage(path + " should have been excluded, but was included")
        .that(filter.allowed(path))
        .isFalse();
  }

  private void assertIncluded(String path) {
    assertWithMessage(path + " should have been included but was not")
        .that(filter.allowed(path))
        .isTrue();
  }
}
