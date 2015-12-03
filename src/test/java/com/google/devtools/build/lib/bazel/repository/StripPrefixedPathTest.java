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

package com.google.devtools.build.lib.bazel.repository;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.google.common.base.Optional;
import com.google.devtools.build.lib.vfs.PathFragment;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests {@link StripPrefixedPath}.
 */
@RunWith(JUnit4.class)
public class StripPrefixedPathTest {
  @Test
  public void testStrip() {
    StripPrefixedPath result = StripPrefixedPath.maybeDeprefix("foo/bar", Optional.of("foo"));
    assertEquals(result.getPathFragment(), new PathFragment("bar"));
    assertTrue(result.foundPrefix());
    assertFalse(result.skip());

    result = StripPrefixedPath.maybeDeprefix("foo", Optional.of("foo"));
    assertTrue(result.skip());

    result = StripPrefixedPath.maybeDeprefix("bar/baz", Optional.of("foo"));
    assertFalse(result.foundPrefix());

    result = StripPrefixedPath.maybeDeprefix("foof/bar", Optional.of("foo"));
    assertFalse(result.foundPrefix());
  }
}
