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

package com.google.devtools.build.lib.bazel.rules.java;

import static org.junit.Assert.assertEquals;

import com.google.devtools.build.lib.vfs.PathFragment;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for {@link BazelJavaSemantics}.
 */
@RunWith(JUnit4.class)
public class BazelJavaSemanticsTest {
  @Test
  public void testFindingResources() {
    BazelJavaSemantics semantics = BazelJavaSemantics.INSTANCE;
    assertEquals(
        PathFragment.EMPTY_FRAGMENT,
        semantics.getDefaultJavaResourcePath(PathFragment.create("x/y/src/main/resources")));
    assertEquals(
        PathFragment.create("foo"),
        semantics.getDefaultJavaResourcePath(PathFragment.create("x/y/src/main/resources/foo")));
    assertEquals(
        PathFragment.create("foo"),
        semantics.getDefaultJavaResourcePath(
            PathFragment.create("java/x/y/src/main/resources/foo")));
    assertEquals(
        PathFragment.create("foo/java/bar"),
        semantics.getDefaultJavaResourcePath(PathFragment.create("java/foo/java/bar")));
    assertEquals(
        PathFragment.create("foo/java/bar"),
        semantics.getDefaultJavaResourcePath(PathFragment.create("javatests/foo/java/bar")));
  }
}
