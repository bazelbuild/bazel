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
package com.google.devtools.build.lib.actions;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link Root}. */
@RunWith(JUnit4.class)
public class RootTest {
  private Scratch scratch = new Scratch();

  @Test
  public void testAsSourceRoot() throws IOException {
    Path sourceDir = scratch.dir("/source");
    Root root = Root.asSourceRoot(sourceDir);
    assertThat(root.isSourceRoot()).isTrue();
    assertThat(root.getExecPath()).isEqualTo(PathFragment.EMPTY_FRAGMENT);
    assertThat(root.getPath()).isEqualTo(sourceDir);
    assertThat(root.toString()).isEqualTo("/source[source]");
  }

  @Test
  public void testBadAsSourceRoot() {
    try {
      Root.asSourceRoot(null);
      fail();
    } catch (NullPointerException expected) {
    }
  }

  @Test
  public void testAsDerivedRoot() throws IOException {
    Path execRoot = scratch.dir("/exec");
    Path rootDir = scratch.dir("/exec/root");
    Root root = Root.asDerivedRoot(execRoot, rootDir);
    assertThat(root.isSourceRoot()).isFalse();
    assertThat(root.getExecPath()).isEqualTo(PathFragment.create("root"));
    assertThat(root.getPath()).isEqualTo(rootDir);
    assertThat(root.toString()).isEqualTo("/exec/root[derived]");
  }

  @Test
  public void testBadAsDerivedRoot() throws IOException {
    try {
      Path execRoot = scratch.dir("/exec");
      Path outsideDir = scratch.dir("/not_exec");
      Root.asDerivedRoot(execRoot, outsideDir);
      fail();
    } catch (IllegalArgumentException expected) {
    }
  }

  @Test
  public void testBadAsDerivedRootSameForBoth() throws IOException {
    try {
      Path execRoot = scratch.dir("/exec");
      Root.asDerivedRoot(execRoot, execRoot);
      fail();
    } catch (IllegalArgumentException expected) {
    }
  }

  @Test
  public void testBadAsDerivedRootNullDir() throws IOException {
    try {
      Path execRoot = scratch.dir("/exec");
      Root.asDerivedRoot(execRoot, null);
      fail();
    } catch (NullPointerException expected) {
    }
  }

  @Test
  public void testBadAsDerivedRootNullExecRoot() throws IOException {
    try {
      Path execRoot = scratch.dir("/exec");
      Root.asDerivedRoot(null, execRoot);
      fail();
    } catch (NullPointerException expected) {
    }
  }

  @Test
  public void testEquals() throws IOException {
    Path execRoot = scratch.dir("/exec");
    Path rootDir = scratch.dir("/exec/root");
    Path otherRootDir = scratch.dir("/");
    Path sourceDir = scratch.dir("/source");
    Root rootA = Root.asDerivedRoot(execRoot, rootDir);
    assertEqualsAndHashCode(true, rootA, Root.asDerivedRoot(execRoot, rootDir));
    assertEqualsAndHashCode(false, rootA, Root.asSourceRoot(sourceDir));
    assertEqualsAndHashCode(false, rootA, Root.asSourceRoot(rootDir));
    assertEqualsAndHashCode(false, rootA, Root.asDerivedRoot(otherRootDir, rootDir));
  }

  public void assertEqualsAndHashCode(boolean expected, Object a, Object b) {
    if (expected) {
      new EqualsTester().addEqualityGroup(b, a).testEquals();
    } else {
      assertThat(a.equals(b)).isFalse();
      assertThat(a.hashCode() == b.hashCode()).isFalse();
    }
  }
}
