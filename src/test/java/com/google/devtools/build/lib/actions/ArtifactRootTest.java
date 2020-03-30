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
import static org.junit.Assert.assertThrows;

import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ArtifactRoot}. */
@RunWith(JUnit4.class)
public class ArtifactRootTest {
  private Scratch scratch = new Scratch();

  @Test
  public void testAsSourceRoot() throws IOException {
    Path sourceDir = scratch.dir("/source");
    ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(sourceDir));
    assertThat(root.isSourceRoot()).isTrue();
    assertThat(root.getExecPath()).isEqualTo(PathFragment.EMPTY_FRAGMENT);
    assertThat(root.getRoot()).isEqualTo(Root.fromPath(sourceDir));
    assertThat(root.toString()).isEqualTo("/source[source]");
  }

  @Test
  public void testBadAsSourceRoot() {
    assertThrows(NullPointerException.class, () -> ArtifactRoot.asSourceRoot(null));
  }

  @Test
  public void testAsDerivedRoot() throws IOException {
    Path execRoot = scratch.dir("/exec");
    Path rootDir = scratch.dir("/exec/root");
    ArtifactRoot root = ArtifactRoot.asDerivedRoot(execRoot, "root");
    assertThat(root.isSourceRoot()).isFalse();
    assertThat(root.getExecPath()).isEqualTo(PathFragment.create("root"));
    assertThat(root.getRoot()).isEqualTo(Root.fromPath(rootDir));
    assertThat(root.toString()).isEqualTo("/exec/root[derived]");
  }

  @Test
  public void emptyExecPathNotOk() throws IOException {
    Path execRoot = scratch.dir("/exec");
    assertThrows(IllegalArgumentException.class, () -> ArtifactRoot.asDerivedRoot(execRoot, ""));
  }

  @Test
  public void emptySegmentOk() throws IOException {
    Path execRoot = scratch.dir("/exec");
    assertThat(ArtifactRoot.asDerivedRoot(execRoot, "", "suffix", ""))
        .isEqualTo(ArtifactRoot.asDerivedRoot(execRoot, "suffix"));
  }

  @Test
  public void segmentsAreSingles() throws IOException {
    Path execRoot = scratch.dir("/exec");
    assertThrows(
        IllegalArgumentException.class, () -> ArtifactRoot.asDerivedRoot(execRoot, "suffix/"));
  }

  @Test
  public void testBadAsDerivedRootIsExecRoot() throws IOException {
    Path execRoot = scratch.dir("/exec");
    assertThrows(IllegalArgumentException.class, () -> ArtifactRoot.asDerivedRoot(execRoot));
  }

  @Test
  public void testBadAsDerivedRootNullExecRoot() {
    assertThrows(NullPointerException.class, () -> ArtifactRoot.asDerivedRoot(null, "exec"));
  }

  @Test
  public void testEquals() throws IOException {
    Path execRoot = scratch.dir("/exec");
    String rootSegment = "root";
    Path rootDir = execRoot.getChild(rootSegment);
    rootDir.createDirectoryAndParents();
    Path otherRootDir = scratch.dir("/");
    Path sourceDir = scratch.dir("/source");
    ArtifactRoot rootA = ArtifactRoot.asDerivedRoot(execRoot, rootSegment);
    assertEqualsAndHashCode(true, rootA, ArtifactRoot.asDerivedRoot(execRoot, rootSegment));
    assertEqualsAndHashCode(false, rootA, ArtifactRoot.asSourceRoot(Root.fromPath(sourceDir)));
    assertEqualsAndHashCode(false, rootA, ArtifactRoot.asSourceRoot(Root.fromPath(rootDir)));
    assertEqualsAndHashCode(
        false, rootA, ArtifactRoot.asDerivedRoot(otherRootDir, "exec", rootSegment));
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
