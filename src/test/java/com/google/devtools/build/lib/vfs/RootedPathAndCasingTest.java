// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.vfs;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RootedPathAndCasing}. */
@RunWith(JUnit4.class)
public class RootedPathAndCasingTest {
  private final Path root = new InMemoryFileSystem(BlazeClock.instance()).getPath("c:/");

  @Test
  public void testSanityCheckFilesystemIsCaseInsensitive() {
    Path p1 = root.getRelative("Foo/Bar");
    Path p2 = root.getRelative("FOO/BAR");
    Path p3 = root.getRelative("control");
    assertThat(p1).isNotSameInstanceAs(p2);
    assertThat(p1).isNotSameInstanceAs(p3);
    assertThat(p2).isNotSameInstanceAs(p3);
    assertThat(p1).isEqualTo(p2);
    assertThat(p1).isNotEqualTo(p3);
  }

  @Test
  public void testEqualsAndHashCodeContract() {
    PathFragment pf1 = PathFragment.create("Foo/Bar");
    PathFragment pf2 = PathFragment.create("FOO/baR");
    PathFragment pf3 = PathFragment.create("FOO/baR");
    assertThat(pf1).isNotSameInstanceAs(pf2);
    assertThat(pf2).isNotSameInstanceAs(pf3);
    assertThat(pf1).isEqualTo(pf2);
    assertThat(pf1).isEqualTo(pf3);

    RootedPath rp1 = RootedPath.toRootedPath(Root.fromPath(root), pf1);
    RootedPath rp2 = RootedPath.toRootedPath(Root.fromPath(root), pf2);
    RootedPath rp3 = RootedPath.toRootedPath(Root.fromPath(root), pf3);
    assertThat(rp1).isNotSameInstanceAs(rp2);
    assertThat(rp2).isNotSameInstanceAs(rp3);
    assertThat(rp1).isEqualTo(rp2);
    assertThat(rp1).isEqualTo(rp3);
    assertThat(rp1.hashCode()).isEqualTo(rp2.hashCode());
    assertThat(rp1.hashCode()).isEqualTo(rp3.hashCode());

    RootedPathAndCasing rpac1 = RootedPathAndCasing.create(rp1);
    RootedPathAndCasing rpac2 = RootedPathAndCasing.create(rp2);
    RootedPathAndCasing rpac3 = RootedPathAndCasing.create(rp3);
    assertThat(rpac1).isNotSameInstanceAs(rpac2);
    assertThat(rpac2).isNotSameInstanceAs(rpac3);
    assertThat(rpac1).isNotEqualTo(rpac2);
    assertThat(rpac2).isEqualTo(rpac3);
    assertThat(rpac1.hashCode()).isNotEqualTo(rpac2.hashCode());
    assertThat(rpac2.hashCode()).isEqualTo(rpac3.hashCode());
  }

  @Test
  public void testToStringRespectsCasing() {
    RootedPath rp1 = RootedPath.toRootedPath(Root.fromPath(root), PathFragment.create("Foo/Bar"));
    RootedPath rp2 = RootedPath.toRootedPath(Root.fromPath(root), PathFragment.create("FOO/baR"));
    RootedPathAndCasing rpac1 = RootedPathAndCasing.create(rp1);
    RootedPathAndCasing rpac2 = RootedPathAndCasing.create(rp2);
    assertThat(rpac1.toString()).contains("Foo/Bar");
    assertThat(rpac2.toString()).contains("FOO/baR");
  }
}
