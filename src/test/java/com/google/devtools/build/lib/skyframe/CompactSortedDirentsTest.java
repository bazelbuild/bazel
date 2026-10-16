// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.vfs.Dirent;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link CompactSortedDirents}. */
@RunWith(JUnit4.class)
public final class CompactSortedDirentsTest {

  @Test
  public void emptySingleton() {
    assertThat(CompactSortedDirents.create(ImmutableList.of()))
        .isSameInstanceAs(CompactSortedDirents.EMPTY);
    assertThat(CompactSortedDirents.EMPTY).isEmpty();
    assertThat(CompactSortedDirents.EMPTY.maybeGetDirent("foo")).isNull();
  }

  @Test
  public void instanceReuse() {
    CompactSortedDirents csd =
        CompactSortedDirents.create(
            ImmutableList.of(
                new Dirent("b", Dirent.Type.FILE), new Dirent("a", Dirent.Type.DIRECTORY)));
    assertThat(CompactSortedDirents.create(csd)).isSameInstanceAs(csd);
  }

  @Test
  public void sortingAndTypePreservation() {
    CompactSortedDirents csd =
        CompactSortedDirents.create(
            ImmutableList.of(
                new Dirent("c", Dirent.Type.SYMLINK),
                new Dirent("a", Dirent.Type.DIRECTORY),
                new Dirent("d", Dirent.Type.UNKNOWN),
                new Dirent("b", Dirent.Type.FILE)));

    assertThat(csd).hasSize(4);
    assertThat(csd)
        .containsExactly(
            new Dirent("a", Dirent.Type.DIRECTORY),
            new Dirent("b", Dirent.Type.FILE),
            new Dirent("c", Dirent.Type.SYMLINK),
            new Dirent("d", Dirent.Type.UNKNOWN))
        .inOrder();

    assertThat(csd.maybeGetDirent("a")).isEqualTo(new Dirent("a", Dirent.Type.DIRECTORY));
    assertThat(csd.maybeGetDirent("b")).isEqualTo(new Dirent("b", Dirent.Type.FILE));
    assertThat(csd.maybeGetDirent("c")).isEqualTo(new Dirent("c", Dirent.Type.SYMLINK));
    assertThat(csd.maybeGetDirent("d")).isEqualTo(new Dirent("d", Dirent.Type.UNKNOWN));
    assertThat(csd.maybeGetDirent("missing")).isNull();

    assertThat(csd.maybeGetDirentType("a")).isEqualTo(Dirent.Type.DIRECTORY);
    assertThat(csd.maybeGetDirentType("b")).isEqualTo(Dirent.Type.FILE);
    assertThat(csd.maybeGetDirentType("c")).isEqualTo(Dirent.Type.SYMLINK);
    assertThat(csd.maybeGetDirentType("d")).isEqualTo(Dirent.Type.UNKNOWN);
    assertThat(csd.maybeGetDirentType("missing")).isNull();
  }

  @Test
  public void equalsAndHashCode() {
    CompactSortedDirents csd1 =
        CompactSortedDirents.create(
            ImmutableList.of(
                new Dirent("b", Dirent.Type.FILE), new Dirent("a", Dirent.Type.DIRECTORY)));
    CompactSortedDirents csd2 =
        CompactSortedDirents.create(
            ImmutableList.of(
                new Dirent("a", Dirent.Type.DIRECTORY), new Dirent("b", Dirent.Type.FILE)));

    assertThat(csd1).isEqualTo(csd2);
    assertThat(csd1.hashCode()).isEqualTo(csd2.hashCode());
  }
}
