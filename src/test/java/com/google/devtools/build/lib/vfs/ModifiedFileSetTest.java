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
package com.google.devtools.build.lib.vfs;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.testing.EqualsTester;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import com.google.testing.junit.testparameterinjector.TestParameters;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link ModifiedFileSet}. */
@RunWith(TestParameterInjector.class)
public class ModifiedFileSetTest {

  @Test
  public void testHashCodeAndEqualsContract() throws Exception {
    PathFragment fragA = PathFragment.create("a");
    PathFragment fragB = PathFragment.create("b");

    ModifiedFileSet empty1 = ModifiedFileSet.NOTHING_MODIFIED;
    ModifiedFileSet empty2 = ModifiedFileSet.builder().build();
    ModifiedFileSet empty3 = ModifiedFileSet.builder().modifyAll(
        ImmutableList.<PathFragment>of()).build();

    ModifiedFileSet nonEmpty1 = ModifiedFileSet.builder().modifyAll(
        ImmutableList.of(fragA, fragB)).build();
    ModifiedFileSet nonEmpty2 = ModifiedFileSet.builder().modifyAll(
        ImmutableList.of(fragB, fragA)).build();
    ModifiedFileSet nonEmpty3 = ModifiedFileSet.builder().modify(fragA).modify(fragB).build();
    ModifiedFileSet nonEmpty4 = ModifiedFileSet.builder().modify(fragB).modify(fragA).build();

    ModifiedFileSet everythingModified = ModifiedFileSet.EVERYTHING_MODIFIED;

    new EqualsTester()
        .addEqualityGroup(empty1, empty2, empty3)
        .addEqualityGroup(nonEmpty1, nonEmpty2, nonEmpty3, nonEmpty4)
        .addEqualityGroup(everythingModified)
        .testEquals();
  }

  @TestParameters({
    "{set1: false, set2: false, union: false}",
    "{set1: true,  set2: false, union: false}",
    "{set1: false, set2: true,  union: false}",
    "{set1: true,  set2: true,  union: true}"
  })
  @Test
  public void union_returnsConjunctionOfIncludesAncestorDirectories(
      boolean set1, boolean set2, boolean union) {
    ModifiedFileSet fileSet1 =
        ModifiedFileSet.builder()
            .modify(PathFragment.create("a"))
            .setIncludesAncestorDirectories(set1)
            .build();
    ModifiedFileSet fileSet2 =
        ModifiedFileSet.builder()
            .modify(PathFragment.create("b"))
            .setIncludesAncestorDirectories(set2)
            .build();

    ModifiedFileSet unionSet = ModifiedFileSet.union(fileSet1, fileSet2);

    assertThat(unionSet.modifiedSourceFiles())
        .containsExactly(PathFragment.create("a"), PathFragment.create("b"));
    assertThat(unionSet.includesAncestorDirectories()).isEqualTo(union);
  }
}
