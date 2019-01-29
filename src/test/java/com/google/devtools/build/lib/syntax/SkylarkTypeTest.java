// Copyright 2018 The Bazel Authors. All Rights Reserved.
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

package com.google.devtools.build.lib.syntax;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for Skylark's type system. */
@RunWith(JUnit4.class)
public final class SkylarkTypeTest {

  @SkylarkModule(
      name = "ParentType",
      doc = "A parent class annotated with @SkylarkModule."
  )
  private static class ParentClassWithSkylarkModule {}

  private static class ChildClass extends ParentClassWithSkylarkModule {}

  @Test
  public void simpleTypeOfChild_UsesTypeOfAncestorHavingSkylarkModule() {
    assertThat(SkylarkType.of(ChildClass.class).getType())
        .isEqualTo(ParentClassWithSkylarkModule.class);
  }

  @Test
  public void typeHierarchy_Basic() throws Exception {
    assertThat(SkylarkType.INT.includes(SkylarkType.BOTTOM)).isTrue();
    assertThat(SkylarkType.BOTTOM.includes(SkylarkType.INT)).isFalse();
    assertThat(SkylarkType.TOP.includes(SkylarkType.INT)).isTrue();
  }

  @Test
  public void typeHierarchy_Combination() throws Exception {
    SkylarkType combo = SkylarkType.Combination.of(SkylarkType.LIST, SkylarkType.INT);
    assertThat(SkylarkType.LIST.includes(combo)).isTrue();
  }

  @Test
  public void typeHierarchy_Union() throws Exception {
    SkylarkType combo = SkylarkType.Combination.of(SkylarkType.LIST, SkylarkType.INT);
    SkylarkType union = SkylarkType.Union.of(SkylarkType.DICT, SkylarkType.LIST);
    assertThat(union.includes(SkylarkType.DICT)).isTrue();
    assertThat(union.includes(combo)).isTrue();
    assertThat(union.includes(SkylarkType.STRING)).isFalse();
  }

  @Test
  public void typeHierarchy_Intersection() throws Exception {
    SkylarkType combo = SkylarkType.Combination.of(SkylarkType.LIST, SkylarkType.INT);
    SkylarkType union1 = SkylarkType.Union.of(SkylarkType.DICT, SkylarkType.LIST);
    SkylarkType union2 =
        SkylarkType.Union.of(
            SkylarkType.LIST, SkylarkType.DICT, SkylarkType.STRING, SkylarkType.INT);
    SkylarkType inter = SkylarkType.intersection(union1, union2);
    assertThat(inter.includes(SkylarkType.DICT)).isTrue();
    assertThat(inter.includes(SkylarkType.LIST)).isTrue();
    assertThat(inter.includes(combo)).isTrue();
    assertThat(inter.includes(SkylarkType.INT)).isFalse();
  }

  @Test
  public void testStringPairTuple() {
    assertThat(SkylarkType.intersection(SkylarkType.STRING_PAIR, SkylarkType.TUPLE))
        .isEqualTo(SkylarkType.STRING_PAIR);
  }
}
