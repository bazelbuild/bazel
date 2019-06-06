// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.importdeps;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.importdeps.AbstractClassEntryState.ExistingState;
import com.google.devtools.build.importdeps.AbstractClassEntryState.IncompleteState;
import com.google.devtools.build.importdeps.AbstractClassEntryState.MissingState;
import com.google.devtools.build.importdeps.ClassInfo.MemberInfo;
import java.nio.file.Paths;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for {@link AbstractClassEntryState} */
@RunWith(JUnit4.class)
public class LazyClassEntryStateTest {

  public static final String LIST_CLASS_NAME = "java/util/List";
  public static final ImmutableSet<MemberInfo> METHOD_LIST =
      ImmutableSet.of(MemberInfo.create("hashCode", "()I"));
  public static final ClassInfo LIST_CLASS_INFO =
      ClassInfo.create(LIST_CLASS_NAME, Paths.get("a"), true, ImmutableList.of(), METHOD_LIST);

  @Test
  public void testExistingState() {
    ExistingState state = ExistingState.create(LIST_CLASS_INFO);

    assertThat(state.isExistingState()).isTrue();
    assertThat(state.isIncompleteState()).isFalse();
    assertThat(state.isMissingState()).isFalse();

    assertThat(state.asExistingState()).isSameInstanceAs(state);
    assertThrows(IllegalStateException.class, () -> state.asIncompleteState());
    assertThrows(IllegalStateException.class, () -> state.asMissingState());

    ClassInfo classInfo = state.classInfo().get();
    assertThat(classInfo.internalName()).isEqualTo("java/util/List");
    assertThat(classInfo.declaredMembers()).hasSize(1);
    assertThat(classInfo.declaredMembers()).containsExactly(MemberInfo.create("hashCode", "()I"));
  }

  @Test
  public void testIncompleteState() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            IncompleteState.create(
                LIST_CLASS_INFO,
                ResolutionFailureChain.createWithParent(LIST_CLASS_INFO, ImmutableList.of())));
    IncompleteState state =
        IncompleteState.create(
            LIST_CLASS_INFO, ResolutionFailureChain.createMissingClass("java/lang/Object"));

    assertThat(state.isExistingState()).isFalse();
    assertThat(state.isIncompleteState()).isTrue();
    assertThat(state.isMissingState()).isFalse();

    assertThat(state.asIncompleteState()).isSameInstanceAs(state);
    assertThrows(IllegalStateException.class, () -> state.asExistingState());
    assertThrows(IllegalStateException.class, () -> state.asMissingState());

    ClassInfo classInfo = state.classInfo().get();
    assertThat(classInfo.internalName()).isEqualTo("java/util/List");
    assertThat(classInfo.declaredMembers()).hasSize(1);
    assertThat(classInfo.declaredMembers()).containsExactly(MemberInfo.create("hashCode", "()I"));

    assertThat(state.resolutionFailureChain().getMissingClassesWithSubclasses()).isEmpty();
    assertThat(state.missingAncestors()).hasSize(1);
    assertThat(state.missingAncestors()).containsExactly("java/lang/Object");
  }

  @Test
  public void testMissingState() {
    MissingState state = MissingState.singleton();

    assertThat(state.isMissingState()).isTrue();
    assertThat(state.isExistingState()).isFalse();
    assertThat(state.isIncompleteState()).isFalse();

    assertThat(state.asMissingState()).isSameInstanceAs(state);
    assertThrows(IllegalStateException.class, () -> state.asExistingState());
    assertThrows(IllegalStateException.class, () -> state.asIncompleteState());
  }
}
