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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.importdeps.AbstractClassEntryState.IncompleteState;
import com.google.devtools.build.importdeps.AbstractClassEntryState.MissingState;
import com.google.devtools.build.importdeps.ClassInfo.MemberInfo;
import java.io.IOException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for {@link ResultCollector}. */
@RunWith(JUnit4.class)
public class ResultCollectorTest {

  private ResultCollector collector;

  @Before
  public void setup() throws IOException {
    collector = new ResultCollector();
  }

  @Test
  public void testEmptyCollector() throws IOException {
    assertThat(collector.getSortedMissingClassInternalNames()).isEmpty();
    assertThat(collector.getSortedMissingMembers()).isEmpty();
    assertThat(collector.getSortedIncompleteClasses()).isEmpty();
  }

  @Test
  public void testOneMissingClass() throws IOException {
    collector.addMissingOrIncompleteClass("java.lang.String", MissingState.singleton());
    assertThat(collector.getSortedMissingClassInternalNames()).containsExactly("java.lang.String");
    assertThat(collector.getSortedMissingMembers()).isEmpty();

    collector.addMissingMember(MemberInfo.create("java/lang/Object", "field", "I"));
    assertThat(collector.getSortedMissingMembers())
        .containsExactly(MemberInfo.create("java/lang/Object", "field", "I"));
    assertThat(collector.getSortedMissingClassInternalNames()).containsExactly("java.lang.String");
  }

  @Test
  public void testIncompleteClasses() throws IOException {
    collector.addMissingOrIncompleteClass(
        "java/lang/String",
        IncompleteState.create(
            ClassInfo.create("java/lang/String", ImmutableList.of(), ImmutableSet.of()),
            ImmutableList.of("java/lang/Object")));
    assertThat(collector.getSortedIncompleteClasses()).hasSize(1);
    assertThat(collector.getSortedIncompleteClasses().get(0).classInfo().get())
        .isEqualTo(ClassInfo.create("java/lang/String", ImmutableList.of(), ImmutableSet.of()));
  }

  @Test
  public void testResultsAreSorted() throws IOException {
    collector.addMissingOrIncompleteClass("java.lang.String", MissingState.singleton());
    collector.addMissingOrIncompleteClass("java.lang.Integer", MissingState.singleton());
    collector.addMissingOrIncompleteClass("java.lang.Object", MissingState.singleton());

    assertThat(collector.getSortedMissingClassInternalNames())
        .containsExactly("java.lang.String", "java.lang.Integer", "java.lang.Object");
    assertThat(collector.getSortedMissingMembers()).isEmpty();
  }
}
