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
import com.google.devtools.build.importdeps.ResultCollector.MissingMember;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
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
    assertThat(collector.getSortedIndirectDeps()).isEmpty();
    assertThat(collector.isEmpty()).isTrue();
  }

  @Test
  public void testOneMissingClass() throws IOException {
    collector.addMissingOrIncompleteClass("java.lang.String", MissingState.singleton());
    assertThat(collector.getSortedMissingClassInternalNames()).containsExactly("java.lang.String");
    assertThat(collector.getSortedMissingMembers()).isEmpty();

    collector.addMissingMember("java/lang/Object", MemberInfo.create("field", "I"));
    assertThat(collector.getSortedMissingMembers())
        .containsExactly(MissingMember.create("java/lang/Object", "field", "I"));
    assertThat(collector.getSortedMissingClassInternalNames()).containsExactly("java.lang.String");
    assertThat(collector.isEmpty()).isFalse();
  }

  @Test
  public void testIncompleteClasses() throws IOException {
    collector.addMissingOrIncompleteClass(
        "java/lang/String",
        IncompleteState.create(
            ClassInfo.create(
                "java/lang/String", Paths.get("a"), true, ImmutableList.of(), ImmutableSet.of()),
            ImmutableList.of("java/lang/Object")));
    assertThat(collector.getSortedIncompleteClasses()).hasSize(1);
    assertThat(collector.getSortedIncompleteClasses().get(0).classInfo().get())
        .isEqualTo(
            ClassInfo.create(
                "java/lang/String", Paths.get("a"), true, ImmutableList.of(), ImmutableSet.of()));
    assertThat(collector.isEmpty()).isFalse();
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

  @Test
  public void testIndirectDeps() {
    Path a = Paths.get("a");
    Path b = Paths.get("b");
    collector.addIndirectDep(b);
    collector.addIndirectDep(a);
    collector.addIndirectDep(b);
    assertThat(collector.getSortedIndirectDeps()).containsExactly(a, b).inOrder();
    assertThat(collector.isEmpty()).isFalse();
  }

  @Test
  public void testMissingMember() {
    String owner = "owner";
    String name = "name";
    String desc = "desc";
    MissingMember member = MissingMember.create(owner, name, desc);
    assertThat(member.owner()).isEqualTo(owner);
    assertThat(member.memberName()).isEqualTo(name);
    assertThat(member.descriptor()).isEqualTo(desc);
    assertThat(member.member()).isEqualTo(MemberInfo.create(name, desc));

    MissingMember member2 = MissingMember.create(owner, MemberInfo.create(name, desc));
    assertThat(member2).isEqualTo(member);
  }

  @Test
  public void testMemberComparison() {
    MissingMember member1 = MissingMember.create("A", MemberInfo.create("B", "C"));
    MissingMember member2 = MissingMember.create("A", MemberInfo.create("B", "C"));
    assertThat(member1.compareTo(member2)).isEqualTo(0);

    MissingMember member3 = MissingMember.create("B", MemberInfo.create("B", "C"));
    assertThat(member1.compareTo(member3)).isEqualTo(-1);
    assertThat(member3.compareTo(member1)).isEqualTo(1);

    MissingMember member4 = MissingMember.create("A", MemberInfo.create("C", "C"));
    assertThat(member1.compareTo(member4)).isEqualTo(-1);

    assertThat(member3.compareTo(member4)).isEqualTo(1);
  }
}
