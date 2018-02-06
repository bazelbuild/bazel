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
import com.google.devtools.build.importdeps.ClassInfo.MemberInfo;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for {@link ClassInfo} */
@RunWith(JUnit4.class)
public class ClassInfoTest {

  public static final String JAVA_LANG_OBJECT = "java/lang/Object";
  private final MemberInfo hashCodeMethod = MemberInfo.create(JAVA_LANG_OBJECT, "hashCode", "()I");
  private final MemberInfo sizeMethod = MemberInfo.create(JAVA_LANG_OBJECT, "clear", "()V");

  private final ClassInfo objectClass =
      ClassInfo.create(JAVA_LANG_OBJECT, ImmutableList.of(), ImmutableSet.of(hashCodeMethod));

  private final ClassInfo listClass =
      ClassInfo.create(
          "java/util/List", ImmutableList.of(objectClass), ImmutableSet.of(sizeMethod));

  @Test
  public void testMemberInfo() {
    MemberInfo memberInfo = MemberInfo.create(JAVA_LANG_OBJECT, "a", "I");
    assertThat(memberInfo.memberName()).isEqualTo("a");
    assertThat(memberInfo.descriptor()).isEqualTo("I");
    assertThat(memberInfo).isEqualTo(MemberInfo.create("java/lang/Object", "a", "I"));

    assertThat(hashCodeMethod).isEqualTo(MemberInfo.create("java/lang/Object", "hashCode", "()I"));
    assertThat(sizeMethod).isEqualTo(MemberInfo.create("java/lang/Object", "clear", "()V"));
  }

  @Test
  public void testClassInfoCorrectlySet() {
    assertThat(objectClass.internalName()).isEqualTo("java/lang/Object");
    assertThat(objectClass.declaredMembers())
        .containsExactly(MemberInfo.create("java/lang/Object", "hashCode", "()I"))
        .inOrder();
    assertThat(objectClass.containsMember(MemberInfo.create("java/lang/Object", "hashCode", "()I")))
        .isTrue();

    assertThat(listClass.internalName()).isEqualTo("java/util/List");
    assertThat(listClass.declaredMembers()).containsExactly(sizeMethod);
    assertThat(listClass.containsMember(hashCodeMethod)).isTrue();
  }

  @Test
  public void testContainsMember() {
    ClassInfo parent = objectClass;
    ClassInfo child = listClass;
    assertThat(child.superClasses()).contains(parent);
    assertThat(parent.containsMember(MemberInfo.create("java/lang/Object", "hashCode", "()I")))
        .isTrue();
    assertThat(parent.containsMember(MemberInfo.create("java/lang/Object", "size", "()I")))
        .isFalse();
    assertThat(parent.containsMember(MemberInfo.create("java/lang/Object", "clear", "()V")))
        .isFalse();

    assertThat(child.containsMember(MemberInfo.create("java/lang/Object", "hashCode", "()I")))
        .isTrue();
    assertThat(child.containsMember(MemberInfo.create("java/lang/Object", "clear", "()V")))
        .isTrue();
  }
}
