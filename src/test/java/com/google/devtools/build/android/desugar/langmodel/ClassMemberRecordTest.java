/*
 * Copyright 2019 The Bazel Authors. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.devtools.build.android.desugar.langmodel;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.android.desugar.langmodel.ClassMemberRecord.ClassMemberRecordBuilder;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.objectweb.asm.Opcodes;

/** Tests for {@link ClassMemberRecord}. */
@RunWith(JUnit4.class)
public final class ClassMemberRecordTest {

  private final ClassMemberRecordBuilder classMemberRecord = ClassMemberRecord.builder();

  @Test
  public void trackFieldUse() {
    FieldKey classMemberKey =
        FieldKey.create(ClassName.create("package/path/OwnerClass"), "fieldOfPrimitiveLong", "J");
    classMemberRecord.logMemberUse(classMemberKey, Opcodes.GETFIELD);
    classMemberRecord.logMemberUse(classMemberKey, Opcodes.PUTFIELD);
    classMemberRecord.logMemberUse(classMemberKey, Opcodes.H_GETFIELD);
    classMemberRecord.logMemberUse(classMemberKey, Opcodes.H_PUTFIELD);

    assertThat(classMemberRecord.build().findAllMemberUseKind(classMemberKey))
        .containsExactly(
            MemberUseKind.GETFIELD,
            MemberUseKind.PUTFIELD,
            MemberUseKind.H_GETFIELD,
            MemberUseKind.H_PUTFIELD);
  }

  @Test
  public void trackConstructorUse() {
    MethodKey classMemberKey =
        MethodKey.create(ClassName.create("package/path/OwnerClass"), "<init>", "()V");
    classMemberRecord.logMemberUse(classMemberKey, Opcodes.INVOKESPECIAL);
    classMemberRecord.logMemberUse(classMemberKey, Opcodes.H_NEWINVOKESPECIAL);

    assertThat(classMemberRecord.build().findAllMemberUseKind(classMemberKey))
        .containsExactly(MemberUseKind.INVOKESPECIAL, MemberUseKind.H_NEWINVOKESPECIAL);
  }

  @Test
  public void trackMethodUse() {
    MethodKey classMemberKey =
        MethodKey.create(ClassName.create("package/path/OwnerClass"), "method", "(II)I");
    classMemberRecord.logMemberUse(classMemberKey, Opcodes.INVOKEVIRTUAL);
    classMemberRecord.logMemberUse(classMemberKey, Opcodes.INVOKESPECIAL);
    classMemberRecord.logMemberUse(classMemberKey, Opcodes.INVOKESTATIC);
    classMemberRecord.logMemberUse(classMemberKey, Opcodes.INVOKEINTERFACE);
    classMemberRecord.logMemberUse(classMemberKey, Opcodes.INVOKEDYNAMIC);
    classMemberRecord.logMemberUse(classMemberKey, Opcodes.H_INVOKEVIRTUAL);
    classMemberRecord.logMemberUse(classMemberKey, Opcodes.H_INVOKESTATIC);
    classMemberRecord.logMemberUse(classMemberKey, Opcodes.H_INVOKESPECIAL);
    classMemberRecord.logMemberUse(classMemberKey, Opcodes.H_NEWINVOKESPECIAL);
    classMemberRecord.logMemberUse(classMemberKey, Opcodes.H_INVOKEINTERFACE);

    assertThat(classMemberRecord.build().findAllMemberUseKind(classMemberKey))
        .containsExactly(
            MemberUseKind.INVOKEVIRTUAL,
            MemberUseKind.INVOKESPECIAL,
            MemberUseKind.INVOKESTATIC,
            MemberUseKind.INVOKEINTERFACE,
            MemberUseKind.INVOKEDYNAMIC,
            MemberUseKind.H_INVOKEVIRTUAL,
            MemberUseKind.H_INVOKESTATIC,
            MemberUseKind.H_INVOKESPECIAL,
            MemberUseKind.H_NEWINVOKESPECIAL,
            MemberUseKind.H_INVOKEINTERFACE);
  }

  @Test
  public void trackMemberDeclaration() {
    MethodKey classMemberKey =
        MethodKey.create(ClassName.create("package/path/OwnerClass"), "method", "(II)I");
    classMemberRecord.logMemberDecl(
        classMemberKey, Opcodes.ACC_PUBLIC | Opcodes.ACC_SUPER, Opcodes.ACC_PRIVATE);

    ClassMemberRecord readOnlyClassMemberRecord = classMemberRecord.build();
    assertThat(readOnlyClassMemberRecord.findOwnerAccessCode(classMemberKey))
        .isEqualTo(Opcodes.ACC_PUBLIC | Opcodes.ACC_SUPER);
    assertThat(readOnlyClassMemberRecord.findMemberAccessCode(classMemberKey))
        .isEqualTo(Opcodes.ACC_PRIVATE);
  }

  @Test
  public void trackMemberDeclaration_withDeprecatedAnnotation() {
    MethodKey classMemberKey =
        MethodKey.create(ClassName.create("package/path/OwnerClass"), "method", "(II)I");
    classMemberRecord.logMemberDecl(
        classMemberKey,
        Opcodes.ACC_DEPRECATED | Opcodes.ACC_PUBLIC | Opcodes.ACC_SUPER,
        Opcodes.ACC_DEPRECATED | Opcodes.ACC_PRIVATE);
    ClassMemberRecord rawClassMemberRecord = classMemberRecord.build();
    assertThat(rawClassMemberRecord.findOwnerAccessCode(classMemberKey))
        .isEqualTo(Opcodes.ACC_DEPRECATED | Opcodes.ACC_PUBLIC | Opcodes.ACC_SUPER);
    assertThat(rawClassMemberRecord.findMemberAccessCode(classMemberKey))
        .isEqualTo(Opcodes.ACC_DEPRECATED | Opcodes.ACC_PRIVATE);
  }

  @Test
  public void mergeRecord_trackingReasons() {
    ClassMemberRecordBuilder otherClassMemberRecord = ClassMemberRecord.builder();

    MethodKey method1 =
        MethodKey.create(ClassName.create("package/path/OwnerClass"), "method1", "(II)I");
    MethodKey method2 =
        MethodKey.create(ClassName.create("package/path/OwnerClass"), "method2", "(II)I");
    MethodKey method3 =
        MethodKey.create(ClassName.create("package/path/OwnerClass"), "method3", "(II)I");

    classMemberRecord.logMemberDecl(method1, Opcodes.ACC_SUPER, Opcodes.ACC_PRIVATE);
    classMemberRecord.logMemberUse(method2, Opcodes.INVOKEVIRTUAL);

    otherClassMemberRecord.logMemberDecl(method2, Opcodes.ACC_SUPER, Opcodes.ACC_PRIVATE);
    otherClassMemberRecord.logMemberUse(method2, Opcodes.INVOKESPECIAL);
    otherClassMemberRecord.logMemberUse(method3, Opcodes.INVOKEVIRTUAL);

    ClassMemberRecord mergedRecord =
        classMemberRecord.mergeFrom(otherClassMemberRecord.build()).build();

    assertThat(mergedRecord.hasDeclReason(method1)).isTrue();
    assertThat(mergedRecord.findOwnerAccessCode(method1)).isEqualTo(Opcodes.ACC_SUPER);
    assertThat(mergedRecord.findMemberAccessCode(method1)).isEqualTo(Opcodes.ACC_PRIVATE);
    assertThat(mergedRecord.findAllMemberUseKind(method1)).isEmpty();

    assertThat(mergedRecord.hasDeclReason(method2)).isTrue();
    assertThat(mergedRecord.findOwnerAccessCode(method2)).isEqualTo(Opcodes.ACC_SUPER);
    assertThat(mergedRecord.findMemberAccessCode(method2)).isEqualTo(Opcodes.ACC_PRIVATE);
    assertThat(mergedRecord.findAllMemberUseKind(method2))
        .containsExactly(MemberUseKind.INVOKEVIRTUAL, MemberUseKind.INVOKESPECIAL);

    assertThat(mergedRecord.hasDeclReason(method3)).isFalse();
    assertThat(mergedRecord.findOwnerAccessCode(method3)).isEqualTo(0);
    assertThat(mergedRecord.findMemberAccessCode(method3)).isEqualTo(0);
    assertThat(mergedRecord.findAllMemberUseKind(method3))
        .containsExactly(MemberUseKind.INVOKEVIRTUAL);
  }

  @Test
  public void filterUsedMemberWithTrackedDeclaration_noMemberDeclaration() {
    MethodKey classMemberKey =
        MethodKey.create(ClassName.create("package/path/OwnerClass"), "method", "(II)I");

    classMemberRecord.logMemberUse(classMemberKey, Opcodes.INVOKEVIRTUAL);
    ClassMemberRecord rawClassMemberRecord = classMemberRecord.build();
    assertThat(rawClassMemberRecord.hasTrackingReason(classMemberKey)).isTrue();

    ClassMemberRecord filteredRecord =
        rawClassMemberRecord.filterUsedMemberWithTrackedDeclaration();
    assertThat(filteredRecord.hasTrackingReason(classMemberKey)).isFalse();
  }

  @Test
  public void filterUsedMemberWithTrackedDeclaration_noMemberUse() {
    MethodKey classMemberKey =
        MethodKey.create(ClassName.create("package/path/OwnerClass"), "method", "(II)I");

    classMemberRecord.logMemberDecl(
        classMemberKey, Opcodes.ACC_PUBLIC | Opcodes.ACC_SUPER, Opcodes.ACC_PRIVATE);
    ClassMemberRecord rawClassMemberRecord = classMemberRecord.build();
    assertThat(rawClassMemberRecord.hasTrackingReason(classMemberKey)).isTrue();

    ClassMemberRecord filteredRecord =
        rawClassMemberRecord.filterUsedMemberWithTrackedDeclaration();
    assertThat(filteredRecord.hasTrackingReason(classMemberKey)).isFalse();
  }

  @Test
  public void filterUsedMemberWithTrackedDeclaration_interfaceMemberWithoutUse_shouldTrack() {
    MethodKey classMemberKey =
        MethodKey.create(ClassName.create("package/path/OwnerClass"), "method", "(II)I");

    classMemberRecord.logMemberDecl(
        classMemberKey, Opcodes.ACC_PUBLIC | Opcodes.ACC_INTERFACE, Opcodes.ACC_PRIVATE);
    ClassMemberRecord rawClassMemberRecord = classMemberRecord.build();
    assertThat(rawClassMemberRecord.hasTrackingReason(classMemberKey)).isTrue();

    ClassMemberRecord filteredRecord =
        rawClassMemberRecord.filterUsedMemberWithTrackedDeclaration();
    assertThat(filteredRecord.hasTrackingReason(classMemberKey)).isTrue();
  }
}
