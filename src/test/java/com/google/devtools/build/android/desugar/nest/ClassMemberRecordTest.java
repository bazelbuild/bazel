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

package com.google.devtools.build.android.desugar.nest;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.android.desugar.nest.ClassMemberKey.FieldKey;
import com.google.devtools.build.android.desugar.nest.ClassMemberKey.MethodKey;
import com.google.devtools.build.android.desugar.nest.ClassMemberTrackReason.MemberUseKind;
import com.google.testing.testsize.SmallTest;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.objectweb.asm.Opcodes;

/** Tests for {@link ClassMemberRecord}. */
@RunWith(JUnit4.class)
@SmallTest
public class ClassMemberRecordTest {

  private final ClassMemberRecord classMemberRecord = ClassMemberRecord.create();

  @Test
  public void trackFieldUse() {
    ClassMemberKey classMemberKey =
        FieldKey.create("package.path.OwnerClass", "fieldOfPrimitiveLong", "J");
    classMemberRecord.logMemberUse(classMemberKey, Opcodes.GETFIELD);
    classMemberRecord.logMemberUse(classMemberKey, Opcodes.PUTFIELD);
    classMemberRecord.logMemberUse(classMemberKey, Opcodes.H_GETFIELD);
    classMemberRecord.logMemberUse(classMemberKey, Opcodes.H_PUTFIELD);

    assertThat(classMemberRecord.findAllMemberUseKind(classMemberKey))
        .containsExactly(
            MemberUseKind.GETFIELD,
            MemberUseKind.PUTFIELD,
            MemberUseKind.H_GETFIELD,
            MemberUseKind.H_PUTFIELD);
  }

  @Test
  public void trackConstructorUse() {
    ClassMemberKey classMemberKey = MethodKey.create("package.path.OwnerClass", "<init>", "()V");
    classMemberRecord.logMemberUse(classMemberKey, Opcodes.INVOKESPECIAL);
    classMemberRecord.logMemberUse(classMemberKey, Opcodes.H_NEWINVOKESPECIAL);

    assertThat(classMemberRecord.findAllMemberUseKind(classMemberKey))
        .containsExactly(MemberUseKind.INVOKESPECIAL, MemberUseKind.H_NEWINVOKESPECIAL);
  }

  @Test
  public void trackMethodUse() {
    ClassMemberKey classMemberKey = MethodKey.create("package.path.OwnerClass", "method", "(II)I");
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

    assertThat(classMemberRecord.findAllMemberUseKind(classMemberKey))
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
    ClassMemberKey classMemberKey = MethodKey.create("package.path.OwnerClass", "method", "(II)I");
    classMemberRecord.logMemberDecl(
        classMemberKey, Opcodes.ACC_PUBLIC | Opcodes.ACC_SUPER, Opcodes.ACC_PRIVATE);

    assertThat(classMemberRecord.findOwnerAccessCode(classMemberKey))
        .isEqualTo(Opcodes.ACC_PUBLIC | Opcodes.ACC_SUPER);
    assertThat(classMemberRecord.findMemberAccessCode(classMemberKey))
        .isEqualTo(Opcodes.ACC_PRIVATE);
  }

  @Test
  public void trackMemberDeclaration_withDeprecatedAnnotation() {
    ClassMemberKey classMemberKey = MethodKey.create("package.path.OwnerClass", "method", "(II)I");
    classMemberRecord.logMemberDecl(
        classMemberKey,
        Opcodes.ACC_DEPRECATED | Opcodes.ACC_PUBLIC | Opcodes.ACC_SUPER,
        Opcodes.ACC_DEPRECATED | Opcodes.ACC_PRIVATE);
    assertThat(classMemberRecord.findOwnerAccessCode(classMemberKey))
        .isEqualTo(Opcodes.ACC_DEPRECATED | Opcodes.ACC_PUBLIC | Opcodes.ACC_SUPER);
    assertThat(classMemberRecord.findMemberAccessCode(classMemberKey))
        .isEqualTo(Opcodes.ACC_DEPRECATED | Opcodes.ACC_PRIVATE);
  }

  @Test
  public void mergeRecord_trackingReasons() {
    ClassMemberRecord otherClassMemberRecord = ClassMemberRecord.create();

    ClassMemberKey method1 = MethodKey.create("package.path.OwnerClass", "method1", "(II)I");
    ClassMemberKey method2 = MethodKey.create("package.path.OwnerClass", "method2", "(II)I");
    ClassMemberKey method3 = MethodKey.create("package.path.OwnerClass", "method3", "(II)I");

    classMemberRecord.logMemberDecl(method1, Opcodes.ACC_SUPER, Opcodes.ACC_PRIVATE);
    classMemberRecord.logMemberUse(method2, Opcodes.INVOKEVIRTUAL);

    otherClassMemberRecord.logMemberDecl(method2, Opcodes.ACC_SUPER, Opcodes.ACC_PRIVATE);
    otherClassMemberRecord.logMemberUse(method2, Opcodes.INVOKESPECIAL);
    otherClassMemberRecord.logMemberUse(method3, Opcodes.INVOKEVIRTUAL);

    classMemberRecord.mergeFrom(otherClassMemberRecord);

    assertThat(classMemberRecord.hasDeclReason(method1)).isTrue();
    assertThat(classMemberRecord.findOwnerAccessCode(method1)).isEqualTo(Opcodes.ACC_SUPER);
    assertThat(classMemberRecord.findMemberAccessCode(method1)).isEqualTo(Opcodes.ACC_PRIVATE);
    assertThat(classMemberRecord.findAllMemberUseKind(method1)).isEmpty();

    assertThat(classMemberRecord.hasDeclReason(method2)).isTrue();
    assertThat(classMemberRecord.findOwnerAccessCode(method2)).isEqualTo(Opcodes.ACC_SUPER);
    assertThat(classMemberRecord.findMemberAccessCode(method2)).isEqualTo(Opcodes.ACC_PRIVATE);
    assertThat(classMemberRecord.findAllMemberUseKind(method2))
        .containsExactly(MemberUseKind.INVOKEVIRTUAL, MemberUseKind.INVOKESPECIAL);

    assertThat(classMemberRecord.hasDeclReason(method3)).isFalse();
    assertThat(classMemberRecord.findOwnerAccessCode(method3)).isEqualTo(0);
    assertThat(classMemberRecord.findMemberAccessCode(method3)).isEqualTo(0);
    assertThat(classMemberRecord.findAllMemberUseKind(method3))
        .containsExactly(MemberUseKind.INVOKEVIRTUAL);
  }

  @Test
  public void filterUsedMemberWithTrackedDeclaration_noMemberDeclaration() {
    ClassMemberKey classMemberKey = MethodKey.create("package.path.OwnerClass", "method", "(II)I");

    classMemberRecord.logMemberUse(classMemberKey, Opcodes.INVOKEVIRTUAL);
    assertThat(classMemberRecord.hasTrackingReason(classMemberKey)).isTrue();

    classMemberRecord.filterUsedMemberWithTrackedDeclaration();
    assertThat(classMemberRecord.hasTrackingReason(classMemberKey)).isFalse();
  }

  @Test
  public void filterUsedMemberWithTrackedDeclaration_noMemberUse() {
    ClassMemberKey classMemberKey = MethodKey.create("package.path.OwnerClass", "method", "(II)I");

    classMemberRecord.logMemberDecl(
        classMemberKey, Opcodes.ACC_PUBLIC | Opcodes.ACC_SUPER, Opcodes.ACC_PRIVATE);
    assertThat(classMemberRecord.hasTrackingReason(classMemberKey)).isTrue();

    classMemberRecord.filterUsedMemberWithTrackedDeclaration();
    assertThat(classMemberRecord.hasTrackingReason(classMemberKey)).isFalse();
  }

  @Test
  public void filterUsedMemberWithTrackedDeclaration_interfaceMemberWithoutUse_shouldTrack() {
    ClassMemberKey classMemberKey = MethodKey.create("package.path.OwnerClass", "method", "(II)I");

    classMemberRecord.logMemberDecl(
        classMemberKey, Opcodes.ACC_PUBLIC | Opcodes.ACC_INTERFACE, Opcodes.ACC_PRIVATE);
    assertThat(classMemberRecord.hasTrackingReason(classMemberKey)).isTrue();

    classMemberRecord.filterUsedMemberWithTrackedDeclaration();
    assertThat(classMemberRecord.hasTrackingReason(classMemberKey)).isTrue();
  }
}
