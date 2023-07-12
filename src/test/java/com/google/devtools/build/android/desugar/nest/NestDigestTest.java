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

import com.google.devtools.build.android.desugar.langmodel.ClassAttributeRecord;
import com.google.devtools.build.android.desugar.langmodel.ClassAttributeRecord.ClassAttributeRecordBuilder;
import com.google.devtools.build.android.desugar.langmodel.ClassAttributes;
import com.google.devtools.build.android.desugar.langmodel.ClassMemberRecord;
import com.google.devtools.build.android.desugar.langmodel.ClassMemberRecord.ClassMemberRecordBuilder;
import com.google.devtools.build.android.desugar.langmodel.ClassName;
import com.google.devtools.build.android.desugar.langmodel.MethodKey;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.objectweb.asm.Opcodes;

/** The tests for {@link NestDigest}. */
@RunWith(JUnit4.class)
public final class NestDigestTest {

  private final ClassMemberRecordBuilder classMemberRecord = ClassMemberRecord.builder();
  private final ClassAttributeRecordBuilder classAttributeRecord = ClassAttributeRecord.builder();

  @Test
  public void prepareCompanionClassWriters_noCompanionClassesGenerated() {
    classMemberRecord.logMemberDecl(
        MethodKey.create(ClassName.create("package/path/OwnerClass"), "method", "(II)I"),
        /* ownerAccess= */ Opcodes.ACC_PUBLIC | Opcodes.ACC_INTERFACE,
        /* memberDeclAccess= */ Opcodes.ACC_PRIVATE);

    NestDigest nestDigest =
        NestDigest.builder()
            .setClassMemberRecord(classMemberRecord.build())
            .setClassAttributeRecord(classAttributeRecord.build())
            .build();

    assertThat(nestDigest.getAllCompanionClassNames()).isEmpty();
  }

  @Test
  public void prepareCompanionClassWriters_companionClassesGenerated() {
    MethodKey constructor =
        MethodKey.create(ClassName.create("package/path/OwnerClass"), "<init>", "()V");
    classMemberRecord.logMemberDecl(
        constructor, Opcodes.ACC_PUBLIC | Opcodes.ACC_SUPER, Opcodes.ACC_PRIVATE);
    classMemberRecord.logMemberUse(constructor, Opcodes.INVOKESPECIAL);
    classAttributeRecord.addClassAttributes(
        ClassAttributes.builder()
            .setClassBinaryName(ClassName.create("package/path/OwnerClass$NestedClass"))
            .setNestHost(ClassName.create("package/path/OwnerClass"))
            .build());
    classAttributeRecord.addClassAttributes(
        ClassAttributes.builder()
            .setClassBinaryName(ClassName.create("package/path/OwnerClass"))
            .addNestMember(ClassName.create("package/path/OwnerClass$NestedClass"))
            .build());

    NestDigest nestDigest =
        NestDigest.builder()
            .setClassMemberRecord(classMemberRecord.build())
            .setClassAttributeRecord(classAttributeRecord.build())
            .build();

    assertThat(nestDigest.getAllCompanionClassNames())
        .containsExactly("package/path/OwnerClass$NestCC");
  }

  @Test
  public void preparCompanionClassWriters_multipleCompanionClassesGenerated() {
    classMemberRecord.logMemberDecl(
        MethodKey.create(ClassName.create("package/path/OwnerClassA"), "<init>", "()V"),
        Opcodes.ACC_PUBLIC | Opcodes.ACC_SUPER,
        Opcodes.ACC_PRIVATE);
    classMemberRecord.logMemberDecl(
        MethodKey.create(ClassName.create("package/path/OwnerClassA"), "method", "(II)I"),
        Opcodes.ACC_PUBLIC | Opcodes.ACC_SUPER,
        Opcodes.ACC_PRIVATE);
    classMemberRecord.logMemberDecl(
        MethodKey.create(ClassName.create("package/path/OwnerClassB"), "<init>", "()V"),
        Opcodes.ACC_PUBLIC | Opcodes.ACC_SUPER,
        Opcodes.ACC_PRIVATE);

    classMemberRecord.logMemberUse(
        MethodKey.create(ClassName.create("package/path/OwnerClassA"), "<init>", "()V"),
        Opcodes.INVOKESPECIAL);
    classMemberRecord.logMemberUse(
        MethodKey.create(ClassName.create("package/path/OwnerClassA"), "method", "(II)I"),
        Opcodes.INVOKESPECIAL);
    classMemberRecord.logMemberUse(
        MethodKey.create(ClassName.create("package/path/OwnerClassB"), "<init>", "()V"),
        Opcodes.INVOKESPECIAL);

    classAttributeRecord.addClassAttributes(
        ClassAttributes.builder()
            .setClassBinaryName(ClassName.create("package/path/OwnerClassA$NestedClass"))
            .setNestHost(ClassName.create("package/path/OwnerClassA"))
            .build());
    classAttributeRecord.addClassAttributes(
        ClassAttributes.builder()
            .setClassBinaryName(ClassName.create("package/path/OwnerClassB$NestedClass"))
            .setNestHost(ClassName.create("package/path/OwnerClassB"))
            .build());

    classAttributeRecord.addClassAttributes(
        ClassAttributes.builder()
            .setClassBinaryName(ClassName.create("package/path/OwnerClassA"))
            .addNestMember(ClassName.create("package/path/OwnerClassA$NestedClass"))
            .build());
    classAttributeRecord.addClassAttributes(
        ClassAttributes.builder()
            .setClassBinaryName(ClassName.create("package/path/OwnerClassB"))
            .addNestMember(ClassName.create("package/path/OwnerClassB$NestedClass"))
            .build());

    NestDigest nestDigest =
        NestDigest.builder()
            .setClassMemberRecord(classMemberRecord.build())
            .setClassAttributeRecord(classAttributeRecord.build())
            .build();

    assertThat(nestDigest.getAllCompanionClassNames())
        .containsExactly("package/path/OwnerClassA$NestCC", "package/path/OwnerClassB$NestCC");
  }

  @Test
  public void prepareCompanionClassWriters_classNameWithDollarSign() {
    MethodKey constructor =
        MethodKey.create(ClassName.create("package/path/$Owner$Class$"), "<init>", "()V");

    classMemberRecord.logMemberDecl(
        constructor, Opcodes.ACC_PUBLIC | Opcodes.ACC_SUPER, Opcodes.ACC_PRIVATE);
    classMemberRecord.logMemberUse(constructor, Opcodes.INVOKESPECIAL);

    classAttributeRecord.addClassAttributes(
        ClassAttributes.builder()
            .setClassBinaryName(ClassName.create(constructor.ownerName() + "$NestClass"))
            .setNestHost(ClassName.create(constructor.ownerName()))
            .build());
    classAttributeRecord.addClassAttributes(
        ClassAttributes.builder()
            .setClassBinaryName(ClassName.create(constructor.ownerName()))
            .addNestMember(ClassName.create(constructor.ownerName() + "$NestClass"))
            .build());

    NestDigest nestDigest =
        NestDigest.builder()
            .setClassMemberRecord(classMemberRecord.build())
            .setClassAttributeRecord(classAttributeRecord.build())
            .build();

    assertThat(nestDigest.getAllCompanionClassNames())
        .containsExactly("package/path/$Owner$Class$$NestCC");
  }
}
