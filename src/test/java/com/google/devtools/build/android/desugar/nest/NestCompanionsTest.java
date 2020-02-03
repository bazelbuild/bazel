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
import com.google.devtools.build.android.desugar.langmodel.ClassAttributes;
import com.google.devtools.build.android.desugar.langmodel.ClassMemberRecord;
import com.google.devtools.build.android.desugar.langmodel.MethodKey;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.objectweb.asm.Opcodes;

/** The tests for {@link NestCompanions}. */
@RunWith(JUnit4.class)
public class NestCompanionsTest {

  private final ClassMemberRecord classMemberRecord = ClassMemberRecord.create();
  private final ClassAttributeRecord classAttributeRecord = ClassAttributeRecord.create();
  private final NestCompanions nestCompanions =
      NestCompanions.create(classMemberRecord, classAttributeRecord);

  @Test
  public void prepareCompanionClassWriters_noCompanionClassesGenerated() {
    classMemberRecord.logMemberDecl(
        MethodKey.create("package/path/OwnerClass", "method", "(II)I"),
        /* ownerAccess= */ Opcodes.ACC_PUBLIC | Opcodes.ACC_INTERFACE,
        /* memberDeclAccess= */ Opcodes.ACC_PRIVATE);

    nestCompanions.prepareCompanionClasses();

    assertThat(nestCompanions.getAllCompanionClasses()).isEmpty();
  }

  @Test
  public void prepareCompanionClassWriters_companionClassesGenerated() {
    MethodKey constructor = MethodKey.create("package/path/OwnerClass", "<init>", "()V");
    classMemberRecord.logMemberDecl(
        constructor, Opcodes.ACC_PUBLIC | Opcodes.ACC_SUPER, Opcodes.ACC_PRIVATE);
    classMemberRecord.logMemberUse(constructor, Opcodes.INVOKESPECIAL);
    classAttributeRecord.setClassAttributes(
        ClassAttributes.builder()
            .setClassBinaryName("package/path/OwnerClass$NestedClass")
            .setNestHost("package/path/OwnerClass")
            .build());
    classAttributeRecord.setClassAttributes(
        ClassAttributes.builder()
            .setClassBinaryName("package/path/OwnerClass")
            .addNestMember("package/path/OwnerClass$NestedClass")
            .build());

    nestCompanions.prepareCompanionClasses();

    assertThat(nestCompanions.getAllCompanionClasses())
        .containsExactly("package/path/OwnerClass$NestCC");
  }

  @Test
  public void preparCompanionClassWriters_multipleCompanionClassesGenerated() {
    classMemberRecord.logMemberDecl(
        MethodKey.create("package/path/OwnerClassA", "<init>", "()V"),
        Opcodes.ACC_PUBLIC | Opcodes.ACC_SUPER,
        Opcodes.ACC_PRIVATE);
    classMemberRecord.logMemberDecl(
        MethodKey.create("package/path/OwnerClassA", "method", "(II)I"),
        Opcodes.ACC_PUBLIC | Opcodes.ACC_SUPER,
        Opcodes.ACC_PRIVATE);
    classMemberRecord.logMemberDecl(
        MethodKey.create("package/path/OwnerClassB", "<init>", "()V"),
        Opcodes.ACC_PUBLIC | Opcodes.ACC_SUPER,
        Opcodes.ACC_PRIVATE);

    classMemberRecord.logMemberUse(
        MethodKey.create("package/path/OwnerClassA", "<init>", "()V"), Opcodes.INVOKESPECIAL);
    classMemberRecord.logMemberUse(
        MethodKey.create("package/path/OwnerClassA", "method", "(II)I"), Opcodes.INVOKESPECIAL);
    classMemberRecord.logMemberUse(
        MethodKey.create("package/path/OwnerClassB", "<init>", "()V"), Opcodes.INVOKESPECIAL);

    classAttributeRecord.setClassAttributes(
        ClassAttributes.builder()
            .setClassBinaryName("package/path/OwnerClassA$NestedClass")
            .setNestHost("package/path/OwnerClassA")
            .build());
    classAttributeRecord.setClassAttributes(
        ClassAttributes.builder()
            .setClassBinaryName("package/path/OwnerClassB$NestedClass")
            .setNestHost("package/path/OwnerClassB")
            .build());

    classAttributeRecord.setClassAttributes(
        ClassAttributes.builder()
            .setClassBinaryName("package/path/OwnerClassA")
            .addNestMember("package/path/OwnerClassA$NestedClass")
            .build());
    classAttributeRecord.setClassAttributes(
        ClassAttributes.builder()
            .setClassBinaryName("package/path/OwnerClassB")
            .addNestMember("package/path/OwnerClassB$NestedClass")
            .build());

    nestCompanions.prepareCompanionClasses();

    assertThat(nestCompanions.getAllCompanionClasses())
        .containsExactly("package/path/OwnerClassA$NestCC", "package/path/OwnerClassB$NestCC");
  }

  @Test
  public void prepareCompanionClassWriters_classNameWithDollarSign() {
    MethodKey constructor = MethodKey.create("package/path/$Owner$Class$", "<init>", "()V");

    classMemberRecord.logMemberDecl(
        constructor, Opcodes.ACC_PUBLIC | Opcodes.ACC_SUPER, Opcodes.ACC_PRIVATE);
    classMemberRecord.logMemberUse(constructor, Opcodes.INVOKESPECIAL);

    classAttributeRecord.setClassAttributes(
        ClassAttributes.builder()
            .setClassBinaryName(constructor.owner() + "$NestClass")
            .setNestHost(constructor.owner())
            .build());
    classAttributeRecord.setClassAttributes(
        ClassAttributes.builder()
            .setClassBinaryName(constructor.owner())
            .addNestMember(constructor.owner() + "$NestClass")
            .build());

    nestCompanions.prepareCompanionClasses();

    assertThat(nestCompanions.getAllCompanionClasses())
        .containsExactly("package/path/$Owner$Class$$NestCC");
  }
}
