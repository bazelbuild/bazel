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

import com.google.devtools.build.android.desugar.nest.ClassMemberKey.MethodKey;
import com.google.testing.testsize.SmallTest;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.objectweb.asm.Opcodes;

/** The tests for {@link NestCompanions}. */
@RunWith(JUnit4.class)
@SmallTest
public class NestCompanionsTest {

  private final ClassMemberRecord classMemberRecord = ClassMemberRecord.create();
  private final NestCompanions nestCompanions = NestCompanions.create(classMemberRecord);

  @Test
  public void prepareCompanionClassWriters_noCompanionClassesGenerated() {
    classMemberRecord.logMemberDecl(
        MethodKey.create("package/path/OwnerClass", "method", "(II)I"),
        Opcodes.ACC_PUBLIC | Opcodes.ACC_INTERFACE,
        Opcodes.ACC_PRIVATE);

    nestCompanions.prepareCompanionClassWriters();

    assertThat(nestCompanions.getAllCompanionClasses()).isEmpty();
  }

  @Test
  public void prepareCompanionClassWriters_companionClassesGenerated() {
    classMemberRecord.logMemberDecl(
        MethodKey.create("package/path/OwnerClass", "<init>", "()V"),
        Opcodes.ACC_PUBLIC | Opcodes.ACC_INTERFACE,
        Opcodes.ACC_PRIVATE);

    nestCompanions.prepareCompanionClassWriters();

    assertThat(nestCompanions.getAllCompanionClasses())
        .containsExactly("package/path/OwnerClass$NestCC");
  }

  @Test
  public void preparCompanionClassWriters_multipleCompanionClassesGenerated() {
    classMemberRecord.logMemberDecl(
        MethodKey.create("package/path/OwnerClassA", "<init>", "()V"),
        Opcodes.ACC_PUBLIC | Opcodes.ACC_INTERFACE,
        Opcodes.ACC_PRIVATE);
    classMemberRecord.logMemberDecl(
        MethodKey.create("package/path/OwnerClassA", "method", "(II)I"),
        Opcodes.ACC_PUBLIC | Opcodes.ACC_INTERFACE,
        Opcodes.ACC_PRIVATE);
    classMemberRecord.logMemberDecl(
        MethodKey.create("package/path/OwnerClassB", "<init>", "()V"),
        Opcodes.ACC_PUBLIC | Opcodes.ACC_INTERFACE,
        Opcodes.ACC_PRIVATE);

    nestCompanions.prepareCompanionClassWriters();

    assertThat(nestCompanions.getAllCompanionClasses())
        .containsExactly("package/path/OwnerClassA$NestCC", "package/path/OwnerClassB$NestCC");
  }
}
