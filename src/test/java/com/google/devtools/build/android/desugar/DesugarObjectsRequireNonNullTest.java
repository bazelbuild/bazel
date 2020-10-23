// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.desugar;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.fail;
import static org.objectweb.asm.Opcodes.INVOKESTATIC;

import com.google.devtools.build.android.desugar.testdata.ClassCallingRequireNonNull;
import java.io.IOException;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.objectweb.asm.ClassReader;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;

/**
 * This test case tests the desugaring feature for Objects.requireNonNull. This feature replaces any
 * call to this method with o.getClass() to check whether 'o' is null.
 */
@RunWith(JUnit4.class)
public class DesugarObjectsRequireNonNullTest {

  @Test
  public void testClassCallingRequireNonNullHasNoReferenceToRequiresNonNull() {
    try {
      ClassReader reader = new ClassReader(ClassCallingRequireNonNull.class.getName());

      AtomicInteger counterForSingleArgument = new AtomicInteger(0);
      AtomicInteger counterForString = new AtomicInteger(0);
      AtomicInteger counterForSupplier = new AtomicInteger(0);

      reader.accept(
          new ClassVisitor(Opcodes.ASM8) {
            @Override
            public MethodVisitor visitMethod(
                int access, String name, String desc, String signature, String[] exceptions) {
              return new MethodVisitor(api) {
                @Override
                public void visitMethodInsn(
                    int opcode, String owner, String name, String desc, boolean itf) {
                  if (opcode == INVOKESTATIC
                      && owner.equals("java/util/Objects")
                      && name.equals("requireNonNull")) {
                    switch (desc) {
                      case "(Ljava/lang/Object;)Ljava/lang/Object;":
                        counterForSingleArgument.incrementAndGet();
                        break;
                      case "(Ljava/lang/Object;Ljava/lang/String;)Ljava/lang/Object;":
                        counterForString.incrementAndGet();
                        break;
                      case "(Ljava/lang/Object;Ljava/util/function/Supplier;)Ljava/lang/Object;":
                        counterForSupplier.incrementAndGet();
                        break;
                      default:
                        fail("Unknown overloaded requireNonNull is found: " + desc);
                    }
                  }
                }
              };
            }
          },
          0);
      assertThat(counterForSingleArgument.get()).isEqualTo(0);
      // we do not desugar requireNonNull(Object, String) or requireNonNull(Object, Supplier)
      assertThat(counterForString.get()).isEqualTo(1);
      assertThat(counterForSupplier.get()).isEqualTo(1);
    } catch (IOException e) {
      fail();
    }
  }

  @Test
  public void testInliningImplicitCallToObjectsRequireNonNull() {
    assertThrows(
        NullPointerException.class,
        () -> ClassCallingRequireNonNull.getStringLengthWithMethodReference(null));

    assertThat(ClassCallingRequireNonNull.getStringLengthWithMethodReference("")).isEqualTo(0);
    assertThat(ClassCallingRequireNonNull.getStringLengthWithMethodReference("1")).isEqualTo(1);

    assertThrows(
        NullPointerException.class,
        () ->
            ClassCallingRequireNonNull.getStringLengthWithLambdaAndExplicitCallToRequireNonNull(
                null));

    assertThat(
            ClassCallingRequireNonNull.getStringLengthWithLambdaAndExplicitCallToRequireNonNull(""))
        .isEqualTo(0);
    assertThat(
            ClassCallingRequireNonNull.getStringLengthWithLambdaAndExplicitCallToRequireNonNull(
                "1"))
        .isEqualTo(1);
  }

  @Test
  public void testInliningExplicitCallToObjectsRequireNonNull() {
    assertThrows(
        NullPointerException.class, () -> ClassCallingRequireNonNull.getFirstCharVersionOne(null));

    assertThrows(
        NullPointerException.class, () -> ClassCallingRequireNonNull.getFirstCharVersionTwo(null));

    assertThrows(
        NullPointerException.class,
        () -> ClassCallingRequireNonNull.callRequireNonNullWithArgumentString(null));

    assertThrows(
        NullPointerException.class,
        () -> ClassCallingRequireNonNull.callRequireNonNullWithArgumentSupplier(null));

    assertThat(ClassCallingRequireNonNull.getFirstCharVersionOne("hello")).isEqualTo('h');
    assertThat(ClassCallingRequireNonNull.getFirstCharVersionTwo("hello")).isEqualTo('h');

    assertThat(ClassCallingRequireNonNull.callRequireNonNullWithArgumentString("hello"))
        .isEqualTo('h');
    assertThat(ClassCallingRequireNonNull.callRequireNonNullWithArgumentSupplier("hello"))
        .isEqualTo('h');
  }
}
