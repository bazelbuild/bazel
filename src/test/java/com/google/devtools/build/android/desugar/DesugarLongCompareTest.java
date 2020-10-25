// Copyright 2017 The Bazel Authors. All rights reserved.
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
import static org.junit.Assert.fail;
import static org.objectweb.asm.Opcodes.INVOKESTATIC;

import com.google.devtools.build.android.desugar.testdata.ClassCallingLongCompare;
import java.io.IOException;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.objectweb.asm.ClassReader;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;

/** The test case for the rewriter rewriting a call of Long.compare(long, long) to lcmp. */
@RunWith(JUnit4.class)
public class DesugarLongCompareTest {

  @Test
  public void testClassCallingLongCompareHasNoReferenceToLong_compare() {
    try {
      ClassReader reader = new ClassReader(ClassCallingLongCompare.class.getName());

      AtomicInteger counter = new AtomicInteger(0);

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
                      && owner.equals("java/lang/Long")
                      && name.equals("compare")
                      && desc.equals("(JJ)I")) {
                    counter.incrementAndGet();
                  }
                }
              };
            }
          },
          0);
      assertThat(counter.get()).isEqualTo(0);
    } catch (IOException e) {
      fail();
    }
  }

  @Test
  public void testCompareLongWithLambda() {
    assertThat(ClassCallingLongCompare.compareLongWithLambda(1, 0)).isEqualTo(1);
    assertThat(ClassCallingLongCompare.compareLongWithLambda(1, 1)).isEqualTo(0);
    assertThat(ClassCallingLongCompare.compareLongWithLambda(1, 2)).isEqualTo(-1);
    assertThat(ClassCallingLongCompare.compareLongWithLambda(Long.MAX_VALUE, Long.MIN_VALUE))
        .isEqualTo(1);
    assertThat(ClassCallingLongCompare.compareLongWithLambda(Long.MAX_VALUE, Long.MAX_VALUE))
        .isEqualTo(0);
    assertThat(ClassCallingLongCompare.compareLongWithLambda(Long.MIN_VALUE, Long.MAX_VALUE))
        .isEqualTo(-1);
  }

  @Test
  public void testCompareLongWithMethodReference() {
    assertThat(ClassCallingLongCompare.compareLongWithMethodReference(1, 0)).isEqualTo(1);
    assertThat(ClassCallingLongCompare.compareLongWithMethodReference(1, 1)).isEqualTo(0);
    assertThat(ClassCallingLongCompare.compareLongWithMethodReference(1, 2)).isEqualTo(-1);
    assertThat(
            ClassCallingLongCompare.compareLongWithMethodReference(Long.MAX_VALUE, Long.MIN_VALUE))
        .isEqualTo(1);
    assertThat(
            ClassCallingLongCompare.compareLongWithMethodReference(Long.MAX_VALUE, Long.MAX_VALUE))
        .isEqualTo(0);
    assertThat(
            ClassCallingLongCompare.compareLongWithMethodReference(Long.MIN_VALUE, Long.MAX_VALUE))
        .isEqualTo(-1);
  }

  @Test
  public void testcompareLongByCallingLong_compare() {
    assertThat(ClassCallingLongCompare.compareLongByCallingLong_compare(1, 0)).isEqualTo(1);
    assertThat(ClassCallingLongCompare.compareLongByCallingLong_compare(1, 1)).isEqualTo(0);
    assertThat(ClassCallingLongCompare.compareLongByCallingLong_compare(1, 2)).isEqualTo(-1);
    assertThat(
            ClassCallingLongCompare.compareLongByCallingLong_compare(
                Long.MAX_VALUE, Long.MIN_VALUE))
        .isEqualTo(1);
    assertThat(
            ClassCallingLongCompare.compareLongByCallingLong_compare(
                Long.MAX_VALUE, Long.MAX_VALUE))
        .isEqualTo(0);
    assertThat(
            ClassCallingLongCompare.compareLongByCallingLong_compare(
                Long.MIN_VALUE, Long.MAX_VALUE))
        .isEqualTo(-1);
  }

  @Test
  public void testcompareLongByCallingLong_compare2() {
    assertThat(ClassCallingLongCompare.compareLongByCallingLong_compare2(1, 0)).isEqualTo("g");
    assertThat(ClassCallingLongCompare.compareLongByCallingLong_compare2(1, 1)).isEqualTo("e");
    assertThat(ClassCallingLongCompare.compareLongByCallingLong_compare2(0, 1)).isEqualTo("l");

    assertThat(
            ClassCallingLongCompare.compareLongByCallingLong_compare2(
                Long.MAX_VALUE, Long.MIN_VALUE))
        .isEqualTo("g");
    assertThat(
            ClassCallingLongCompare.compareLongByCallingLong_compare2(
                Long.MAX_VALUE, Long.MAX_VALUE))
        .isEqualTo("e");
    assertThat(
            ClassCallingLongCompare.compareLongByCallingLong_compare2(
                Long.MIN_VALUE, Long.MAX_VALUE))
        .isEqualTo("l");
  }
}
