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
package com.google.devtools.build.android.desugar;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.android.desugar.io.CoreLibraryRewriter;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;

// TODO(b/134636762): Test override preservation logic somehow (needs to class-load test input)
@RunWith(JUnit4.class)
public class CorePackageRenamerTest {

  @Test
  public void testSymbolRewrite() throws Exception {
    MockClassVisitor out = new MockClassVisitor();
    CorePackageRenamer renamer =
        new CorePackageRenamer(
            out,
            new CoreLibrarySupport(
                new CoreLibraryRewriter(""),
                null,
                ImmutableList.of("java/time/"),
                ImmutableList.of(),
                ImmutableList.of("java/util/A#m->java/time/B"),
                ImmutableList.of(),
                ImmutableList.of(),
                ImmutableList.of()));
    MethodVisitor mv = renamer.visitMethod(0, "test", "()V", null, null);

    mv.visitMethodInsn(
        Opcodes.INVOKESTATIC, "java/time/Instant", "now", "()Ljava/time/Instant;", false);
    assertThat(out.mv.owner).isEqualTo("j$/time/Instant");
    assertThat(out.mv.desc).isEqualTo("()Lj$/time/Instant;");

    // Ignore moved methods but not their descriptors
    mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/util/A", "m", "()Ljava/time/Instant;", false);
    assertThat(out.mv.owner).isEqualTo("java/util/A");
    assertThat(out.mv.desc).isEqualTo("()Lj$/time/Instant;");

    // Ignore arbitrary other methods but not their descriptors
    mv.visitMethodInsn(
        Opcodes.INVOKESTATIC, "other/time/Instant", "now", "()Ljava/time/Instant;", false);
    assertThat(out.mv.owner).isEqualTo("other/time/Instant");
    assertThat(out.mv.desc).isEqualTo("()Lj$/time/Instant;");

    mv.visitFieldInsn(Opcodes.GETFIELD, "other/time/Instant", "now", "Ljava/time/Instant;");
    assertThat(out.mv.owner).isEqualTo("other/time/Instant");
    assertThat(out.mv.desc).isEqualTo("Lj$/time/Instant;");
  }

  @Test
  public void testCorePackageCheck() throws Exception {
    MockClassVisitor out = new MockClassVisitor();
    CorePackageRenamer renamer =
        new CorePackageRenamer(
            out,
            new CoreLibrarySupport(
                new CoreLibraryRewriter(""),
                null,
                ImmutableList.of("java/time/"),
                ImmutableList.of(),
                ImmutableList.of(),
                ImmutableList.of(),
                ImmutableList.of(),
                ImmutableList.of()));
    MethodVisitor mv = renamer.visitMethod(0, "test", "()V", null, null);

    mv.visitMethodInsn(
        Opcodes.INVOKESTATIC, "android/support/Instant", "now", "()Ljava/time/Instant;", false);
    assertThat(out.mv.owner).isEqualTo("android/support/Instant");
    assertThat(out.mv.desc).isEqualTo("()Lj$/time/Instant;");

    mv.visitMethodInsn(
        Opcodes.INVOKESTATIC, "android/arch/Instant", "now", "()Ljava/time/Instant;", false);
    assertThat(out.mv.owner).isEqualTo("android/arch/Instant");
    assertThat(out.mv.desc).isEqualTo("()Lj$/time/Instant;");

    assertThrows(
        IllegalStateException.class,
        () ->
            mv.visitMethodInsn(
                Opcodes.INVOKESTATIC,
                "android/time/Instant",
                "now",
                "()Ljava/time/Instant;",
                false));
    assertThrows(
        IllegalStateException.class,
        () ->
            mv.visitFieldInsn(
                Opcodes.GETFIELD, "android/time/Instant", "now", "Ljava/time/Instant;"));
  }

  private static class MockClassVisitor extends ClassVisitor {

    final MockMethodVisitor mv = new MockMethodVisitor();

    public MockClassVisitor() {
      super(Opcodes.ASM7);
    }

    @Override
    public MethodVisitor visitMethod(
        int access, String name, String desc, String signature, String[] exceptions) {
      return mv;
    }
  }

  private static class MockMethodVisitor extends MethodVisitor {

    String owner;
    String desc;

    public MockMethodVisitor() {
      super(Opcodes.ASM7);
    }

    @Override
    public void visitMethodInsn(int opcode, String owner, String name, String desc, boolean itf) {
      this.owner = owner;
      this.desc = desc;
    }

    @Override
    public void visitFieldInsn(int opcode, String owner, String name, String desc) {
      this.owner = owner;
      this.desc = desc;
    }
  }
}
