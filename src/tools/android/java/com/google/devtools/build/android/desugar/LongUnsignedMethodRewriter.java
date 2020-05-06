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

import com.google.devtools.build.android.desugar.io.CoreLibraryRewriter;
import java.util.concurrent.atomic.AtomicInteger;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;

/** This class rewrites calls to supported Long.unsigned* methods to use desugar's runtime lib. */
public class LongUnsignedMethodRewriter extends ClassVisitor {

  private final CoreLibraryRewriter rewriter;
  private final AtomicInteger numOfUnsignedLongsInvoked;

  public LongUnsignedMethodRewriter(
      ClassVisitor cv, CoreLibraryRewriter rewriter, AtomicInteger numOfUnsignedLongsInvoked) {
    super(Opcodes.ASM8, cv);
    this.rewriter = rewriter;
    this.numOfUnsignedLongsInvoked = numOfUnsignedLongsInvoked;
  }

  @Override
  public MethodVisitor visitMethod(
      int access, String name, String desc, String signature, String[] exceptions) {
    MethodVisitor visitor = super.cv.visitMethod(access, name, desc, signature, exceptions);
    return visitor == null ? visitor : new LongUnsignedMethodVisitor(visitor);
  }

  private class LongUnsignedMethodVisitor extends MethodVisitor {

    public LongUnsignedMethodVisitor(MethodVisitor visitor) {
      super(Opcodes.ASM8, visitor);
    }

    @Override
    public void visitMethodInsn(int opcode, String owner, String name, String desc, boolean itf) {
      if (opcode == Opcodes.INVOKESTATIC
          && rewriter.unprefix(owner).equals("java/lang/Long")
          && desc.equals("(JJ)J")) {
        if (name.equals("divideUnsigned") || name.equals("remainderUnsigned")) {
          numOfUnsignedLongsInvoked.incrementAndGet();
          owner = "com/google/devtools/build/android/desugar/runtime/UnsignedLongs";
        }
      }
      super.visitMethodInsn(opcode, owner, name, desc, itf);
    }
  }
}
