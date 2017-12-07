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

import static org.objectweb.asm.Opcodes.ASM6;
import static org.objectweb.asm.Opcodes.DUP;
import static org.objectweb.asm.Opcodes.INVOKESTATIC;
import static org.objectweb.asm.Opcodes.INVOKEVIRTUAL;
import static org.objectweb.asm.Opcodes.POP;

import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.MethodVisitor;

/**
 * This class desugars any call to Objects.requireNonNull(Object o), Objects.requireNonNull(Object
 * o, String msg), and Objects.requireNonNull(Object o, Supplier msg), by replacing the call with
 * o.getClass().
 */
public class ObjectsRequireNonNullMethodRewriter extends ClassVisitor {

  public ObjectsRequireNonNullMethodRewriter(ClassVisitor cv) {
    super(ASM6, cv);
  }

  @Override
  public MethodVisitor visitMethod(
      int access, String name, String desc, String signature, String[] exceptions) {
    MethodVisitor visitor = super.cv.visitMethod(access, name, desc, signature, exceptions);
    return visitor == null ? visitor : new ObjectsMethodInlinerMethodVisitor(visitor);
  }

  private static class ObjectsMethodInlinerMethodVisitor extends MethodVisitor {

    public ObjectsMethodInlinerMethodVisitor(MethodVisitor mv) {
      super(ASM6, mv);
    }

    @Override
    public void visitMethodInsn(int opcode, String owner, String name, String desc, boolean itf) {
      if (opcode != INVOKESTATIC
          || !owner.equals("java/util/Objects")
          || !name.equals("requireNonNull")
          || !desc.equals("(Ljava/lang/Object;)Ljava/lang/Object;")) {
        super.visitMethodInsn(opcode, owner, name, desc, itf);
        return;
      }

      // a call to Objects.requireNonNull(Object o)
      // duplicate the first argument 'o', as this method returns 'o'.
      super.visitInsn(DUP);
      super.visitMethodInsn(
          INVOKEVIRTUAL, "java/lang/Object", "getClass", "()Ljava/lang/Class;", false);
      super.visitInsn(POP);
    }
  }
}
