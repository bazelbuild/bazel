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

import static org.objectweb.asm.Opcodes.INVOKESTATIC;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.android.desugar.io.CoreLibraryRewriter;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;

/**
 * This class rewrites (or removes) some trivial primitive wrapper methods.
 */
public class PrimitiveWrapperRewriter extends ClassVisitor {

  /** Classes with no-op {@code hashCode(primitive)} static methods, and their descriptors. */
  private static final ImmutableMap<String, String> NOOP_HASHCODE_DESC =
      ImmutableMap.of(
          "java/lang/Integer", "(I)I",
          "java/lang/Short", "(S)I",
          "java/lang/Byte", "(B)I",
          "java/lang/Character", "(C)I");

  private final CoreLibraryRewriter rewriter;

  public PrimitiveWrapperRewriter(ClassVisitor cv, CoreLibraryRewriter rewriter) {
    super(Opcodes.ASM7, cv);
    this.rewriter = rewriter;
  }

  @Override
  public MethodVisitor visitMethod(
      int access, String name, String desc, String signature, String[] exceptions) {
    MethodVisitor visitor = super.cv.visitMethod(access, name, desc, signature, exceptions);
    return visitor == null ? visitor : new PrimitiveWrapperMethodVisitor(visitor);
  }

  private class PrimitiveWrapperMethodVisitor extends MethodVisitor {

    public PrimitiveWrapperMethodVisitor(MethodVisitor visitor) {
      super(Opcodes.ASM7, visitor);
    }

    @Override
    public void visitMethodInsn(int opcode, String owner, String name, String desc, boolean itf) {
      String noopDescriptor = NOOP_HASHCODE_DESC.get(rewriter.unprefix(owner));
      if (opcode == INVOKESTATIC
          && !itf
          && name.equals("hashCode")
          && noopDescriptor != null
          && noopDescriptor.equals(desc)) {
        return;  // skip: use original primitive value as its hash (b/147139686)
      }
      super.visitMethodInsn(opcode, owner, name, desc, itf);
    }
  }
}
