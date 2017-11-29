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

import com.google.common.base.Preconditions;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.Label;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;

/**
 * A class scanner to check whether the class has the synthetic method $closeResource(Throwable,
 * AutoCloseable).
 */
public class CloseResourceMethodScanner extends ClassVisitor {

  private boolean hasCloseResourceMethod;
  private String internalName;
  private int classFileVersion;

  public CloseResourceMethodScanner() {
    super(Opcodes.ASM6);
  }

  @Override
  public void visit(
      int version,
      int access,
      String name,
      String signature,
      String superName,
      String[] interfaces) {
    Preconditions.checkState(internalName == null, "This scanner has been used.");
    this.internalName = name;
    this.classFileVersion = version;
    super.visit(version, access, name, signature, superName, interfaces);
  }

  public boolean hasCloseResourceMethod() {
    return hasCloseResourceMethod;
  }

  @Override
  public MethodVisitor visitMethod(
      int access, String name, String desc, String signature, String[] exceptions) {
    if (classFileVersion <= 50) {
      // A Java 6 or below class file should not have $closeResource method.
      return null;
    }
    if (!hasCloseResourceMethod) {
      hasCloseResourceMethod =
          TryWithResourcesRewriter.isSyntheticCloseResourceMethod(access, name, desc);
    }
    return new StackMapFrameCollector(name, desc);
  }

  private class StackMapFrameCollector extends MethodVisitor {

    private final String methodSignature;
    private boolean hasCallToCloseResourceMethod;
    private boolean hasJumpInstructions;
    private boolean hasStackMapFrame;

    public StackMapFrameCollector(String name, String desc) {
      super(Opcodes.ASM6);
      methodSignature = internalName + '.' + name + desc;
    }

    @Override
    public void visitEnd() {
      if (!hasCallToCloseResourceMethod) {
        return;
      }
      if (hasJumpInstructions && !hasStackMapFrame) {
        throw new UnsupportedOperationException(
            "The method "
                + methodSignature
                + " calls $closeResource(Throwable, AutoCloseable), "
                + "and Desugar thus needs to perform type inference for it "
                + "to rewrite $closeResourceMethod. "
                + "However, this method has jump instructions, but does not have stack map frames. "
                + "Please recompile this class with stack map frames.");
      }
    }

    @Override
    public void visitMethodInsn(int opcode, String owner, String name, String desc, boolean itf) {
      if (!hasCallToCloseResourceMethod
          && TryWithResourcesRewriter.isCallToSyntheticCloseResource(
              internalName, opcode, owner, name, desc)) {
        hasCallToCloseResourceMethod = true;
      }
    }

    @Override
    public void visitFrame(int type, int nLocal, Object[] local, int nStack, Object[] stack) {
      hasStackMapFrame = true;
    }

    @Override
    public void visitJumpInsn(int opcode, Label label) {
      hasJumpInstructions = true;
    }
  }
}
