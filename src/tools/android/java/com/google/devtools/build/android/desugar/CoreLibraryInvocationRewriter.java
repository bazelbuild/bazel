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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;

/**
 * Rewriter of default and static interface methods defined in some core libraries.
 *
 * <p>This is conceptually similar to call site rewriting in {@link InterfaceDesugaring} but here
 * we're doing it for certain bootclasspath methods and in particular for invokeinterface and
 * invokevirtual, which are ignored in regular {@link InterfaceDesugaring}.
 */
public class CoreLibraryInvocationRewriter extends ClassVisitor {

  private final CoreLibrarySupport support;

  public CoreLibraryInvocationRewriter(ClassVisitor cv, CoreLibrarySupport support) {
    super(Opcodes.ASM8, cv);
    this.support = support;
  }

  @Override
  public MethodVisitor visitMethod(
      int access, String name, String desc, String signature, String[] exceptions) {
    MethodVisitor result = super.visitMethod(access, name, desc, signature, exceptions);
    return result != null ? new CoreLibraryMethodInvocationRewriter(result) : null;
  }

  private class CoreLibraryMethodInvocationRewriter extends MethodVisitor {

    public CoreLibraryMethodInvocationRewriter(MethodVisitor mv) {
      super(Opcodes.ASM8, mv);
    }

    @Override
    public void visitMethodInsn(int opcode, String owner, String name, String desc, boolean itf) {
      Class<?> coreInterface =
          support.getCoreInterfaceRewritingTarget(opcode, owner, name, desc, itf);
      if (coreInterface != null) {
        String coreInterfaceName = coreInterface.getName().replace('.', '/');
        if (opcode == Opcodes.INVOKESTATIC) {
          checkState(owner.equals(coreInterfaceName));
        } else {
          desc = InterfaceDesugaring.companionDefaultMethodDescriptor(coreInterfaceName, desc);
        }

        if (opcode == Opcodes.INVOKESTATIC || opcode == Opcodes.INVOKESPECIAL) {
          checkArgument(
              itf || opcode == Opcodes.INVOKESPECIAL,
              "Expected interface to rewrite %s.%s : %s",
              owner,
              name,
              desc);
          if (coreInterface.isInterface()) {
            owner = InterfaceDesugaring.getCompanionClassName(coreInterfaceName);
            name =
                InterfaceDesugaring.normalizeInterfaceMethodName(
                    name, name.startsWith("lambda$"), opcode);
          } else {
            owner = checkNotNull(support.getMoveTarget(coreInterfaceName, name));
          }
        } else {
          checkState(coreInterface.isInterface());
          owner = coreInterfaceName + "$$Dispatch";
        }

        opcode = Opcodes.INVOKESTATIC;
        itf = false;
      } else {
        String newOwner = support.getMoveTarget(owner, name);
        if (newOwner != null) {
          if (opcode != Opcodes.INVOKESTATIC) {
            // assuming a static method
            desc = InterfaceDesugaring.companionDefaultMethodDescriptor(owner, desc);
            opcode = Opcodes.INVOKESTATIC;
          }
          owner = newOwner;
          itf = false; // assuming a class
        }
      }
      super.visitMethodInsn(opcode, owner, name, desc, itf);
    }
  }
}
