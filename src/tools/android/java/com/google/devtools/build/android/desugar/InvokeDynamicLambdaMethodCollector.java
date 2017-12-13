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

import com.google.common.collect.ImmutableSet;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.Handle;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;

/**
 * Class visitor to collect all the lambda methods that are used in invokedynamic instructions.
 *
 * <p>Note that this class only collects lambda methods. If the invokedynamic is used for other
 * purposes, the methods used in the instruction are NOT collected.
 */
class InvokeDynamicLambdaMethodCollector extends ClassVisitor {

  private final ImmutableSet.Builder<MethodInfo> lambdaMethodsUsedInInvokeDynamic =
      ImmutableSet.builder();
  private boolean needOuterClassRewrite = false;

  public InvokeDynamicLambdaMethodCollector() {
    super(Opcodes.ASM6);
  }

  /**
   * Returns whether the visited class is declared in the scope of a lambda.  In that case
   * {@link LambdaDesugaring} will want to rewrite the EnclosingMethod attribute of the class.
   */
  public boolean needOuterClassRewrite() {
    return needOuterClassRewrite;
  }

  /** Returns methods referenced in invokedynamic instructions that use LambdaMetafactory. */
  public ImmutableSet<MethodInfo> getLambdaMethodsUsedInInvokeDynamics() {
    return lambdaMethodsUsedInInvokeDynamic.build();
  }

  @Override
  public MethodVisitor visitMethod(
      int access, String name, String desc, String signature, String[] exceptions) {
    MethodVisitor mv = super.visitMethod(access, name, desc, signature, exceptions);
    return new LambdaMethodCollector(mv);
  }

  @Override
  public void visitOuterClass(String owner, String name, String desc) {
    needOuterClassRewrite = needOuterClassRewrite || (name != null && name.startsWith("lambda$"));
    super.visitOuterClass(owner, name, desc);
  }

  private class LambdaMethodCollector extends MethodVisitor {

    public LambdaMethodCollector(MethodVisitor dest) {
      super(Opcodes.ASM6, dest);
    }

    @Override
    public void visitInvokeDynamicInsn(String name, String desc, Handle bsm, Object... bsmArgs) {
      if (!"java/lang/invoke/LambdaMetafactory".equals(bsm.getOwner())) {
        // Not an invokedynamic for a lambda expression
        return;
      }
      Handle handle = (Handle) bsmArgs[1];
      lambdaMethodsUsedInInvokeDynamic.add(
          MethodInfo.create(handle.getOwner(), handle.getName(), handle.getDesc()));
      super.visitInvokeDynamicInsn(name, desc, bsm, bsmArgs);
    }
  }
}
