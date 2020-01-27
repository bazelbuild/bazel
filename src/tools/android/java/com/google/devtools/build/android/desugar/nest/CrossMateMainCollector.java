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

import com.google.devtools.build.android.desugar.langmodel.ClassMemberKey;
import com.google.devtools.build.android.desugar.langmodel.ClassMemberRecord;
import com.google.devtools.build.android.desugar.langmodel.FieldKey;
import com.google.devtools.build.android.desugar.langmodel.LangModelHelper;
import com.google.devtools.build.android.desugar.langmodel.MethodKey;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.FieldVisitor;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;

/**
 * A visitor that collects declared private members (fields, constructors and methods), and
 * delegates to a method visitor to track the references to these members.
 */
final class CrossMateMainCollector extends ClassVisitor {

  /** The project-wise class member records. */
  private final ClassMemberRecord memberRecord;

  /**
   * An class member record to stage member record candidates, merging into the project-wise member
   * record during the {@link #visitEnd()} where eligible conditions are specified.
   */
  private final ClassMemberRecord stagingMemberRecord = ClassMemberRecord.create();

  private String className;
  private int classAccessCode;
  private boolean isInNest;

  public CrossMateMainCollector(ClassMemberRecord memberRecord) {
    super(Opcodes.ASM7);
    this.memberRecord = memberRecord;
  }

  @Override
  public void visit(
      int version,
      int access,
      String name,
      String signature,
      String superName,
      String[] interfaces) {
    className = name;
    classAccessCode = access;
    super.visit(
        Math.min(version, NestDesugarConstants.MIN_VERSION),
        access,
        name,
        signature,
        superName,
        interfaces);
  }

  @Override
  public FieldVisitor visitField(
      int access, String name, String descriptor, String signature, Object value) {
    if ((access & Opcodes.ACC_PRIVATE) != 0) {
      stagingMemberRecord.logMemberDecl(
          FieldKey.create(className, name, descriptor), classAccessCode, access);
    }
    return super.visitField(access, name, descriptor, signature, value);
  }

  @Override
  public MethodVisitor visitMethod(
      int access, String name, String descriptor, String signature, String[] exceptions) {
    MethodKey methodKey = MethodKey.create(className, name, descriptor);
    if ((access & Opcodes.ACC_PRIVATE) != 0 && !isInDeclOmissionList(methodKey)) {
      stagingMemberRecord.logMemberDecl(methodKey, classAccessCode, access);
    }
    return new CrossMateRefCollector(
        super.visitMethod(access, name, descriptor, signature, exceptions),
        methodKey,
        stagingMemberRecord);
  }

  /**
   * The method declaration will be explicitly omitted.
   *
   * <p>TODO(deltazulu): Refine this list and condition check. e.g. check ACC_SYNTHETIC flag.
   */
  private static boolean isInDeclOmissionList(MethodKey methodKey) {
    return methodKey.name().startsWith("lambda$") // handled by LambdaDesugaring.
        || methodKey.name().equals("$deserializeLambda$") // handled by LambdaDesugaring.
        || methodKey.name().contains("jacoco$") // handled by InterfaceDesugaring.
        || methodKey.name().contains("$jacoco"); // handled by InterfaceDesugaring.
  }

  @Override
  public void visitNestHost(String nestHost) {
    isInNest = true;
    super.visitNestHost(nestHost);
  }

  @Override
  public void visitNestMember(String nestMember) {
    isInNest = true;
    super.visitNestMember(nestMember);
  }

  @Override
  public void visitEnd() {
    if (isInNest || (classAccessCode & Opcodes.ACC_INTERFACE) != 0) {
      memberRecord.mergeFrom(stagingMemberRecord);
    }
    super.visitEnd();
  }

  /**
   * A visitor that collects privately referenced class members (fields/constructors/methods) within
   * a nest.
   */
  private static class CrossMateRefCollector extends MethodVisitor {

    /** The current enclosing the method. */
    private final MethodKey enclosingMethodKey;

    /**
     * A per-class class member record and will determined to merge or not into the main member
     * record at visitEnd of its associated class visitor.
     *
     * <p>@see CrossMateMainCollector#stagingMemberRecord for more details.
     */
    private final ClassMemberRecord memberRecord;

    CrossMateRefCollector(
        MethodVisitor methodVisitor, MethodKey enclosingMethodKey, ClassMemberRecord memberRecord) {
      super(Opcodes.ASM7, methodVisitor);
      this.enclosingMethodKey = enclosingMethodKey;
      this.memberRecord = memberRecord;
    }

    @Override
    public void visitFieldInsn(int opcode, String owner, String name, String descriptor) {
      ClassMemberKey memberKey = FieldKey.create(owner, name, descriptor);
      if (LangModelHelper.isCrossMateRefInNest(memberKey, enclosingMethodKey)) {
        memberRecord.logMemberUse(memberKey, opcode);
      }
      super.visitFieldInsn(opcode, owner, name, descriptor);
    }

    @Override
    public void visitMethodInsn(
        int opcode, String owner, String name, String descriptor, boolean isInterface) {
      ClassMemberKey memberKey = MethodKey.create(owner, name, descriptor);
      if (isInterface || LangModelHelper.isCrossMateRefInNest(memberKey, enclosingMethodKey)) {
        memberRecord.logMemberUse(memberKey, opcode);
      }
      super.visitMethodInsn(opcode, owner, name, descriptor, isInterface);
    }
  }
}
