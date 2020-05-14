/*
 * Copyright 2020 The Bazel Authors. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.devtools.build.android.desugar.preanalysis;

import com.google.devtools.build.android.desugar.langmodel.ClassAttributeRecord.ClassAttributeRecordBuilder;
import com.google.devtools.build.android.desugar.langmodel.ClassAttributes;
import com.google.devtools.build.android.desugar.langmodel.ClassAttributes.ClassAttributesBuilder;
import com.google.devtools.build.android.desugar.langmodel.ClassMemberRecord;
import com.google.devtools.build.android.desugar.langmodel.ClassMemberRecord.ClassMemberRecordBuilder;
import com.google.devtools.build.android.desugar.langmodel.ClassName;
import com.google.devtools.build.android.desugar.langmodel.DesugarClassAttribute;
import com.google.devtools.build.android.desugar.langmodel.DesugarMethodAttribute;
import com.google.devtools.build.android.desugar.langmodel.FieldKey;
import com.google.devtools.build.android.desugar.langmodel.LangModelHelper;
import com.google.devtools.build.android.desugar.langmodel.MethodKey;
import com.google.devtools.build.android.desugar.langmodel.SyntheticMethod;
import com.google.devtools.build.android.desugar.langmodel.SyntheticMethod.SyntheticReason;
import org.objectweb.asm.Attribute;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.FieldVisitor;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;

/**
 * A visitor that performs pre-transformation phase analysis that collects relevant class file
 * information for the next transformation phase.
 */
public final class ClassMetadataCollector extends ClassVisitor {

  /** Class member records from a single input for nest-based access control analysis. */
  private final ClassMemberRecordBuilder nestAnalysisBasedMemberRecord;

  /**
   * Record that tracks class-level attributes, such nest members, nest mates and desugar-custom
   * attributes.
   */
  private final ClassAttributeRecordBuilder classAttributeRecord;

  /**
   * An class member record to stage member record candidates, merging into the project-wise member
   * record during the {@link #visitEnd()} where eligible conditions are specified.
   */
  private final ClassMemberRecordBuilder stagingMemberRecord = ClassMemberRecord.builder();

  private final ClassAttributesBuilder classAttributesBuilder;

  private ClassName className;
  private int classAccessCode;
  private boolean isInNest;

  ClassMetadataCollector(
      ClassMemberRecordBuilder nestAnalysisBasedMemberRecord,
      ClassAttributeRecordBuilder classAttributeRecord) {
    super(Opcodes.ASM8);
    this.nestAnalysisBasedMemberRecord = nestAnalysisBasedMemberRecord;
    this.classAttributeRecord = classAttributeRecord;
    this.classAttributesBuilder = ClassAttributes.builder();
  }

  @Override
  public void visit(
      int version,
      int access,
      String name,
      String signature,
      String superName,
      String[] interfaces) {
    className = ClassName.create(name);
    classAccessCode = access;
    classAttributesBuilder.setClassBinaryName(className);
    classAttributesBuilder.setMajorVersion(version & 0xffff);
    super.visit(version, access, name, signature, superName, interfaces);
  }

  @Override
  public void visitAttribute(Attribute attribute) {
    if (attribute instanceof DesugarClassAttribute) {
      DesugarClassAttribute desugarClassAttribute = (DesugarClassAttribute) attribute;
      for (SyntheticMethod syntheticMethod :
          desugarClassAttribute.getDesugarClassInfo().getSyntheticMethodList()) {
        if (SyntheticReason.OVERRIDING_BRIDGE.equals(syntheticMethod.getReason())) {
          classAttributesBuilder.addDesugarIgnoredMethods(
              MethodKey.from(syntheticMethod.getMethod()));
        }
      }
    }
    super.visitAttribute(attribute);
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
    if ((access & Opcodes.ACC_PRIVATE) != 0 && (access & Opcodes.ACC_STATIC) == 0) {
      classAttributesBuilder.addPrivateInstanceMethod(methodKey);
    }
    if ((access & Opcodes.ACC_PRIVATE) != 0 && !isInDeclOmissionList(methodKey)) {
      stagingMemberRecord.logMemberDecl(methodKey, classAccessCode, access);
    }
    return new MethodMetadataCollector(
        super.visitMethod(access, name, descriptor, signature, exceptions),
        methodKey,
        classAttributesBuilder,
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
    classAttributesBuilder.setNestHost(ClassName.create(nestHost));
    super.visitNestHost(nestHost);
  }

  @Override
  public void visitNestMember(String nestMember) {
    isInNest = true;
    classAttributesBuilder.addNestMember(ClassName.create(nestMember));
    super.visitNestMember(nestMember);
  }

  @Override
  public void visitEnd() {
    if (isInNest || (classAccessCode & Opcodes.ACC_INTERFACE) != 0) {
      nestAnalysisBasedMemberRecord.mergeFrom(stagingMemberRecord.build());
    }
    classAttributeRecord.addClassAttributes(classAttributesBuilder.build());
    super.visitEnd();
  }

  /**
   * A visitor that collects privately referenced class members (fields/constructors/methods) within
   * a nest.
   */
  private static class MethodMetadataCollector extends MethodVisitor {

    /** The current enclosing the method. */
    private final MethodKey enclosingMethodKey;

    private final ClassAttributesBuilder classAttributesBuilder;

    /**
     * A per-class class member record and will determined to merge or not into the main member
     * record at visitEnd of its associated class visitor.
     *
     * <p>@see {@link ClassMetadataCollector#stagingMemberRecord} for more details.
     */
    private final ClassMemberRecordBuilder nestAnalysisBasedMemberRecord;

    MethodMetadataCollector(
        MethodVisitor methodVisitor,
        MethodKey enclosingMethodKey,
        ClassAttributesBuilder classAttributesBuilder,
        ClassMemberRecordBuilder nestAnalysisBasedMemberRecord) {
      super(Opcodes.ASM8, methodVisitor);
      this.enclosingMethodKey = enclosingMethodKey;
      this.classAttributesBuilder = classAttributesBuilder;
      this.nestAnalysisBasedMemberRecord = nestAnalysisBasedMemberRecord;
    }

    @Override
    public void visitAttribute(Attribute attribute) {
      if (attribute instanceof DesugarMethodAttribute) {
        classAttributesBuilder.addDesugarIgnoredMethods(enclosingMethodKey);
      }
      super.visitAttribute(attribute);
    }

    @Override
    public void visitFieldInsn(int opcode, String owner, String name, String descriptor) {
      FieldKey memberKey = FieldKey.create(ClassName.create(owner), name, descriptor);
      if (LangModelHelper.isCrossMateRefInNest(memberKey, enclosingMethodKey)) {
        nestAnalysisBasedMemberRecord.logMemberUse(memberKey, opcode);
      }
      super.visitFieldInsn(opcode, owner, name, descriptor);
    }

    @Override
    public void visitMethodInsn(
        int opcode, String owner, String name, String descriptor, boolean isInterface) {
      MethodKey memberKey = MethodKey.create(ClassName.create(owner), name, descriptor);
      if (isInterface || LangModelHelper.isCrossMateRefInNest(memberKey, enclosingMethodKey)) {
        nestAnalysisBasedMemberRecord.logMemberUse(memberKey, opcode);
      }
      super.visitMethodInsn(opcode, owner, name, descriptor, isInterface);
    }
  }
}
