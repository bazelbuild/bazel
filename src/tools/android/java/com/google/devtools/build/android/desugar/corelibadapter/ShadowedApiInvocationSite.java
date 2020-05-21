/*
 * Copyright 2020 The Bazel Authors. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.devtools.build.android.desugar.corelibadapter;

import static com.google.common.base.Preconditions.checkState;
import static com.google.devtools.build.android.desugar.corelibadapter.InvocationSiteTransformationReason.InvocationSiteTransformationKind.INLINE_PARAM_TYPE_CONVERSION;
import static com.google.devtools.build.android.desugar.corelibadapter.ShadowedApiAdapterHelper.shouldUseApiTypeAdapter;
import static com.google.devtools.build.android.desugar.corelibadapter.ShadowedApiAdapterHelper.shouldUseInlineTypeConversion;
import static com.google.devtools.build.android.desugar.langmodel.ClassName.IMMUTABLE_LABEL_ATTACHER;
import static com.google.devtools.build.android.desugar.langmodel.ClassName.IN_PROCESS_LABEL_STRIPPER;
import static com.google.devtools.build.android.desugar.langmodel.ClassName.SHADOWED_TO_MIRRORED_TYPE_MAPPER;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.android.desugar.corelibadapter.InvocationSiteTransformationRecord.InvocationSiteTransformationRecordBuilder;
import com.google.devtools.build.android.desugar.io.BootClassPathDigest;
import com.google.devtools.build.android.desugar.langmodel.ClassAttributeRecord;
import com.google.devtools.build.android.desugar.langmodel.ClassName;
import com.google.devtools.build.android.desugar.langmodel.DesugarMethodAttribute;
import com.google.devtools.build.android.desugar.langmodel.DesugarMethodInfo;
import com.google.devtools.build.android.desugar.langmodel.LangModelHelper;
import com.google.devtools.build.android.desugar.langmodel.MemberUseKind;
import com.google.devtools.build.android.desugar.langmodel.MethodDeclInfo;
import com.google.devtools.build.android.desugar.langmodel.MethodInvocationSite;
import com.google.devtools.build.android.desugar.langmodel.MethodKey;
import com.google.devtools.build.android.desugar.langmodel.SwitchableTypeMapper;
import com.google.devtools.build.android.desugar.typehierarchy.TypeHierarchy;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;
import org.objectweb.asm.commons.ClassRemapper;
import org.objectweb.asm.commons.MethodRemapper;
import org.objectweb.asm.tree.MethodNode;

/**
 * Desugars the bytecode instructions that interacts with desugar-shadowed APIs, which is a method
 * with desugar-shadowed types, e.g. {@code java.time.MonthDay}.
 */
public final class ShadowedApiInvocationSite extends ClassVisitor {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();
  private static final SwitchableTypeMapper<InvocationSiteTransformationReason>
      immutableLabelApplicator =
          new SwitchableTypeMapper<>(IN_PROCESS_LABEL_STRIPPER.andThen(IMMUTABLE_LABEL_ATTACHER));

  /** An evolving record that collects the adapter method requests from invocation sites. */
  private final InvocationSiteTransformationRecordBuilder invocationSiteRecord;

  private final BootClassPathDigest bootClassPathDigest;
  private final TypeHierarchy typeHierarchy;
  private final ClassAttributeRecord classAttributeRecord;

  private int classAccess;
  private ClassName className;
  private ImmutableSet<MethodKey> desugarIgnoredMethods;

  public ShadowedApiInvocationSite(
      ClassVisitor classVisitor,
      InvocationSiteTransformationRecordBuilder invocationSiteRecord,
      BootClassPathDigest bootClassPathDigest,
      ClassAttributeRecord classAttributeRecord,
      TypeHierarchy typeHierarchy) {
    super(Opcodes.ASM8, classVisitor);
    this.invocationSiteRecord = invocationSiteRecord;
    this.bootClassPathDigest = bootClassPathDigest;
    this.classAttributeRecord = classAttributeRecord;
    this.typeHierarchy = typeHierarchy;
  }

  @Override
  public void visit(
      int version,
      int access,
      String name,
      String signature,
      String superName,
      String[] interfaces) {
    this.classAccess = access;
    this.className = ClassName.create(name);
    this.desugarIgnoredMethods =
        classAttributeRecord.hasAttributeRecordFor(className)
            ? classAttributeRecord.getDesugarIgnoredMethods(className)
            : ImmutableSet.of();
    super.visit(version, access, name, signature, superName, interfaces);
  }

  @Override
  public MethodVisitor visitMethod(
      int access, String name, String descriptor, String signature, String[] exceptions) {
    MethodDeclInfo verbatimMethod =
        MethodDeclInfo.create(
                MethodKey.create(className, name, descriptor),
                classAccess,
                access,
                signature,
                exceptions)
            .acceptTypeMapper(IN_PROCESS_LABEL_STRIPPER);
    if (desugarIgnoredMethods.contains(verbatimMethod.methodKey())) {
      MethodVisitor bridgeMethodVisitor =
          verbatimMethod.acceptTypeMapper(IMMUTABLE_LABEL_ATTACHER).accept(cv);
      return new MethodRemapper(
          bridgeMethodVisitor, IN_PROCESS_LABEL_STRIPPER.andThen(IMMUTABLE_LABEL_ATTACHER));
    }
    if (ShadowedApiAdapterHelper.shouldEmitApiOverridingBridge(
        verbatimMethod, typeHierarchy, bootClassPathDigest)) {
      MethodNode bridgeMethodNode = new MethodNode();

      bridgeMethodNode.visitAttribute(
          new DesugarMethodAttribute(
              DesugarMethodInfo.newBuilder()
                  .setDesugarToolIgnore(true)
                  .setSyntheticReason(DesugarMethodInfo.SyntheticReason.OVERRIDING_BRIDGE)
                  .build()));

      bridgeMethodNode.visitCode();

      int slotOffset = 0;
      bridgeMethodNode.visitVarInsn(Opcodes.ALOAD, slotOffset++);
      for (Type argType : verbatimMethod.argumentTypes()) {
        ClassName argTypeName = ClassName.create(argType);
        bridgeMethodNode.visitVarInsn(argType.getOpcode(Opcodes.ILOAD), slotOffset);
        if (argTypeName.isDesugarShadowedType()) {
          MethodInvocationSite conversion =
              ShadowedApiAdapterHelper.shadowedToMirroredTypeConversionSite(argTypeName);
          conversion.accept(bridgeMethodNode);
        }
        slotOffset += argType.getSize();
      }

      // revisit
      MethodInvocationSite shadowedInvocation =
          MethodInvocationSite.builder()
              .setInvocationKind(MemberUseKind.INVOKEVIRTUAL)
              .setMethod(verbatimMethod.methodKey())
              .setIsInterface(false)
              .build();
      MethodInvocationSite mirroredInvocationSite =
          shadowedInvocation.acceptTypeMapper(SHADOWED_TO_MIRRORED_TYPE_MAPPER);
      mirroredInvocationSite.accept(bridgeMethodNode);

      // TODO(deltazulu): Refine forward / backward conversions.
      invocationSiteRecord.addInlineConversion(shadowedInvocation);

      ClassName adapterReturnTypeName = verbatimMethod.returnTypeName();
      if (adapterReturnTypeName.isDesugarShadowedType()) {
        MethodInvocationSite conversion =
            ShadowedApiAdapterHelper.mirroredToShadowedTypeConversionSite(
                adapterReturnTypeName.shadowedToMirrored());
        conversion.accept(bridgeMethodNode);
      }

      bridgeMethodNode.visitInsn(verbatimMethod.returnType().getOpcode(Opcodes.IRETURN));

      bridgeMethodNode.visitMaxs(slotOffset, slotOffset);
      bridgeMethodNode.visitEnd();

      MethodVisitor bridgeMethodVisitor =
          verbatimMethod.acceptTypeMapper(IMMUTABLE_LABEL_ATTACHER).accept(cv);

      MethodRemapper methodRemapper =
          new MethodRemapper(
              bridgeMethodVisitor, IN_PROCESS_LABEL_STRIPPER.andThen(IMMUTABLE_LABEL_ATTACHER));

      bridgeMethodNode.accept(methodRemapper);
    }
    MethodVisitor mv = super.visitMethod(access, name, descriptor, signature, exceptions);
    return mv == null
        ? null
        : new ShadowedApiInvocationSiteMethodVisitor(
            api, mv, verbatimMethod, invocationSiteRecord, typeHierarchy, bootClassPathDigest);
  }

  /** Desugars instructions for the enclosing class visitor. */
  private static class ShadowedApiInvocationSiteMethodVisitor extends MethodRemapper {

    private static final String BEGIN_TAG = "BEGIN:";
    private static final String END_TAG = "END:";

    private final MethodDeclInfo enclosingMethod;
    private final InvocationSiteTransformationRecordBuilder invocationSiteRecord;
    private final TypeHierarchy typeHierarchy;
    private final BootClassPathDigest bootClassPathDigest;

    /**
     * The transformation-in-process invocation site while visiting. The value can be {@code null}
     * if no in-line type conversion is in process.
     */
    private InvocationSiteTransformationReason inProcessTransformation;

    private ShadowedApiInvocationSiteMethodVisitor(
        int api,
        MethodVisitor methodVisitor,
        MethodDeclInfo enclosingMethod,
        InvocationSiteTransformationRecordBuilder invocationSiteRecord,
        TypeHierarchy typeHierarchy,
        BootClassPathDigest bootClassPathDigest) {
      super(api, methodVisitor, immutableLabelApplicator);
      this.enclosingMethod = enclosingMethod;
      this.invocationSiteRecord = invocationSiteRecord;
      this.typeHierarchy = typeHierarchy;
      this.bootClassPathDigest = bootClassPathDigest;
    }

    @Override
    public void visitLdcInsn(Object value) {
      if (value instanceof String) {
        String paramTypeConversionTag = (String) value;
        if (paramTypeConversionTag.startsWith(BEGIN_TAG + INLINE_PARAM_TYPE_CONVERSION)) {
          checkState(inProcessTransformation == null);
          inProcessTransformation =
              InvocationSiteTransformationReason.decode(
                  paramTypeConversionTag.substring(BEGIN_TAG.length()));
          immutableLabelApplicator.turnOn(inProcessTransformation);
        } else if (paramTypeConversionTag.startsWith(END_TAG + INLINE_PARAM_TYPE_CONVERSION)) {
          checkState(inProcessTransformation != null);
          InvocationSiteTransformationReason reasonToClose =
              InvocationSiteTransformationReason.decode(
                  paramTypeConversionTag.substring(END_TAG.length()));
          immutableLabelApplicator.turnOff(reasonToClose);
          inProcessTransformation = null;
        }
      }
      super.visitLdcInsn(value);
    }

    @Override
    public void visitMethodInsn(
        int opcode, String owner, String name, String descriptor, boolean isInterface) {
      MethodInvocationSite verbatimInvocationSite =
          MethodInvocationSite.create(opcode, owner, name, descriptor, isInterface)
              .acceptTypeMapper(IN_PROCESS_LABEL_STRIPPER);

      if (inProcessTransformation != null) {
        super.visitMethodInsn(opcode, owner, name, descriptor, isInterface);
        return;
      }

      if (shouldUseInlineTypeConversion(
          verbatimInvocationSite, typeHierarchy, bootClassPathDigest, enclosingMethod)) {
        logger.atInfo().log(
            "----> Inline Type Conversion performed for %s", verbatimInvocationSite);
        InvocationSiteTransformationReason transformationReason =
            InvocationSiteTransformationReason.create(
                INLINE_PARAM_TYPE_CONVERSION, verbatimInvocationSite.method());
        LangModelHelper.mapOperandStackValues(
            this,
            ImmutableList.of(ShadowedApiAdapterHelper::anyMirroredToBuiltinTypeConversion),
            SHADOWED_TO_MIRRORED_TYPE_MAPPER.map(verbatimInvocationSite.argumentTypeNames()),
            verbatimInvocationSite.argumentTypeNames(),
            /* beginningMarker= */ BEGIN_TAG + transformationReason.encode());

        verbatimInvocationSite.accept(this);

        ClassName verbatimReturnTypeName = verbatimInvocationSite.returnTypeName();
        if (verbatimReturnTypeName.isDesugarShadowedType()) {
          MethodInvocationSite conversion =
              ShadowedApiAdapterHelper.shadowedToMirroredTypeConversionSite(verbatimReturnTypeName);
          conversion.accept(this);
        }
        visitLdcInsn(END_TAG + transformationReason.encode());
        visitInsn(Opcodes.POP);
        invocationSiteRecord.addInlineConversion(verbatimInvocationSite);
        return;
      }

      if (shouldUseApiTypeAdapter(verbatimInvocationSite, bootClassPathDigest)) {
        checkState(!immutableLabelApplicator.isSwitchOn());
        MethodInvocationSite adapterSite =
            ShadowedApiAdapterHelper.getAdapterInvocationSite(verbatimInvocationSite);
        invocationSiteRecord.addAdapterReplacement(verbatimInvocationSite);
        adapterSite.accept(this);
        return;
      }

      super.visitMethodInsn(opcode, owner, name, descriptor, isInterface);
    }
  }

  /** Strips out immutable labels at the end of desugar pipeline. */
  public static class ImmutableLabelRemover extends ClassRemapper {
    public ImmutableLabelRemover(ClassVisitor cv) {
      super(cv, ClassName.IMMUTABLE_LABEL_STRIPPER);
    }
  }
}
