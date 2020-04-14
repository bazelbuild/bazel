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
import com.google.devtools.build.android.desugar.corelibadapter.InvocationSiteTransformationRecord.InvocationSiteTransformationRecordBuilder;
import com.google.devtools.build.android.desugar.langmodel.ClassName;
import com.google.devtools.build.android.desugar.langmodel.LangModelHelper;
import com.google.devtools.build.android.desugar.langmodel.MethodInvocationSite;
import com.google.devtools.build.android.desugar.langmodel.SwitchableTypeMapper;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.commons.ClassRemapper;
import org.objectweb.asm.commons.MethodRemapper;

/**
 * Desugars the bytecode instructions that interacts with desugar-shadowed APIs, which is a method
 * with desugar-shadowed types, e.g. {@code java.time.MonthDay}.
 */
public final class ShadowedApiInvocationSite extends ClassVisitor {

  /** An evolving record that collects the adapter method requests from invocation sites. */
  private final InvocationSiteTransformationRecordBuilder invocationSiteRecord;

  public ShadowedApiInvocationSite(
      ClassVisitor classVisitor, InvocationSiteTransformationRecordBuilder invocationSiteRecord) {
    super(Opcodes.ASM7, classVisitor);
    this.invocationSiteRecord = invocationSiteRecord;
  }

  @Override
  public MethodVisitor visitMethod(
      int access, String name, String descriptor, String signature, String[] exceptions) {
    MethodVisitor mv = super.visitMethod(access, name, descriptor, signature, exceptions);
    return mv == null
        ? null
        : ShadowedApiInvocationSiteMethodVisitor.create(api, mv, invocationSiteRecord);
  }

  /** Desugars instructions for the enclosing class visitor. */
  private static class ShadowedApiInvocationSiteMethodVisitor extends MethodRemapper {

    private final SwitchableTypeMapper<InvocationSiteTransformationReason> immutableLabelApplicator;

    private final InvocationSiteTransformationRecordBuilder invocationSiteRecord;

    /**
     * The transformation-in-process invocation site while visiting. The value can be {@code null}
     * if no in-line type conversion is in process.
     */
    private InvocationSiteTransformationReason inProcessTransformation;

    private ShadowedApiInvocationSiteMethodVisitor(
        int api,
        MethodVisitor methodVisitor,
        InvocationSiteTransformationRecordBuilder invocationSiteRecord,
        SwitchableTypeMapper<InvocationSiteTransformationReason> immutableLabelApplicator) {
      super(api, methodVisitor, immutableLabelApplicator);
      this.invocationSiteRecord = invocationSiteRecord;
      this.immutableLabelApplicator = immutableLabelApplicator;
    }

    static ShadowedApiInvocationSiteMethodVisitor create(
        int api,
        MethodVisitor methodVisitor,
        InvocationSiteTransformationRecordBuilder invocationSiteRecord) {
      return new ShadowedApiInvocationSiteMethodVisitor(
          api,
          methodVisitor,
          invocationSiteRecord,
          new SwitchableTypeMapper<>(IN_PROCESS_LABEL_STRIPPER.andThen(IMMUTABLE_LABEL_ATTACHER)));
    }

    @Override
    public void visitLdcInsn(Object value) {
      if (value instanceof String
          && ((String) value).startsWith(INLINE_PARAM_TYPE_CONVERSION.toString())) {
        String paramTypeConversionTag = (String) value;
        checkState(inProcessTransformation == null);
        inProcessTransformation = InvocationSiteTransformationReason.decode(paramTypeConversionTag);
        immutableLabelApplicator.turnOn(inProcessTransformation);
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
        if (verbatimInvocationSite.method().equals(inProcessTransformation.method())) {
          immutableLabelApplicator.turnOff(inProcessTransformation);
          inProcessTransformation = null;
        }
        return;
      }

      if (shouldUseInlineTypeConversion(verbatimInvocationSite)) {
        InvocationSiteTransformationReason transformationReason =
            InvocationSiteTransformationReason.create(
                INLINE_PARAM_TYPE_CONVERSION, verbatimInvocationSite.method());
        LangModelHelper.mapOperandStackValues(
            this,
            ImmutableList.of(ShadowedApiAdapterHelper::anyMirroredToBuiltinTypeConversion),
            SHADOWED_TO_MIRRORED_TYPE_MAPPER.map(verbatimInvocationSite.argumentTypeNames()),
            verbatimInvocationSite.argumentTypeNames(),
            /* beginningMarker= */ transformationReason.encode());
        invocationSiteRecord.addInlineConversion(verbatimInvocationSite);
        verbatimInvocationSite.accept(this);
        return;
      }

      if (shouldUseApiTypeAdapter(verbatimInvocationSite)) {
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
