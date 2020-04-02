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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;
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
import com.google.devtools.build.android.desugar.langmodel.MethodKey;
import com.google.devtools.build.android.desugar.langmodel.SwitchableTypeMapper;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.commons.ClassRemapper;
import org.objectweb.asm.commons.MethodRemapper;
import org.objectweb.asm.tree.MethodNode;

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
        : AndroidJdkLibInvocationSiteMethodVisitor.create(api, mv, invocationSiteRecord);
  }

  /** Desugars instructions for the enclosing class visitor. */
  private static class AndroidJdkLibInvocationSiteMethodVisitor extends MethodRemapper {

    /** The tag included in the marker instruction. */
    private static final String INLINE_PARAM_TYPE_CONVERSION_TAG = "__desugar_param_conversion__:";

    private final SwitchableTypeMapper immutableLabelApplicator;

    private final InvocationSiteTransformationRecordBuilder invocationSiteRecord;

    /**
     * The served method under parameter type conversion while visiting. The value can be {@code
     * null} if no in-line type conversion is in process.
     */
    private MethodKey methodInParamInlineTypeConversionService;

    private AndroidJdkLibInvocationSiteMethodVisitor(
        int api,
        MethodVisitor methodVisitor,
        InvocationSiteTransformationRecordBuilder invocationSiteRecord,
        SwitchableTypeMapper immutableLabelApplicator) {
      super(api, methodVisitor, immutableLabelApplicator);
      this.invocationSiteRecord = invocationSiteRecord;
      this.immutableLabelApplicator = immutableLabelApplicator;
    }

    static AndroidJdkLibInvocationSiteMethodVisitor create(
        int api,
        MethodVisitor methodVisitor,
        InvocationSiteTransformationRecordBuilder invocationSiteRecord) {
      return new AndroidJdkLibInvocationSiteMethodVisitor(
          api,
          methodVisitor,
          invocationSiteRecord,
          new SwitchableTypeMapper(IN_PROCESS_LABEL_STRIPPER.andThen(IMMUTABLE_LABEL_ATTACHER)));
    }

    @Override
    public void visitLdcInsn(Object value) {
      if (value instanceof String
          && ((String) value).startsWith(INLINE_PARAM_TYPE_CONVERSION_TAG)) {
        String paramTypeConversionTag = (String) value;
        visitParamInlineTypeConversionStart(
            MethodKey.decode(
                paramTypeConversionTag.substring(INLINE_PARAM_TYPE_CONVERSION_TAG.length())));
      }
      super.visitLdcInsn(value);
    }

    @Override
    public void visitMethodInsn(
        int opcode, String owner, String name, String descriptor, boolean isInterface) {
      MethodInvocationSite verbatimInvocationSite =
          MethodInvocationSite.create(opcode, owner, name, descriptor, isInterface)
              .acceptTypeMapper(IN_PROCESS_LABEL_STRIPPER);

      if (isInParamInlineTypeConversion()) {
        super.visitMethodInsn(opcode, owner, name, descriptor, isInterface);
        if (verbatimInvocationSite.method().equals(methodInParamInlineTypeConversionService)) {
          visitParamInlineTypeConversionEnd(methodInParamInlineTypeConversionService);
        }
        return;
      }

      if (shouldUseInlineTypeConversion(verbatimInvocationSite)) {
        MethodNode paramInlineInstructionsContainer = new MethodNode();
        LangModelHelper.mapOperandStackValues(
            paramInlineInstructionsContainer,
            ImmutableList.of(ShadowedApiAdapterHelper::anyMirroredToBuiltinTypeConversion),
            SHADOWED_TO_MIRRORED_TYPE_MAPPER.map(verbatimInvocationSite.argumentTypeNames()),
            verbatimInvocationSite.argumentTypeNames(),
            INLINE_PARAM_TYPE_CONVERSION_TAG + verbatimInvocationSite.method().encode());
        invocationSiteRecord.addInlineConversion(verbatimInvocationSite);
        verbatimInvocationSite.accept(paramInlineInstructionsContainer);

        MethodRemapper methodRemapper = new MethodRemapper(mv, IMMUTABLE_LABEL_ATTACHER);
        paramInlineInstructionsContainer.accept(methodRemapper);
        return;
      }

      if (shouldUseApiTypeAdapter(verbatimInvocationSite)) {
        checkState(!immutableLabelApplicator.isSwitchOn());
        MethodInvocationSite adapterSite =
            ShadowedApiAdapterHelper.getAdapterInvocationSite(verbatimInvocationSite);
        invocationSiteRecord.addAdapterReplacement(verbatimInvocationSite);
        adapterSite.acceptTypeMapper(IMMUTABLE_LABEL_ATTACHER).accept(mv);
        return;
      }

      super.visitMethodInsn(opcode, owner, name, descriptor, isInterface);
    }

    private boolean isInParamInlineTypeConversion() {
      boolean isInParamInlineTypeConversion = methodInParamInlineTypeConversionService != null;
      if (isInParamInlineTypeConversion) {
        checkState(
            immutableLabelApplicator.isSwitchOn(),
            "Inconsistent state: Expected immutable label applicator is turned ON");
      } else {
        checkState(
            !immutableLabelApplicator.isSwitchOn(),
            "Inconsistent state: Expected immutable label applicator is turned OFF");
      }
      return isInParamInlineTypeConversion;
    }

    private void visitParamInlineTypeConversionStart(MethodKey method) {
      checkState(!isInParamInlineTypeConversion());
      methodInParamInlineTypeConversionService = checkNotNull(method);
      immutableLabelApplicator.turnOn();
    }

    private void visitParamInlineTypeConversionEnd(MethodKey method) {
      checkState(isInParamInlineTypeConversion());
      checkState(
          method.equals(methodInParamInlineTypeConversionService),
          "Inconsistent state: Expected to end %s, but % is in service.",
          method,
          methodInParamInlineTypeConversionService);
      methodInParamInlineTypeConversionService = null;
      immutableLabelApplicator.turnOff();
    }
  }

  /** Strips out immutable labels at the end of desugar pipeline. */
  public static class ImmutableLabelRemover extends ClassRemapper {
    public ImmutableLabelRemover(ClassVisitor cv) {
      super(cv, ClassName.IMMUTABLE_LABEL_STRIPPER);
    }
  }
}
