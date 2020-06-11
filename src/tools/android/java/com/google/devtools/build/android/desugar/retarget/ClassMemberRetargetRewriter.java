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
package com.google.devtools.build.android.desugar.retarget;

import static com.google.devtools.build.android.desugar.langmodel.ClassName.IN_PROCESS_LABEL_STRIPPER;

import com.google.common.collect.ImmutableSet;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.android.desugar.langmodel.ClassName;
import com.google.devtools.build.android.desugar.langmodel.MemberUseKind;
import com.google.devtools.build.android.desugar.langmodel.MethodInvocationSite;
import com.google.devtools.build.android.desugar.langmodel.MethodKey;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;

/** This class rewrites (or removes) some trivial primitive wrapper methods. */
public class ClassMemberRetargetRewriter extends ClassVisitor {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  /** The configuration for invocation retargeting. */
  private final ClassMemberRetargetConfig classMemberRetargetConfig;

  /**
   * A collector that gathers runtime library classes referenced by the user and generate code. The
   * desugar tool copies these classes into output.
   */
  private final ImmutableSet.Builder<ClassName> requiredRuntimeSupportTypes;

  public ClassMemberRetargetRewriter(
      ClassVisitor cv,
      ClassMemberRetargetConfig classMemberRetargetConfig,
      ImmutableSet.Builder<ClassName> requiredRuntimeSupportTypes) {
    super(Opcodes.ASM8, cv);
    this.classMemberRetargetConfig = classMemberRetargetConfig;
    this.requiredRuntimeSupportTypes = requiredRuntimeSupportTypes;
  }

  @Override
  public MethodVisitor visitMethod(
      int access, String name, String desc, String signature, String[] exceptions) {
    MethodVisitor visitor = super.visitMethod(access, name, desc, signature, exceptions);
    return visitor == null ? null : new ClassMemberRetargetMethodVisitor(visitor);
  }

  private class ClassMemberRetargetMethodVisitor extends MethodVisitor {

    ClassMemberRetargetMethodVisitor(MethodVisitor visitor) {
      super(Opcodes.ASM8, visitor);
    }

    @Override
    public void visitMethodInsn(
        int opcode, String owner, String name, String desc, boolean isInterface) {
      MethodInvocationSite verbatimInvocationSite =
          MethodInvocationSite.create(opcode, owner, name, desc, isInterface)
              .acceptTypeMapper(IN_PROCESS_LABEL_STRIPPER);
      if (replaceExactMatchedInvocationSites(verbatimInvocationSite)) {
        return;
      }
      if (replacePatternMatchedInvocationSites(verbatimInvocationSite)) {
        return;
      }
      super.visitMethodInsn(opcode, owner, name, desc, isInterface);
    }

    private boolean replaceExactMatchedInvocationSites(
        MethodInvocationSite verbatimInvocationSite) {
      MethodInvocationSite replacementSite =
          classMemberRetargetConfig.findReplacementSite(verbatimInvocationSite);
      if (replacementSite != null) {
        if (replacementSite.name().equals("identityAsHashCode")) {
          return true;
        }
        ClassName successor = replacementSite.owner();
        if (successor.isInDesugarRuntimeLibrary()) {
          requiredRuntimeSupportTypes.add(successor);
        }
        replacementSite.accept(mv); // (b/147139686)
        return true;
      }
      return false;
    }

    private boolean replacePatternMatchedInvocationSites(
        MethodInvocationSite verbatimInvocationSite) {
      MethodInvocationSite invocationSiteOwnerAndNameRepresentative =
          MethodInvocationSite.builder()
              .setInvocationKind(MemberUseKind.UNKNOWN)
              .setMethod(
                  MethodKey.create(
                      verbatimInvocationSite.owner(), verbatimInvocationSite.name(), ""))
              .setIsInterface(false)
              .build();
      MethodInvocationSite replacementSiteOwnerAndNameRepresentative =
          classMemberRetargetConfig.findReplacementSite(invocationSiteOwnerAndNameRepresentative);
      if (replacementSiteOwnerAndNameRepresentative != null) {
        // The invocation site successor under pattern-matching mode is a static invocation.
        MethodInvocationSite replacementSite =
            MethodInvocationSite.builder()
                .setInvocationKind(MemberUseKind.INVOKESTATIC)
                .setMethod(
                    MethodKey.create(
                        replacementSiteOwnerAndNameRepresentative.owner(),
                        replacementSiteOwnerAndNameRepresentative.name(),
                        verbatimInvocationSite.isStaticInvocation()
                            ? verbatimInvocationSite.descriptor()
                            : verbatimInvocationSite.method().instanceMethodToStaticDescriptor()))
                .setIsInterface(false)
                .build();

        ClassName successor = replacementSiteOwnerAndNameRepresentative.owner();
        logger.atInfo().log("--> Representative-based replacementSite: %s", replacementSite);
        if (successor.isInDesugarRuntimeLibrary()) {
          requiredRuntimeSupportTypes.add(successor);
        }
        replacementSite.acceptTypeMapper(ClassName.SHADOWED_TO_MIRRORED_TYPE_MAPPER).accept(mv);
        return true;
      }
      return false;
    }
  }
}
