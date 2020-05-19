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
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.common.flogger.GoogleLogger;
import com.google.common.io.Resources;
import com.google.devtools.build.android.desugar.langmodel.ClassName;
import com.google.devtools.build.android.desugar.langmodel.MethodInvocationSite;
import com.google.protobuf.ExtensionRegistry;
import com.google.protobuf.TextFormat;
import java.io.IOError;
import java.io.IOException;
import java.net.URL;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;

/** This class rewrites (or removes) some trivial primitive wrapper methods. */
public class ClassMemberRetargetRewriter extends ClassVisitor {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  /** The configuration file that defines class member retargeting. */
  private final URL invocationReplacementConfigUrl =
      getClass().getResource("retarget_config.textproto");

  /**
   * The enabled configuration flags for invocation replacements. A replacement takes effect if and
   * only if the set intersection of this flag and the {@code range} field of {@code
   * MethodInvocations#replacements} is non-empty.
   */
  private final ImmutableSet<ReplacementRange> enabledInvocationReplacementRanges;

  /**
   * A collector that gathers runtime library classes referenced by the user and generate code. The
   * desugar tool copies these classes into output.
   */
  private final ImmutableSet.Builder<ClassName> requiredRuntimeSupportTypes;

  /**
   * The parsed invocation replacement configuration from {@link #invocationReplacementConfigUrl}.
   */
  private ImmutableMap<MethodInvocationSite, MethodInvocationSite> invocationReplacements;

  /**
   * The parsed effective range flags from {@link #invocationReplacementConfigUrl} for each
   * invocation source.
   */
  private ImmutableSetMultimap<MethodInvocationSite, ReplacementRange>
      invocationReplacementRangeConfigs;

  public ClassMemberRetargetRewriter(
      ClassVisitor cv,
      ImmutableSet<ReplacementRange> enabledInvocationReplacementRanges,
      ImmutableSet.Builder<ClassName> requiredRuntimeSupportTypes) {
    super(Opcodes.ASM8, cv);
    this.enabledInvocationReplacementRanges = enabledInvocationReplacementRanges;
    this.requiredRuntimeSupportTypes = requiredRuntimeSupportTypes;
  }

  @Override
  public void visit(
      int version,
      int access,
      String name,
      String signature,
      String superName,
      String[] interfaces) {
    try {
      parseMethodInvocationReplacements();
    } catch (IOException e) {
      throw new IOError(e);
    }
    super.visit(version, access, name, signature, superName, interfaces);
  }

  private void parseMethodInvocationReplacements() throws IOException {
    String protoText = Resources.toString(invocationReplacementConfigUrl, UTF_8);
    MethodInvocations parsedInvocations =
        TextFormat.parse(protoText, ExtensionRegistry.getEmptyRegistry(), MethodInvocations.class);
    ImmutableMap.Builder<MethodInvocationSite, MethodInvocationSite> replacementsBuilder =
        ImmutableMap.builder();
    ImmutableSetMultimap.Builder<MethodInvocationSite, ReplacementRange> replacementsRangeBuilder =
        ImmutableSetMultimap.builder();
    for (MethodInvocationReplacement replacement : parsedInvocations.getReplacementsList()) {
      MethodInvocationSite invocationSite = MethodInvocationSite.fromProto(replacement.getSource());
      replacementsBuilder.put(
          invocationSite, MethodInvocationSite.fromProto(replacement.getDestination()));
      replacementsRangeBuilder.putAll(invocationSite, replacement.getRangeList());
    }
    invocationReplacements = replacementsBuilder.build();
    invocationReplacementRangeConfigs = replacementsRangeBuilder.build();
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

      ImmutableSet<ReplacementRange> replacementRangeConfig =
          invocationReplacementRangeConfigs.get(verbatimInvocationSite);
      if (replacementRangeConfig.contains(ReplacementRange.ALL)
          || replacementRangeConfig.stream()
              .anyMatch(enabledInvocationReplacementRanges::contains)) {
        MethodInvocationSite replacementSite = invocationReplacements.get(verbatimInvocationSite);
        logger.atInfo().log("replacementSite: %s", replacementSite);
        if (replacementSite.name().equals("identityAsHashCode")) {
          return; // skip: use original primitive value as its hash (b/147139686)
        }
        requiredRuntimeSupportTypes.add(replacementSite.owner());
        replacementSite.accept(this); // (b/147139686)
        return;
      }

      super.visitMethodInsn(opcode, owner, name, desc, isInterface);
    }
  }
}
