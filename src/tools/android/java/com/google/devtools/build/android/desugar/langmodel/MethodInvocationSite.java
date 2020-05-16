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

package com.google.devtools.build.android.desugar.langmodel;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.android.desugar.langmodel.MethodInvocationSite.MethodInvocationSiteBuilder;
import org.objectweb.asm.MethodVisitor;

/** A value object that represents an method invocation site. */
@AutoValue
public abstract class MethodInvocationSite
    extends ClassMemberUse<MethodKey, MethodInvocationSiteBuilder, MethodInvocationSite> {

  public final MemberUseKind invocationKind() {
    return useKind();
  }

  public final MethodKey method() {
    return member();
  }

  // TODO(deltazulu): remove once bazel has been updated to the most recent autovalue library.
  @Override
  public abstract MethodKey member();

  public abstract boolean isInterface();

  public static MethodInvocationSiteBuilder builder() {
    return new AutoValue_MethodInvocationSite.Builder();
  }

  /** Convenient factory method for use in the callback of {@link MethodVisitor#visitMethodInsn}. */
  public static MethodInvocationSite create(
      int opcode, String owner, String name, String descriptor, boolean isInterface) {
    return builder()
        .setInvocationKind(MemberUseKind.fromValue(opcode))
        .setMethod(MethodKey.create(ClassName.create(owner), name, descriptor))
        .setIsInterface(isInterface)
        .build();
  }

  public abstract MethodInvocationSiteBuilder toBuilder();

  public final int invokeOpcode() {
    return invocationKind().getOpcode();
  }

  public final ClassName owner() {
    return method().owner();
  }

  public final String name() {
    return method().name();
  }

  public final String descriptor() {
    return method().descriptor();
  }

  public final ClassName returnTypeName() {
    return method().getReturnTypeName();
  }

  public final ImmutableList<ClassName> argumentTypeNames() {
    return method().getArgumentTypeNames();
  }

  public final boolean isStaticInvocation() {
    return invocationKind() == MemberUseKind.INVOKESTATIC;
  }

  public final boolean isConstructorInvocation() {
    return method().isConstructor();
  }

  @Override
  public final MethodInvocationSite acceptTypeMapper(TypeMapper typeMapper) {
    return toBuilder()
        .setMethod(method().acceptTypeMapper(typeMapper))
        .setInvocationKind(invocationKind())
        .setIsInterface(isInterface())
        .build();
  }

  public final MethodVisitor accept(MethodVisitor mv) {
    mv.visitMethodInsn(invokeOpcode(), owner().binaryName(), name(), descriptor(), isInterface());
    return mv;
  }

  /** The builder for {@link MethodInvocationSite}. */
  @AutoValue.Builder
  public abstract static class MethodInvocationSiteBuilder
      extends ClassMemberUseBuilder<MethodKey, MethodInvocationSiteBuilder, MethodInvocationSite> {

    public final MethodInvocationSiteBuilder setInvocationKind(MemberUseKind value) {
      return setUseKind(value);
    }

    public final MethodInvocationSiteBuilder setMethod(MethodKey value) {
      return setMember(value);
    }

    // TODO(deltazulu): remove once bazel has been updated to the most recent autovalue library.
    @Override
    abstract MethodInvocationSiteBuilder setMember(MethodKey value);

    public abstract MethodInvocationSiteBuilder setIsInterface(boolean value);

    public abstract MethodInvocationSite build();
  }
}
