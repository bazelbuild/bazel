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
import org.objectweb.asm.MethodVisitor;

/** A value object that represents an method invocation site. */
@AutoValue
public abstract class MethodInvocationSite implements TypeMappable<MethodInvocationSite> {

  public abstract MemberUseKind invocationKind();

  public abstract MethodKey method();

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

  public final boolean isStaticInvocation() {
    return invocationKind() == MemberUseKind.INVOKESTATIC;
  }

  public final boolean isConstructorInvocation() {
    return method().isConstructor();
  }

  public final MethodVisitor accept(MethodVisitor mv) {
    mv.visitMethodInsn(invokeOpcode(), owner().binaryName(), name(), descriptor(), isInterface());
    return mv;
  }

  @Override
  public MethodInvocationSite acceptTypeMapper(TypeMapper typeMapper) {
    return toBuilder().setMethod(method().acceptTypeMapper(typeMapper)).build();
  }

  public final MethodInvocationSite toAdapterInvocationSite() {
    return MethodInvocationSite.builder()
        .setInvocationKind(MemberUseKind.INVOKESTATIC)
        .setMethod(method().toArgumentTypeAdapter(isStaticInvocation()))
        .setIsInterface(false)
        .build();
  }

  /** The builder for {@link MethodInvocationSite}. */
  @AutoValue.Builder
  public abstract static class MethodInvocationSiteBuilder {

    public abstract MethodInvocationSiteBuilder setInvocationKind(MemberUseKind value);

    public abstract MethodInvocationSiteBuilder setMethod(MethodKey value);

    public abstract MethodInvocationSiteBuilder setIsInterface(boolean value);

    public abstract MethodInvocationSite build();
  }
}
