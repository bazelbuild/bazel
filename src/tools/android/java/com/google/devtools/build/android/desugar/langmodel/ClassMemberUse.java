/*
 * Copyright 2019 The Bazel Authors. All rights reserved.
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

/**
 * Identifies the way a class member (field, method) is used, including method invocation and field
 * access.
 */
@AutoValue
public abstract class ClassMemberUse implements TypeMappable<ClassMemberUse> {

  public abstract ClassMemberKey<?> method();

  public abstract MemberUseKind useKind();

  public static ClassMemberUse create(ClassMemberKey<?> memberKey, MemberUseKind memberUseKind) {
    return new AutoValue_ClassMemberUse(memberKey, memberUseKind);
  }

  // Performs the current member use on the given class visitor.
  public final void acceptClassMethodInsn(MethodVisitor mv) {
    ClassMemberKey<?> method = method();
    mv.visitMethodInsn(
        useKind().getOpcode(),
        method.ownerName(),
        method.name(),
        method.descriptor(),
        /* isInterface= */ false);
  }

  @Override
  public ClassMemberUse acceptTypeMapper(TypeMapper typeMapper) {
    return create(method().acceptTypeMapper(typeMapper), useKind());
  }
}
