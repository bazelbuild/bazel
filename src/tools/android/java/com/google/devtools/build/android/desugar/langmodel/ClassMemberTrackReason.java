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
import com.google.common.collect.ImmutableSet;
import java.util.Collection;
import org.objectweb.asm.Opcodes;

/**
 * Used to track the declaration and invocation information of a class member, including fields,
 * constructors and methods.
 */
@AutoValue
abstract class ClassMemberTrackReason {

  abstract boolean hasDeclReason();

  abstract int ownerAccess();

  abstract int memberAccess();

  abstract ImmutableSet<MemberUseKind> useAccesses();

  public static ClassMemberTrackReasonBuilder builder() {
    return new AutoValue_ClassMemberTrackReason.Builder()
        .setHasDeclReason(false)
        .setOwnerAccess(0)
        .setMemberAccess(0);
  }

  abstract ClassMemberTrackReasonBuilder toBuilder();

  final boolean hasInterfaceDeclReason() {
    return hasDeclReason() && (ownerAccess() & Opcodes.ACC_INTERFACE) != 0;
  }

  final boolean hasMemberUseReason() {
    return !useAccesses().isEmpty();
  }

  /** The builder for {@link ClassMemberTrackReason}. */
  @AutoValue.Builder
  abstract static class ClassMemberTrackReasonBuilder {

    abstract ClassMemberTrackReasonBuilder setHasDeclReason(boolean value);

    abstract ClassMemberTrackReasonBuilder setOwnerAccess(int value);

    abstract ClassMemberTrackReasonBuilder setMemberAccess(int value);

    abstract ClassMemberTrackReasonBuilder setUseAccesses(Collection<MemberUseKind> value);

    abstract ImmutableSet.Builder<MemberUseKind> useAccessesBuilder();

    final ClassMemberTrackReasonBuilder setDeclAccess(int ownerAccess, int memberAccess) {
      return setHasDeclReason(true).setOwnerAccess(ownerAccess).setMemberAccess(memberAccess);
    }

    final ClassMemberTrackReasonBuilder addUseAccess(int invokeOpcode) {
      useAccessesBuilder().add(MemberUseKind.fromValue(invokeOpcode));
      return this;
    }

    final ClassMemberTrackReasonBuilder addAllUseAccesses(Collection<MemberUseKind> values) {
      useAccessesBuilder().addAll(values);
      return this;
    }

    final ClassMemberTrackReasonBuilder mergeFrom(ClassMemberTrackReason otherReason) {
      if (otherReason.hasDeclReason()) {
        setDeclAccess(otherReason.ownerAccess(), otherReason.memberAccess());
      }
      addAllUseAccesses(otherReason.useAccesses());
      return this;
    }

    abstract ClassMemberTrackReason build();
  }
}
