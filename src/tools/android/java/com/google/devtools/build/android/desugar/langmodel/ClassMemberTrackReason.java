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

import com.google.common.hash.Hashing;
import java.util.EnumSet;
import org.objectweb.asm.Opcodes;

/**
 * Used to track the declaration and invocation information of a class member, including fields,
 * constructors and methods.
 */
public final class ClassMemberTrackReason {

  private boolean hasDeclReason;
  private int ownerAccess;
  private int memberAccess;
  private final EnumSet<MemberUseKind> useAccesses = EnumSet.noneOf(MemberUseKind.class);

  ClassMemberTrackReason setDeclAccess(int ownerAccess, int memberAccess) {
    this.ownerAccess = ownerAccess;
    this.memberAccess = memberAccess;
    this.hasDeclReason = true;
    return this;
  }

  ClassMemberTrackReason addUseAccess(int invokeOpcode) {
    this.useAccesses.add(MemberUseKind.fromValue(invokeOpcode));
    return this;
  }

  boolean hasDeclReason() {
    return hasDeclReason;
  }

  boolean hasInterfaceDeclReason() {
    return hasDeclReason && (ownerAccess & Opcodes.ACC_INTERFACE) != 0;
  }

  boolean hasMemberUseReason() {
    return !useAccesses.isEmpty();
  }

  int getOwnerAccess() {
    return ownerAccess;
  }

  int getMemberAccess() {
    return memberAccess;
  }

  EnumSet<MemberUseKind> getUseAccesses() {
    return useAccesses;
  }

  ClassMemberTrackReason mergeFrom(ClassMemberTrackReason otherClassMemberTrackReason) {
    if (!hasDeclReason() && otherClassMemberTrackReason.hasDeclReason()) {
      ownerAccess = otherClassMemberTrackReason.getOwnerAccess();
      memberAccess = otherClassMemberTrackReason.getMemberAccess();
      hasDeclReason = true;
    }
    useAccesses.addAll(otherClassMemberTrackReason.getUseAccesses());
    return this;
  }

  @Override
  public int hashCode() {
    return Hashing.sha256()
        .newHasher()
        .putBoolean(hasDeclReason)
        .putInt(ownerAccess)
        .putInt(memberAccess)
        .putInt(useAccesses.hashCode())
        .hash()
        .asInt();
  }

  @Override
  public boolean equals(Object obj) {
    if (obj == this) {
      return true;
    }
    if (obj instanceof ClassMemberTrackReason) {
      ClassMemberTrackReason other = (ClassMemberTrackReason) obj;
      return this.hasDeclReason == other.hasDeclReason
          && this.ownerAccess == other.ownerAccess
          && this.memberAccess == other.memberAccess
          && this.useAccesses.equals(other.useAccesses);
    }
    return false;
  }

  @Override
  public String toString() {
    return String.format(
        "%s{hasDeclReason=%s, ownerAccess=%d, memberAccess=%d, useAccesses=%s}",
        getClass().getSimpleName(), hasDeclReason, ownerAccess, memberAccess, useAccesses);
  }
}
