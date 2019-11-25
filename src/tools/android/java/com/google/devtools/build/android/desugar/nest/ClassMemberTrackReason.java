// Copyright 2019 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.android.desugar.nest;

import static com.google.common.collect.ImmutableMap.toImmutableMap;

import com.google.common.collect.ImmutableMap;
import com.google.common.hash.Hashing;
import java.util.Arrays;
import java.util.EnumSet;
import org.objectweb.asm.Opcodes;

/**
 * Used to track the declaration and invocation information of a class member, including fields,
 * constructors and methods.
 */
final class ClassMemberTrackReason {

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

  public enum MemberUseKind {
    H_GETFIELD(Opcodes.H_GETFIELD),
    H_GETSTATIC(Opcodes.H_GETSTATIC),
    H_PUTFIELD(Opcodes.H_PUTFIELD),
    H_PUTSTATIC(Opcodes.H_PUTSTATIC),
    H_INVOKEVIRTUAL(Opcodes.H_INVOKEVIRTUAL),
    H_INVOKESTATIC(Opcodes.H_INVOKESTATIC),
    H_INVOKESPECIAL(Opcodes.H_INVOKESPECIAL),
    H_NEWINVOKESPECIAL(Opcodes.H_NEWINVOKESPECIAL),
    H_INVOKEINTERFACE(Opcodes.H_INVOKEINTERFACE),

    GETSTATIC(Opcodes.GETSTATIC),
    PUTSTATIC(Opcodes.PUTSTATIC),
    GETFIELD(Opcodes.GETFIELD),
    PUTFIELD(Opcodes.PUTFIELD),
    INVOKEVIRTUAL(Opcodes.INVOKEVIRTUAL),
    INVOKESPECIAL(Opcodes.INVOKESPECIAL),
    INVOKESTATIC(Opcodes.INVOKESTATIC),
    INVOKEINTERFACE(Opcodes.INVOKEINTERFACE),
    INVOKEDYNAMIC(Opcodes.INVOKEDYNAMIC);

    private static final ImmutableMap<Integer, MemberUseKind> VALUE_TO_MEMBER_USE_KIND_MAP =
        Arrays.stream(MemberUseKind.values())
            .collect(toImmutableMap(kind -> kind.opcode, kind -> kind));

    private final int opcode;

    MemberUseKind(int opcode) {
      this.opcode = opcode;
    }

    static MemberUseKind fromValue(int opcode) {
      return VALUE_TO_MEMBER_USE_KIND_MAP.get(opcode);
    }

    public int getOpcode() {
      return opcode;
    }
  }
}
