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

import static com.google.common.collect.ImmutableMap.toImmutableMap;

import com.google.common.collect.ImmutableMap;
import java.util.Arrays;
import org.objectweb.asm.Opcodes;

/** The categorized usages of class members. */
public enum MemberUseKind {
  UNKNOWN(0),

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

  public static MemberUseKind fromValue(int opcode) {
    return VALUE_TO_MEMBER_USE_KIND_MAP.get(opcode);
  }

  public int getOpcode() {
    return opcode;
  }
}
