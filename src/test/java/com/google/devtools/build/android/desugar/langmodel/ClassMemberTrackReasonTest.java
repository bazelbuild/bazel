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

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.objectweb.asm.Opcodes;

/** Tests for {@link ClassMemberTrackReason}. */
@RunWith(JUnit4.class)
public class ClassMemberTrackReasonTest {

  private static final ImmutableList<Integer> MEMBER_USE_OPCODES =
      ImmutableList.of(
          0, // default unknown value.
          Opcodes.H_GETFIELD,
          Opcodes.H_GETSTATIC,
          Opcodes.H_PUTFIELD,
          Opcodes.H_PUTSTATIC,
          Opcodes.H_INVOKEVIRTUAL,
          Opcodes.H_INVOKESTATIC,
          Opcodes.H_INVOKESPECIAL,
          Opcodes.H_NEWINVOKESPECIAL,
          Opcodes.H_INVOKEINTERFACE,
          Opcodes.GETSTATIC,
          Opcodes.PUTSTATIC,
          Opcodes.GETFIELD,
          Opcodes.PUTFIELD,
          Opcodes.INVOKEVIRTUAL,
          Opcodes.INVOKESPECIAL,
          Opcodes.INVOKESTATIC,
          Opcodes.INVOKEINTERFACE,
          Opcodes.INVOKEDYNAMIC);

  @Test
  public void memberUseKind_toOpcodeValues() {
    List<Integer> backingOpcodes =
        Arrays.stream(MemberUseKind.values())
            .map(MemberUseKind::getOpcode)
            .collect(Collectors.toList());
    assertThat(backingOpcodes).containsExactlyElementsIn(MEMBER_USE_OPCODES).inOrder();
  }

  @Test
  public void memberUseKind_fromOpcodeValues() {
    List<MemberUseKind> allParsedMemberUseKinds =
        MEMBER_USE_OPCODES.stream().map(MemberUseKind::fromValue).collect(Collectors.toList());
    assertThat(allParsedMemberUseKinds).containsExactlyElementsIn(MemberUseKind.values()).inOrder();
  }

  @Test
  public void memberUseKind_invertibleMapping() {
    assertThat(
            MEMBER_USE_OPCODES.stream()
                .map(MemberUseKind::fromValue)
                .map(MemberUseKind::getOpcode)
                .collect(Collectors.toList()))
        .containsExactlyElementsIn(MEMBER_USE_OPCODES)
        .inOrder();
  }
}
