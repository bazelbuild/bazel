// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.serialization;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link NoSuchTargetException} serialization. */
@RunWith(JUnit4.class)
public class NoSuchTargetExceptionCodecTest {
  @Test
  public void smoke() throws Exception {
    new SerializationTester(
            new NoSuchTargetException("sup"),
            new NoSuchTargetException(Label.parseCanonical("//foo:bar"), "busted"),
            new NoSuchTargetException(mockTarget("//broken:target")))
        .makeMemoizing()
        .setVerificationFunction(verifyDeserialization)
        .runTests();
  }

  private static Target mockTarget(String label) throws LabelSyntaxException {
    Target mockTarget = mock(Target.class);
    when(mockTarget.getLabel()).thenReturn(Label.parseCanonical(label));
    return mockTarget;
  }

  private static final SerializationTester.VerificationFunction<NoSuchTargetException>
      verifyDeserialization =
          (deserialized, subject) -> {
            assertThat(deserialized).hasMessageThat().isEqualTo(subject.getMessage());
            assertThat(deserialized.getLabel()).isEqualTo(subject.getLabel());
            assertThat(deserialized.hasTarget()).isEqualTo(subject.hasTarget());
          };
}
