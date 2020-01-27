// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.importdeps;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import java.nio.file.Paths;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for {@link ResolutionFailureChain} */
@RunWith(JUnit4.class)
public class ResolutionFailureChainTest {

  private final ClassInfo charSequenceClass =
      ClassInfo.create(
          "java/lang/CharSequence",
          Paths.get("string.jar"),
          false,
          ImmutableList.of(),
          ImmutableSet.of());
  private final ResolutionFailureChain objectMissingChain =
      ResolutionFailureChain.createMissingClass("java/lang/Object");
  private final ResolutionFailureChain charSequenceFailureChain =
      ResolutionFailureChain.createWithParent(
          charSequenceClass, ImmutableList.of(objectMissingChain));

  @Test
  public void testArgumentCheck() {
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> ResolutionFailureChain.createWithParent(charSequenceClass, ImmutableList.of()));
  }

  @Test
  public void testMissingClassChain() {
    assertThat(objectMissingChain.missingClasses()).containsExactly("java/lang/Object");
    assertThat(objectMissingChain.getMissingClassesWithSubclasses()).isEmpty();
    assertThat(objectMissingChain.parentChains()).isEmpty();
    assertThat(objectMissingChain.resolutionStartClass()).isNull();
  }

  @Test
  public void testChainWithOneHead() {
    assertThat(charSequenceFailureChain.missingClasses()).containsExactly("java/lang/Object");
    assertThat(charSequenceFailureChain.resolutionStartClass()).isEqualTo(charSequenceClass);
    assertThat(charSequenceFailureChain.getMissingClassesWithSubclasses().values())
        .containsExactly(charSequenceClass);
    assertThat(charSequenceFailureChain.parentChains()).containsExactly(objectMissingChain);
  }

  @Test
  public void testChainWithTwoHeads() {
    ClassInfo stringClass =
        ClassInfo.create(
            "java/lang/String",
            Paths.get("string.jar"),
            false,
            ImmutableList.of(),
            ImmutableSet.of());
    ResolutionFailureChain chain =
        ResolutionFailureChain.createWithParent(
            stringClass, ImmutableList.of(objectMissingChain, charSequenceFailureChain));
    assertThat(chain.missingClasses()).containsExactly("java/lang/Object");
    assertThat(chain.parentChains()).containsExactly(objectMissingChain, charSequenceFailureChain);
    assertThat(chain.getMissingClassesWithSubclasses().values())
        .containsExactly(stringClass, charSequenceClass);
    assertThat(chain.resolutionStartClass()).isEqualTo(stringClass);
  }
}
