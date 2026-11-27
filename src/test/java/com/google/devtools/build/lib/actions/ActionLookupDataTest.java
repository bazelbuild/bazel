// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.actions;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.skyframe.serialization.AutoRegistry;
import com.google.devtools.build.lib.skyframe.serialization.DeferredObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ActionLookupData}. */
@RunWith(JUnit4.class)
public final class ActionLookupDataTest {

  @Test
  public void generatedCodecs_areDeferredObjectCodecs() throws Exception {
    assertThat(Class.forName(ActionLookupData.class.getName() + "_AutoCodec"))
        .isAssignableTo(DeferredObjectCodec.class);
    assertThat(
            Class.forName(
                ActionLookupData.class.getName() + "_UnshareableActionLookupData_AutoCodec"))
        .isAssignableTo(DeferredObjectCodec.class);
  }

  @Test
  public void shareableAndUnshareable_areDistinct() {
    assertThat(ActionsTestUtil.NULL_ARTIFACT_OWNER.mayOwnShareableActions()).isTrue();

    ActionLookupData shareable = ActionLookupData.create(ActionsTestUtil.NULL_ARTIFACT_OWNER, 1);
    ActionLookupData unshareableOnShareableOwner =
        ActionLookupData.createUnshareable(ActionsTestUtil.NULL_ARTIFACT_OWNER, 1);

    assertThat(shareable.valueIsShareable()).isTrue();
    assertThat(unshareableOnShareableOwner.valueIsShareable()).isFalse();
    assertThat(shareable).isNotEqualTo(unshareableOnShareableOwner);
  }

  @Test
  public void serialization_preservesShareabilityAndFields() throws Exception {
    ActionLookupData shareable = ActionLookupData.create(ActionsTestUtil.NULL_ARTIFACT_OWNER, 1);
    ActionLookupData unshareableOnShareableOwner =
        ActionLookupData.createUnshareable(ActionsTestUtil.NULL_ARTIFACT_OWNER, 1);
    ActionLookupData anotherShareable =
        ActionLookupData.create(ActionsTestUtil.YET_ANOTHER_NULL_ARTIFACT_OWNER, 2);

    ObjectCodecs codecs =
        new ObjectCodecs(AutoRegistry.get().getBuilder().setAllowDefaultCodec(false).build());

    new SerializationTester(shareable, unshareableOnShareableOwner, anotherShareable)
        .setObjectCodecs(codecs)
        .makeMemoizingAndAllowFutureBlocking(/* allowFutureBlocking= */ true)
        .setVerificationFunction((ActionLookupData original, ActionLookupData deserialized) -> {
          assertThat(deserialized).isEqualTo(original);
          assertThat(deserialized.getClass()).isEqualTo(original.getClass());
          assertThat(deserialized.valueIsShareable()).isEqualTo(original.valueIsShareable());
          assertThat(deserialized.getActionLookupKey()).isEqualTo(original.getActionLookupKey());
          assertThat(deserialized.getActionIndex()).isEqualTo(original.getActionIndex());
        })
        .runTests();
  }
}
