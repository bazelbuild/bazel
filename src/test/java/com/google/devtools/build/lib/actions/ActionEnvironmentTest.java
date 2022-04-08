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

package com.google.devtools.build.lib.actions;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import java.util.HashMap;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** {@link ActionEnvironment}Test */
@RunWith(JUnit4.class)
public final class ActionEnvironmentTest {

  @Test
  public void compoundEnvOrdering() {
    ActionEnvironment env1 =
        ActionEnvironment.create(
            ImmutableMap.of("FOO", "foo1", "BAR", "bar"), ImmutableSet.of("baz"));
    // entries added by env2 override the existing entries
    ActionEnvironment env2 = env1.withAdditionalFixedVariables(ImmutableMap.of("FOO", "foo2"));

    assertThat(env1.getFixedEnv()).containsExactly("FOO", "foo1", "BAR", "bar");
    assertThat(env1.getInheritedEnv()).containsExactly("baz");

    assertThat(env2.getFixedEnv()).containsExactly("FOO", "foo2", "BAR", "bar");
    assertThat(env2.getInheritedEnv()).containsExactly("baz");
  }

  @Test
  public void fixedInheritedInteraction() {
    ActionEnvironment env =
        ActionEnvironment.create(
                ImmutableMap.of("FIXED_ONLY", "fixed"), ImmutableSet.of("INHERITED_ONLY"))
            .withAdditionalVariables(
                ImmutableMap.of("FIXED_AND_INHERITED", "fixed"),
                ImmutableSet.of("FIXED_AND_INHERITED"));
    Map<String, String> clientEnv =
        ImmutableMap.of("INHERITED_ONLY", "inherited", "FIXED_AND_INHERITED", "inherited");
    Map<String, String> result = new HashMap<>();
    env.resolve(result, clientEnv);

    assertThat(result)
        .containsExactly(
            "FIXED_ONLY",
            "fixed",
            "FIXED_AND_INHERITED",
            "inherited",
            "INHERITED_ONLY",
            "inherited");
  }

  @Test
  public void emptyEnvironmentInterning() {
    ActionEnvironment emptyEnvironment =
        ActionEnvironment.create(ImmutableMap.of(), ImmutableSet.of());
    assertThat(emptyEnvironment).isSameInstanceAs(ActionEnvironment.EMPTY);

    ActionEnvironment base =
        ActionEnvironment.create(ImmutableMap.of("FOO", "foo1"), ImmutableSet.of("baz"));
    assertThat(base.withAdditionalFixedVariables(ImmutableMap.of())).isSameInstanceAs(base);
    assertThat(base.withAdditionalVariables(ImmutableMap.of(), ImmutableSet.of()))
        .isSameInstanceAs(base);
  }
}
