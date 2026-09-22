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
import com.google.devtools.build.lib.util.Fingerprint;
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
                ImmutableMap.of("FIXED_ONLY", "fixed"),
                ImmutableSet.of("INHERITED_ONLY", "FIXED_AND_INHERITED"))
            .withAdditionalFixedVariables(ImmutableMap.of("FIXED_AND_INHERITED", "fixed"));
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
  public void unsetRemovesFixedInheritedAndExistingVariables() {
    ActionEnvironment env =
        ActionEnvironment.create(
            ImmutableMap.of("FIXED", "fixed", "FIXED_UNSET", "fixed"),
            ImmutableSet.of("INHERITED", "INHERITED_UNSET"),
            ImmutableSet.of("FIXED_UNSET", "INHERITED_UNSET", "EXISTING_UNSET", "MISSING"));
    Map<String, String> clientEnv =
        ImmutableMap.of("INHERITED", "inherited", "INHERITED_UNSET", "inherited");
    Map<String, String> result = new HashMap<>();
    result.put("EXISTING", "existing");
    result.put("EXISTING_UNSET", "existing");
    env.resolve(result, clientEnv);

    assertThat(env.getUnsetEnv())
        .containsExactly("FIXED_UNSET", "INHERITED_UNSET", "EXISTING_UNSET", "MISSING");
    assertThat(result)
        .containsExactly("FIXED", "fixed", "INHERITED", "inherited", "EXISTING", "existing");
  }

  @Test
  public void unsetRemovesVariablesFromEarlierLayer() {
    ActionEnvironment defaultEnv =
        ActionEnvironment.create(ImmutableMap.of("HOME", "/tmp", "TZ", "UTC"));
    ActionEnvironment userEnv =
        ActionEnvironment.create(
            ImmutableMap.of("FOO", "foo"), ImmutableSet.of(), ImmutableSet.of("HOME"));
    Map<String, String> result = new HashMap<>();
    defaultEnv.resolve(result, ImmutableMap.of());
    userEnv.resolve(result, ImmutableMap.of());

    assertThat(result).containsExactly("TZ", "UTC", "FOO", "foo");
  }

  @Test
  public void additionalFixedVariablesOverrideUnset() {
    ActionEnvironment env =
        ActionEnvironment.create(
                ImmutableMap.of(), ImmutableSet.of(), ImmutableSet.of("FOO", "BAR"))
            .withAdditionalFixedVariables(ImmutableMap.of("FOO", "foo"));
    Map<String, String> result = new HashMap<>();
    result.put("BAR", "bar");
    env.resolve(result, ImmutableMap.of());

    assertThat(env.getFixedEnv()).containsExactly("FOO", "foo");
    assertThat(env.getUnsetEnv()).containsExactly("BAR");
    assertThat(result).containsExactly("FOO", "foo");
  }

  @Test
  public void splitWithUnset() {
    Map<String, String> env = new HashMap<>();
    env.put("FIXED", "fixed");
    env.put("INHERITED", null);
    ActionEnvironment actionEnv = ActionEnvironment.split(env, ImmutableSet.of("UNSET"));

    assertThat(actionEnv.getFixedEnv()).containsExactly("FIXED", "fixed");
    assertThat(actionEnv.getInheritedEnv()).containsExactly("INHERITED");
    assertThat(actionEnv.getUnsetEnv()).containsExactly("UNSET");
  }

  @Test
  public void unsetAffectsEqualityAndFingerprint() {
    ActionEnvironment withoutUnset =
        ActionEnvironment.create(ImmutableMap.of("FOO", "foo"), ImmutableSet.of("BAR"));
    ActionEnvironment withEmptyUnset =
        ActionEnvironment.create(
            ImmutableMap.of("FOO", "foo"), ImmutableSet.of("BAR"), ImmutableSet.of());
    ActionEnvironment withUnset =
        ActionEnvironment.create(
            ImmutableMap.of("FOO", "foo"), ImmutableSet.of("BAR"), ImmutableSet.of("BAZ"));

    assertThat(withEmptyUnset).isSameInstanceAs(withoutUnset);
    assertThat(withUnset).isNotEqualTo(withoutUnset);
    assertThat(fingerprint(withUnset)).isNotEqualTo(fingerprint(withoutUnset));
    assertThat(fingerprint(withEmptyUnset)).isEqualTo(fingerprint(withoutUnset));
  }

  private static String fingerprint(ActionEnvironment env) {
    Fingerprint fp = new Fingerprint();
    env.addTo(fp);
    return fp.hexDigestAndReset();
  }

  @Test
  public void emptyEnvironmentInterning() {
    ActionEnvironment emptyEnvironment =
        ActionEnvironment.create(ImmutableMap.of(), ImmutableSet.of());
    assertThat(emptyEnvironment).isSameInstanceAs(ActionEnvironment.EMPTY);
    assertThat(ActionEnvironment.create(ImmutableMap.of(), ImmutableSet.of(), ImmutableSet.of()))
        .isSameInstanceAs(ActionEnvironment.EMPTY);

    ActionEnvironment base =
        ActionEnvironment.create(ImmutableMap.of("FOO", "foo1"), ImmutableSet.of("baz"));
    assertThat(base.withAdditionalFixedVariables(ImmutableMap.of())).isSameInstanceAs(base);
  }
}
