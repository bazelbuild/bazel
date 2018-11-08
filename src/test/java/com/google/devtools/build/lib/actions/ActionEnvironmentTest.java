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
    ActionEnvironment env2 = env1.addFixedVariables(ImmutableMap.of("FOO", "foo2"));

    assertThat(env1.getFixedEnv().toMap()).containsExactly("FOO", "foo1", "BAR", "bar");
    assertThat(env1.getInheritedEnv()).containsExactly("baz");

    assertThat(env2.getFixedEnv().toMap()).containsExactly("FOO", "foo2", "BAR", "bar");
    assertThat(env2.getInheritedEnv()).containsExactly("baz");
  }
}
