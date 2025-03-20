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
package com.google.devtools.build.lib.exec;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.runtime.proto.MnemonicPolicy;
import com.google.devtools.build.lib.runtime.proto.StrategiesForMnemonic;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class SpawnStrategyPolicyTest {

  @Test
  public void applyEmptyPolicyListAllowsEverything() {
    SpawnStrategyPolicy underTest = SpawnStrategyPolicy.create(MnemonicPolicy.getDefaultInstance());

    ImmutableList<String> strategies = ImmutableList.of("foo", "bar");
    assertThat(underTest.apply("mnemonic1", strategies))
        .containsExactlyElementsIn(strategies)
        .inOrder();
  }

  @Test
  public void applyNonOverriddenMnemonicUsesDefaultAllowList() {
    SpawnStrategyPolicy underTest =
        SpawnStrategyPolicy.create(
            mnemonicPolicy(
                ImmutableList.of(strategiesForMnemonic("mnemonic1", "baz")),
                ImmutableList.of("foo", "bar")));

    assertThat(underTest.apply("not-mnemonic1", ImmutableList.of("foo", "bar", "baz")))
        .containsExactly("foo", "bar")
        .inOrder();
  }

  @Test
  public void applyPerStrategyAllowListUsedToFilterStrategies() {
    SpawnStrategyPolicy underTest =
        SpawnStrategyPolicy.create(
            mnemonicPolicy(
                ImmutableList.of(strategiesForMnemonic("mnemonic1", "baz")),
                ImmutableList.of("foo", "bar")));

    assertThat(underTest.apply("mnemonic1", ImmutableList.of("foo", "bar", "baz")))
        .containsExactly("baz");
  }

  @Test
  public void applyPerStrategyAllowListLastListPerMnemonicWins() {
    SpawnStrategyPolicy underTest =
        SpawnStrategyPolicy.create(
            mnemonicPolicy(
                ImmutableList.of(
                    strategiesForMnemonic("mnemonic1", "bar"),
                    strategiesForMnemonic("mnemonic1", "foo", "bar")),
                ImmutableList.of("boom")));

    assertThat(underTest.apply("mnemonic1", ImmutableList.of("foo", "bar", "baz")))
        .containsExactly("foo", "bar")
        .inOrder();
  }

  @Test
  public void applyDefaultAllowList() {
    SpawnStrategyPolicy underTest =
        SpawnStrategyPolicy.create(
            mnemonicPolicy(ImmutableList.of(), ImmutableList.of("foo", "baz")));

    assertThat(underTest.apply(ImmutableList.of("foo", "bar", "baz")))
        .containsExactly("foo", "baz")
        .inOrder();
  }

  private static MnemonicPolicy mnemonicPolicy(
      List<StrategiesForMnemonic> strategyAllowList, List<String> defaultAllowlist) {
    return MnemonicPolicy.newBuilder()
        .addAllStrategyAllowlist(strategyAllowList)
        .addAllDefaultAllowlist(defaultAllowlist)
        .build();
  }

  private static StrategiesForMnemonic strategiesForMnemonic(
      String mnemonic, String... strategies) {
    return StrategiesForMnemonic.newBuilder()
        .setMnemonic(mnemonic)
        .addAllStrategy(ImmutableList.copyOf(strategies))
        .build();
  }
}
