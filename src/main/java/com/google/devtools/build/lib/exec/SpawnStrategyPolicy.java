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

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.runtime.proto.MnemonicPolicy;
import com.google.devtools.build.lib.runtime.proto.StrategiesForMnemonic;
import java.util.List;

/** Policy for filtering spawn strategies. */
public interface SpawnStrategyPolicy {

  /** Returns result of applying policy to per-mnemonic strategies. */
  ImmutableList<String> apply(String mnemonic, List<String> strategies);

  /** Creates new policy from proto descriptor. Empty proto policy implies everything allowed. */
  static SpawnStrategyPolicy create(MnemonicPolicy policy) {
    if (MnemonicPolicy.getDefaultInstance().equals(policy)) {
      return new AllowAllStrategiesPolicy();
    }

    ImmutableMap.Builder<String, ImmutableSet<String>> perMnemonicAllowList =
        ImmutableMap.builder();
    for (StrategiesForMnemonic strategiesForMnemonic : policy.getStrategyAllowlistList()) {
      perMnemonicAllowList.put(
          strategiesForMnemonic.getMnemonic(),
          ImmutableSet.copyOf(strategiesForMnemonic.getStrategyList()));
    }
    return new SpawnStrategyPolicyImpl(
        perMnemonicAllowList.buildKeepingLast(),
        ImmutableSet.copyOf(policy.getDefaultAllowlistList()));
  }

  /** Allows all strategies - effectively a no-op strategy. */
  class AllowAllStrategiesPolicy implements SpawnStrategyPolicy {

    private AllowAllStrategiesPolicy() {}

    @Override
    public ImmutableList<String> apply(String mnemonic, List<String> strategies) {
      return ImmutableList.copyOf(strategies);
    }
  }

  /** Enforces a real strategy policy based on provided config. */
  class SpawnStrategyPolicyImpl implements SpawnStrategyPolicy {

    private final ImmutableMap<String, ImmutableSet<String>> perMnemonicAllowList;
    private final ImmutableSet<String> defaultAllowList;

    private SpawnStrategyPolicyImpl(
        ImmutableMap<String, ImmutableSet<String>> perMnemonicAllowList,
        ImmutableSet<String> defaultAllowList) {
      this.perMnemonicAllowList = perMnemonicAllowList;
      this.defaultAllowList = defaultAllowList;
    }

    @Override
    public ImmutableList<String> apply(String mnemonic, List<String> strategies) {
      ImmutableSet<String> allowList =
          perMnemonicAllowList.getOrDefault(mnemonic, defaultAllowList);
      return strategies.stream().filter(allowList::contains).collect(toImmutableList());
    }
  }
}
