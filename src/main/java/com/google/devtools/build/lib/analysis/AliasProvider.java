// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.util.Preconditions;

/**
 * A provider that gives information about the aliases a rule was resolved through.
 */
@Immutable
public final class AliasProvider implements TransitiveInfoProvider {
  // We don't expect long alias chains, so it's better to have a list instead of a nested set
  private final ImmutableList<Label> aliasChain;

  public AliasProvider(ImmutableList<Label> aliasChain) {
    Preconditions.checkState(!aliasChain.isEmpty());
    this.aliasChain = aliasChain;
  }

  public static AliasProvider fromAliasRule(Label label, ConfiguredTarget actual) {
    ImmutableList.Builder<Label> chain = ImmutableList.builder();
    chain.add(label);
    AliasProvider dep = actual.getProvider(AliasProvider.class);
    if (dep != null) {
      chain.addAll(dep.getAliasChain());
    }

    return new AliasProvider(chain.build());
  }

  /**
   * Returns the label by which it was referred to in the BUILD file.
   *
   * <p>For non-alias rules, it's the label of the rule itself, for alias rules, it's the label of
   * the alias rule.
   */
  public static Label getDependencyLabel(TransitiveInfoCollection dep) {
    AliasProvider aliasProvider = dep.getProvider(AliasProvider.class);
    return aliasProvider != null
        ? aliasProvider.getAliasChain().get(0)
        : dep.getLabel();
  }

  /**
   * Returns the list of aliases from top to bottom (i.e. the last alias depends on the actual
   * resolved target and the first alias is the one that was in the attribute of the rule currently
   * being analyzed)
   */
  public ImmutableList<Label> getAliasChain() {
    return aliasChain;
  }

  public static String printLabelWithAliasChain(ConfiguredTarget target) {
    AliasProvider aliasProvider = target.getProvider(AliasProvider.class);
    String suffix = aliasProvider == null
        ? ""
        : " (aliased through '" + Joiner.on("' -> '").join(aliasProvider.getAliasChain()) + "')";

    return "'" + target.getLabel() + "'" + suffix;
  }
}
