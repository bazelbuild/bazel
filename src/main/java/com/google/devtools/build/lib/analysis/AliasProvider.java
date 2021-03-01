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
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;

/** A provider that gives information about the aliases a rule was resolved through. */
@AutoCodec
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
   * Returns all labels by which it can be referred to in the BUILD file.
   *
   * <p>For non-alias rules, it's the label of the rule itself. For alias rules, they're the label
   * of the alias and the label of alias' target rule.
   */
  public static ImmutableList<Label> getDependencyLabels(TransitiveInfoCollection dep) {
    AliasProvider aliasProvider = dep.getProvider(AliasProvider.class);
    return aliasProvider != null
        ? ImmutableList.of(aliasProvider.getAliasChain().get(0), dep.getLabel())
        : ImmutableList.of(dep.getLabel());
  }

  /**
   * Returns the list of aliases from top to bottom (i.e. the last alias depends on the actual
   * resolved target and the first alias is the one that was in the attribute of the rule currently
   * being analyzed)
   */
  public ImmutableList<Label> getAliasChain() {
    return aliasChain;
  }

  /** The way {@link #describeTargetWithAliases(ConfiguredTargetAndData, TargetMode) reports the
   * kind of a target. */
  public enum TargetMode {
    WITH_KIND,      // Specify the kind of the target
    WITHOUT_KIND,   // Only say "target"
  }

  /**
   * Prints a nice description of a target.
   *
   * Also adds the aliases it was reached through, if any.
   *
   * @param target the target to describe
   * @param targetMode how to express the kind of the target
   * @return
   */
  public static String describeTargetWithAliases(
      ConfiguredTargetAndData target, TargetMode targetMode) {
    String kind = targetMode == TargetMode.WITH_KIND
        ? target.getTarget().getTargetKind() : "target";
    AliasProvider aliasProvider = target.getConfiguredTarget().getProvider(AliasProvider.class);
    if (aliasProvider == null) {
      return kind + " '" + target.getTarget().getLabel() + "'";
    }

    ImmutableList<Label> aliasChain = aliasProvider.getAliasChain();
    StringBuilder result = new StringBuilder();
    result.append("alias '" + aliasChain.get(0) + "'");
    result.append(" referring to " + kind  + " '" + target.getTarget().getLabel() + "'");
    if (aliasChain.size() > 1) {
      result.append(" through '"
          + Joiner.on("' -> '").join(aliasChain.subList(1, aliasChain.size()))
          + "'");
    }

    return result.toString();
  }

  /**
   * Returns {@code true} iff the given {@link TransitiveInfoCollection} has an {@link
   * AliasProvider}.
   */
  public static boolean isAlias(TransitiveInfoCollection dep) {
    return dep.getProvider(AliasProvider.class) != null;
  }
}
