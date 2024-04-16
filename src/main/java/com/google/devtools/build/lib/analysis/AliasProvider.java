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

import static com.google.common.base.Preconditions.checkArgument;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.AdvertisedProviderSet;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;

/** A provider that gives information about the aliases a rule was resolved through. */
@Immutable
public final class AliasProvider implements TransitiveInfoProvider {
  // We don't expect long alias chains, so it's better to have a list instead of a nested set
  private final ImmutableList<Label> aliasChain;

  private AliasProvider(ImmutableList<Label> aliasChain) {
    this.aliasChain = aliasChain;
  }

  /**
   * Creates an alias provider indicating that the given rule is an alias to {@code actual}.
   *
   * <p>The given rule must either explicitly advertise {@link AliasProvider} or advertise that it
   * {@linkplain AdvertisedProviderSet#canHaveAnyProvider() can have any provider}.
   */
  public static AliasProvider fromAliasRule(Rule rule, ConfiguredTarget actual) {
    checkArgument(mayBeAlias(rule), "%s does not advertise AliasProvider", rule);

    ImmutableList<Label> chain;
    AliasProvider dep = actual.getProvider(AliasProvider.class);
    if (dep == null) {
      chain = ImmutableList.of(rule.getLabel());
    } else {
      chain =
          ImmutableList.<Label>builderWithExpectedSize(dep.aliasChain.size() + 1)
              .add(rule.getLabel())
              .addAll(dep.aliasChain)
              .build();
    }
    return new AliasProvider(chain);
  }

  /**
   * Returns the label by which it was referred to in the BUILD file.
   *
   * <p>For non-alias rules, it's the label of the rule itself, for alias rules, it's the label of
   * the alias rule.
   */
  public static Label getDependencyLabel(TransitiveInfoCollection dep) {
    AliasProvider aliasProvider = dep.getProvider(AliasProvider.class);
    return aliasProvider != null ? aliasProvider.aliasChain.get(0) : dep.getLabel();
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
        ? ImmutableList.of(aliasProvider.aliasChain.get(0), dep.getLabel())
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
   */
  public static String describeTargetWithAliases(
      ConfiguredTargetAndData target, TargetMode targetMode) {
    String kind = targetMode == TargetMode.WITH_KIND ? target.getTargetKind() : "target";
    AliasProvider aliasProvider = target.getConfiguredTarget().getProvider(AliasProvider.class);
    if (aliasProvider == null) {
      return kind + " '" + target.getTargetLabel() + "'";
    }

    ImmutableList<Label> aliasChain = aliasProvider.aliasChain;
    StringBuilder result = new StringBuilder();
    result.append("alias '").append(aliasChain.get(0)).append("'");
    result
        .append(" referring to ")
        .append(kind)
        .append(" '")
        .append(target.getTargetLabel())
        .append("'");
    if (aliasChain.size() > 1) {
      result
          .append(" through '")
          .append(Joiner.on("' -> '").join(aliasChain.subList(1, aliasChain.size())))
          .append("'");
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

  /**
   * Returns {@code true} if the given target <em>may</em> contain an {@link AliasProvider} when
   * analyzed.
   *
   * <p>This method returns {@code true} for the {@code alias} rule as well as some other alias-like
   * rules such as {@code bind}.
   *
   * <p>Note that due to the presence of late-bound aliases, this may return {@code true} even if
   * {@link #isAlias} on the configured target returns {@code false}.
   */
  public static boolean mayBeAlias(Target target) {
    if (!(target instanceof Rule rule)) {
      return false;
    }
    AdvertisedProviderSet providerSet = rule.getRuleClassObject().getAdvertisedProviders();
    return providerSet.canHaveAnyProvider()
        || providerSet.getBuiltinProviders().contains(AliasProvider.class);
  }
}
