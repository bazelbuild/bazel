// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.License;
import com.google.devtools.build.lib.packages.Rule;

/**
 * A {@link ConfiguredTarget} that has licensed targets in its transitive closure.
 */
@Immutable
public final class LicensesProviderImpl implements LicensesProvider {
  public static final LicensesProvider EMPTY =
      new LicensesProviderImpl(NestedSetBuilder.<TargetLicense>emptySet(Order.LINK_ORDER));

  private final NestedSet<TargetLicense> transitiveLicenses;

  public LicensesProviderImpl(NestedSet<TargetLicense> transitiveLicenses) {
    this.transitiveLicenses = transitiveLicenses;
  }

  /**
   * Create the appropriate {@link LicensesProvider} for a rule based on its {@code RuleContext}
   */
  public static LicensesProvider of(RuleContext ruleContext) {
    if (!ruleContext.getConfiguration().checkLicenses()) {
      return EMPTY;
    }

    NestedSetBuilder<TargetLicense> builder = NestedSetBuilder.linkOrder();
    BuildConfiguration configuration = ruleContext.getConfiguration();
    Rule rule = ruleContext.getRule();
    License toolOutputLicense = rule.getToolOutputLicense(ruleContext.attributes());
    if (configuration.isHostConfiguration() && toolOutputLicense != null) {
      if (toolOutputLicense != License.NO_LICENSE) {
        builder.add(new TargetLicense(rule.getLabel(), toolOutputLicense));
      }
    } else {
      if (rule.getLicense() != License.NO_LICENSE) {
        builder.add(new TargetLicense(rule.getLabel(), rule.getLicense()));
      }

      for (TransitiveInfoCollection dep : ruleContext.getConfiguredTargetMap().values()) {
        LicensesProvider provider = dep.getProvider(LicensesProvider.class);
        if (provider != null) {
          builder.addTransitive(provider.getTransitiveLicenses());
        }
      }
    }

    return new LicensesProviderImpl(builder.build());
  }

  @Override
  public NestedSet<TargetLicense> getTransitiveLicenses() {
    return transitiveLicenses;
  }
}
