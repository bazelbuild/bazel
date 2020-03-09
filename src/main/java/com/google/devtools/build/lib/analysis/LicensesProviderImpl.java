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

import com.google.common.collect.ListMultimap;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.License;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;

/** A {@link ConfiguredTarget} that has licensed targets in its transitive closure. */
@Immutable
@AutoCodec
public final class LicensesProviderImpl implements LicensesProvider {
  public static final LicensesProvider EMPTY =
      new LicensesProviderImpl(NestedSetBuilder.<TargetLicense>emptySet(Order.LINK_ORDER), null);

  private final NestedSet<TargetLicense> transitiveLicenses;
  private final TargetLicense outputLicenses;

  public LicensesProviderImpl(
      NestedSet<TargetLicense> transitiveLicenses, TargetLicense outputLicenses) {
    this.transitiveLicenses = transitiveLicenses;
    this.outputLicenses = outputLicenses;
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
    AttributeMap attributes = ruleContext.attributes();
    License toolOutputLicense = rule.getToolOutputLicense(attributes);
    TargetLicense outputLicenses =
        toolOutputLicense == null ? null : new TargetLicense(rule.getLabel(), toolOutputLicense);

    if (configuration.isToolConfiguration() && toolOutputLicense != null) {
      if (toolOutputLicense != License.NO_LICENSE) {
        builder.add(outputLicenses);
      }
    } else {
      if (rule.getLicense() != License.NO_LICENSE) {
        builder.add(new TargetLicense(rule.getLabel(), rule.getLicense()));
      }

      ListMultimap<String, ? extends TransitiveInfoCollection> configuredMap =
          ruleContext.getConfiguredTargetMap();

      for (String depAttrName : attributes.getAttributeNames()) {
        // Only add the transitive licenses for the attributes that do not have the output_licenses.
        Attribute attribute = attributes.getAttributeDefinition(depAttrName);
        for (TransitiveInfoCollection dep : configuredMap.get(depAttrName)) {
          LicensesProvider provider = dep.getProvider(LicensesProvider.class);
          if (provider == null) {
            continue;
          }
          if (useOutputLicenses(attribute, configuration) && provider.hasOutputLicenses()) {
              builder.add(provider.getOutputLicenses());
          } else {
            builder.addTransitive(provider.getTransitiveLicenses());
          }
        }
      }
    }

    return new LicensesProviderImpl(builder.build(), outputLicenses);
  }

  private static boolean useOutputLicenses(Attribute attribute, BuildConfiguration configuration) {
    return configuration.isToolConfiguration() || attribute.useOutputLicenses();
  }

  @Override
  public NestedSet<TargetLicense> getTransitiveLicenses() {
    return transitiveLicenses;
  }

  @Override
  public TargetLicense getOutputLicenses() {
    return outputLicenses;
  }

  @Override
  public boolean hasOutputLicenses() {
    return outputLicenses != null;
  }
}
