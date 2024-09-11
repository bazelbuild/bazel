// Copyright 2024 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.starlarkdocextract;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleClassFunctions.StarlarkRuleFunction;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.StarlarkProviderIdentifier;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.OriginKey;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.RuleInfo;

/** API documentation extractor for a Starlark rule. */
public final class RuleInfoExtractor {

  public static RuleInfo buildRuleInfo(
      ExtractorContext context, String qualifiedName, StarlarkRuleFunction ruleFunction)
      throws ExtractionException {
    RuleInfo.Builder ruleInfoBuilder = RuleInfo.newBuilder();
    // Record the name under which this symbol is made accessible, which may differ from the
    // symbol's exported name
    ruleInfoBuilder.setRuleName(qualifiedName);
    // ... but record the origin rule key for cross references.
    ruleInfoBuilder.setOriginKey(
        OriginKey.newBuilder()
            .setName(ruleFunction.getName())
            .setFile(context.getLabelRenderer().render(ruleFunction.getExtensionLabel())));
    ruleFunction.getDocumentation().ifPresent(ruleInfoBuilder::setDocString);

    RuleClass ruleClass = ruleFunction.getRuleClass();
    if (ruleClass.getRuleClassType() == RuleClassType.TEST) {
      ruleInfoBuilder.setTest(true);
    }
    if (ruleClass.hasAttr(Rule.IS_EXECUTABLE_ATTRIBUTE_NAME, Type.BOOLEAN)) {
      ruleInfoBuilder.setExecutable(true);
    }

    ruleInfoBuilder.addAttribute(
        AttributeInfoExtractor.IMPLICIT_NAME_ATTRIBUTE_INFO); // name comes first
    AttributeInfoExtractor.addDocumentableAttributes(
        context, ruleClass.getAttributes(), ruleInfoBuilder::addAttribute, "rule " + qualifiedName);
    ImmutableSet<StarlarkProviderIdentifier> advertisedProviders =
        ruleClass.getAdvertisedProviders().getStarlarkProviders();
    if (!advertisedProviders.isEmpty()) {
      ruleInfoBuilder.setAdvertisedProviders(
          ProviderNameGroupExtractor.buildProviderNameGroup(context, advertisedProviders));
    }
    return ruleInfoBuilder.build();
  }

  private RuleInfoExtractor() {}
}
