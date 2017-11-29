// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.java;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.ComputedDefault;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.SkylarkProviderIdentifier;

/** Common rule class definitions for Java rules. */
public final class JavaRuleClasses {
  private JavaRuleClasses() {}

  /**
   * Common attributes for rules that depend on ijar.
   */
  public static final class IjarBaseRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
      return builder
          .add(
              attr(":java_toolchain", LABEL)
                  .useOutputLicenses()
                  .mandatoryProviders(ToolchainInfo.PROVIDER.id())
                  .value(JavaSemantics.JAVA_TOOLCHAIN))
          .setPreferredDependencyPredicate(JavaSemantics.JAVA_SOURCE)
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("$ijar_base_rule")
          .type(RuleClassType.ABSTRACT)
          .build();
    }
  }

  public static Attribute.ComputedDefault createUseTestrunnerComputedDefault() {
    return new ComputedDefault() {
      @Override
      public Object getDefault(AttributeMap rule) {
        return !rule.isAttributeValueExplicitlySpecified("main_class");
      }
    };
  }

  /**
   * Meant to be an element of {@code mandatoryProvidersLists} in order to accept rules providing a
   * {@link JavaInfo} through an attribute. Other providers can be included in {@code
   * mandatoryProvidersLists} as well.
   */
  public static final ImmutableList<SkylarkProviderIdentifier> CONTAINS_JAVA_PROVIDER =
      ImmutableList.of(SkylarkProviderIdentifier.forKey(JavaInfo.PROVIDER.getKey()));
}
