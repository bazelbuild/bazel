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
import com.google.devtools.build.lib.analysis.config.HostTransition;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.StarlarkProviderIdentifier;

/** Common rule class definitions for Java rules. */
public class JavaRuleClasses {

  public static final String JAVA_TOOLCHAIN_ATTRIBUTE_NAME = "$java_toolchain";
  public static final String JAVA_RUNTIME_ATTRIBUTE_NAME = "$jvm";
  public static final String HOST_JAVA_RUNTIME_ATTRIBUTE_NAME = "$host_jdk";

  /** Common attributes for rules that depend on ijar. */
  public static final class IjarBaseRule implements RuleDefinition {
    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
      return builder.setPreferredDependencyPredicate(JavaSemantics.JAVA_SOURCE).build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("$ijar_base_rule")
          .type(RuleClassType.ABSTRACT)
          .ancestors(JavaToolchainBaseRule.class)
          .build();
    }
  }

  /** Common attributes for rules that use the Java toolchain. */
  public static final class JavaToolchainBaseRule implements RuleDefinition {
    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
      return builder
          .add(
              attr(JAVA_TOOLCHAIN_ATTRIBUTE_NAME, LABEL)
                  .useOutputLicenses()
                  .mandatoryProviders(ToolchainInfo.PROVIDER.id())
                  .value(JavaSemantics.javaToolchainAttribute(env)))
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("$java_toolchain_base_rule")
          .type(RuleClassType.ABSTRACT)
          .build();
    }
  }

  /** Common attributes for rules that use the Java runtime. */
  public static final class JavaRuntimeBaseRule implements RuleDefinition {
    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
      return builder
          .add(
              attr(JAVA_RUNTIME_ATTRIBUTE_NAME, LABEL)
                  .value(JavaSemantics.jvmAttribute(env))
                  .mandatoryProviders(ToolchainInfo.PROVIDER.id())
                  .useOutputLicenses())
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("$java_runtime_toolchain_base_rule")
          .type(RuleClassType.ABSTRACT)
          .ancestors(JavaHostRuntimeBaseRule.class)
          .build();
    }
  }

  /** Common attributes for rules that use the host Java runtime. */
  public static final class JavaHostRuntimeBaseRule implements RuleDefinition {
    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
      return builder
          .add(
              attr(HOST_JAVA_RUNTIME_ATTRIBUTE_NAME, LABEL)
                  .cfg(HostTransition.createFactory())
                  .value(JavaSemantics.hostJdkAttribute(env))
                  .mandatoryProviders(ToolchainInfo.PROVIDER.id()))
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("$java_host_runtime_toolchain_base_rule")
          .type(RuleClassType.ABSTRACT)
          .build();
    }
  }

  /**
   * Meant to be an element of {@code mandatoryProvidersLists} in order to accept rules providing a
   * {@link JavaInfo} through an attribute. Other providers can be included in {@code
   * mandatoryProvidersLists} as well.
   */
  public static final ImmutableList<StarlarkProviderIdentifier> CONTAINS_JAVA_PROVIDER =
      ImmutableList.of(StarlarkProviderIdentifier.forKey(JavaInfo.PROVIDER.getKey()));
}
