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
package com.google.devtools.build.lib.analysis.util;

import static com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition.HOST;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.BuildType.NODEP_LABEL_LIST;
import static com.google.devtools.build.lib.syntax.Type.BOOLEAN;
import static com.google.devtools.build.lib.syntax.Type.STRING;
import static com.google.devtools.build.lib.syntax.Type.STRING_LIST;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.util.FileTypeSet;

import java.util.Arrays;
import java.util.List;

/**
 * Provides a simple API for creating custom rule classes for tests.
 *
 * <p>Usage:
 *
 * <pre>
 *   MockRule fooRule = () -> MockRule.define("foo_rule");
 *   MockRule ruleWithCustomAttr = () -> MockRule.define("attr_rule", attr("myattr", Type.STRING));
 * </pre>
 *
 * <p>If you need special behavior beyond custom attributes:
 *
 * <pre>
 *   class MyCustomRuleClass implements MockRule {
 *     &#064;Override
 *     public MockRule.State define() {
 *      return MockRule.define("my_custom_rule");
 *     }
 *
 *     &#064;Override
 *     public void customize(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
 *       builder.depsCfg(HostTransition.INSTANCE);
 *     }
 *  }
 * </pre>
 *
 * <p>We use lambdas for custom rule classes because {@link ConfiguredRuleClassProvider} indexes
 * rule class definitions by their Java class names. So each definition has to have its own
 * unique Java class.
 */
public interface MockRule extends RuleDefinition {
  /**
   * Container for the desired name and custom attributes for this rule class.
   */
  class State {
    private final String name;
    private final List<Attribute.Builder<?>> attributes;

    State(String ruleClassName, Attribute.Builder<?>... attributes) {
      this.name = ruleClassName;
      this.attributes = Arrays.asList(attributes);
    }
  }

  /**
   * Returns a new {@link State} for this rule class. This is a convenience method for lambda
   * definitions:
   *
   * <pre>
   *   MockRule myRule = () -> MockRule.define("my_rule", attr("myattr", Type.STRING));
   * </pre>
   */
  static State define(String ruleClassName, Attribute.Builder<?>... attributes) {
    return new State(ruleClassName, attributes);
  }

  /**
   * Returns the basic state that defines this rule class. This is the only interface method
   * implementers must override.
   */
  State define();

  /**
   * Allows for custom builder configuration beyond setting attributes.
   */
  default void customize(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
  }

  /**
   * Default <code>"deps"</code> attribute for rule classes that don't need special behavior.
   */
  Attribute.Builder<?> DEPS_ATTRIBUTE = attr("deps", BuildType.LABEL_LIST).allowedFileTypes();

  /**
   * Builds out this rule with default attributes Blaze expects of all rules plus the custom
   * attributes defined by this implementation's {@link State}.
   *
   * <p>Do not override this method. For extra custom behavior, override {@link #customize}.
   */
  @Override
  default RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
    builder
        .add(attr("testonly", BOOLEAN).nonconfigurable("test").value(false))
        .add(attr("deprecation", STRING).nonconfigurable("test").value((String) null))
        .add(attr("tags", STRING_LIST))
        .add(attr("visibility", NODEP_LABEL_LIST).orderIndependent().cfg(HOST)
            .nonconfigurable("test"))
        .add(attr(RuleClass.COMPATIBLE_ENVIRONMENT_ATTR, LABEL_LIST)
            .allowedFileTypes(FileTypeSet.NO_FILE))
        .add(attr(RuleClass.RESTRICTED_ENVIRONMENT_ATTR, LABEL_LIST)
            .allowedFileTypes(FileTypeSet.NO_FILE));
    for (Attribute.Builder<?> customAttribute : define().attributes) {
      builder.add(customAttribute);
    }
    customize(builder, environment);
    return builder.build();
  }

  /**
   * Sets this rule class's metadata with the name defined by {@link State}.
   */
  @Override
  default RuleDefinition.Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name(define().name)
        .type(RuleClass.Builder.RuleClassType.NORMAL)
        .factoryClass(MockConfiguredTargetFactory.class)
        .ancestors(BaseRuleClasses.RootRule.class)
        .build();
  }
}
