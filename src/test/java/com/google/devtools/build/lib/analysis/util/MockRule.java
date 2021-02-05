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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import java.util.Arrays;

/**
 * Provides a simple API for creating custom rule classes for tests.
 *
 * <p>Use this whenever you want to test language-agnostic Bazel functionality, i.e. behavior that
 * isn't specific to individual rule implementations. If you find yourself searching through rule
 * implementations trying to find one that matches whatever you're trying to test, you probably
 * want this instead.
 *
 * <p>This prevents the anti-pattern of tests with commingled dependencies. For example, when a test
 * uses <code>cc_library</code> to test generic logic that <code>cc_library</code> happens to
 * provide, the test can break if the <code>cc_library</code> implementation changes. This means C++
 * rule developers have to understand the test to change C++ logic: a dependency that helps no one.
 *
 * <p>Even if C++ logic doesn't change, <code>cc_library</code> may not make it clear what's being
 * tested (e.g. "why is the "malloc" attribute used here?"). Using a mock rule class offers the
 * ability to write a clearer, more focused, easier to understand test (e.g.
 * <code>mock_rule(name = "foo", attr_that_tests_this_specific_test_logic = ":bar")</code).
 *
 * <p>Usage for a custom rule type that just needs to exist (no special attributes or behavior
 * needed):
 *
 * <pre>
 *   MockRule fooRule = () -> MockRule.define("foo_rule");
 * </pre>
 *
 * <p>Usage for custom attributes:
 *
 * <pre>
 *   MockRule fooRule = () -> MockRule.define("foo_rule", attr("some_attr", Type.STRING));
 * </pre>
 *
 * <p>Usage for arbitrary customization:
 *
 * <pre>
 *   MockRule fooRule = () -> MockRule.define(
 *       "foo_rule",
 *       (builder, env) ->
 *           builder
 *               .removeAttribute("tags")
 *               .requiresConfigurationFragments(FooConfiguration.class);
 *       );
 * </pre>
 *
 * Custom {@link RuleDefinition} ancestors and {@link RuleConfiguredTargetFactory} implementations
 * can also be specified:
 *
 * <pre>
 *   MockRule customAncestor = () -> MockRule.ancestor(BaseRule.class).define(...);
 *   MockRule customImpl = () -> MockRule.factory(FooRuleFactory.class).define(...);
 *   MockRule customEverything = () ->
 *       MockRule.ancestor(BaseRule.class).factory(FooRuleFactory.class).define(...);
 * </pre>
 *
 * When unspecified, {@link State#DEFAULT_ANCESTOR} and {@link State#DEFAULT_FACTORY} apply.
 *
 * <p>We use lambdas for custom rule classes because {@link ConfiguredRuleClassProvider} indexes
 * rule class definitions by their Java class names. So each definition has to have its own
 * unique Java class.
 *
 * <p>Both of the following forms are valid:
 *
 * <pre>MockRule fooRule = () -> MockRule.define("foo_rule");</pre>
 * <pre>RuleDefinition fooRule = (MockRule) () -> MockRule.define("foo_rule");</pre>
 *
 * <p>Use discretion in choosing your preferred form. The first is more compact. The second makes
 * it clearer that <code>fooRule</code> is a proper rule class definition.
 */
public interface MockRule extends RuleDefinition {
  // MockRule is designed to be easy to use. That doesn't necessarily mean its implementation is
  // easy to undestand.
  //
  // If you just want to mock a rule, it's best to rely on the interface javadoc above, rather than
  // trying to parse what's going on below. You really only need to understand the below if you want
  // to customize MockRule itself.

  /**
   * Container for the desired name and custom settings for this rule class.
   */
  class State {
    private final String name;
    private final MockRuleCustomBehavior customBehavior;
    private final Class<? extends RuleConfiguredTargetFactory> factory;
    private final ImmutableList<Class<? extends RuleDefinition>> ancestors;
    private final RuleClassType type;

    /** The default {@link RuleConfiguredTargetFactory} for this rule class. */
    private static final Class<? extends RuleConfiguredTargetFactory> DEFAULT_FACTORY =
        MockRuleDefaults.DefaultConfiguredTargetFactory.class;
    /** The default {@link RuleDefinition} for this rule class. */
    private static final ImmutableList<Class<? extends RuleDefinition>> DEFAULT_ANCESTORS =
        ImmutableList.of();

    State(String ruleClassName, MockRuleCustomBehavior customBehavior,
        Class<? extends RuleConfiguredTargetFactory> factory,
        ImmutableList<Class<? extends RuleDefinition>> ancestors,
        RuleClassType type) {
      this.name = Preconditions.checkNotNull(ruleClassName);
      this.customBehavior = Preconditions.checkNotNull(customBehavior);
      this.factory = factory;
      this.ancestors = ancestors;
      this.type = type;
    }

    public static class Builder {
      private Class<? extends RuleConfiguredTargetFactory> factory = DEFAULT_FACTORY;
      private ImmutableList<Class<? extends RuleDefinition>> ancestors = DEFAULT_ANCESTORS;
      private RuleClassType type = RuleClassType.NORMAL;

      public Builder factory(Class<? extends RuleConfiguredTargetFactory> factory) {
        this.factory = factory;
        return this;
      }

      public Builder ancestor(Class<? extends RuleDefinition>... ancestor) {
        this.ancestors = ImmutableList.copyOf(ancestor);
        return this;
      }

      public Builder type(RuleClassType type) {
        this.type = type;
        return this;
      }

      public State define(String ruleClassName, Attribute.Builder<?>... attributes) {
        return build(ruleClassName,
            new MockRuleCustomBehavior.CustomAttributes(Arrays.asList(attributes)));
      }

      public State define(String ruleClassName, MockRuleCustomBehavior customBehavior) {
        return build(ruleClassName, customBehavior);
      }

      private State build(String ruleClassName, MockRuleCustomBehavior customBehavior) {
        return new State(ruleClassName, customBehavior, factory, ancestors, type);
      }
    }
  }

  /**
   * Sets a custom {@link RuleConfiguredTargetFactory} for this mock rule.
   *
   * <p>If not set, {@link State#DEFAULT_FACTORY} is used.
   */
  static State.Builder factory(Class<? extends RuleConfiguredTargetFactory> factory) {
    return new State.Builder().factory(factory);
  }

  /**
   * Sets a custom ancestor {@link RuleDefinition} for this mock rule.
   *
   * <p>If not set, {@link State#DEFAULT_ANCESTORS} is used.
   */
  static State.Builder ancestor(Class<? extends RuleDefinition>... ancestor) {
    return new State.Builder().ancestor(ancestor);
  }

  /**
   * Sets a custom {@link RuleClassType} for this mock rule.
   *
   * <p>If not set, {@link RuleClassType#NORMAL} is used.
   */
  static State.Builder type(RuleClassType type) {
    return new State.Builder().type(type);
  }

  /**
   * Returns a new {@link State} for this rule class with custom attributes. This is a convenience
   * method for lambda definitions:
   *
   * <pre>
   *   MockRule myRule = () -> MockRule.define("my_rule", attr("myattr", Type.STRING));
   * </pre>
   */
  static State define(String ruleClassName, Attribute.Builder<?>... attributes) {
    return new State.Builder().define(ruleClassName, attributes);
  }

  /**
   * Returns a new {@link State} for this rule class with arbitrary custom behavior. This is a
   * convenience method for lambda definitions:
   *
   * <pre>
   *   MockRule myRule = () -> MockRule.define(
   *       "my_rule",
   *       (builder, env) -> builder.requiresConfigurationFragments(FooConfiguration.class));
   * </pre>
   */
  static State define(String ruleClassName, MockRuleCustomBehavior customBehavior) {
    return new State.Builder().define(ruleClassName, customBehavior);
  }

  /**
   * Returns the basic state that defines this rule class. This is the only interface method
   * implementers must override.
   */
  State define();

  /**
   * Builds out this rule with default attributes Blaze expects of all rules
   * ({@link MockRuleDefaults#DEFAULT_ATTRIBUTES}) plus the custom attributes defined by this
   * implementation's {@link State}.
   *
   * <p>Do not override this method. For extra custom behavior, use
   * {@link #define(String, MockRuleCustomBehavior)}
   */
  @Override
  default RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
    State state = define();
    if (State.DEFAULT_ANCESTORS.equals(state.ancestors)) {
      MockRuleDefaults.DEFAULT_ATTRIBUTES.stream().forEach(builder::add);
    }
    state.customBehavior.customize(builder, environment);
    return builder.build();
  }

  /**
   * Sets this rule class's metadata with the name defined by {@link State}, configured target
   * factory declared by {@link State.Builder#factory}, and ancestor rule class declared by
   * {@link State.Builder#ancestor}.
   */
  @Override
  default RuleDefinition.Metadata getMetadata() {
    State state = define();
    return RuleDefinition.Metadata.builder()
        .name(state.name)
        .type(state.type)
        .factoryClass(state.factory)
        .ancestors(state.ancestors)
        .build();
  }
}
