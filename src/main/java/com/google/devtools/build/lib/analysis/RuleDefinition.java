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

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * This class is a common ancestor for every rule object.
 *
 * <p>Implementors are also required to have the {@link Metadata} annotation
 * set.
 */
public interface RuleDefinition {
  /**
   * This method should return a RuleClass object that represents the rule. The usual pattern is
   * that various setter methods are called on the builder object passed in as the argument, then
   * the object that is built by the builder is returned.
   *
   * @param builder A {@link com.google.devtools.build.lib.packages.RuleClass.Builder} object
   *     already preloaded with the attributes of the ancestors specified in the {@link
   *     Metadata} annotation.
   * @param environment The services Blaze provides to rule definitions.
   *
   * @return the {@link RuleClass} representing the rule.
   */
  RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment);

  /**
   * Returns metadata for this rule.
   */
  Metadata getMetadata();

  /**
   * Value class that contains the name, type, ancestors of a rule, as well as a reference to the
   * configured target factory.
   */
  @AutoValue
  public abstract static class Metadata {
    /**
     * The name of the rule, as it appears in the BUILD file. If it starts with
     * '$', the rule will be hidden from users and will only be usable from
     * inside Blaze.
     */
    public abstract String name();

    /**
     * The type of the rule. It can be an abstract rule, a normal rule or a test
     * rule. If the rule type is abstract, the configured class must not be set.
     */
    public abstract RuleClassType type();

    /**
     * The {@link RuleConfiguredTargetFactory} class that implements this rule. If the rule is
     * abstract, this must not be set.
     */
    public abstract Class<? extends RuleConfiguredTargetFactory> factoryClass();

    /**
     * The list of other rule classes this rule inherits from.
     */
    public abstract List<Class<? extends RuleDefinition>> ancestors();

    public static Builder builder() {
      return new AutoValue_RuleDefinition_Metadata.Builder()
          .type(RuleClassType.NORMAL)
          .factoryClass(RuleConfiguredTargetFactory.class)
          .ancestors(Collections.<Class<? extends RuleDefinition>>emptyList());
    }

    public static Metadata empty() {
      return builder().build();
    }

    /**
     * Builder class for the Metadata class.
     */
    @AutoValue.Builder
    public abstract static class Builder {
      public abstract Builder name(String s);
      public abstract Builder type(RuleClassType type);
      public abstract Builder factoryClass(Class<? extends RuleConfiguredTargetFactory> factory);
      public abstract Builder ancestors(List<Class<? extends RuleDefinition>> ancestors);
      @SafeVarargs
      public final Builder ancestors(Class<? extends RuleDefinition>... ancstrs) {
        return ancestors(Arrays.asList(ancstrs));
      }
      public abstract Metadata build();
    }
  }
}
