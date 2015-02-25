// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * An annotation for rule classes.
 */
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
public @interface BlazeRule {
  /**
   * The name of the rule, as it appears in the BUILD file. If it starts with
   * '$', the rule will be hidden from users and will only be usable from
   * inside Blaze.
   */
  String name();

  /**
   * The type of the rule. It can be an abstract rule, a normal rule or a test
   * rule. If the rule type is abstract, the configured class must not be set.
   */
  RuleClassType type() default RuleClassType.NORMAL;

  /**
   * The {@link RuleConfiguredTargetFactory} class that implements this rule. If the rule is
   * abstract, this must not be set.
   */
  Class<? extends RuleConfiguredTargetFactory> factoryClass()
      default RuleConfiguredTargetFactory.class;

  /**
   * The list of other rule classes this rule inherits from.
   */
  Class<? extends RuleDefinition>[] ancestors() default {};
}
