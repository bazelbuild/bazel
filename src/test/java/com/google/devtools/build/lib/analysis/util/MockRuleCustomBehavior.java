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

import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.RuleClass;

/**
 * Interface for supporting arbitrary custom behavior in mock rule classes.
 *
 * <p>See {@link MockRule} for details and usage instructions.
 */
public interface MockRuleCustomBehavior {

  /**
   * Adds custom behavior to a mock rule class.
   *
   * <p>It's not necessary to call {@link RuleClass.Builder#build} here.
   */
  void customize(RuleClass.Builder builder, RuleDefinitionEnvironment env);

  /* Predefined no-op behavior.  */
  MockRuleCustomBehavior NOOP = (builder, env) -> {};

  /**
   * Predefined behavior that populates a list of attributes.
   */
  class CustomAttributes implements MockRuleCustomBehavior {
    private final Iterable<Attribute.Builder<?>> attributes;

    CustomAttributes(Iterable<Attribute.Builder<?>> attributes) {
      this.attributes = attributes;
    }

    @Override
    public void customize(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
      for (Attribute.Builder<?> attribute : attributes) {
        builder.add(attribute);
      }
    }
  }
}
