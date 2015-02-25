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

import com.google.devtools.build.lib.packages.RuleClass;

/**
 * This class is a common ancestor for every rule object.
 *
 * <p>Implementors are also required to have the {@link BlazeRule} annotation
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
   *     BlazeRule} annotation.
   * @param environment The services Blaze provides to rule definitions.
   *
   * @return the {@link RuleClass} representing the rule.
   */
  RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment);
}
