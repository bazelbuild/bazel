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

package com.google.devtools.build.lib.packages;

/**
 * A thin interface exposing only the warning and error reporting functionality
 * of a rule.
 *
 * <p>When a class or a method needs only this functionality but not the whole
 * {@code RuleContext}, it can use this thin interface instead.
 *
 * <p>This interface should only be implemented by {@code RuleContext}.
 */
public interface RuleErrorConsumer {
  /**
   * Consume a non-attribute-specific warning in a rule.
   */
  void ruleWarning(String message);

  /**
   * Consume a non-attribute-specific error in a rule.
   */
  void ruleError(String message);

  /**
   * Consume an attribute-specific warning in a rule.
   */
  void attributeWarning(String attrName, String message);

  /**
   * Consume an attribute-specific error in a rule.
   */
  void attributeError(String attrName, String message);
}
