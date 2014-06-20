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
package com.google.devtools.build.lib.rules;

import com.google.devtools.build.lib.view.RuleConfiguredTarget;
import com.google.devtools.build.lib.view.RuleContext;

/**
 * A factory or builder class for rule implementations.
 */
public interface RuleConfiguredTargetFactory {

  /**
   * Returns a fully initialized configured target instance using the given context.
   */
  // TODO(bazel-team): The InterruptedException declaration is only used by GenQuery.
  RuleConfiguredTarget create(RuleContext ruleContext) throws InterruptedException;
}
