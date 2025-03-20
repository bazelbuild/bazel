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

package com.google.devtools.build.lib.rules.repository;

import com.google.devtools.build.lib.actions.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.VisibilityProvider;
import com.google.devtools.build.lib.rules.AliasConfiguredTarget;
import javax.annotation.Nullable;

/** Implementation for the {@code bind} rule. */
public final class Bind implements RuleConfiguredTargetFactory {

  @Override
  @Nullable
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    if (ruleContext.getPrerequisite("actual") == null) {
      ruleContext.ruleError(String.format("The external label '%s' is not bound to anything",
          ruleContext.getLabel()));
      return null;
    }

    ConfiguredTarget actual = (ConfiguredTarget) ruleContext.getPrerequisite("actual");
    return AliasConfiguredTarget.create(ruleContext, actual, VisibilityProvider.PUBLIC_VISIBILITY);
  }
}
