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
package com.google.devtools.build.lib.rules.java;

import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;

/** Implementation for the java_plugin rule. */
public class JavaPlugin implements RuleConfiguredTargetFactory {

  private final JavaSemantics semantics;

  protected JavaPlugin(JavaSemantics semantics) {
    this.semantics = semantics;
  }

  @Override
  public final ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    return new JavaLibrary(semantics)
        .init(
            ruleContext,
            new JavaCommon(ruleContext, semantics),
            /* includeGeneratedExtensionRegistry = */ true,
            /* isJavaPluginRule= */ true);
  }
}
