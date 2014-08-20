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

import static com.google.devtools.build.lib.syntax.SkylarkFunction.cast;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.syntax.ClassObject.SkylarkClassObject;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Function;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.view.ConfiguredTarget;
import com.google.devtools.build.lib.view.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.view.RuleContext;
import com.google.devtools.build.lib.view.RunfilesProvider;
import com.google.devtools.build.lib.view.RunfilesSupport;

/**
 * A helper class to build Rule Configured Targets via runtime loaded rule implementations
 * defined using the Skylark Build Extension Language. This is experimental code.
 */
public final class SkylarkRuleConfiguredTargetBuilder {

  /**
   * Create a Rule Configured Target from the ruleContext and the ruleImplementation.
   */
  public static ConfiguredTarget buildRule(RuleContext ruleContext,
      Function ruleImplementation) {
    String expectError = ruleContext.attributes().get("expect_failure", Type.STRING);
    try {
      SkylarkRuleContext skylarkRuleContext = new SkylarkRuleContext(ruleContext);
      Environment env =
          SkylarkRuleImplementationFunctions.getNewEnvironment(skylarkRuleContext);

      Object target = ruleImplementation.call(ImmutableList.<Object>of(skylarkRuleContext),
          ImmutableMap.<String, Object>of(), null, env);

      if (ruleContext.hasErrors()) {
        return null;
      } else if (!(target instanceof SkylarkClassObject)) {
        ruleContext.ruleError("Rule implementation doesn't return a struct");
        return null;
      } else if (!expectError.isEmpty()) {
        ruleContext.ruleError("Expected error not found: " + expectError);
        return null;
      }
      return createTarget(ruleContext, (SkylarkClassObject) target);

    } catch (InterruptedException e) {
      ruleContext.ruleError(e.getMessage());
      return null;
    } catch (EvalException e) {
      // If the error was expected, return an empty target.
      if (!expectError.isEmpty() && e.getMessage().matches(expectError)) {
        return new com.google.devtools.build.lib.view.RuleConfiguredTargetBuilder(ruleContext)
            .add(RunfilesProvider.class, RunfilesProvider.EMPTY)
            .build();
      }
      ruleContext.ruleError("\n" + e.print());
      return null;
    }
  }

  private static ConfiguredTarget createTarget(
      RuleContext ruleContext, SkylarkClassObject struct) throws EvalException {
    try {
      RuleConfiguredTargetBuilder builder = new RuleConfiguredTargetBuilder(ruleContext);
      // Every target needs runfiles provider by default.
      builder.add(RunfilesProvider.class, RunfilesProvider.EMPTY);
      Location loc = struct.getCreationLoc();
      for (String key : struct.getKeys()) {
        if (key.equals("files_to_build")) {
          builder.setFilesToBuild(cast(struct.getValue("files_to_build"),
              SkylarkNestedSet.class, "files_to_build", loc).getSet(Artifact.class));
        } else if (key.equals("runfiles")) {
          builder.add(RunfilesProvider.class, cast(struct.getValue("runfiles"),
              RunfilesProvider.class, "runfiles", loc));
        } else if (key.equals("runfiles_support")) {
          RunfilesSupport runfilesSupport = cast(struct.getValue("runfiles_support"),
              RunfilesSupport.class, "runfiles support", loc);
          builder.setRunfilesSupport(runfilesSupport, runfilesSupport.getExecutable());
        } else if (key.equals("executable")) {
          builder.setRunfilesSupport(null,
              cast(struct.getValue("executable"), Artifact.class, "executable", loc));
        } else {
          builder.addSkylarkTransitiveInfo(key, struct.getValue(key));
        }
      } 
      return builder.build();
    } catch (IllegalArgumentException e) {
      throw new EvalException(struct.getCreationLoc(), e.getMessage());
    }
  }
}
