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
package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredAspectFactory;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.SkylarkProviderValidationUtil;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.SkylarkAspect;
import com.google.devtools.build.lib.rules.SkylarkRuleContext;
import com.google.devtools.build.lib.rules.SkylarkRuleContext.Kind;
import com.google.devtools.build.lib.syntax.ClassObject.SkylarkClassObject;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalExceptionWithStackTrace;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.syntax.SkylarkType;

import java.util.Map;

/**
 * A factory for aspects that are defined in Skylark.
 */
public class SkylarkAspectFactory implements ConfiguredAspectFactory {

  private final SkylarkAspect skylarkAspect;

  public SkylarkAspectFactory(SkylarkAspect skylarkAspect) {
    this.skylarkAspect = skylarkAspect;
  }

  @Override
  public ConfiguredAspect create(
      ConfiguredTarget base, RuleContext ruleContext, AspectParameters parameters)
      throws InterruptedException {
    try (Mutability mutability = Mutability.create("aspect")) {
      SkylarkRuleContext skylarkRuleContext;
      try {
        skylarkRuleContext = new SkylarkRuleContext(ruleContext, Kind.ASPECT);
      } catch (EvalException e) {
        ruleContext.ruleError(e.getMessage());
        return null;
      }
      Environment env =
          Environment.builder(mutability)
              .setSkylark()
              .setGlobals(skylarkAspect.getFuncallEnv().getGlobals())
              .setEventHandler(ruleContext.getAnalysisEnvironment().getEventHandler())
              .build(); // NB: loading phase functions are not available: this is analysis already,
                        // so we do *not* setLoadingPhase().
      Object aspectSkylarkObject;
      try {
        aspectSkylarkObject =
            skylarkAspect
                .getImplementation()
                .call(
                    ImmutableList.<Object>of(base, skylarkRuleContext),
                    ImmutableMap.<String, Object>of(),
                    /*ast=*/ null,
                    env);

        if (ruleContext.hasErrors()) {
          return null;
        } else if (!(aspectSkylarkObject instanceof SkylarkClassObject)) {
          ruleContext.ruleError("Aspect implementation doesn't return a struct");
          return null;
        }

        ConfiguredAspect.Builder builder = new ConfiguredAspect.Builder(
            skylarkAspect.getName(), ruleContext);

        SkylarkClassObject struct = (SkylarkClassObject) aspectSkylarkObject;
        Location loc = struct.getCreationLoc();
        for (String key : struct.getKeys()) {
          if (key.equals("output_groups")) {
            addOutputGroups(struct.getValue(key), loc, builder);
          }
          builder.addSkylarkTransitiveInfo(key, struct.getValue(key), loc);
        }
        ConfiguredAspect configuredAspect = builder.build();
        SkylarkProviderValidationUtil.checkOrphanArtifacts(ruleContext);
        return configuredAspect;
      } catch (EvalException e) {
        addAspectToStackTrace(base, e);
        ruleContext.ruleError("\n" + e.print());
        return null;
      }

    }
  }

  private static void addOutputGroups(Object value, Location loc,
      ConfiguredAspect.Builder builder)
      throws EvalException {
    Map<String, SkylarkNestedSet> outputGroups = SkylarkType
        .castMap(value, String.class, SkylarkNestedSet.class, "output_groups");

    for (String outputGroup : outputGroups.keySet()) {
      SkylarkNestedSet objects = outputGroups.get(outputGroup);
      builder.addOutputGroup(outputGroup,
          SkylarkType.cast(objects, SkylarkNestedSet.class, Artifact.class, loc,
              "Output group '%s'", outputGroup).getSet(Artifact.class));
    }
  }

  private void addAspectToStackTrace(ConfiguredTarget base, EvalException e) {
    if (e instanceof EvalExceptionWithStackTrace) {
      ((EvalExceptionWithStackTrace) e)
          .registerPhantomFuncall(
              String.format("%s(...)", skylarkAspect.getName()),
              base.getTarget().getAssociatedRule().getLocation(),
              skylarkAspect.getImplementation());
    }
  }
}
