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
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.AnalysisEnvironment;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredAspectFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.StarlarkProviderValidationUtil;
import com.google.devtools.build.lib.analysis.skylark.StarlarkRuleConfiguredTargetUtil;
import com.google.devtools.build.lib.analysis.skylark.StarlarkRuleContext;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.BazelStarlarkContext;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.StarlarkDefinedAspect;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.packages.StructProvider;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.syntax.Dict;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalExceptionWithStackTrace;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.lib.syntax.StarlarkValue;
import java.util.Map;

/** A factory for aspects that are defined in Starlark. */
public class SkylarkAspectFactory implements ConfiguredAspectFactory {
  private final StarlarkDefinedAspect skylarkAspect;

  SkylarkAspectFactory(StarlarkDefinedAspect skylarkAspect) {
    this.skylarkAspect = skylarkAspect;
  }

  @Override
  public ConfiguredAspect create(
      ConfiguredTargetAndData ctadBase,
      RuleContext ruleContext,
      AspectParameters parameters,
      String toolsRepository)
      throws InterruptedException, ActionConflictException {
    StarlarkRuleContext starlarkRuleContext = null;
    try (Mutability mutability = Mutability.create("aspect")) {
      AspectDescriptor aspectDescriptor =
          new AspectDescriptor(skylarkAspect.getAspectClass(), parameters);
      AnalysisEnvironment analysisEnv = ruleContext.getAnalysisEnvironment();
      try {
        starlarkRuleContext =
            new StarlarkRuleContext(
                ruleContext, aspectDescriptor, analysisEnv.getSkylarkSemantics());
      } catch (EvalException | RuleErrorException e) {
        ruleContext.ruleError(e.getMessage());
        return null;
      }
      StarlarkThread thread =
          StarlarkThread.builder(mutability)
              .setSemantics(analysisEnv.getSkylarkSemantics())
              .build();
      thread.setPrintHandler(Event.makeDebugPrintHandler(analysisEnv.getEventHandler()));

      new BazelStarlarkContext(
              BazelStarlarkContext.Phase.ANALYSIS,
              toolsRepository,
              /*fragmentNameToClass=*/ null,
              ruleContext.getRule().getPackage().getRepositoryMapping(),
              ruleContext.getSymbolGenerator(),
              ruleContext.getLabel())
          .storeInThread(thread);

      try {
        Object aspectSkylarkObject =
            Starlark.call(
                thread,
                skylarkAspect.getImplementation(),
                /*args=*/ ImmutableList.of(ctadBase.getConfiguredTarget(), starlarkRuleContext),
                /*kwargs=*/ ImmutableMap.of());

        // If allowing analysis failures, targets should be created somewhat normally, and errors
        // will be propagated via a hook elsewhere as AnalysisFailureInfo.
        boolean allowAnalysisFailures = ruleContext.getConfiguration().allowAnalysisFailures();

        if (ruleContext.hasErrors() && !allowAnalysisFailures) {
          return null;
        } else if (!(aspectSkylarkObject instanceof StructImpl)
            && !(aspectSkylarkObject instanceof Iterable)
            && !(aspectSkylarkObject instanceof Info)) {
          ruleContext.ruleError(
              String.format(
                  "Aspect implementation should return a struct, a list, or a provider "
                      + "instance, but got %s",
                  Starlark.type(aspectSkylarkObject)));
          return null;
        }
        return createAspect(aspectSkylarkObject, ruleContext);
      } catch (EvalException e) {
        addAspectToStackTrace(ctadBase.getTarget(), e);
        ruleContext.ruleError("\n" + e.print());
        return null;
      }
    } finally {
      if (starlarkRuleContext != null) {
        starlarkRuleContext.nullify();
      }
    }
  }

  private static ConfiguredAspect createAspect(Object aspectSkylarkObject, RuleContext ruleContext)
      throws EvalException, ActionConflictException {

    ConfiguredAspect.Builder builder = new ConfiguredAspect.Builder(ruleContext);

    if (aspectSkylarkObject instanceof Iterable) {
      addDeclaredProviders(builder, (Iterable) aspectSkylarkObject);
    } else {
      // Either an old-style struct or a single declared provider (not in a list)
      Info info = (Info) aspectSkylarkObject;
      if (info.getProvider().getKey().equals(StructProvider.STRUCT.getKey())) {
        // Old-style struct, that may contain declared providers.
        StructImpl struct = (StructImpl) aspectSkylarkObject;
        for (String field : struct.getFieldNames()) {
          if (field.equals("output_groups")) {
            addOutputGroups(struct.getValue(field), builder);
          } else if (field.equals("providers")) {
            Object providers = struct.getValue(field);
            // TODO(adonovan): can we be more specific than iterable, and use Sequence.cast?
            if (!(providers instanceof Iterable)) {
              throw Starlark.errorf(
                  "The value for \"providers\" should be a list of declared providers, "
                      + "got %s instead",
                  Starlark.type(providers));
            }
            addDeclaredProviders(builder, (Iterable<?>) providers);
          } else {
            builder.addSkylarkTransitiveInfo(field, struct.getValue(field));
          }
        }
      } else {
        builder.addSkylarkDeclaredProvider(info);
      }
    }

    ConfiguredAspect configuredAspect = builder.build();
    StarlarkProviderValidationUtil.validateArtifacts(ruleContext);
    return configuredAspect;
  }

  private static void addDeclaredProviders(
      ConfiguredAspect.Builder builder, Iterable<?> aspectSkylarkObject) throws EvalException {
    int i = 0;
    for (Object o : aspectSkylarkObject) {
      if (!(o instanceof Info)) {
        throw Starlark.errorf(
            "A return value of an aspect implementation function should be "
                + "a sequence of declared providers, instead got a %s at index %d",
            Starlark.type(o), i);
      }
      builder.addSkylarkDeclaredProvider((Info) o);
      i++;
    }
  }

  private static void addOutputGroups(Object outputGroups, ConfiguredAspect.Builder builder)
      throws EvalException {
    for (Map.Entry<String, StarlarkValue> entry :
        Dict.cast(outputGroups, String.class, StarlarkValue.class, "output_groups").entrySet()) {
      builder.addOutputGroup(
          entry.getKey(),
          StarlarkRuleConfiguredTargetUtil.convertToOutputGroupValue(
              entry.getKey(), entry.getValue()));
    }
  }

  private void addAspectToStackTrace(Target base, EvalException e) {
    if (e instanceof EvalExceptionWithStackTrace) {
      ((EvalExceptionWithStackTrace) e)
          .registerPhantomCall(
              String.format("%s(...)", skylarkAspect.getName()),
              base.getAssociatedRule().getLocation(),
              skylarkAspect.getImplementation());
    }
  }
}
