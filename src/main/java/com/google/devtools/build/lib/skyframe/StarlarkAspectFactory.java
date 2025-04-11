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

import com.google.devtools.build.lib.actions.ActionConflictException;
import com.google.devtools.build.lib.analysis.CachingAnalysisEnvironment;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredAspectFactory;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RequiredConfigFragmentsProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.StarlarkProviderValidationUtil;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.StarlarkDefinedAspect;
import com.google.devtools.build.lib.packages.StarlarkInfo;
import com.google.devtools.build.lib.packages.StructProvider;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;

/** A factory for aspects that are defined in Starlark. */
public class StarlarkAspectFactory implements ConfiguredAspectFactory {
  private final StarlarkDefinedAspect starlarkAspect;

  StarlarkAspectFactory(StarlarkDefinedAspect starlarkAspect) {
    this.starlarkAspect = starlarkAspect;
  }

  @Override
  public ConfiguredAspect create(
      Label targetLabel,
      ConfiguredTarget ct,
      RuleContext ruleContext,
      AspectParameters parameters,
      RepositoryName toolsRepository)
      throws InterruptedException, ActionConflictException {
    RequiredConfigFragmentsProvider requiredConfigFragments;
    Object aspectStarlarkObject;
    try {
      StarlarkRuleContext ctx = ruleContext.initStarlarkRuleContext();
      aspectStarlarkObject =
          Starlark.positionalOnlyCall(
              ruleContext.getStarlarkThread(), starlarkAspect.getImplementation(), ct, ctx);
    } catch (RuleErrorException e) {
      // TODO(bazel-team): Doesn't this double-log the message, if the exception was created by
      // RuleContext#throwWithRuleError?
      ruleContext.ruleError(e.getMessage());
      return errorConfiguredAspect(ruleContext);
    } catch (Starlark.UncheckedEvalException ex) {
      // MissingDepException is expected to transit through Starlark execution.
      throw ex.getCause() instanceof CachingAnalysisEnvironment.MissingDepException
          ? (CachingAnalysisEnvironment.MissingDepException) ex.getCause()
          : ex;
    } catch (EvalException e) {
      ruleContext.ruleError("\n" + e.getMessageWithStack());
      return errorConfiguredAspect(ruleContext);
    } finally {
      requiredConfigFragments = ruleContext.getRequiredConfigFragments();
      // freeze mutability to allow optimizing StarlarkInfo instances
      ruleContext.close();
    }
    // If allowing analysis failures, targets should be created somewhat normally, and errors
    // will be propagated via a hook elsewhere as AnalysisFailureInfo.
    boolean allowAnalysisFailures = ruleContext.getConfiguration().allowAnalysisFailures();

    if (ruleContext.hasErrors() && !allowAnalysisFailures) {
      return errorConfiguredAspect(ruleContext, requiredConfigFragments);
    } else if (aspectStarlarkObject instanceof Info info
        && info.getProvider().getKey().equals(StructProvider.STRUCT.getKey())) {
      ruleContext.ruleError(
          "Returning a struct from an aspect implementation function is deprecated.");
    } else if (!(aspectStarlarkObject instanceof Iterable)
        && !(aspectStarlarkObject instanceof Info)) {
      ruleContext.ruleError(
          String.format(
              "Aspect implementation should return a list, or a provider instance, but got %s",
              Starlark.type(aspectStarlarkObject)));
      return errorConfiguredAspect(ruleContext, requiredConfigFragments);
    }
    try {
      return createAspect(aspectStarlarkObject, ruleContext, requiredConfigFragments);
    } catch (EvalException e) {
      ruleContext.ruleError("\n" + e.getMessageWithStack());
      return errorConfiguredAspect(ruleContext, requiredConfigFragments);
    }
  }

  private static ConfiguredAspect errorConfiguredAspect(RuleContext ruleContext)
      throws ActionConflictException, InterruptedException {
    return errorConfiguredAspect(ruleContext, ruleContext.getRequiredConfigFragments());
  }

  private static ConfiguredAspect errorConfiguredAspect(
      RuleContext ruleContext, RequiredConfigFragmentsProvider requiredConfigFragmentsProvider)
      throws ActionConflictException, InterruptedException {
    return ConfiguredTargetFactory.erroredConfiguredAspect(
        ruleContext, requiredConfigFragmentsProvider);
  }

  private static ConfiguredAspect createAspect(
      Object aspectStarlarkObject,
      RuleContext ruleContext,
      @Nullable RequiredConfigFragmentsProvider requiredConfigFragments)
      throws EvalException, ActionConflictException, InterruptedException {

    ConfiguredAspect.Builder builder = new ConfiguredAspect.Builder(ruleContext);
    if (requiredConfigFragments != null) {
      builder.addProvider(requiredConfigFragments);
    }
    // not instanceof Info, because OutputGroupInfo is both Iterable and Info
    if (!(aspectStarlarkObject instanceof Info)
        && aspectStarlarkObject instanceof Iterable<?> iterable) {
      addDeclaredProviders(builder, iterable);
    } else {
      // A single declared provider (not in a list)
      Info info = (Info) aspectStarlarkObject;
      if (info instanceof StarlarkInfo starlarkInfo) {
        info = starlarkInfo.unsafeOptimizeMemoryLayout();
      }
      builder.addStarlarkDeclaredProvider(info);
    }

    ConfiguredAspect configuredAspect = builder.build();
    StarlarkProviderValidationUtil.validateArtifacts(ruleContext);
    return configuredAspect;
  }

  private static void addDeclaredProviders(
      ConfiguredAspect.Builder builder, Iterable<?> aspectStarlarkObject) throws EvalException {
    int i = 0;
    for (Object o : aspectStarlarkObject) {
      if (!(o instanceof Info)) {
        throw Starlark.errorf(
            "A return value of an aspect implementation function should be "
                + "a sequence of declared providers, instead got a %s at index %d",
            Starlark.type(o), i);
      }
      if (o instanceof StarlarkInfo starlarkInfo) {
        o = starlarkInfo.unsafeOptimizeMemoryLayout();
      }
      builder.addStarlarkDeclaredProvider((Info) o);
      i++;
    }
  }
}
