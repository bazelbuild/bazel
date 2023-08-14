// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.android;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredAspectFactory;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.NativeAspectClass;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.StarlarkProviderIdentifier;
import com.google.devtools.build.lib.rules.java.JavaCommon;
import com.google.devtools.build.lib.rules.java.JavaInfo;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;

/**
 * An aspect that collects neverlink libraries in the transitive closure.
 *
 * <p>Used for determining the -libraryjars argument of the Proguard invocation in Android binaries.
 *
 * <p>One would think that using the compile time classpath would be enough, but alas, those are
 * ijars,
 */
public class AndroidNeverlinkAspect extends NativeAspectClass implements ConfiguredAspectFactory {
  public static final String NAME = "AndroidNeverlinkAspect";
  private static final ImmutableList<String> ATTRIBUTES =
      ImmutableList.of(
          "deps", "exports", "runtime_deps", "binary_under_test", "$instrumentation_test_runner");

  @Override
  @Nullable
  public ConfiguredAspect create(
      Label targetLabel,
      ConfiguredTarget ct,
      RuleContext ruleContext,
      AspectParameters parameters,
      RepositoryName toolsRepository)
      throws ActionConflictException, InterruptedException {
    if (!JavaCommon.getConstraints(ruleContext).contains("android")
        && !ruleContext.getRule().getRuleClass().startsWith("android_")) {
      return new ConfiguredAspect.Builder(ruleContext).build();
    }

    List<TransitiveInfoCollection> deps = new ArrayList<>();

    // This is probably an overestimate, but it's fine -- Proguard doesn't care if there are more
    // jars in -libraryjars than is required. The alternative would be somehow getting
    // JavaCommon.getDependencies() here, which would be fugly.
    for (String attribute : ATTRIBUTES) {
      if (!ruleContext.getRule().getRuleClassObject().hasAttr(attribute, BuildType.LABEL_LIST)) {
        continue;
      }

      deps.addAll(ruleContext.getPrerequisites(attribute));
    }

    NestedSetBuilder<Artifact> runtimeJars = NestedSetBuilder.naiveLinkOrder();
    try {
      runtimeJars.addAll(JavaInfo.getJavaInfo(ct).getDirectRuntimeJars());
    } catch (RuleErrorException e) {
      ruleContext.ruleError(e.getMessage());
      return null;
    }
    AndroidLibraryResourceClassJarProvider provider =
        AndroidLibraryResourceClassJarProvider.getProvider(ct);
    if (provider != null) {
      runtimeJars.addTransitive(provider.getResourceClassJars());
    }
    NestedSet<Artifact> neverLinkLibraries;
    try {
      neverLinkLibraries =
          AndroidCommon.collectTransitiveNeverlinkLibraries(ruleContext, deps, runtimeJars.build());
    } catch (RuleErrorException e) {
      ruleContext.ruleError(e.getMessage());
      return null;
    }
    return new ConfiguredAspect.Builder(ruleContext)
        .addNativeDeclaredProvider(AndroidNeverLinkLibrariesProvider.create(neverLinkLibraries))
        .build();
  }

  @Override
  public AspectDefinition getDefinition(AspectParameters aspectParameters) {
    AspectDefinition.Builder builder = new AspectDefinition.Builder(this);
    for (String attribute : ATTRIBUTES) {
      builder.propagateAlongAttribute(attribute);
    }

    return builder
        .requireStarlarkProviders(StarlarkProviderIdentifier.forKey(JavaInfo.PROVIDER.getKey()))
        .requireStarlarkProviders(
            StarlarkProviderIdentifier.forKey(JavaInfo.PROVIDER.getKey()),
            StarlarkProviderIdentifier.forKey(
                AndroidLibraryResourceClassJarProvider.PROVIDER.getKey()))
        .requiresConfigurationFragments()
        .build();
  }
}
