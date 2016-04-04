// Copyright 2016 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkState;
import static com.google.devtools.build.lib.Constants.TOOLS_REPOSITORY;
import static com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition.HOST;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredAspectFactory;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.NativeAspectClass.NativeAspectFactory;
import com.google.devtools.build.lib.rules.java.JavaCommon;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaRuntimeJarProvider;

/**
 * Aspect to {@link DexArchiveProvider build .dex Archives} from Jars.
 */
public final class DexArchiveAspect implements NativeAspectFactory, ConfiguredAspectFactory {
  private static final String NAME = "DexArchiveAspect";
  private static final ImmutableList<String> TRANSITIVE_ATTRIBUTES =
      ImmutableList.of("deps", "exports", "runtime_deps");

  @Override
  public AspectDefinition getDefinition(AspectParameters params) {
    AspectDefinition.Builder result = new AspectDefinition.Builder(NAME)
        // Actually we care about JavaRuntimeJarProvider, but rules don't advertise that provider.
        .requireProvider(JavaCompilationArgsProvider.class)
        .add(attr("$dexbuilder", LABEL).cfg(HOST).exec()
        // Parse label here since we don't have RuleDefinitionEnvironment.getLabel like in a rule
            .value(Label.parseAbsoluteUnchecked(TOOLS_REPOSITORY + "//tools/android:dexbuilder")))
        .requiresConfigurationFragments(AndroidConfiguration.class);
    for (String attr : TRANSITIVE_ATTRIBUTES) {
      result.attributeAspect(attr, DexArchiveAspect.class);
    }
    return result.build();
  }

  @Override
  public ConfiguredAspect create(ConfiguredTarget base, RuleContext ruleContext,
      AspectParameters params) throws InterruptedException {
    if (AndroidCommon.getAndroidConfig(ruleContext).getIncrementalDexingBinaries().isEmpty()) {
      // Dex archives will never be used, so don't bother setting them up.
      return new ConfiguredAspect.Builder(NAME, ruleContext).build();
    }
    checkState(base.getProvider(DexArchiveProvider.class) == null,
        "dex archive natively generated: %s", ruleContext.getLabel());

    if (JavaCommon.isNeverLink(ruleContext)) {
      return new ConfiguredAspect.Builder(NAME, ruleContext)
          .addProvider(DexArchiveProvider.class, DexArchiveProvider.NEVERLINK)
          .build();
    }

    DexArchiveProvider.Builder result = createArchiveProviderBuilderFromDeps(ruleContext);
    JavaRuntimeJarProvider jarProvider = base.getProvider(JavaRuntimeJarProvider.class);
    if (jarProvider != null) {
      for (Artifact jar : jarProvider.getRuntimeJars()) {
        Artifact dexArchive = createDexArchiveAction(ruleContext, jar);
        result.addDexArchive(dexArchive, jar);
      }
    }
    return new ConfiguredAspect.Builder(NAME, ruleContext)
        .addProvider(DexArchiveProvider.class, result.build())
        .build();
  }

  private static DexArchiveProvider.Builder createArchiveProviderBuilderFromDeps(
      RuleContext ruleContext) {
    DexArchiveProvider.Builder result = new DexArchiveProvider.Builder();
    for (String attr : TRANSITIVE_ATTRIBUTES) {
      if (ruleContext.getRule().getRuleClassObject().hasAttr(attr, LABEL_LIST)) {
        result.addTransitiveProviders(
            ruleContext.getPrerequisites(attr, Mode.TARGET, DexArchiveProvider.class));
      }
    }
    return result;
  }

  private static Artifact createDexArchiveAction(RuleContext ruleContext, Artifact jar) {
    Artifact result = AndroidBinary.getDxArtifact(ruleContext, jar.getFilename() + ".dex.zip");
    createDexArchiveAction(ruleContext, jar, result);
    return result;
  }

  // Package-private methods for use in AndroidBinary

  static void createDexArchiveAction(RuleContext ruleContext, Artifact jar, Artifact dexArchive) {
    SpawnAction.Builder dexbuilder = new SpawnAction.Builder()
        .setExecutable(ruleContext.getExecutablePrerequisite("$dexbuilder", Mode.HOST))
        .addArgument("--input_jar")
        .addInputArgument(jar)
        .addArgument("--output_zip")
        .addOutputArgument(dexArchive)
        .setMnemonic("DexBuilder")
        .setProgressMessage("Dexing " + jar.prettyPrint());
    if (ruleContext.getConfiguration().isCodeCoverageEnabled()) {
      // Match what we do in AndroidCommon.createDexAction
      dexbuilder.addArgument("--nolocals"); // TODO(bazel-team): Still needed? See createDexAction
    }
    ruleContext.registerAction(dexbuilder.build(ruleContext));
  }
}
