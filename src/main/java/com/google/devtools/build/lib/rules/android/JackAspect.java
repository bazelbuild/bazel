// Copyright 2015 Google Inc. All rights reserved.
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

import static com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition.HOST;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.Type.LABEL;
import static com.google.devtools.build.lib.packages.Type.LABEL_LIST;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.Aspect;
import com.google.devtools.build.lib.analysis.ConfiguredAspectFactory;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.rules.java.JavaCommon;
import com.google.devtools.build.lib.rules.java.JavaSourceInfoProvider;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.syntax.Label.SyntaxException;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.List;

/** Aspect to provide Jack support to rules which have java sources. */
public final class JackAspect implements ConfiguredAspectFactory {
  private Label getLabel(String path) {
    try {
      return Label.parseAbsolute(path);
    } catch (SyntaxException ex) {
      throw new IllegalArgumentException(ex);
    }
  }

  @Override
  public AspectDefinition getDefinition() {
    return new AspectDefinition.Builder("JackAspect")
        .requireProvider(JavaSourceInfoProvider.class)
        .add(attr("$jack", LABEL).cfg(HOST).exec().value(getLabel("//tools/android/jack:jack")))
        .add(attr("$jill", LABEL).cfg(HOST).exec().value(getLabel("//tools/android/jack:jill")))
        .add(
            attr("$resource_extractor", LABEL)
                .cfg(HOST)
                .exec()
                .value(getLabel("//tools/android/jack:resource_extractor")))
        .add(
            attr("$java_jack", LABEL)
                .cfg(HOST)
                .value(getLabel("//tools/android/jack:android_jack")))
        .attributeAspect("deps", JackAspect.class)
        .attributeAspect("exports", JackAspect.class)
        .attributeAspect("runtime_deps", JackAspect.class)
        .build();
  }

  @Override
  public Aspect create(ConfiguredTarget base, RuleContext ruleContext) {
    JavaSourceInfoProvider sourceProvider = base.getProvider(JavaSourceInfoProvider.class);

    PathFragment rulePath = ruleContext.getLabel().toPathFragment();

    PathFragment jackLibraryPath = rulePath.replaceName("lib" + rulePath.getBaseName() + ".jack");

    Artifact jackLibraryOutput =
        ruleContext
            .getAnalysisEnvironment()
            .getDerivedArtifact(jackLibraryPath, ruleContext.getBinOrGenfilesDirectory());

    JackCompilationHelper jackHelper =
        new JackCompilationHelper.Builder()
            // blaze infrastructure
            .setRuleContext(ruleContext)
            // configuration
            .setOutputArtifact(jackLibraryOutput)
            // tools
            .setAndroidBaseLibraryForJack(ruleContext.getHostPrerequisiteArtifact("$java_jack"))
            // sources
            .addJavaSources(sourceProvider.getSourceFiles())
            .addSourceJars(sourceProvider.getSourceJars())
            .addCompiledJars(sourceProvider.getJarFiles())
            .addResources(sourceProvider.getResources())
            .addProcessorNames(sourceProvider.getProcessorNames())
            .addProcessorClasspathJars(sourceProvider.getProcessorPath())
            // dependencies
            .addExports(getPotentialDependency(ruleContext, "exports"))
            .addDeps(getPotentialDependency(ruleContext, "deps"))
            .addRuntimeDeps(getPotentialDependency(ruleContext, "runtime_deps"))
            .build();

    JackLibraryProvider result =
        JavaCommon.isNeverLink(ruleContext)
            ? jackHelper.compileAsNeverlinkLibrary()
            : jackHelper.compileAsLibrary();

    return new Aspect.Builder().addProvider(JackLibraryProvider.class, result).build();
  }

  /** Gets a list of targets on the given LABEL_LIST attribute if it exists, else an empty list. */
  private static List<? extends TransitiveInfoCollection> getPotentialDependency(
      RuleContext context, String attribute) {
    if (!context.getRule().getRuleClassObject().hasAttr(attribute, LABEL_LIST)) {
      return ImmutableList.of();
    }
    return context.getPrerequisites(attribute, Mode.TARGET);
  }
}
