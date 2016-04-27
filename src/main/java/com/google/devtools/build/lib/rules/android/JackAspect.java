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
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.NativeAspectClass;
import com.google.devtools.build.lib.rules.android.AndroidRuleClasses.AndroidSdkLabel;
import com.google.devtools.build.lib.rules.java.JavaCommon;
import com.google.devtools.build.lib.rules.java.JavaSourceInfoProvider;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;

/** Aspect to provide Jack support to rules which have java sources. */
public final class JackAspect extends NativeAspectClass implements ConfiguredAspectFactory {
  public static final String NAME = "JackAspect";
  private final String toolsRepository;

  /**
   * Creates a JackAspect using the provided tools repository path.
   *
   * @param toolsRepository the path to the tools repository
   */
  public JackAspect(String toolsRepository) {
    this.toolsRepository = toolsRepository;
  }

  @Override
  public AspectDefinition getDefinition(AspectParameters params) {
    Label androidSdk;
    try {
      androidSdk = Label.parseAbsolute(toolsRepository + AndroidRuleClasses.DEFAULT_SDK);
    } catch (LabelSyntaxException e) {
      throw new IllegalStateException(e);
    }

    return new AspectDefinition.Builder("JackAspect")
        .requireProvider(JavaSourceInfoProvider.class)
        .add(attr(":android_sdk", LABEL)
              .allowedRuleClasses("android_sdk")
              .value(new AndroidSdkLabel(androidSdk)))
        .attributeAspect("deps", this)
        .attributeAspect("exports", this)
        .attributeAspect("runtime_deps", this)
        .requiresConfigurationFragments(AndroidConfiguration.class)
        .build();
  }

  @Override
  public ConfiguredAspect create(
      ConfiguredTarget base, RuleContext ruleContext, AspectParameters params) {
    JavaSourceInfoProvider sourceProvider = base.getProvider(JavaSourceInfoProvider.class);
    PathFragment rulePath = ruleContext.getLabel().toPathFragment();
    PathFragment jackLibraryPath = rulePath.replaceName("lib" + rulePath.getBaseName() + ".jack");
    Artifact jackLibraryOutput =
        ruleContext
            .getAnalysisEnvironment()
            .getDerivedArtifact(jackLibraryPath, ruleContext.getBinOrGenfilesDirectory());
    if (!AndroidSdkProvider.verifyPresence(ruleContext)) {
      return null;
    }
    AndroidSdkProvider androidSdk = AndroidSdkProvider.fromRuleContext(ruleContext);
    JackCompilationHelper jackHelper =
        new JackCompilationHelper.Builder()
            // blaze infrastructure
            .setRuleContext(ruleContext)
            // configuration
            .setOutputArtifact(jackLibraryOutput)
            // tools
            .setJackBinary(androidSdk.getJack())
            .setJillBinary(androidSdk.getJill())
            .setResourceExtractorBinary(androidSdk.getResourceExtractor())
            .setJackBaseClasspath(androidSdk.getJavaBaseClasspathForJack())
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
    return new ConfiguredAspect.Builder(NAME, ruleContext)
        .addProvider(JackLibraryProvider.class, result)
        .build();
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
