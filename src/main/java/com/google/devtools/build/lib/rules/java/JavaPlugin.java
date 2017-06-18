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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.syntax.Type;

/**
 * Implementation for the java_plugin rule.
 */
public class JavaPlugin implements RuleConfiguredTargetFactory {

  private final JavaSemantics semantics;

  protected JavaPlugin(JavaSemantics semantics) {
    this.semantics = semantics;
  }

  @Override
  public final ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
    JavaLibrary javaLibrary = new JavaLibrary(semantics);
    JavaCommon common = new JavaCommon(ruleContext, semantics);
    RuleConfiguredTargetBuilder builder =
        javaLibrary.init(ruleContext, common, true /* includeGeneratedExtensionRegistry */);
    if (builder == null) {
      return null;
    }
    ImmutableSet<String> processorClasses = getProcessorClasses(ruleContext);
    NestedSet<Artifact> processorClasspath = common.getRuntimeClasspath();
    ImmutableSet<String> apiGeneratingProcessorClasses;
    NestedSet<Artifact> apiGeneratingProcessorClasspath;
    if (ruleContext.attributes().get("generates_api", Type.BOOLEAN)) {
      apiGeneratingProcessorClasses = processorClasses;
      apiGeneratingProcessorClasspath = processorClasspath;
    } else {
      apiGeneratingProcessorClasses = ImmutableSet.of();
      apiGeneratingProcessorClasspath = NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
    }
    builder.addProvider(
        new JavaPluginInfoProvider(
            processorClasses,
            processorClasspath,
            apiGeneratingProcessorClasses,
            apiGeneratingProcessorClasspath));
    return builder.build();
  }

  /**
   * Returns the class that should be passed to javac in order to run the annotation processor this
   * class represents.
   */
  private static ImmutableSet<String> getProcessorClasses(RuleContext ruleContext) {
    if (ruleContext.getRule().isAttributeValueExplicitlySpecified("processor_class")) {
      return ImmutableSet.of(ruleContext.attributes().get("processor_class", Type.STRING));
    }
    return ImmutableSet.of();
  }
}
