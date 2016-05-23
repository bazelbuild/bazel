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
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
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
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
    JavaLibrary javaLibrary = new JavaLibrary(semantics);
    JavaCommon common = new JavaCommon(ruleContext, semantics);
    RuleConfiguredTargetBuilder builder = javaLibrary.init(ruleContext, common);
    if (builder == null) {
      return null;
    }
    builder.add(JavaPluginInfoProvider.class, new JavaPluginInfoProvider(
        getProcessorClasses(ruleContext), common.getRuntimeClasspath()));
    return builder.build();
  }

  /**
   * Returns the class that should be passed to javac in order
   * to run the annotation processor this class represents.
   */
  private ImmutableSet<String> getProcessorClasses(RuleContext ruleContext) {
    if (ruleContext.getRule().isAttributeValueExplicitlySpecified("processor_class")) {
      return ImmutableSet.of(ruleContext.attributes().get("processor_class", Type.STRING));
    }
    return ImmutableSet.of();
  }
}
