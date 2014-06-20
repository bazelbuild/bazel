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

package com.google.devtools.build.lib.view.test;

import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.packages.TestSize;
import com.google.devtools.build.lib.packages.TestTimeout;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.view.RuleContext;

import java.util.List;

/**
 * Container for test target properties available to the
 * TestRunnerAction instance.
 */
public class TestTargetProperties {

  /**
   * Resources used by local tests of various sizes.
   */
  private static final ResourceSet SMALL_RESOURCES = new ResourceSet(20, 0.9, 0.05);
  private static final ResourceSet MEDIUM_RESOURCES = new ResourceSet(100, 0.9, 0.1);
  private static final ResourceSet LARGE_RESOURCES = new ResourceSet(300, 0.8, 0.3);
  private static final ResourceSet ENORMOUS_RESOURCES = new ResourceSet(800, 0.7, 0.4);

  private static ResourceSet getResourceSetFromSize(TestSize size) {
    switch (size) {
      case SMALL: return SMALL_RESOURCES;
      case MEDIUM: return MEDIUM_RESOURCES;
      case LARGE: return LARGE_RESOURCES;
      default: return ENORMOUS_RESOURCES;
    }
  }

  private final TestSize size;
  private final TestTimeout timeout;
  private final List<String> tags;
  private final boolean isLocal;
  private final boolean isFlaky;
  private final boolean isExternal;
  private final ResourceSet resources;
  private final String language;
  private final ImmutableSet<String> requirements;

  /**
   * Creates test target properties instance. Constructor expects that it
   * will be called only for test configured targets.
   */
  TestTargetProperties(RuleContext ruleContext,
      ExecutionRequirementProvider executionRequirements) {
    Rule rule = ruleContext.getRule();

    Preconditions.checkState(TargetUtils.isTestRule(rule));
    size = TestSize.getTestSize(rule);
    timeout = TestTimeout.getTestTimeout(rule);
    tags = ruleContext.attributes().get("tags", Type.STRING_LIST);
    isLocal = TargetUtils.isLocalTestRule(rule) || TargetUtils.isExclusiveTestRule(rule);

    // We need to use method on ruleConfiguredTarget to perform validation.
    isFlaky = ruleContext.attributes().get("flaky", Type.BOOLEAN);
    isExternal = TargetUtils.isExternalTestRule(rule);

    if (executionRequirements != null) {
      ImmutableSet.Builder<String> builder = ImmutableSet.builder();
      builder.addAll(TargetUtils.constraintKeywords(rule));
      builder.addAll(executionRequirements.getRequirements());
      requirements = builder.build();
    } else {
      requirements = TargetUtils.constraintKeywords(rule);
    }

    language = TargetUtils.getRuleLanguage(rule);
    resources = TestTargetProperties.getResourceSetFromSize(size);
  }

  public TestSize getSize() {
    return size;
  }

  public TestTimeout getTimeout() {
    return timeout;
  }

  public List<String> getTags() {
    return tags;
  }

  public boolean isLocal() {
    return isLocal;
  }

  public boolean isFlaky() {
    return isFlaky;
  }

  public boolean isExternal() {
    return isExternal;
  }

  public ResourceSet getLocalResourceUsage() {
    return resources;
  }

  /**
   * Returns a set of strings describing constraints on the execution environment.
   */
  public ImmutableSet<String> getRequirements() {
    return requirements;
  }

  /**
   * Returns a map of execution info. Includes getRequirements() as keys.
   */
  public ImmutableMap<String, String> getExecutionInfo() {
    ImmutableMap.Builder<String, String> builder = ImmutableMap.builder();
    for (String key : requirements) {
      builder.put(key, "");
    }
    return builder.build();
  }

  public String getLanguage() {
    return language;
  }
}
