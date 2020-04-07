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

package com.google.devtools.build.lib.analysis.test;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.ExecutionRequirements.ParseableRequirement.ValidationException;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.packages.TestSize;
import com.google.devtools.build.lib.packages.TestTimeout;
import com.google.devtools.build.lib.packages.Type;
import java.util.List;
import java.util.Map;

/**
 * Container for test target properties available to the
 * TestRunnerAction instance.
 */
public class TestTargetProperties {

  /**
   * Resources used by local tests of various sizes.
   *
   * <p>When changing these values, remember to update the documentation at
   * attributes/test/size.html.
   */
  private static final ResourceSet SMALL_RESOURCES = ResourceSet.create(20, 1, 1);
  private static final ResourceSet MEDIUM_RESOURCES = ResourceSet.create(100, 1, 1);
  private static final ResourceSet LARGE_RESOURCES = ResourceSet.create(300, 1, 1);
  private static final ResourceSet ENORMOUS_RESOURCES = ResourceSet.create(800, 1, 1);
  private static final ResourceSet LOCAL_TEST_JOBS_BASED_RESOURCES =
      ResourceSet.createWithLocalTestCount(1);

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
  private final boolean isRemotable;
  private final boolean isFlaky;
  private final boolean isExternal;
  private final String language;
  private final ImmutableMap<String, String> executionInfo;

  /**
   * Creates test target properties instance. Constructor expects that it
   * will be called only for test configured targets.
   */
  TestTargetProperties(RuleContext ruleContext,
      ExecutionInfo executionRequirements) {
    Rule rule = ruleContext.getRule();

    Preconditions.checkState(TargetUtils.isTestRule(rule));
    size = TestSize.getTestSize(rule);
    timeout = TestTimeout.getTestTimeout(rule);
    tags = ruleContext.attributes().get("tags", Type.STRING_LIST);

    // We need to use method on ruleConfiguredTarget to perform validation.
    isFlaky = ruleContext.attributes().get("flaky", Type.BOOLEAN);
    isExternal = TargetUtils.isExternalTestRule(rule);

    Map<String, String> executionInfo = Maps.newLinkedHashMap();
    executionInfo.putAll(TargetUtils.getExecutionInfo(rule));
    if (TargetUtils.isLocalTestRule(rule) || TargetUtils.isExclusiveTestRule(rule)) {
      executionInfo.put(ExecutionRequirements.LOCAL, "");
    }

    if (executionRequirements != null) {
      // This will overwrite whatever TargetUtils put there, which might be confusing.
      executionInfo.putAll(executionRequirements.getExecutionInfo());
    }
    ruleContext.getConfiguration().modifyExecutionInfo(executionInfo, TestRunnerAction.MNEMONIC);
    this.executionInfo = ImmutableMap.copyOf(executionInfo);

    isRemotable =
        !executionInfo.containsKey(ExecutionRequirements.LOCAL)
            && !executionInfo.containsKey(ExecutionRequirements.NO_REMOTE)
            && !executionInfo.containsKey(ExecutionRequirements.NO_REMOTE_EXEC);

    language = TargetUtils.getRuleLanguage(rule);
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

  public boolean isRemotable() {
    return isRemotable;
  }

  public boolean isFlaky() {
    return isFlaky;
  }

  public boolean isExternal() {
    return isExternal;
  }

  public ResourceSet getLocalResourceUsage(Label label, boolean usingLocalTestJobs)
      throws UserExecException {
    if (usingLocalTestJobs) {
      return LOCAL_TEST_JOBS_BASED_RESOURCES;
    }

    ResourceSet testResourcesFromSize = TestTargetProperties.getResourceSetFromSize(size);

    // Tests can override their CPU reservation with a "cpus:<n>" tag.
    ResourceSet testResourcesFromTag = null;
    for (String tag : executionInfo.keySet()) {
      try {
        String cpus = ExecutionRequirements.CPU.parseIfMatches(tag);
        if (cpus != null) {
          if (testResourcesFromTag != null) {
            throw new UserExecException(
                String.format(
                    "%s has more than one '%s' tag, but duplicate tags aren't allowed",
                    label, ExecutionRequirements.CPU.userFriendlyName()));
          }
          testResourcesFromTag =
              ResourceSet.create(
                  testResourcesFromSize.getMemoryMb(),
                  Float.parseFloat(cpus),
                  testResourcesFromSize.getLocalTestCount());
        }
      } catch (ValidationException e) {
        throw new UserExecException(
            String.format(
                "%s has a '%s' tag, but its value '%s' didn't pass validation: %s",
                label,
                ExecutionRequirements.CPU.userFriendlyName(),
                e.getTagValue(),
                e.getMessage()));
      }
    }

    return testResourcesFromTag != null ? testResourcesFromTag : testResourcesFromSize;
  }

  /**
   * Returns a map of execution info. See {@link
   * com.google.devtools.build.lib.actions.Spawn#getExecutionInfo}.
   */
  public ImmutableMap<String, String> getExecutionInfo() {
    return executionInfo;
  }

  public String getLanguage() {
    return language;
  }
}
