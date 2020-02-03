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

package com.google.devtools.build.lib.rules.apple;


import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;

/** Implementation for the {@code available_xcodes} rule. */
public class AvailableXcodes implements RuleConfiguredTargetFactory {

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {

    Iterable<XcodeVersionRuleData> availableVersions =
        ruleContext.getPrerequisites(
            AvailableXcodesRule.VERSIONS_ATTR_NAME,
            RuleConfiguredTarget.Mode.TARGET,
            XcodeVersionRuleData.class);
    XcodeVersionRuleData defaultVersion =
        ruleContext.getPrerequisite(
            AvailableXcodesRule.DEFAULT_ATTR_NAME,
            RuleConfiguredTarget.Mode.TARGET,
            XcodeVersionRuleData.class);
    AvailableXcodesInfo availableXcodes =
        new AvailableXcodesInfo(availableVersions, defaultVersion);

    return new RuleConfiguredTargetBuilder(ruleContext)
        .addProvider(RunfilesProvider.class, RunfilesProvider.EMPTY)
        .addNativeDeclaredProvider(availableXcodes)
        .build();
  }
}
