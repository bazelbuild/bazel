// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.platform;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.platform.ConstraintCollection;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformProviderUtils;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.packages.Types;
import java.util.List;
import java.util.Map;

/** Defines a platform for execution contexts. */
public class Platform implements RuleConfiguredTargetFactory {
  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {

    PlatformInfo.Builder platformBuilder = PlatformInfo.builder().setLabel(ruleContext.getLabel());

    ImmutableList<PlatformInfo> parentPlatforms =
        PlatformProviderUtils.platforms(
            ruleContext.getPrerequisites(PlatformRule.PARENTS_PLATFORM_ATTR));

    if (parentPlatforms.size() > 1) {
      throw ruleContext.throwWithAttributeError(
          PlatformRule.PARENTS_PLATFORM_ATTR,
          PlatformRule.PARENTS_PLATFORM_ATTR + " attribute must have a single value");
    }
    PlatformInfo parentPlatform = Iterables.getFirst(parentPlatforms, null);
    platformBuilder.setParent(parentPlatform);

    // Add the declared constraints. Because setting the host_platform or target_platform attribute
    // to true on a platform automatically includes the detected CPU and OS constraints, if the
    // constraint_values attribute tries to add those, this will throw an error.
    platformBuilder.addConstraints(
        PlatformProviderUtils.constraintValues(
            ruleContext.getPrerequisites(PlatformRule.CONSTRAINT_VALUES_ATTR)));

    String remoteExecutionProperties =
        ruleContext.attributes().get(PlatformRule.REMOTE_EXECUTION_PROPS_ATTR, Type.STRING);
    platformBuilder.setRemoteExecutionProperties(remoteExecutionProperties);

    Map<String, String> execProperties =
        ruleContext.attributes().get(PlatformRule.EXEC_PROPS_ATTR, Types.STRING_DICT);
    if (execProperties != null && !execProperties.isEmpty()) {
      platformBuilder.setExecProperties(ImmutableMap.copyOf(execProperties));
    }

    List<String> flags = ruleContext.attributes().get(PlatformRule.FLAGS_ATTR, Types.STRING_LIST);
    if (flags != null && !flags.isEmpty()) {
      platformBuilder.addFlags(flags);
    }

    PlatformInfo platformInfo;
    try {
      platformInfo = platformBuilder.build();
    } catch (ConstraintCollection.DuplicateConstraintException e) {
      throw ruleContext.throwWithAttributeError(
          PlatformRule.CONSTRAINT_VALUES_ATTR, e.getMessage());
    } catch (PlatformInfo.ExecPropertiesException e) {
      throw ruleContext.throwWithAttributeError(PlatformRule.EXEC_PROPS_ATTR, e.getMessage());
    }

    return new RuleConfiguredTargetBuilder(ruleContext)
        .addProvider(RunfilesProvider.class, RunfilesProvider.EMPTY)
        .addProvider(FileProvider.class, FileProvider.EMPTY)
        .addProvider(FilesToRunProvider.class, FilesToRunProvider.EMPTY)
        .addNativeDeclaredProvider(platformInfo)
        .build();
  }
}
