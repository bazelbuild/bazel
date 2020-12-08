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

import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.platform.FatPlatformInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformProviderUtils;
import java.util.List;

/** Defines a platform for execution contexts for multi-architecture artifacts. */
public class FatPlatform implements RuleConfiguredTargetFactory {
  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {

    List<PlatformInfo> platforms =
        Lists.newArrayList(
            PlatformProviderUtils.platforms(
                ruleContext.getPrerequisites(FatPlatformRule.PLATFORMS_ATTR)));

    if (platforms.size() == 0) {
      throw ruleContext.throwWithAttributeError(
          FatPlatformRule.PLATFORMS_ATTR,
          FatPlatformRule.PLATFORMS_ATTR + " attribute must have at least one platform");
    }

    FatPlatformInfo fatPlatformInfo = new FatPlatformInfo(ruleContext.getLabel(), platforms);
    PlatformInfo defaultPlatformInfo = fatPlatformInfo.getDefaultPlatform();

    return new RuleConfiguredTargetBuilder(ruleContext)
        .addProvider(RunfilesProvider.class, RunfilesProvider.EMPTY)
        .addProvider(FileProvider.class, FileProvider.EMPTY)
        .addProvider(FilesToRunProvider.class, FilesToRunProvider.EMPTY)
        .addNativeDeclaredProvider(fatPlatformInfo)
        .addNativeDeclaredProvider(defaultPlatformInfo)
        .build();
  }
}
