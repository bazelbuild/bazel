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

package com.google.devtools.build.lib.rules.cpp.proto;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode.TARGET;

import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.rules.cpp.CcSkylarkApiProvider;

/** Part of the implementation of cc_proto_library. */
public class CcProtoLibrary implements RuleConfiguredTargetFactory {
  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {

    if (ruleContext.getPrerequisites("deps", TARGET).size() != 1) {
      ruleContext.throwWithAttributeError(
          "deps",
          "'deps' attribute must contain exactly one label "
              + "(we didn't name it 'dep' for consistency). "
              + "The main use-case for multiple deps is to create a rule that contains several "
              + "other targets. This makes dependency bloat more likely. It also makes it harder"
              + "to remove unused deps.");
    }

    CcProtoLibraryProviders depProviders =
        checkNotNull(ruleContext.getPrerequisite("deps", TARGET))
            .getProvider(CcProtoLibraryProviders.class);

    RuleConfiguredTargetBuilder ruleConfiguredTargetBuilder = new RuleConfiguredTargetBuilder(
        ruleContext)
        .setFilesToBuild(depProviders.filesBuilder)
        .addProvider(
            RunfilesProvider.class, RunfilesProvider.withData(Runfiles.EMPTY, Runfiles.EMPTY))
        .addProviders(depProviders.providerMap);

    for (String groupName : depProviders.outputGroupInfo) {
      ruleConfiguredTargetBuilder.addOutputGroup(groupName,
          depProviders.outputGroupInfo.getOutputGroup(groupName));
    }

    CcSkylarkApiProvider.maybeAdd(ruleContext, ruleConfiguredTargetBuilder);
    return ruleConfiguredTargetBuilder.build();
  }
}
