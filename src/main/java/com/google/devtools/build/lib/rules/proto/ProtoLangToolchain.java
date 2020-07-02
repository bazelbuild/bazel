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

package com.google.devtools.build.lib.rules.proto;

import static com.google.devtools.build.lib.analysis.TransitionMode.HOST;
import static com.google.devtools.build.lib.analysis.TransitionMode.TARGET;
import static com.google.devtools.build.lib.collect.nestedset.Order.STABLE_ORDER;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.Type;

/** Implements {code proto_lang_toolchain}. */
public class ProtoLangToolchain implements RuleConfiguredTargetFactory {
  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    ProtoCommon.checkRuleHasValidMigrationTag(ruleContext);
    NestedSetBuilder<Artifact> blacklistedProtos = NestedSetBuilder.stableOrder();
    for (TransitiveInfoCollection protos :
        ruleContext.getPrerequisites("blacklisted_protos", TARGET)) {
      ProtoInfo protoInfo = protos.get(ProtoInfo.PROVIDER);
      if (protoInfo == null
          && ruleContext
              .getFragment(ProtoConfiguration.class)
              .blacklistedProtosRequiresProtoInfo()) {
        ruleContext.ruleError(
            "'" + ruleContext.getLabel() + "' does not have mandatory provider 'ProtoInfo'.");
      }
      if (protoInfo != null) {
        blacklistedProtos.addTransitive(protoInfo.getOriginalTransitiveProtoSources());
      } else {
        // Only add files from FileProvider if |protos| is not a proto_library to avoid adding
        // the descriptor_set of proto_library to the list of blacklisted files.
        blacklistedProtos.addTransitive(protos.getProvider(FileProvider.class).getFilesToBuild());
      }
    }

    return new RuleConfiguredTargetBuilder(ruleContext)
        .addProvider(
            ProtoLangToolchainProvider.create(
                ruleContext.attributes().get("command_line", Type.STRING),
                ruleContext.getPrerequisite("plugin", HOST, FilesToRunProvider.class),
                ruleContext.getPrerequisite("runtime", TARGET),
                blacklistedProtos.build()))
        .setFilesToBuild(NestedSetBuilder.<Artifact>emptySet(STABLE_ORDER))
        .addProvider(RunfilesProvider.simple(Runfiles.EMPTY))
        .build();
  }
}
