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

package com.google.devtools.build.lib.rules.proto;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;

/**
 * An implementation for the "proto_library" rule.
 */
public class BazelProtoLibrary implements RuleConfiguredTargetFactory {

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
    ImmutableList<Artifact> protoSources =
        ruleContext.getPrerequisiteArtifacts("srcs", Mode.TARGET).list();
    ImmutableList<Artifact> checkDepsProtoSources = ProtoCommon.getCheckDepsProtoSources(
        ruleContext, protoSources);
    ProtoCommon.checkSourceFilesAreInSamePackage(ruleContext);

    NestedSet<Artifact> transitiveImports =
        ProtoCommon.collectTransitiveImports(ruleContext, protoSources);

    RunfilesProvider runfilesProvider =
        ProtoCommon.createRunfilesProvider(transitiveImports, ruleContext);
    // TODO(bazel-team): this second constructor argument is superfluous and should be removed.
    ProtoSourcesProvider sourcesProvider =
        ProtoSourcesProvider.create(
            transitiveImports, transitiveImports, protoSources, checkDepsProtoSources);

    return new RuleConfiguredTargetBuilder(ruleContext)
        .add(RunfilesProvider.class, runfilesProvider)
        .addProvider(ProtoSourcesProvider.class, sourcesProvider)
        .addSkylarkTransitiveInfo(ProtoSourcesProvider.SKYLARK_NAME, sourcesProvider)
        .build();
  }
}
