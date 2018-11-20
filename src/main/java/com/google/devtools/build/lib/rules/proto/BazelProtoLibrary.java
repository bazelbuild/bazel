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

import static com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode.TARGET;
import static com.google.devtools.build.lib.collect.nestedset.Order.STABLE_ORDER;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;

/** An implementation for the "proto_library" rule. */
public class BazelProtoLibrary implements RuleConfiguredTargetFactory {

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    ImmutableList<Artifact> protoSources =
        ruleContext.getPrerequisiteArtifacts("srcs", TARGET).list();
    NestedSet<Artifact> checkDepsProtoSources =
        ProtoCommon.getCheckDepsProtoSources(ruleContext, protoSources);
    ProtoCommon.checkSourceFilesAreInSamePackage(ruleContext);

    NestedSet<Artifact> transitiveImports =
        ProtoCommon.collectTransitiveImports(ruleContext, protoSources);
    NestedSet<Artifact> protosInDirectDeps = ProtoCommon.computeProtosInDirectDeps(ruleContext);

    NestedSet<Artifact> protosInExports = ProtoCommon.computeProtosInExportedDeps(ruleContext);

    String protoSourceRoot = ProtoCommon.getProtoSourceRoot(ruleContext);
    NestedSet<String> directProtoSourceRoots =
        ProtoCommon.getProtoSourceRootsOfDirectDependencies(ruleContext, protoSourceRoot);
    NestedSet<String> exportedProtoSourceRoots =
        ProtoCommon.getProtoSourceRootsOfExportedDependencies(ruleContext, protoSourceRoot);
    NestedSet<String> protoPathFlags =
        ProtoCommon.collectTransitiveProtoPathFlags(ruleContext, protoSourceRoot);

    final SupportData supportData =
        SupportData.create(
            protoSources,
            protosInDirectDeps,
            transitiveImports,
            protoPathFlags,
            protoSourceRoot,
            directProtoSourceRoots,
            !protoSources.isEmpty(),
            protosInExports,
            exportedProtoSourceRoots);

    Artifact descriptorSetOutput =
        ruleContext.getGenfilesArtifact(
            ruleContext.getLabel().getName() + "-descriptor-set.proto.bin");
    NestedSet<Artifact> dependenciesDescriptorSets =
        ProtoCommon.collectDependenciesDescriptorSets(ruleContext);
    NestedSet<Artifact> transitiveDescriptorSetOutput =
        NestedSetBuilder.fromNestedSet(dependenciesDescriptorSets).add(descriptorSetOutput).build();

    ProtoCompileActionBuilder.writeDescriptorSet(
        ruleContext,
        descriptorSetOutput.getExecPathString(),
        protoSources,
        transitiveImports,
        protosInDirectDeps,
        descriptorSetOutput,
        /* allowServices= */ true,
        dependenciesDescriptorSets,
        protoPathFlags,
        directProtoSourceRoots);

    Runfiles dataRunfiles =
        ProtoCommon.createDataRunfilesProvider(transitiveImports, ruleContext)
            .addArtifact(descriptorSetOutput)
            .build();

    // TODO(bazel-team): this second constructor argument is superfluous and should be removed.
    ProtoSourcesProvider sourcesProvider =
        ProtoSourcesProvider.create(
            transitiveImports,
            transitiveImports,
            protoSources,
            checkDepsProtoSources,
            protosInDirectDeps,
            descriptorSetOutput,
            transitiveDescriptorSetOutput,
            protoSourceRoot,
            directProtoSourceRoots,
            protoPathFlags,
            protosInExports);

    return new RuleConfiguredTargetBuilder(ruleContext)
        .setFilesToBuild(NestedSetBuilder.create(STABLE_ORDER, descriptorSetOutput))
        .addProvider(RunfilesProvider.withData(Runfiles.EMPTY, dataRunfiles))
        .addProvider(ProtoSourcesProvider.class, sourcesProvider)
        .addProvider(ProtoSupportDataProvider.class, new ProtoSupportDataProvider(supportData))
        .addSkylarkTransitiveInfo(ProtoSourcesProvider.SKYLARK_NAME, sourcesProvider)
        .build();
  }
}
