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

package com.google.devtools.build.lib.rules.java.proto;

import static com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode.TARGET;
import static com.google.devtools.build.lib.collect.nestedset.Order.STABLE_ORDER;
import static com.google.devtools.build.lib.rules.java.proto.JplCcLinkParams.createCcLinkParamsStore;
import static com.google.devtools.build.lib.rules.java.proto.StrictDepsUtils.constructJcapFromAspectDeps;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.OutputGroupProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.WrappingProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;
import com.google.devtools.build.lib.rules.java.JavaInfo;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaRunfilesProvider;
import com.google.devtools.build.lib.rules.java.JavaSkylarkApiProvider;
import com.google.devtools.build.lib.rules.java.JavaSourceJarsProvider;
import com.google.devtools.build.lib.rules.java.ProtoJavaApiInfoAspectProvider;

/** Implementation of the java_proto_library rule. */
public class JavaProtoLibrary implements RuleConfiguredTargetFactory {

  @Override
  public ConfiguredTarget create(final RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {

    Iterable<JavaProtoLibraryAspectProvider> javaProtoLibraryAspectProviders =
        ruleContext.getPrerequisites("deps", TARGET, JavaProtoLibraryAspectProvider.class);

    JavaCompilationArgsProvider dependencyArgsProviders =
        constructJcapFromAspectDeps(ruleContext, javaProtoLibraryAspectProviders);

    // We assume that the runtime jars will not have conflicting artifacts
    // with the same root relative path
    Runfiles runfiles =
        new Runfiles.Builder(ruleContext.getWorkspaceName())
            .addTransitiveArtifactsWrappedInStableOrder(
                dependencyArgsProviders.getRecursiveJavaCompilationArgs().getRuntimeJars())
            .build();

    JavaSourceJarsProvider sourceJarsProvider =
        JavaSourceJarsProvider.merge(
            WrappingProvider.Helper.unwrapProviders(
                javaProtoLibraryAspectProviders, JavaSourceJarsProvider.class));

    NestedSetBuilder<Artifact> filesToBuild = NestedSetBuilder.stableOrder();

    filesToBuild.addAll(sourceJarsProvider.getSourceJars());

    for (JavaProtoLibraryAspectProvider provider : javaProtoLibraryAspectProviders) {
      filesToBuild.addTransitive(provider.getJars());
    }

    JavaRunfilesProvider javaRunfilesProvider = new JavaRunfilesProvider(runfiles);

    JavaInfo javaInfo =
        JavaInfo.Builder.create()
            .addProvider(JavaCompilationArgsProvider.class, dependencyArgsProviders)
            .addProvider(JavaSourceJarsProvider.class, sourceJarsProvider)
            .addProvider(
                ProtoJavaApiInfoAspectProvider.class,
                ProtoJavaApiInfoAspectProvider.merge(
                    JavaInfo.getProvidersFromListOfTargets(
                        ProtoJavaApiInfoAspectProvider.class,
                        ruleContext.getPrerequisites("deps", TARGET))))
            .addProvider(JavaRuleOutputJarsProvider.class, JavaRuleOutputJarsProvider.EMPTY)
            .addProvider(JavaRunfilesProvider.class, javaRunfilesProvider)
            .build();

    RuleConfiguredTargetBuilder result =
        new RuleConfiguredTargetBuilder(ruleContext)
            .setFilesToBuild(filesToBuild.build())
            .addSkylarkTransitiveInfo(
                JavaSkylarkApiProvider.NAME, JavaSkylarkApiProvider.fromRuleContext())
            .addProvider(RunfilesProvider.withData(Runfiles.EMPTY, runfiles))
            .addOutputGroup(
                OutputGroupProvider.DEFAULT, NestedSetBuilder.<Artifact>emptySet(STABLE_ORDER))
            .addProvider(dependencyArgsProviders)
            .addProvider(sourceJarsProvider)
            .addProvider(javaRunfilesProvider)
            .addProvider(JavaRuleOutputJarsProvider.EMPTY)
            .addNativeDeclaredProvider(javaInfo);

    if (ruleContext.getFragment(JavaConfiguration.class).jplPropagateCcLinkParamsStore()) {
      result.addProvider(createCcLinkParamsStore(ruleContext, ImmutableList.of()));
    }

    return result.build();
  }
}
