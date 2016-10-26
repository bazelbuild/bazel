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

import static com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode.TARGET;
import static com.google.devtools.build.lib.collect.nestedset.Order.STABLE_ORDER;
import static com.google.devtools.build.lib.rules.java.proto.JavaLiteProtoAspect.LITE_PROTO_RUNTIME_ATTR;

import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.OutputGroupProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaRunfilesProvider;
import com.google.devtools.build.lib.rules.java.JavaSkylarkApiProvider;
import com.google.devtools.build.lib.rules.java.JavaSourceJarsProvider;
import com.google.devtools.build.lib.rules.java.ProguardLibrary;
import com.google.devtools.build.lib.rules.java.ProguardSpecProvider;

/** Implementation of the java_lite_proto_library rule. */
public class JavaLiteProtoLibrary implements RuleConfiguredTargetFactory {

  @Override
  public ConfiguredTarget create(final RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {

    JavaCompilationArgsProvider dependencyArgsProviders =
        JavaCompilationArgsProvider.merge(
            Iterables.<JavaCompilationArgsAspectProvider, JavaCompilationArgsProvider>transform(
                getDeps(ruleContext, JavaCompilationArgsAspectProvider.class),
                JavaCompilationArgsAspectProvider.GET_PROVIDER));

    if (!StrictDepsUtils.isStrictDepsJavaProtoLibrary(ruleContext)) {
      dependencyArgsProviders = StrictDepsUtils.makeNonStrict(dependencyArgsProviders);
    }

    Runfiles runfiles = new Runfiles.Builder(ruleContext.getWorkspaceName()).addArtifacts(
        dependencyArgsProviders.getRecursiveJavaCompilationArgs().getRuntimeJars()).build();

    JavaSourceJarsProvider sourceJarsProvider =
        JavaSourceJarsProvider.merge(
            Iterables.<JavaSourceJarsAspectProvider, JavaSourceJarsProvider>transform(
                getDeps(ruleContext, JavaSourceJarsAspectProvider.class),
                JavaSourceJarsAspectProvider.GET_PROVIDER));

    NestedSetBuilder<Artifact> filesToBuild = NestedSetBuilder.stableOrder();

    filesToBuild.addAll(sourceJarsProvider.getSourceJars());

    for (JavaProtoLibraryTransitiveFilesToBuildProvider provider : ruleContext.getPrerequisites(
        "deps", Mode.TARGET, JavaProtoLibraryTransitiveFilesToBuildProvider.class)) {
      filesToBuild.addTransitive(provider.getJars());
    }

    return new RuleConfiguredTargetBuilder(ruleContext)
        .setFilesToBuild(filesToBuild.build())
        .addProvider(RunfilesProvider.class, RunfilesProvider.withData(Runfiles.EMPTY, runfiles))
        .addOutputGroup(
            OutputGroupProvider.DEFAULT, NestedSetBuilder.<Artifact>emptySet(STABLE_ORDER))
        .add(JavaCompilationArgsProvider.class, dependencyArgsProviders)
        .add(JavaSourceJarsProvider.class, sourceJarsProvider)
        .add(JavaRunfilesProvider.class, new JavaRunfilesProvider(runfiles))
        .add(
            ProguardSpecProvider.class,
            new ProguardSpecProvider(
                new ProguardLibrary(ruleContext)
                    .collectProguardSpecs(
                        ImmutableMultimap.of(Mode.TARGET, LITE_PROTO_RUNTIME_ATTR))))
        .add(JavaRuleOutputJarsProvider.class, JavaRuleOutputJarsProvider.builder().build())
        .addSkylarkTransitiveInfo(JavaSkylarkApiProvider.NAME, new JavaSkylarkApiProvider())
        .build();
  }

  private <C extends TransitiveInfoProvider> Iterable<C> getDeps(
      RuleContext ruleContext, Class<C> clazz) {
    return ruleContext.getPrerequisites("deps", TARGET, clazz);
  }
}
