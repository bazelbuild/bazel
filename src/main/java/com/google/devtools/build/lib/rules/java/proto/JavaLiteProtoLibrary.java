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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode.TARGET;
import static com.google.devtools.build.lib.collect.nestedset.Order.STABLE_ORDER;
import static com.google.devtools.build.lib.rules.java.proto.JavaLiteProtoAspect.PROTO_TOOLCHAIN_ATTR;

import com.google.common.collect.ImmutableMultimap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.OutputGroupProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.WrappingProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaProvider;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaRunfilesProvider;
import com.google.devtools.build.lib.rules.java.JavaSkylarkApiProvider;
import com.google.devtools.build.lib.rules.java.JavaSourceJarsProvider;
import com.google.devtools.build.lib.rules.java.ProguardLibrary;
import com.google.devtools.build.lib.rules.java.ProguardSpecProvider;
import com.google.devtools.build.lib.rules.java.ProtoJavaApiInfoAspectProvider;
import com.google.devtools.build.lib.rules.proto.ProtoLangToolchainProvider;

/** Implementation of the java_lite_proto_library rule. */
public class JavaLiteProtoLibrary implements RuleConfiguredTargetFactory {

  @Override
  public ConfiguredTarget create(final RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {

    Iterable<JavaProtoLibraryAspectProvider> javaProtoLibraryAspectProviders =
        ruleContext.getPrerequisites("deps", Mode.TARGET, JavaProtoLibraryAspectProvider.class);

    JavaCompilationArgsProvider dependencyArgsProviders =
        JavaCompilationArgsProvider.merge(
            WrappingProvider.Helper.unwrapProviders(
                javaProtoLibraryAspectProviders, JavaCompilationArgsProvider.class));

    if (!StrictDepsUtils.isStrictDepsJavaProtoLibrary(ruleContext)) {
      dependencyArgsProviders = StrictDepsUtils.makeNonStrict(dependencyArgsProviders);
    }

    Runfiles runfiles =
        new Runfiles.Builder(ruleContext.getWorkspaceName())
            .addArtifacts(
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

    JavaProvider javaProvider =
        JavaProvider.Builder.create()
            .addProvider(JavaCompilationArgsProvider.class, dependencyArgsProviders)
            .addProvider(JavaSourceJarsProvider.class, sourceJarsProvider)
            .addProvider(
                ProtoJavaApiInfoAspectProvider.class,
                ProtoJavaApiInfoAspectProvider.merge(
                    JavaProvider.getProvidersFromListOfTargets(
                        ProtoJavaApiInfoAspectProvider.class,
                        ruleContext.getPrerequisites("deps", TARGET))))
            .addProvider(JavaRuleOutputJarsProvider.class, JavaRuleOutputJarsProvider.EMPTY)
            .addProvider(JavaRunfilesProvider.class, javaRunfilesProvider)
            .build();

    return new RuleConfiguredTargetBuilder(ruleContext)
        .setFilesToBuild(filesToBuild.build())
        .addSkylarkTransitiveInfo(
            JavaSkylarkApiProvider.NAME, JavaSkylarkApiProvider.fromRuleContext())
        .addProvider(RunfilesProvider.withData(Runfiles.EMPTY, runfiles))
        .addOutputGroup(
            OutputGroupProvider.DEFAULT, NestedSetBuilder.<Artifact>emptySet(STABLE_ORDER))
        .addProvider(dependencyArgsProviders)
        .addProvider(sourceJarsProvider)
        .addProvider(javaRunfilesProvider)
        .addProvider(getJavaLiteRuntimeSpec(ruleContext))
        .addProvider(JavaRuleOutputJarsProvider.EMPTY)
        .addNativeDeclaredProvider(javaProvider)
        .build();
  }

  private ProguardSpecProvider getJavaLiteRuntimeSpec(RuleContext ruleContext) {
    NestedSet<Artifact> specs =
        new ProguardLibrary(ruleContext).collectProguardSpecs(ImmutableMultimap.<Mode, String>of());

    TransitiveInfoCollection runtime = getProtoToolchainProvider(ruleContext).runtime();
    if (runtime == null) {
      return new ProguardSpecProvider(specs);
    }

    ProguardSpecProvider specProvider = runtime.getProvider(ProguardSpecProvider.class);
    if (specProvider == null) {
      return new ProguardSpecProvider(specs);
    }

    return new ProguardSpecProvider(
        NestedSetBuilder.fromNestedSet(specs)
            .addTransitive(specProvider.getTransitiveProguardSpecs())
            .build());
  }

  private ProtoLangToolchainProvider getProtoToolchainProvider(RuleContext ruleContext) {
    return checkNotNull(
        ruleContext.getPrerequisite(
            PROTO_TOOLCHAIN_ATTR, TARGET, ProtoLangToolchainProvider.class));
  }
}
