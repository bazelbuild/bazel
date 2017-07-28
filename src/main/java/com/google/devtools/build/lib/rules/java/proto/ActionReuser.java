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

package com.google.devtools.build.lib.rules.java.proto;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.collect.Iterables.getOnlyElement;
import static com.google.common.collect.Iterables.isEmpty;
import static com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode.TARGET;
import static com.google.devtools.build.lib.rules.java.JavaCompilationArgs.ClasspathType.BOTH;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoProviderMap;
import com.google.devtools.build.lib.analysis.TransitiveInfoProviderMapBuilder;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgs;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaCompilationArtifacts;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaSkylarkApiProvider;
import com.google.devtools.build.lib.rules.java.JavaSourceJarsProvider;
import com.google.devtools.build.lib.rules.java.ProtoJavaApiInfoProvider;
import com.google.devtools.build.lib.rules.proto.ProtoConfiguration;

public class ActionReuser {

  /**
   * If the underlying proto_library rule already registers the compile actions we need, just reuse
   * them. This will preserve memory.
   *
   * <p>TODO(b/36191931): Delete when it's time.
   */
  public static boolean reuseExistingActions(
      ConfiguredTarget base, RuleContext ruleContext, ConfiguredAspect.Builder aspect) {
    ProtoJavaApiInfoProvider javaApi = base.getProvider(ProtoJavaApiInfoProvider.class);
    if (javaApi == null) {
      return false;
    }

    JavaCompilationArtifacts directJars = javaApi.getJavaCompilationArtifactsImmutable();
    if (isEmpty(directJars.getCompileTimeJars()) || javaApi.sourceJarImmutable() == null) {
      return false;
    }

    boolean correctRollupTransitiveProtoRuntimes =
        ruleContext
            .getConfiguration()
            .getFragment(ProtoConfiguration.class)
            .correctRollupTransitiveProtoRuntimes();

    JavaCompilationArgs.Builder transitiveJars =
        JavaCompilationArgs.builder()
            .addTransitiveArgs(javaApi.getTransitiveJavaCompilationArgsImmutable(), BOTH)
            .merge(directJars);
    if (correctRollupTransitiveProtoRuntimes) {
      transitiveJars.addTransitiveArgs(javaApi.getTransitiveProtoRuntimeImmutable(), BOTH);
    } else {
      transitiveJars.addTransitiveDependencies(
          javaApi.getProtoRuntimeImmutable(), true /* recursive */);
    }

    Artifact outputJar = getOnlyElement(directJars.getRuntimeJars());
    Artifact compileTimeJar = getOnlyElement(directJars.getCompileTimeJars());
    Artifact sourceJar = checkNotNull(javaApi.sourceJarImmutable());

    JavaCompilationArgsProvider compilationArgsProvider =
        JavaCompilationArgsProvider.create(
            JavaCompilationArgs.builder().merge(directJars).build(),
            transitiveJars.build(),
            NestedSetBuilder.create(
                Order.STABLE_ORDER, directJars.getCompileTimeDependencyArtifact()),
            NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER));

    TransitiveInfoProviderMapBuilder javaProvidersBuilder =
        new TransitiveInfoProviderMapBuilder()
            .add(createOutputJarProvider(outputJar, compileTimeJar, sourceJar))
            .add(createSrcJarProvider(sourceJar))
            .add(compilationArgsProvider);

    NestedSetBuilder<Artifact> transitiveOutputJars = NestedSetBuilder.stableOrder();
    for (JavaProtoLibraryAspectProvider provider :
        ruleContext.getPrerequisites("deps", TARGET, JavaProtoLibraryAspectProvider.class)) {
      transitiveOutputJars.addTransitive(provider.getJars());
    }
    transitiveOutputJars.add(outputJar);

    TransitiveInfoProviderMap javaProviders = javaProvidersBuilder.build();
    aspect
        .addSkylarkTransitiveInfo(
            JavaSkylarkApiProvider.PROTO_NAME.getLegacyId(),
            JavaSkylarkApiProvider.fromProviderMap(javaProviders))
        .addProviders(
            new JavaProtoLibraryAspectProvider(javaProviders, transitiveOutputJars.build()));
    return true;
  }

  private static JavaRuleOutputJarsProvider createOutputJarProvider(
      Artifact outputJar, Artifact compileTimeJar, Artifact sourceJar) {
    return JavaRuleOutputJarsProvider.builder()
        .addOutputJar(outputJar, compileTimeJar, ImmutableList.of(sourceJar))
        .build();
  }

  private static JavaSourceJarsProvider createSrcJarProvider(Artifact sourceJar) {
    return JavaSourceJarsProvider.create(
        NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
        NestedSetBuilder.<Artifact>stableOrder().add(sourceJar).build());
  }
}
