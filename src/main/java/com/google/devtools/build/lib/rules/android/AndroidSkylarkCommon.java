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
package com.google.devtools.build.lib.rules.android;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaInfo;
import com.google.devtools.build.lib.rules.java.JavaStrictCompilationArgsProvider;
import com.google.devtools.build.lib.skylarkbuildapi.android.AndroidSkylarkCommonApi;
import com.google.devtools.build.lib.skylarkbuildapi.android.AndroidSplitTransititionApi;
import com.google.devtools.build.lib.vfs.PathFragment;

/** Common utilities for Skylark rules related to Android. */
public class AndroidSkylarkCommon implements AndroidSkylarkCommonApi<Artifact, JavaInfo> {

  @Override
  public AndroidDeviceBrokerInfo createDeviceBrokerInfo(String deviceBrokerType) {
    return new AndroidDeviceBrokerInfo(deviceBrokerType);
  }

  @Override
  public PathFragment getSourceDirectoryRelativePathFromResource(Artifact resource) {
    return AndroidCommon.getSourceDirectoryRelativePathFromResource(resource);
  }

  @Override
  public AndroidSplitTransititionApi getAndroidSplitTransition() {
    return AndroidRuleClasses.ANDROID_SPLIT_TRANSITION;
  }

  /**
   * TODO(b/14473160): Provides a Starlark compatibility layer for the sourceless deps bug. When a
   * sourceless target is defined, the deps of the target are implicitly exported. Specifically only
   * the {@link JavaCompilationArgsProvider} is propagated. This method takes the existing JavaInfo
   * and produces a new one, only containing the {@link JavaCompilationArgsProvider} to be added to
   * the exports field of the java_common.compile method. Remove this method once the bug has been
   * fixed.
   */
  @Override
  public JavaInfo enableImplicitSourcelessDepsExportsCompatibility(JavaInfo javaInfo) {
    return JavaInfo.Builder.create()
        .addProvider(
            JavaCompilationArgsProvider.class,
            javaInfo.getProvider(JavaCompilationArgsProvider.class))
        .build();
  }

  /**
   * TODO(b/132905414): Provides a Starlark compatibility layer for maintaining the consistency of
   * the Javac actions created by dependers of a Starlark android_library rule. Specifically, this
   * method attempts to keep the --classpath and --direct_dependencies ordering the same by ensuring
   * that the resource jar appears before the other jars in the dependencies.
   */
  @Override
  public JavaInfo addRJarToJavaInfo(JavaInfo javaInfo, Artifact rJar) {
    JavaCompilationArgsProvider javaCompilationArgsProvider =
        javaInfo.getProvider(JavaCompilationArgsProvider.class);
    JavaCompilationArgsProvider compilationArgsProvider =
        JavaCompilationArgsProvider.create(
            NestedSetBuilder.<Artifact>naiveLinkOrder()
                .add(rJar)
                .addTransitive(javaCompilationArgsProvider.getRuntimeJars())
                .build(),
            NestedSetBuilder.<Artifact>naiveLinkOrder()
                .add(rJar)
                .addTransitive(javaCompilationArgsProvider.getDirectCompileTimeJars())
                .build(),
            NestedSetBuilder.<Artifact>naiveLinkOrder()
                .add(rJar)
                .addTransitive(javaCompilationArgsProvider.getTransitiveCompileTimeJars())
                .build(),
            NestedSetBuilder.<Artifact>naiveLinkOrder()
                .add(rJar)
                .addTransitive(javaCompilationArgsProvider.getDirectFullCompileTimeJars())
                .build(),
            NestedSetBuilder.<Artifact>naiveLinkOrder()
                .add(rJar)
                .addTransitive(javaCompilationArgsProvider.getTransitiveFullCompileTimeJars())
                .build(),
            NestedSetBuilder.<Artifact>naiveLinkOrder()
                .addTransitive(javaCompilationArgsProvider.getCompileTimeJavaDependencyArtifacts())
                .build());
    return JavaInfo.Builder.copyOf(javaInfo)
        .addProvider(JavaCompilationArgsProvider.class, compilationArgsProvider)
        .addProvider(
            JavaStrictCompilationArgsProvider.class,
            new JavaStrictCompilationArgsProvider(compilationArgsProvider))
        .build();
  }
}
