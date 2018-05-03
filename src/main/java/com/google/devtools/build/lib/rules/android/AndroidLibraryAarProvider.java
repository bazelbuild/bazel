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
package com.google.devtools.build.lib.rules.android;

import com.google.auto.value.AutoValue;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import javax.annotation.Nullable;

/**
 * A target that can provide the aar artifact of Android libraries and all the manifests that are
 * merged into the main aar manifest.
 */
@AutoValue
@Immutable
public abstract class AndroidLibraryAarProvider implements TransitiveInfoProvider {

  public static AndroidLibraryAarProvider create(
      @Nullable Aar aar,
      NestedSet<Aar> transitiveAars,
      NestedSet<Artifact> transitiveAarArtifacts) {
    return new AutoValue_AndroidLibraryAarProvider(aar, transitiveAars, transitiveAarArtifacts);
  }

  @Nullable
  public abstract Aar getAar();

  public abstract NestedSet<Aar> getTransitiveAars();

  public abstract NestedSet<Artifact> getTransitiveAarArtifacts();

  /** The .aar file and associated AndroidManifest.xml contributed by a single target. */
  @AutoValue
  @Immutable
  public abstract static class Aar {
    @VisibleForTesting
    static Aar create(Artifact aar, Artifact manifest) {
      return new AutoValue_AndroidLibraryAarProvider_Aar(aar, manifest);
    }

    static Aar makeAar(
        RuleContext ruleContext,
        ResourceApk resourceApk,
        ImmutableList<Artifact> localProguardSpecs)
        throws InterruptedException {
      Artifact classesJar =
          ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_LIBRARY_CLASS_JAR);
      Artifact aarOut =
          ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_LIBRARY_AAR);

      new AarGeneratorBuilder(ruleContext)
          .withPrimaryResources(resourceApk.getPrimaryResources())
          .withPrimaryAssets(resourceApk.getPrimaryAssets())
          .withManifest(resourceApk.getManifest())
          .withRtxt(resourceApk.getRTxt())
          .withClasses(classesJar)
          .setAAROut(aarOut)
          .setProguardSpecs(localProguardSpecs)
          .setThrowOnResourceConflict(
              AndroidCommon.getAndroidConfig(ruleContext).throwOnResourceConflict())
          .build(ruleContext);

      return Aar.create(aarOut, resourceApk.getManifest());
    }

    public abstract Artifact getAar();

    public abstract Artifact getManifest();

    Aar() {}

    public AndroidLibraryAarProvider toProvider(
        RuleContext ruleContext, boolean definesLocalResources) {
      return toProvider(
          AndroidCommon.getTransitivePrerequisites(
              ruleContext, Mode.TARGET, AndroidLibraryAarProvider.class),
          definesLocalResources);
    }

    public AndroidLibraryAarProvider toProvider(
        Iterable<AndroidLibraryAarProvider> depProviders, boolean definesLocalResources) {
      NestedSetBuilder<Aar> aarBuilder = NestedSetBuilder.naiveLinkOrder();
      NestedSetBuilder<Artifact> artifactBuilder = NestedSetBuilder.naiveLinkOrder();

      for (AndroidLibraryAarProvider depProvider : depProviders) {
        aarBuilder.addTransitive(depProvider.getTransitiveAars());
        artifactBuilder.addTransitive(depProvider.getTransitiveAarArtifacts());
      }

      if (!definesLocalResources) {
        return AndroidLibraryAarProvider.create(null, aarBuilder.build(), artifactBuilder.build());
      }

      aarBuilder.add(this);
      artifactBuilder.add(getAar()).add(getManifest());

      return AndroidLibraryAarProvider.create(this, aarBuilder.build(), artifactBuilder.build());
    }
  }

  AndroidLibraryAarProvider() {}
}
