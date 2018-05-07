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
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.NativeProvider;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * A target that can provide the aar artifact of Android libraries and all the manifests that are
 * merged into the main aar manifest.
 */
@SkylarkModule(
    name = "AndroidLibraryAarInfo",
    doc = "Android AARs provided by a library rule and its dependencies",
    category = SkylarkModuleCategory.PROVIDER)
@Immutable
public class AndroidLibraryAarInfo extends NativeInfo {
  private static final String SKYLARK_NAME = "AndroidLibraryAarInfo";
  public static final NativeProvider<AndroidLibraryAarInfo> PROVIDER =
      new NativeProvider<AndroidLibraryAarInfo>(AndroidLibraryAarInfo.class, SKYLARK_NAME) {};

  @Nullable private final Aar aar;
  private final NestedSet<Aar> transitiveAars;
  private final NestedSet<Artifact> transitiveAarArtifacts;

  private AndroidLibraryAarInfo(
      @Nullable Aar aar,
      NestedSet<Aar> transitiveAars,
      NestedSet<Artifact> transitiveAarArtifacts) {
    super(PROVIDER);
    this.aar = aar;
    this.transitiveAars = transitiveAars;
    this.transitiveAarArtifacts = transitiveAarArtifacts;
  }

  public static AndroidLibraryAarInfo create(
      @Nullable Aar aar,
      NestedSet<Aar> transitiveAars,
      NestedSet<Artifact> transitiveAarArtifacts) {
    return new AndroidLibraryAarInfo(aar, transitiveAars, transitiveAarArtifacts);
  }

  @Nullable
  public Aar getAar() {
    return aar;
  }

  public NestedSet<Aar> getTransitiveAars() {
    return transitiveAars;
  }

  public NestedSet<Artifact> getTransitiveAarArtifacts() {
    return transitiveAarArtifacts;
  }

  @Override
  public int hashCode() {
    return Objects.hash(aar, transitiveAars, transitiveAarArtifacts);
  }

  @Override
  public boolean equals(Object object) {
    if (!(object instanceof AndroidLibraryAarInfo)) {
      return false;
    }

    AndroidLibraryAarInfo other = (AndroidLibraryAarInfo) object;
    return Objects.equals(aar, other.aar)
        && transitiveAars.equals(other.transitiveAars)
        && transitiveAarArtifacts.equals(other.transitiveAarArtifacts);
  }

  /** The .aar file and associated AndroidManifest.xml contributed by a single target. */
  @AutoValue
  @Immutable
  public abstract static class Aar {
    @VisibleForTesting
    static Aar create(Artifact aar, Artifact manifest) {
      return new AutoValue_AndroidLibraryAarInfo_Aar(aar, manifest);
    }

    static Aar makeAar(
        RuleContext ruleContext,
        ResourceApk resourceApk,
        ImmutableList<Artifact> localProguardSpecs,
        Artifact libraryClassJar)
        throws InterruptedException {
      return makeAar(
          ruleContext,
          resourceApk.getPrimaryResources(),
          resourceApk.getPrimaryAssets(),
          resourceApk.getProcessedManifest(),
          resourceApk.getRTxt(),
          libraryClassJar,
          localProguardSpecs);
    }

    static Aar makeAar(
        RuleContext ruleContext,
        AndroidResources primaryResources,
        AndroidAssets primaryAssets,
        ProcessedAndroidManifest manifest,
        Artifact rTxt,
        Artifact libraryClassJar,
        ImmutableList<Artifact> localProguardSpecs)
        throws InterruptedException {
      Artifact aarOut =
          ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_LIBRARY_AAR);

      new AarGeneratorBuilder(ruleContext)
          .withPrimaryResources(primaryResources)
          .withPrimaryAssets(primaryAssets)
          .withManifest(manifest.getManifest())
          .withRtxt(rTxt)
          .withClasses(libraryClassJar)
          .setAAROut(aarOut)
          .setProguardSpecs(localProguardSpecs)
          .setThrowOnResourceConflict(
              AndroidCommon.getAndroidConfig(ruleContext).throwOnResourceConflict())
          .build(ruleContext);

      return Aar.create(aarOut, manifest.getManifest());
    }

    public abstract Artifact getAar();

    public abstract Artifact getManifest();

    Aar() {}

    public AndroidLibraryAarInfo toProvider(
        RuleContext ruleContext, boolean definesLocalResources) {
      return toProvider(
          AndroidCommon.getTransitivePrerequisites(ruleContext, Mode.TARGET, PROVIDER),
          definesLocalResources);
    }

    public AndroidLibraryAarInfo toProvider(
        Iterable<AndroidLibraryAarInfo> depProviders, boolean definesLocalResources) {
      NestedSetBuilder<Aar> aarBuilder = NestedSetBuilder.naiveLinkOrder();
      NestedSetBuilder<Artifact> artifactBuilder = NestedSetBuilder.naiveLinkOrder();

      for (AndroidLibraryAarInfo depProvider : depProviders) {
        aarBuilder.addTransitive(depProvider.getTransitiveAars());
        artifactBuilder.addTransitive(depProvider.getTransitiveAarArtifacts());
      }

      if (!definesLocalResources) {
        return AndroidLibraryAarInfo.create(null, aarBuilder.build(), artifactBuilder.build());
      }

      aarBuilder.add(this);
      artifactBuilder.add(getAar()).add(getManifest());

      return AndroidLibraryAarInfo.create(this, aarBuilder.build(), artifactBuilder.build());
    }
  }
}
