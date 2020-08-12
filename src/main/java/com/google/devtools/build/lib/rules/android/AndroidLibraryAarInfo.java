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
import com.google.devtools.build.lib.analysis.TransitionMode;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.starlarkbuildapi.android.AndroidLibraryAarInfoApi;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Sequence;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * A target that can provide the aar artifact of Android libraries and all the manifests that are
 * merged into the main aar manifest.
 */
@Immutable
public class AndroidLibraryAarInfo extends NativeInfo
    implements AndroidLibraryAarInfoApi<Artifact> {

  public static final Provider PROVIDER = new Provider();

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

  @Nullable
  @Override
  public Artifact getAarArtifact() {
    if (aar == null) {
      return null;
    }
    return aar.getAar();
  }

  public NestedSet<Aar> getTransitiveAars() {
    return transitiveAars;
  }

  @Override
  public Depset /*<Artifact>*/ getTransitiveAarArtifactsForStarlark() {
    return Depset.of(Artifact.TYPE, transitiveAarArtifacts);
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
        AndroidDataContext dataContext,
        ResourceApk resourceApk,
        ImmutableList<Artifact> localProguardSpecs,
        Artifact libraryClassJar)
        throws InterruptedException {
      return makeAar(
          dataContext,
          resourceApk.getPrimaryResources(),
          resourceApk.getPrimaryAssets(),
          resourceApk.getProcessedManifest().toProvider(),
          resourceApk.getRTxt(),
          libraryClassJar,
          localProguardSpecs);
    }

    static Aar makeAar(
        AndroidDataContext dataContext,
        AndroidResources primaryResources,
        AndroidAssets primaryAssets,
        AndroidManifestInfo manifest,
        Artifact rTxt,
        Artifact libraryClassJar,
        ImmutableList<Artifact> localProguardSpecs)
        throws InterruptedException {
      Artifact aarOut = dataContext.createOutputArtifact(AndroidRuleClasses.ANDROID_LIBRARY_AAR);

      new AarGeneratorBuilder()
          .withPrimaryResources(primaryResources)
          .withPrimaryAssets(primaryAssets)
          .withManifest(manifest.getManifest())
          .withRtxt(rTxt)
          .withClasses(libraryClassJar)
          .setAAROut(aarOut)
          .setProguardSpecs(localProguardSpecs)
          .setThrowOnResourceConflict(dataContext.throwOnResourceConflict())
          .build(dataContext);

      return Aar.create(aarOut, manifest.getManifest());
    }

    public abstract Artifact getAar();

    public abstract Artifact getManifest();

    Aar() {}

    public AndroidLibraryAarInfo toProvider(
        RuleContext ruleContext, boolean definesLocalResources) {
      return toProvider(
          AndroidCommon.getTransitivePrerequisites(ruleContext, TransitionMode.TARGET, PROVIDER),
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

  /** Provider class for {@link AndroidLibraryAarInfo} objects. */
  public static class Provider extends BuiltinProvider<AndroidLibraryAarInfo>
      implements AndroidLibraryAarInfoApi.Provider<Artifact> {

    public Provider() {
      super(NAME, AndroidLibraryAarInfo.class);
    }

    @SuppressWarnings("unchecked")
    @Override
    public AndroidLibraryAarInfoApi<Artifact> create(
        Artifact aarArtifact,
        Artifact manifest,
        Sequence<? extends AndroidLibraryAarInfoApi<Artifact>> infosFromDeps,
        Boolean definesLocalResources)
        throws EvalException {

      Aar aar = Aar.create(aarArtifact, manifest);
      return aar.toProvider((Iterable<AndroidLibraryAarInfo>) infosFromDeps, definesLocalResources);
    }
  }
}
