// Copyright 2018 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.RuleErrorConsumer;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration.AndroidAaptVersion;
import com.google.devtools.build.lib.skylarkbuildapi.android.ValidatedAndroidDataApi;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.SkylarkType;
import com.google.devtools.build.lib.syntax.StarlarkList;
import java.util.Objects;
import java.util.Optional;
import javax.annotation.Nullable;

/** Wraps validated and packaged Android resource information */
public class ValidatedAndroidResources extends MergedAndroidResources
    implements ValidatedAndroidDataApi<Artifact, AndroidResourcesInfo> {

  public static final SkylarkType TYPE = SkylarkType.of(ValidatedAndroidResources.class);

  private final Artifact rTxt;
  private final Artifact sourceJar;
  private final Artifact apk;

  // aapt2 outputs. Will be null if and only if aapt2 is not used for validation.
  @Nullable private final Artifact aapt2SourceJar;
  @Nullable private final Artifact staticLibrary;

  // Opaque file used to trigger resource validation.
  @Nullable private final Artifact aapt2ValidationArtifact;
  private final boolean useRTxtFromMergedResources;

  /**
   * Validates and packages merged resources.
   *
   * <p>Specifically, validates that:
   *
   * <ul>
   *   <li>there are no conflicts between resources (though currently we just warn)
   *   <li>each reference to a resource in resources and manifest are satisfied
   * </ul>
   *
   * <p>And packs resources into:
   *
   * <ul>
   *   <li>R.java and R.txt files
   *   <li>A resource-only APK (deprecated)
   *   <li>When building with aapt2, aapt2 equivalents of the above
   *   <li>When building with aapt2, a compiled symbols zip
   * </ul>
   */
  public static ValidatedAndroidResources validateFrom(
      AndroidDataContext dataContext, MergedAndroidResources merged, AndroidAaptVersion aaptVersion)
      throws InterruptedException {
    if (dataContext.getAndroidConfig().incompatibleProhibitAapt1()) {
      Artifact rTxtOut = dataContext.createOutputArtifact(AndroidRuleClasses.ANDROID_R_TXT);
      Artifact sourceJarOut =
          dataContext.createOutputArtifact(AndroidRuleClasses.ANDROID_JAVA_SOURCE_JAR);
      Artifact apkOut = dataContext.createOutputArtifact(AndroidRuleClasses.ANDROID_LIBRARY_APK);

      BusyBoxActionBuilder.create(dataContext, "LINK_STATIC_LIBRARY")
          .addAapt(AndroidAaptVersion.AAPT2)
          .addInput("--libraries", dataContext.getSdk().getAndroidJar())
          .addInput("--compiled", merged.getCompiledSymbols())
          .addInput("--manifest", merged.getManifest())
          // Sets an alternative java package for the generated R.java
          // this allows android rules to generate resources outside of the java{,tests} tree.
          .maybeAddFlag("--packageForR", merged.getJavaPackage())
          .addTransitiveVectoredInput(
              "--compiledDep", merged.getResourceDependencies().getTransitiveCompiledSymbols())
          .addOutput("--sourceJarOut", sourceJarOut)
          .addOutput("--rTxtOut", rTxtOut)
          .addOutput("--staticLibraryOut", apkOut)
          .buildAndRegister("Linking static android resource library", "AndroidResourceLink");

      return of(
          merged,
          rTxtOut,
          sourceJarOut,
          apkOut,
          // TODO: remove below three when incompatibleProhibitAapt1 is on by default.
          rTxtOut,
          sourceJarOut,
          apkOut,
          dataContext.getAndroidConfig().useRTxtFromMergedResources());
    }

    AndroidResourceValidatorActionBuilder builder =
        new AndroidResourceValidatorActionBuilder()
            .setJavaPackage(merged.getJavaPackage())
            .setDebug(dataContext.useDebug())
            .setMergedResources(merged.getMergedResources())
            .setRTxtOut(dataContext.createOutputArtifact(AndroidRuleClasses.ANDROID_R_TXT))
            .setSourceJarOut(
                dataContext.createOutputArtifact(AndroidRuleClasses.ANDROID_JAVA_SOURCE_JAR))
            // Request an APK so it can be inherited when a library is used in a binary's
            // resources attr.
            // TODO(b/30307842): Remove this once it is no longer needed for resources migration.
            .setApkOut(dataContext.createOutputArtifact(AndroidRuleClasses.ANDROID_LIBRARY_APK))
            .withDependencies(merged.getResourceDependencies());

    if (aaptVersion == AndroidAaptVersion.AAPT2) {
      builder
          .setCompiledSymbols(merged.getCompiledSymbols())
          .setAapt2ValidationArtifactOut(
              dataContext.createOutputArtifact(
                  AndroidRuleClasses.ANDROID_RESOURCES_AAPT2_VALIDATION_ARTIFACT))
          .setAapt2SourceJarOut(
              dataContext.createOutputArtifact(
                  AndroidRuleClasses.ANDROID_RESOURCES_AAPT2_SOURCE_JAR))
          .setStaticLibraryOut(
              dataContext.createOutputArtifact(
                  AndroidRuleClasses.ANDROID_RESOURCES_AAPT2_LIBRARY_APK));
    }

    return builder.build(dataContext, merged);
  }

  static ValidatedAndroidResources of(
      MergedAndroidResources merged,
      Artifact rTxt,
      Artifact sourceJar,
      Artifact apk,
      @Nullable Artifact aapt2ValidationArtifact,
      @Nullable Artifact aapt2SourceJar,
      @Nullable Artifact staticLibrary,
      boolean useRTxtFromMergedResources) {
    return new ValidatedAndroidResources(
        merged,
        rTxt,
        sourceJar,
        apk,
        aapt2ValidationArtifact,
        aapt2SourceJar,
        staticLibrary,
        useRTxtFromMergedResources);
  }

  private ValidatedAndroidResources(
      MergedAndroidResources merged,
      Artifact rTxt,
      Artifact sourceJar,
      Artifact apk,
      @Nullable Artifact aapt2ValidationArtifact,
      @Nullable Artifact aapt2SourceJar,
      @Nullable Artifact staticLibrary,
      boolean useRTxtFromMergedResources) {
    super(merged);
    this.rTxt = rTxt;
    this.sourceJar = sourceJar;
    this.apk = apk;
    this.aapt2ValidationArtifact = aapt2ValidationArtifact;
    this.aapt2SourceJar = aapt2SourceJar;
    this.staticLibrary = staticLibrary;
    this.useRTxtFromMergedResources = useRTxtFromMergedResources;
  }

  @Override
  public AndroidResourcesInfo toProvider() {
    return getResourceDependencies().toInfo(this);
  }

  @Override
  public Artifact getRTxt() {
    return rTxt;
  }

  @Override
  public Artifact getJavaClassJar() {
    return super.getClassJar();
  }

  @Override
  public Artifact getJavaSourceJar() {
    return sourceJar;
  }

  // TODO(b/30307842,b/119560471): remove this; it was added for no reason, but persists because
  // the Starlark API is not noneable.
  @Override
  public Artifact getApk() {
    return apk;
  }

  @Override
  public Artifact getAapt2RTxt() {
    return useRTxtFromMergedResources ? super.getAapt2RTxt() : aapt2ValidationArtifact;
  }

  @Nullable
  Artifact getAapt2ValidationArtifact() {
    return aapt2ValidationArtifact;
  }

  @Override
  @Nullable
  public Artifact getAapt2SourceJar() {
    return aapt2SourceJar;
  }

  @Override
  @Nullable
  public Artifact getStaticLibrary() {
    return staticLibrary;
  }

  @Override
  public Sequence<Artifact> getResourcesList() {
    return StarlarkList.immutableCopyOf(getResources());
  }

  public ValidatedAndroidResources filter(
      RuleErrorConsumer errorConsumer, ResourceFilter resourceFilter, boolean isDependency)
      throws RuleErrorException {
    return maybeFilter(errorConsumer, resourceFilter, isDependency).orElse(this);
  }

  @Override
  public Optional<ValidatedAndroidResources> maybeFilter(
      RuleErrorConsumer errorConsumer, ResourceFilter resourceFilter, boolean isDependency)
      throws RuleErrorException {
    return super.maybeFilter(errorConsumer, resourceFilter, isDependency)
        .map(
            merged ->
                ValidatedAndroidResources.of(
                    merged,
                    rTxt,
                    sourceJar,
                    apk,
                    aapt2ValidationArtifact,
                    aapt2SourceJar,
                    staticLibrary,
                    useRTxtFromMergedResources));
  }

  @Override
  public boolean equals(Object object) {
    if (!super.equals(object) || !(object instanceof ValidatedAndroidResources)) {
      return false;
    }

    ValidatedAndroidResources other = (ValidatedAndroidResources) object;
    return rTxt.equals(other.rTxt)
        && sourceJar.equals(other.sourceJar)
        && apk.equals(other.apk)
        && Objects.equals(aapt2SourceJar, other.aapt2SourceJar)
        && Objects.equals(staticLibrary, other.staticLibrary);
  }

  @Override
  public int hashCode() {
    return Objects.hash(super.hashCode(), rTxt, sourceJar, apk, aapt2SourceJar, staticLibrary);
  }

  public ValidatedAndroidResources export() {
    return new ValidatedAndroidResources(
        new MergedAndroidResources(
            new ParsedAndroidResources(
                new AndroidResources(getResources(), getResourceRoots()),
                getSymbols(),
                getCompiledSymbols(),
                getLabel(),
                getStampedManifest(),
                // Null out databinding to avoid accidentally propagating ActionCreationContext
                null),
            getMergedResources(),
            getClassJar(),
            getAapt2RTxt(),
            getDataBindingInfoZip(),
            getResourceDependencies(),
            getProcessedManifest()),
        getRTxt(),
        getJavaSourceJar(),
        getApk(),
        getAapt2ValidationArtifact(),
        getAapt2SourceJar(),
        getStaticLibrary(),
        useRTxtFromMergedResources);
  }
}
