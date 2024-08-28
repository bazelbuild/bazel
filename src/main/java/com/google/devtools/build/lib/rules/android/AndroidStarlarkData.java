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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.analysis.BazelRuleAnalysisThreadContext;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.analysis.starlark.StarlarkErrorReporter;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.android.AndroidLibraryAarInfo.Aar;
import com.google.devtools.build.lib.rules.android.databinding.DataBinding;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaCompilationInfoProvider;
import com.google.devtools.build.lib.rules.java.JavaInfo;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider.JavaOutput;
import com.google.devtools.build.lib.rules.java.JavaSourceJarsProvider;
import com.google.devtools.build.lib.starlarkbuildapi.android.AndroidDataProcessingApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.StructApi;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import javax.annotation.Nullable;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkThread;

/** Starlark-visible methods for working with Android data (manifests, resources, and assets). */
public abstract class AndroidStarlarkData
    implements AndroidDataProcessingApi<
        AndroidDataContext,
        ConfiguredTarget,
        Artifact,
        SpecialArtifact,
        AndroidAssetsInfo,
        AndroidResourcesInfo,
        AndroidManifestInfo,
        AndroidLibraryAarInfo,
        AndroidBinaryDataInfo,
        ValidatedAndroidResources> {

  public abstract AndroidSemantics getAndroidSemantics();

  @Override
  public AndroidAssetsInfo assetsFromDeps(
      Sequence<?> deps, // <AndroidAssetsInfo>
      boolean neverlink,
      StarlarkThread thread)
      throws EvalException {
    // We assume this is an analysis-phase thread.
    Label label =
        BazelRuleAnalysisThreadContext.fromOrFail(thread, "assets_from_deps")
            .getAnalysisRuleLabel();
    return AssetDependencies.fromProviders(
            Sequence.cast(deps, AndroidAssetsInfo.class, "deps"), neverlink)
        .toInfo(label);
  }

  @Override
  public AndroidResourcesInfo resourcesFromDeps(
      AndroidDataContext ctx,
      Sequence<?> deps, // <AndroidResourcesInfo>
      Sequence<?> assets, // <AndroidAssetsInfo>
      boolean neverlink,
      String customPackage)
      throws InterruptedException, EvalException {
    try (StarlarkErrorReporter errorReporter =
        StarlarkErrorReporter.from(ctx.getRuleErrorConsumer())) {
      return ResourceApk.processFromTransitiveLibraryData(
              ctx,
              DataBinding.getDisabledDataBindingContext(ctx),
              ResourceDependencies.fromProviders(
                  Sequence.cast(deps, AndroidResourcesInfo.class, "deps"),
                  /* neverlink = */ neverlink),
              AssetDependencies.fromProviders(
                  Sequence.cast(assets, AndroidAssetsInfo.class, "assets"),
                  /* neverlink = */ neverlink),
              StampedAndroidManifest.createEmpty(
                  ctx.getActionConstructionContext(), customPackage, /* exported = */ false))
          .toResourceInfo(ctx.getLabel());
    }
  }

  @Override
  public AndroidManifestInfo stampAndroidManifest(
      AndroidDataContext ctx, Object manifest, Object customPackage, boolean exported)
      throws InterruptedException {
    String pkg = fromNoneable(customPackage, String.class);
    return AndroidManifest.from(
            ctx, fromNoneable(manifest, Artifact.class), getAndroidSemantics(), pkg, exported)
        .stamp(ctx)
        .toProvider();
  }

  @Override
  public AndroidAssetsInfo mergeAssets(
      AndroidDataContext ctx,
      Object assets,
      Object assetsDir,
      Sequence<?> deps, // <AndroidAssetsInfo>
      boolean neverlink)
      throws EvalException, InterruptedException {
    StarlarkErrorReporter errorReporter = StarlarkErrorReporter.from(ctx.getRuleErrorConsumer());
    try {
      return AndroidAssets.from(
              errorReporter,
              isNone(assets) ? null : Sequence.cast(assets, ConfiguredTarget.class, "assets"),
              isNone(assetsDir) ? null : PathFragment.create((String) assetsDir))
          .process(
              ctx,
              AssetDependencies.fromProviders(
                  Sequence.cast(deps, AndroidAssetsInfo.class, "deps"), neverlink))
          .toProvider();
    } catch (RuleErrorException e) {
      throw handleRuleException(errorReporter, e);
    }
  }

  @Override
  public ValidatedAndroidResources mergeRes(
      AndroidDataContext ctx,
      AndroidManifestInfo manifest,
      Sequence<?> resources, // <ConfiguredTarget>
      Sequence<?> deps, // <AndroidResourcesInfo>
      Sequence<?> resApkDeps, // <File>
      boolean neverlink,
      boolean enableDataBinding)
      throws EvalException, InterruptedException {
    StarlarkErrorReporter errorReporter = StarlarkErrorReporter.from(ctx.getRuleErrorConsumer());
    try {
      return AndroidResources.from(
              errorReporter,
              getFileProviders(Sequence.cast(resources, ConfiguredTarget.class, "resources")),
              "resources")
          .process(
              ctx,
              manifest.asStampedManifest(),
              Sequence.cast(resApkDeps, Artifact.class, "resource_apks"),
              ResourceDependencies.fromProviders(
                  Sequence.cast(deps, AndroidResourcesInfo.class, "deps"), neverlink),
              DataBinding.contextFrom(
                  enableDataBinding, ctx.getActionConstructionContext(), ctx.getAndroidConfig()));
    } catch (RuleErrorException e) {
      throw handleRuleException(errorReporter, e);
    }
  }

  @Override
  public Dict<Provider, NativeInfo> mergeResources(
      AndroidDataContext ctx,
      AndroidManifestInfo manifest,
      Sequence<?> resources, // <ConfiguredTarget>
      Sequence<?> deps, // <AndroidResourcesInfo>
      boolean neverlink,
      boolean enableDataBinding)
      throws EvalException, InterruptedException, RuleErrorException {
    ValidatedAndroidResources validated =
        mergeRes(
            ctx, manifest, resources, deps, StarlarkList.empty(), neverlink, enableDataBinding);
    JavaInfo javaInfo =
        getJavaInfoForRClassJar(validated.getClassJar(), validated.getJavaSourceJar());
    return Dict.<Provider, NativeInfo>builder()
        .put(AndroidResourcesInfo.PROVIDER, validated.toProvider())
        .put(JavaInfo.PROVIDER, javaInfo)
        .buildImmutable();
  }

  @Override
  public AndroidLibraryAarInfo makeAar(
      AndroidDataContext ctx,
      AndroidResourcesInfo resourcesInfo,
      AndroidAssetsInfo assetsInfo,
      Artifact libraryClassJar,
      Sequence<?> localProguardSpecs, // <Artifact>
      Sequence<?> deps, // <AndroidLibraryAarInfo>
      boolean neverlink)
      throws EvalException, InterruptedException {
    if (neverlink) {
      return AndroidLibraryAarInfo.create(
          null,
          NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER),
          NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER));
    }

    // Get the target's local resources, if defined, from the provider
    Optional<? extends AndroidResources> resources =
        resourcesInfo.getDirectAndroidResources().toList().stream()
            .filter(r -> r.getLabel().equals(ctx.getLabel()))
            .findFirst();
    boolean definesLocalResources = resources.isPresent();

    // Get the target's local assets, if defined, from the provider
    Optional<? extends AndroidAssets> assets =
        assetsInfo.getDirectParsedAssets().toList().stream()
            .filter(a -> a.getLabel().equals(ctx.getLabel()))
            .findFirst();
    // The target might still define an empty list of assets, in which case its information is not
    // propagated for efficiency. If this is the case, we will still have an artifact for the
    // merging output.
    boolean definesLocalAssets = assets.isPresent() || assetsInfo.getValidationResult() != null;

    if (definesLocalResources != definesLocalAssets) {
      throw new EvalException(
          "Must define either both or none of assets and resources. Use the merge_assets and"
              + " merge_resources methods to define them, or assets_from_deps and"
              + " resources_from_deps to inherit without defining them.");
    }

    return Aar.makeAar(
            ctx,
            resources.isPresent() ? resources.get() : AndroidResources.empty(),
            assets.isPresent() ? assets.get() : AndroidAssets.empty(),
            resourcesInfo.getManifest(),
            resourcesInfo.getRTxt(),
            libraryClassJar,
            ImmutableList.copyOf(
                Sequence.cast(localProguardSpecs, Artifact.class, "local_proguard_specs")))
        .toProvider(
            Sequence.cast(deps, AndroidLibraryAarInfo.class, "deps"), definesLocalResources);
  }

  @Override
  public Dict<Provider, StructApi> processAarImportData(
      AndroidDataContext ctx,
      SpecialArtifact resources,
      SpecialArtifact assets,
      Artifact androidManifestArtifact,
      Sequence<?> deps) // <ConfiguredTarget>
      throws InterruptedException, EvalException, RuleErrorException {
    List<ConfiguredTarget> depsTargets = Sequence.cast(deps, ConfiguredTarget.class, "deps");

    ValidatedAndroidResources validatedResources =
        AndroidResources.forAarImport(resources)
            .process(
                ctx,
                AndroidManifest.forAarImport(androidManifestArtifact),
                ImmutableList.of(),
                ResourceDependencies.fromProviders(
                    getProviders(depsTargets, AndroidResourcesInfo.PROVIDER),
                    /* neverlink= */ false),
                DataBinding.getDisabledDataBindingContext(ctx));

    MergedAndroidAssets mergedAssets =
        AndroidAssets.forAarImport(assets)
            .process(
                ctx,
                AssetDependencies.fromProviders(
                    getProviders(depsTargets, AndroidAssetsInfo.PROVIDER),
                    /* neverlink = */ false));

    ResourceApk resourceApk = ResourceApk.of(validatedResources, mergedAssets, null, null);

    return getNativeInfosFrom(resourceApk, ctx.getLabel());
  }

  private static IllegalStateException handleRuleException(
      StarlarkErrorReporter errorReporter, RuleErrorException exception) throws EvalException {
    // The error reporter should have been notified of the rule error, and thus closing it will
    // throw an EvalException.
    errorReporter.close();
    // It's a catastrophic state error if the errorReporter did not pick up the error.
    throw new IllegalStateException("Unhandled RuleErrorException", exception);
  }

  public static Dict<Provider, StructApi> getNativeInfosFrom(ResourceApk resourceApk, Label label)
      throws RuleErrorException {
    Dict.Builder<Provider, StructApi> builder = Dict.builder();

    builder
        .put(AndroidResourcesInfo.PROVIDER, resourceApk.toResourceInfo(label))
        .put(AndroidAssetsInfo.PROVIDER, resourceApk.toAssetsInfo(label));

    resourceApk.toManifestInfo().ifPresent(info -> builder.put(AndroidManifestInfo.PROVIDER, info));

    builder.put(
        JavaInfo.PROVIDER,
        getJavaInfoForRClassJar(
            resourceApk.getResourceJavaClassJar(), resourceApk.getResourceJavaSrcJar()));

    return builder.buildImmutable();
  }

  private static JavaInfo getJavaInfoForRClassJar(Artifact rClassJar, Artifact rClassSrcJar)
      throws RuleErrorException {
    return JavaInfo.Builder.create()
        .setNeverlink(true)
        .javaSourceJars(JavaSourceJarsProvider.builder().addSourceJar(rClassSrcJar).build())
        .javaRuleOutputs(
            JavaRuleOutputJarsProvider.builder()
                .addJavaOutput(
                    JavaOutput.builder().setClassJar(rClassJar).addSourceJar(rClassSrcJar).build())
                .build())
        .javaCompilationArgs(
            JavaCompilationArgsProvider.builder()
                .addDirectCompileTimeJar(rClassJar, rClassJar)
                .build())
        .javaCompilationInfo(
            new JavaCompilationInfoProvider.Builder()
                .setCompilationClasspath(NestedSetBuilder.create(Order.NAIVE_LINK_ORDER, rClassJar))
                .build())
        .build();
  }

  /**
   * Checks if a "Noneable" object passed by Starlark is "None", which Java should treat as null.
   */
  public static boolean isNone(Object object) {
    return object == Starlark.NONE;
  }

  /**
   * Converts a "Noneable" Object passed by Starlark to an nullable object of the appropriate type.
   *
   * <p>Starlark "Noneable" types are passed in as an Object that may be either the correct type or
   * a Starlark.NONE object. Starlark will handle type checking, based on the appropriate @param
   * annotation, but we still need to do the actual cast (or conversion to null) ourselves.
   *
   * @param object the Noneable object
   * @param clazz the correct class, as defined in the @Param annotation
   * @param <T> the type to cast to
   * @return {@code null}, if the noneable argument was None, or the cast object, otherwise.
   */
  @Nullable
  public static <T> T fromNoneable(Object object, Class<T> clazz) {
    if (isNone(object)) {
      return null;
    }

    return clazz.cast(object);
  }

  public static <T> T fromNoneableOrDefault(Object object, Class<T> clazz, T defaultValue) {
    T value = fromNoneable(object, clazz);
    if (value == null) {
      return defaultValue;
    }

    return value;
  }

  @Nullable
  public static NestedSet<Artifact> fromNoneableDepset(Object depset, String what)
      throws EvalException {
    if (isNone(depset)) {
      return null;
    }
    return Depset.cast(depset, Artifact.class, what);
  }

  private static ImmutableList<FileProvider> getFileProviders(List<ConfiguredTarget> targets) {
    return getProviders(targets, FileProvider.class);
  }

  private static <T extends TransitiveInfoProvider> ImmutableList<T> getProviders(
      List<ConfiguredTarget> targets, Class<T> clazz) {
    return targets
        .stream()
        .map(target -> target.getProvider(clazz))
        .filter(Objects::nonNull)
        .collect(ImmutableList.toImmutableList());
  }

  public static <T extends NativeInfo> Sequence<T> getProviders(
      List<ConfiguredTarget> targets, BuiltinProvider<T> provider) {
    return StarlarkList.immutableCopyOf(
        targets.stream()
            .map(target -> target.get(provider))
            .filter(Objects::nonNull)
            .collect(ImmutableList.toImmutableList()));
  }
}
