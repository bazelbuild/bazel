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
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.skylark.SkylarkRuleContext;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;
import java.util.Objects;
import javax.annotation.Nullable;

/** Skylark-visible methods for working with Android data (manifests, resources, and assets). */
@SkylarkModule(
    name = "android_data",
    doc =
        "Utilities for working with Android data (manifests, resources, and assets). "
            + "This API is non-final and subject to change without warning; do not rely on it.")
public class AndroidSkylarkData {

  /**
   * Skylark API for stamping an Android manifest
   *
   * <p>TODO(b/79159379): Stop passing SkylarkRuleContext here
   *
   * @param ctx the SkylarkRuleContext. We will soon change to using an ActionConstructionContext
   *     instead. See b/79159379
   */
  @SkylarkCallable(
      name = "stamp_manifest",
      mandatoryPositionals = 1, // SkylarkRuleContext ctx is mandatory
      parameters = {
        @Param(
            name = "manifest",
            positional = false,
            defaultValue = "None",
            type = Artifact.class,
            noneable = true,
            named = true,
            doc = "The manifest to stamp. If not passed, a dummy manifest will be generated"),
        @Param(
            name = "custom_package",
            positional = false,
            defaultValue = "None",
            type = String.class,
            noneable = true,
            named = true,
            doc =
                "The Android application package to stamp the manifest with. If not provided, the"
                    + " current Java package, derived from the location of this target's BUILD"
                    + " file, will be used. For example, given a BUILD file in"
                    + " 'java/com/foo/bar/BUILD', the package would be 'com.foo.bar'."),
        @Param(
            name = "exports_manifest",
            positional = false,
            defaultValue = "False",
            type = Boolean.class,
            named = true,
            doc =
                "Defaults to False. If passed as True, this manifest will be exported to and"
                    + " eventually merged into targets that depend on it. Otherwise, it won't be"
                    + " inherited."),
      },
      doc = "Stamps a manifest with package information.")
  public AndroidManifestInfo stampAndroidManifest(
      SkylarkRuleContext ctx, Object manifest, Object customPackage, boolean exported) {
    String pkg = fromNoneable(customPackage, String.class);
    if (pkg == null) {
      pkg = AndroidManifest.getDefaultPackage(ctx.getRuleContext());
    }

    Artifact primaryManifest = fromNoneable(manifest, Artifact.class);
    if (primaryManifest == null) {
      return StampedAndroidManifest.createEmpty(ctx.getRuleContext(), pkg, exported).toProvider();
    }

    return new AndroidManifest(primaryManifest, pkg, exported)
        .stamp(ctx.getRuleContext())
        .toProvider();
  }

  /**
   * Skylark API for merging android_library assets
   *
   * <p>TODO(b/79159379): Stop passing SkylarkRuleContext here
   *
   * @param ctx the SkylarkRuleContext. We will soon change to using an ActionConstructionContext
   *     instead. See b/79159379
   */
  @SkylarkCallable(
      name = "merge_assets",
      mandatoryPositionals = 1, // context
      parameters = {
        @Param(
            name = "assets",
            positional = false,
            defaultValue = "None",
            type = SkylarkList.class,
            generic1 = ConfiguredTarget.class,
            noneable = true,
            named = true,
            doc =
                "Targets containing raw assets for this target. If passed, 'assets_dir' must also"
                    + " be passed."),
        @Param(
            name = "assets_dir",
            positional = false,
            defaultValue = "None",
            type = String.class,
            noneable = true,
            named = true,
            doc =
                "Directory the assets are contained in. Must be passed if and only if 'assets' is"
                    + " passed. This path will be split off of the asset paths on the device."),
        @Param(
            name = "deps",
            positional = false,
            defaultValue = "[]",
            type = SkylarkList.class,
            generic1 = AndroidAssetsInfo.class,
            named = true,
            doc =
                "Providers containing assets from dependencies. These assets will be merged"
                    + " together with each other and this target's assets."),
        @Param(
            name = "neverlink",
            positional = false,
            defaultValue = "False",
            type = Boolean.class,
            named = true,
            doc =
                "Defaults to False. If passed as True, these assets will not be inherited by"
                    + " targets that depend on this one.")
      },
      doc =
          "Merges this target's assets together with assets inherited from dependencies. Note that,"
              + " by default, actions for validating the merge are created but may not be called."
              + " You may want to force these actions to be called - see the 'validation_result'"
              + " field in AndroidAssetsInfo")
  public AndroidAssetsInfo mergeAssets(
      SkylarkRuleContext ctx,
      Object assets,
      Object assetsDir,
      SkylarkList<AndroidAssetsInfo> deps,
      boolean neverlink)
      throws EvalException, InterruptedException {
    try {
      return AndroidAssets.from(
              ctx.getRuleContext(),
              listFromNoneable(assets, ConfiguredTarget.class),
              isNone(assetsDir) ? null : PathFragment.create(fromNoneable(assetsDir, String.class)))
          .parse(ctx.getRuleContext())
          .merge(
              ctx.getRuleContext(),
              AssetDependencies.fromProviders(deps.getImmutableList(), neverlink))
          .toProvider();
    } catch (RuleErrorException e) {
      throw new EvalException(Location.BUILTIN, e);
    }
  }

  /**
   * Skylark API for merging android_library resources
   *
   * <p>TODO(b/79159379): Stop passing SkylarkRuleContext here
   *
   * @param ctx the SkylarkRuleContext. We will soon change to using an ActionConstructionContext
   *     instead. See b/79159379
   */
  @SkylarkCallable(
      name = "merge_resources",
      mandatoryPositionals = 2, // context, manifest
      parameters = {
        @Param(
            name = "resources",
            positional = false,
            defaultValue = "[]",
            type = SkylarkList.class,
            generic1 = FileProvider.class,
            named = true,
            doc = "Providers of this target's resources"),
        @Param(
            name = "deps",
            positional = false,
            defaultValue = "[]",
            type = SkylarkList.class,
            generic1 = AndroidResourcesInfo.class,
            named = true,
            doc =
                "Targets containing raw resources from dependencies. These resources will be merged"
                    + " together with each other and this target's resources."),
        @Param(
            name = "neverlink",
            positional = false,
            defaultValue = "False",
            type = Boolean.class,
            named = true,
            doc =
                "Defaults to False. If passed as True, these resources will not be inherited by"
                    + " targets that depend on this one."),
      },
      doc =
          "Merges this target's resources together with resources inherited from dependencies. The"
              + " passed manifest provider is used to get Android package information and to"
              + " validate that all resources it refers to are available. Note that this method"
              + " might do additional processing to this manifest, so in the future, you may want"
              + " to use the manifest contained in this method's output instead of this one.")
  public AndroidResourcesInfo mergeResources(
      SkylarkRuleContext ctx,
      AndroidManifestInfo manifest,
      SkylarkList<ConfiguredTarget> resources,
      SkylarkList<AndroidResourcesInfo> deps,
      boolean neverlink)
      throws EvalException, InterruptedException {

    ImmutableList<FileProvider> fileProviders =
        resources
            .stream()
            .map(target -> target.getProvider(FileProvider.class))
            .filter(Objects::nonNull)
            .collect(ImmutableList.toImmutableList());

    try {
      return AndroidResources.from(ctx.getRuleContext(), fileProviders, "resources")
          .parse(ctx.getRuleContext(), manifest.asStampedManifest())
          .merge(ctx.getRuleContext(), ResourceDependencies.fromProviders(deps, neverlink))
          .validate(ctx.getRuleContext())
          .toProvider();
    } catch (RuleErrorException e) {
      throw new EvalException(Location.BUILTIN, e);
    }
  }

  /** Checks if a "Noneable" object passed by Skylark is "None", which Java should treat as null. */
  private static boolean isNone(Object object) {
    return object == Runtime.NONE;
  }

  /**
   * Converts a "Noneable" Object passed by Skylark to an nullable object of the appropriate type.
   *
   * <p>Skylark "Noneable" types are passed in as an Object that may be either the correct type or a
   * Runtime.NONE object. Skylark will handle type checking, based on the appropriate @param
   * annotation, but we still need to do the actual cast (or conversion to null) ourselves.
   *
   * @param object the Noneable object
   * @param clazz the correct class, as defined in the @Param annotation
   * @param <T> the type to cast to
   * @return {@code null}, if the noneable argument was None, or the cast object, otherwise.
   */
  @Nullable
  private static <T> T fromNoneable(Object object, Class<T> clazz) {
    if (isNone(object)) {
      return null;
    }

    return clazz.cast(object);
  }

  /**
   * Converts a "Noneable" Object passed by Skylark to a List of the appropriate type.
   *
   * <p>This first calls {@link #fromNoneable(Object, Class)} to get a SkylarkList<?>, then safely
   * casts it to a list with the appropriate generic.
   */
  @Nullable
  private static <T> List<T> listFromNoneable(Object object, Class<T> clazz) throws EvalException {
    SkylarkList<?> asList = fromNoneable(object, SkylarkList.class);
    if (asList == null) {
      return null;
    }

    return SkylarkList.castList(asList, clazz, null);
  }
}
