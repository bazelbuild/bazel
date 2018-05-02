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
import com.google.devtools.build.lib.analysis.skylark.SkylarkRuleContext;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.Runtime;
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
    if (object == Runtime.NONE) {
      return null;
    }

    return clazz.cast(object);
  }
}
