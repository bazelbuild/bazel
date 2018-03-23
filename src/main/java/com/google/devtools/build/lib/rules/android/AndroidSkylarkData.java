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
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.skylark.SkylarkRuleContext;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.SkylarkList;
import javax.annotation.Nullable;

/** Skylark-visible methods for working with Android data (manifests, resources, and assets). */
@SkylarkModule(
  name = "android_data",
  doc = "Utilities for working with Android data (manifests, resources, and assets)"
)
public class AndroidSkylarkData {
  @SkylarkCallable(
    name = "merge_manifests",
    mandatoryPositionals = 1, // context is mandatory
    parameters = {
      @Param(
        name = "manifest",
        positional = false,
        defaultValue = "None",
        type = Artifact.class,
        noneable = true,
        named = true,
        doc = "This target's manifest. Optional."
      ),
      @Param(
        name = "deps",
        positional = false,
        defaultValue = "[]",
        type = SkylarkList.class,
        generic1 = ConfiguredTarget.class,
        named = true,
        doc =
            "Targets from which this dependency should inherit manifests (if they are provided)."
                + " Optional."
      ),
      @Param(
        name = "custom_package",
        positional = false,
        defaultValue = "None",
        type = String.class,
        noneable = true,
        named = true,
        doc =
            "If passed, the manifest will be stamped with this package. Otherwise, the manifest"
                + " will be stamped with a package based on the current Java package."
      )
    },
    doc = "Merges manifests from this target and dependencies, and returns a manifest provider."
  )
  public AndroidManifestInfo mergeManifests(
      SkylarkRuleContext ctx,
      Object manifest,
      SkylarkList<ConfiguredTarget> deps,
      Object customPackage)
      throws InterruptedException {
    return AndroidManifest.of(
            ctx.getRuleContext(),
            fromNoneable(Artifact.class, manifest),
            fromNoneable(String.class, customPackage))
        .stampAndMergeWith(deps.getImmutableList())
        .toProvider();
  }

  /**
   * Converts a "Noneable" Object passed by Skylark to the appropriate type or null.
   *
   * <p>Skylark "Noneable" types are passed in as an Object that may be either the correct type or a
   * Runtime.NONE object. Skylark will handle type checking, based on the appropriate @param
   * annotation, but we still need to do the actual cast (or conversion to null) ourselves.
   *
   * @param clazz the correct class, as defined in the @Param annotation
   * @param noneable the Noneable object
   * @param <T> the type to cast to
   * @return null, if the noneable argument was None, or the cast object, otherwise.
   */
  private static @Nullable <T> T fromNoneable(Class<T> clazz, Object noneable) {
    if (noneable == Runtime.NONE) {
      return null;
    }

    return clazz.cast(noneable);
  }
}
