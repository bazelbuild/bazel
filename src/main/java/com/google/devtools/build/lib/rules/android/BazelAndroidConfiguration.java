// Copyright 2022 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.RequiresOptions;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.errorprone.annotations.CheckReturnValue;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;

/** Configuration fragment for Android rules that is specific to Bazel. */
@Immutable
@StarlarkBuiltin(name = "bazel_android", category = DocCategory.CONFIGURATION_FRAGMENT)
@RequiresOptions(options = {BazelAndroidConfiguration.Options.class})
@CheckReturnValue
public class BazelAndroidConfiguration extends Fragment {

  @Override
  public boolean isImmutable() {
    return true; // immutable and Starlark-hashable
  }

  /** Android configuration options that are specific to Bazel. */
  public static class Options extends FragmentOptions {

    @Option(
        name = "merge_android_manifest_permissions",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
        help =
            "If enabled, the manifest merger will merge uses-permission and "
                + "uses-permission-sdk-23 attributes.")
    public boolean mergeAndroidManifestPermissions;
  }

  private final boolean mergeAndroidManifestPermissions;

  public BazelAndroidConfiguration(BuildOptions buildOptions) {
    Options options = buildOptions.get(BazelAndroidConfiguration.Options.class);
    this.mergeAndroidManifestPermissions = options.mergeAndroidManifestPermissions;
  }

  @StarlarkMethod(
      name = "merge_android_manifest_permissions",
      structField = true,
      doc = "The value of --merge_android_manifest_permissions flag.")
  public boolean getMergeAndroidManifestPermissions() {
    return this.mergeAndroidManifestPermissions;
  }
}
