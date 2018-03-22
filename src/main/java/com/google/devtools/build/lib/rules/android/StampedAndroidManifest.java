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
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

/** An {@link AndroidManifest} stamped with the correct package. */
@Immutable
public class StampedAndroidManifest extends AndroidManifest {

  StampedAndroidManifest(
      RuleContext ruleContext, Artifact manifest, String pkg, boolean isDummy) {
    super(ruleContext, manifest, pkg, isDummy);
  }

  @Override
  StampedAndroidManifest stamp() {
    // This manifest is already stamped
    return this;
  }

  /**
   * Gets the manifest artifact wrapped by this object. Stamped manifests are guaranteed to have a
   * non-null manifest artifact.
   */
  @Override
  Artifact getManifest() {
    return super.getManifest();
  }

  public AndroidManifestInfo toProvider() {
    return AndroidManifestInfo.of(getManifest(), getPackage(), isDummy());
  }
}
