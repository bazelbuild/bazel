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
import com.google.devtools.build.lib.skylarkbuildapi.android.AndroidManifestApi;
import javax.annotation.Nullable;

/**
 * A {@link StampedAndroidManifest} with placeholders removed to avoid interfering with the legacy
 * manifest merger.
 *
 * <p>TODO(b/30817309) Just use {@link StampedAndroidManifest} once the legacy merger is removed.
 */
public class ProcessedAndroidManifest extends StampedAndroidManifest implements AndroidManifestApi {

  ProcessedAndroidManifest(Artifact manifest, @Nullable String pkg, boolean exported) {
    super(manifest, pkg, exported);
  }

  @Override
  public boolean equals(Object object) {
    return (object instanceof ProcessedAndroidManifest) && super.equals(object);
  }
}
