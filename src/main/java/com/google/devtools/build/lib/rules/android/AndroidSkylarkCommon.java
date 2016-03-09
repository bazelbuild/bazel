// Copyright 2016 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * Common utilities for Skylark rules related to Android.
 */
@SkylarkModule(
  name = "android_common",
  doc = "Common utilities and fucntionality related to Android rules."
)
public class AndroidSkylarkCommon {
  @SkylarkCallable(
    name = "resource_source_directory",
    allowReturnNones = true,
    doc =
        "Returns a source directory for Android resource file. "
            + "The source directory is a prefix of resource's relative path up to "
            + "a directory that designates resource kind (cf. "
            + "http://developer.android.com/guide/topics/resources/providing-resources.html)."
  )
  public static PathFragment getSourceDirectoryRelativePathFromResource(Artifact resource) {
    return AndroidCommon.getSourceDirectoryRelativePathFromResource(resource);
  }
}
