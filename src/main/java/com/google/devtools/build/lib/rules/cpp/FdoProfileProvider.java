// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.cpp;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.NativeProvider;
import com.google.devtools.build.lib.vfs.PathFragment;

/** Provider that contains the profile used for FDO. */
@Immutable
public final class FdoProfileProvider extends NativeInfo {
  public static final NativeProvider<FdoProfileProvider> PROVIDER =
      new NativeProvider<FdoProfileProvider>(FdoProfileProvider.class, "FdoProfileInfo") {};

  private final Artifact profileArtifact;
  private final PathFragment fdoPath;

  private FdoProfileProvider(Artifact profileArtifact, PathFragment fdoPath) {
    super(PROVIDER);
    this.profileArtifact = profileArtifact;
    this.fdoPath = fdoPath;
  }

  public static FdoProfileProvider fromAbsolutePath(PathFragment fdoPath) {
    return new FdoProfileProvider(/* profileArtifact= */ null, fdoPath);
  }

  public static FdoProfileProvider fromArtifact(Artifact profileArtifact) {
    return new FdoProfileProvider(profileArtifact, /* fdoPath= */ null);
  }

  public Artifact getProfileArtifact() {
    return profileArtifact;
  }

  public PathFragment getFdoPath() {
    return fdoPath;
  }
}
