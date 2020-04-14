// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.buildinfo.BuildInfoCollection;
import com.google.devtools.build.lib.analysis.buildinfo.BuildInfoFactory;
import com.google.devtools.build.lib.analysis.buildinfo.BuildInfoKey;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.RepositoryName;

/**
 * Objc build info creation - passes on BuildInfo output file for consumption from Objc rules.
 *
 * @deprecated The native bundling rules have been deprecated. This class will be removed in the
 *     future.
 */
@Deprecated
public class ObjcBuildInfoFactory implements BuildInfoFactory {

  public static final BuildInfoKey KEY = new BuildInfoKey("ObjC");

  /**
   * Returns no actions, exactly the one BuildInfo artifact, and no buildChangelist artifacts.
   */
  @Override
  public BuildInfoCollection create(BuildInfoContext context, BuildConfiguration config,
      Artifact buildInfo, Artifact buildChangelist, RepositoryName repositoryName) {
    return new BuildInfoCollection(
        ImmutableList.<Action>of(),
        ImmutableList.of(buildInfo),
        ImmutableList.of(buildInfo));
  }

  @Override
  public BuildInfoKey getKey() {
    return KEY;
  }

  @Override
  public boolean isEnabled(BuildConfiguration config) {
    return config.hasFragment(ObjcConfiguration.class);
  }
}
