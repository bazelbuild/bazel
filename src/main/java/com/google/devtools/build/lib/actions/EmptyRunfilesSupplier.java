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

package com.google.devtools.build.lib.actions;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Map;

/** Empty implementation of RunfilesSupplier */
public class EmptyRunfilesSupplier implements RunfilesSupplier {

  @AutoCodec public static final EmptyRunfilesSupplier INSTANCE = new EmptyRunfilesSupplier();

  private EmptyRunfilesSupplier() {}

  @Override
  public NestedSet<Artifact> getArtifacts() {
    return NestedSetBuilder.<Artifact>stableOrder().build();
  }

  @Override
  public ImmutableSet<PathFragment> getRunfilesDirs() {
    return ImmutableSet.of();
  }

  @Override
  public ImmutableMap<PathFragment, Map<PathFragment, Artifact>> getMappings() {
    return ImmutableMap.of();
  }

  @Override
  public ImmutableList<Artifact> getManifests() {
    return ImmutableList.<Artifact>of();
  }

  @Override
  public boolean isBuildRunfileLinks(PathFragment runfilesDir) {
    return false;
  }

  @Override
  public boolean isRunfileLinksEnabled(PathFragment runfilesDir) {
    return false;
  }
}
