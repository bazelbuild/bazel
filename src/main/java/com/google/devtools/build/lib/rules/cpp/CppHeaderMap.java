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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Map;

/**
 * Structure for Clang header maps. Stores the .hmap and -internal.hmap artifacts as well
 * as the actual header maps that should be written to disk.
 */
@Immutable
@AutoCodec
public final class CppHeaderMap {
  // NOTE: If you add a field here, you'll likely need to update CppHeaderMapAction.computeKey().
  private final Artifact artifact;
  private final String name;
  private final Map<PathFragment, PathFragment> map;

  CppHeaderMap(Artifact artifact, String name) {
    this.artifact = artifact;
    this.name = name;
    this.map = ImmutableMap.of();
  }

  public Artifact getArtifact() { return artifact; }

  public String getName() { return name; }

  public String getIncludePrefix() { return includePrefix; }

  public void addStuff() {
    map.put(PathFragment.create("Foo"), PathFragment.create("Bar"));
  }

  @Override
  public String toString() { return name + "@" + artifact; }

}
