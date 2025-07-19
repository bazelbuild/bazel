// Copyright 2025 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis;

import com.google.common.base.Suppliers;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.LocationExpander.LocationFunction;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;

/** Utility for building location functions in tests. */
final class LocationFunctionBuilder {
  private final Label root;
  private final boolean multiple;
  private LocationFunction.PathType pathType = LocationFunction.PathType.LOCATION;
  private final Map<Label, Collection<Artifact>> labelMap = new HashMap<>();

  LocationFunctionBuilder(String rootLabel, boolean multiple) {
    this.root = Label.parseCanonicalUnchecked(rootLabel);
    this.multiple = multiple;
  }

  public LocationFunction build() {
    return new LocationFunction(root, Suppliers.ofInstance(labelMap), pathType, multiple);
  }

  @CanIgnoreReturnValue
  public LocationFunctionBuilder setPathType(LocationFunction.PathType pathType) {
    this.pathType = pathType;
    return this;
  }

  @CanIgnoreReturnValue
  public LocationFunctionBuilder add(String label, String... paths) {
    labelMap.put(
        Label.parseCanonicalUnchecked(label),
        Arrays.stream(paths)
            .map(LocationFunctionBuilder::makeArtifact)
            .collect(Collectors.toList()));
    return this;
  }

  private static Artifact makeArtifact(String path) {
    FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    if (path.startsWith("/exec/out")) {
      return ActionsTestUtil.createArtifact(
          ArtifactRoot.asDerivedRoot(fs.getPath("/exec"), RootType.OUTPUT, "out"),
          fs.getPath(path));
    } else {
      return ActionsTestUtil.createArtifact(
          ArtifactRoot.asSourceRoot(Root.fromPath(fs.getPath("/exec"))), fs.getPath(path));
    }
  }
}
