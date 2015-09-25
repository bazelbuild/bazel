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

package com.google.devtools.build.lib.rules.objc;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.Collection;
import java.util.Map;

/**
 * Utility code for getting information specific to xcdatamodels for a single rule.
 */
class Xcdatamodels {
  private Xcdatamodels() {}

  static final ImmutableList<FileType> CONTAINER_TYPES =
      ImmutableList.of(FileType.of(".xcdatamodeld"), FileType.of(".xcdatamodel"));

  static Iterable<PathFragment> datamodelDirs(Iterable<Artifact> xcdatamodels) {
    ImmutableSet.Builder<PathFragment> result = new ImmutableSet.Builder<>();
    for (Collection<Artifact> artifacts : byContainer(xcdatamodels).asMap().values()) {
      result.addAll(ObjcCommon.uniqueContainers(artifacts, FileType.of(".xcdatamodel")));
    }
    return result.build();
  }

  static Iterable<Xcdatamodel> xcdatamodels(
      IntermediateArtifacts intermediateArtifacts, Iterable<Artifact> xcdatamodels) {
    ImmutableSet.Builder<Xcdatamodel> result = new ImmutableSet.Builder<>();
    Multimap<PathFragment, Artifact> artifactsByContainer = byContainer(xcdatamodels);

    for (Map.Entry<PathFragment, Collection<Artifact>> modelDirEntry :
        artifactsByContainer.asMap().entrySet()) {
      PathFragment container = modelDirEntry.getKey();
      Artifact outputZip = intermediateArtifacts.compiledMomZipArtifact(container);
      result.add(
          new Xcdatamodel(outputZip, ImmutableSet.copyOf(modelDirEntry.getValue()), container));
    }

    return result.build();
  }


  /**
   * Arrange a sequence of artifacts into entries of a multimap by their nearest container
   * directory, preferring {@code .xcdatamodeld} over {@code .xcdatamodel}.
   * If an artifact is not inside any containing directory, then it is not present in the returned
   * map.
   */
  static Multimap<PathFragment, Artifact> byContainer(Iterable<Artifact> artifacts) {
    ImmutableSetMultimap.Builder<PathFragment, Artifact> result =
        new ImmutableSetMultimap.Builder<>();
    for (Artifact artifact : artifacts) {
      for (PathFragment modelDir :
          ObjcCommon.nearestContainerMatching(CONTAINER_TYPES, artifact).asSet()) {
        result.put(modelDir, artifact);
      }
    }
    return result.build();
  }
}
