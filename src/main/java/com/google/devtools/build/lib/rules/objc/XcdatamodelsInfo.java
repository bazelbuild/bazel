// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.common.base.Optional;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.view.RuleContext;
import com.google.devtools.build.xcode.util.Value;

import java.util.Collection;
import java.util.Map;

/**
 * Contains information specific to xcdatamodels for a single rule.
 */
class XcdatamodelsInfo extends Value<XcdatamodelsInfo> {
  private ImmutableSet<Artifact> notInXcdatamodelDir;
  private ImmutableSet<Xcdatamodel> xcdatamodels;

  private XcdatamodelsInfo(
      ImmutableSet<Artifact> notInXcdatamodelDir, ImmutableSet<Xcdatamodel> xcdatamodels) {
    super(ImmutableMap.of(
        "notInXcdatamodelDir", notInXcdatamodelDir,
        "xcdatamodels", xcdatamodels));
    this.notInXcdatamodelDir = notInXcdatamodelDir;
    this.xcdatamodels = xcdatamodels;
  }

  /**
   * Returns all Artifacts specified in the datamodels attribute that do not have an appropriate
   * container. It is considered a rule error for any Artifact to fall in this category.
   */
  public ImmutableSet<Artifact> getNotInXcdatamodelDir() {
    return notInXcdatamodelDir;
  }

  public ImmutableSet<Xcdatamodel> getXcdatamodels() {
    return xcdatamodels;
  }

  /**
   * For all artifacts specified by the datamodels attribute, returns a grouping of the using a
   * multimap according to their container. If the map key is {@link Optional#absent()}, then the
   * artifact is not inside an appropriate .xcdatamodel[d] container.
   */
  private static Multimap<Optional<PathFragment>, Artifact> artifactsByContainer(
      RuleContext context) {
    ImmutableSetMultimap.Builder<Optional<PathFragment>, Artifact> result =
        new ImmutableSetMultimap.Builder<>();
    for (Artifact datamodelArtifact : context.getPrerequisiteArtifacts("datamodels", Mode.TARGET)) {
      Optional<PathFragment> modelDir =
          Xcdatamodel.nearestContainerEndingWith(".xcdatamodeld", datamodelArtifact)
              .or(Xcdatamodel.nearestContainerEndingWith(".xcdatamodel", datamodelArtifact));
      result.put(modelDir, datamodelArtifact);
    }
    return result.build();
  }

  /**
   * Creates an instance populated with the datamodels information of the rule corresponding to
   * some {@code RuleContext}.
   */
  public static XcdatamodelsInfo fromRule(RuleContext context) {
    ImmutableSet.Builder<Xcdatamodel> result = new ImmutableSet.Builder<>();
    Multimap<Optional<PathFragment>, Artifact> artifactsByContainer = artifactsByContainer(context);

    for (Map.Entry<Optional<PathFragment>, Collection<Artifact>> modelDirEntry :
        artifactsByContainer.asMap().entrySet()) {
      for (PathFragment container : modelDirEntry.getKey().asSet()) {
        Artifact outputZip = ObjcRuleClasses.compiledMomZipArtifact(context, container);
        result.add(
            new Xcdatamodel(outputZip, ImmutableSet.copyOf(modelDirEntry.getValue()), container));
      }
    }

    return new XcdatamodelsInfo(
        ImmutableSet.copyOf(artifactsByContainer.get(Optional.<PathFragment>absent())),
        result.build());
  }
}
