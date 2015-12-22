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

import com.google.common.io.ByteSource;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.xcode.plmerge.proto.PlMergeProtos;
import com.google.devtools.build.xcode.plmerge.proto.PlMergeProtos.Control;

import java.io.IOException;
import java.io.InputStream;

/**
 * A byte source that can be used the generate a control file for the tool plmerge.
 */
public final class PlMergeControlBytes extends ByteSource {

  private final Bundling bundling;
  private final Artifact mergedPlist;

  /**
   * @param bundling  the {@code Bundling} instance describing the bundle
   *     for which to create a merged plist
   * @param mergedPlist the merged plist that should be bundled as Info.plist
   */
  public PlMergeControlBytes(Bundling bundling, Artifact mergedPlist) {
    this.bundling = bundling;
    this.mergedPlist = mergedPlist;
  }

  @Override
  public InputStream openStream() throws IOException {
    return control(bundling).toByteString().newInput();
  }

  private Control control(Bundling bundling) {
    PlMergeProtos.Control.Builder control =
        PlMergeProtos.Control.newBuilder()
            .addAllSourceFile(Artifact.toExecPaths(bundling.getBundleInfoplistInputs()))
            .setOutFile(mergedPlist.getExecPathString());

    if (bundling.getAutomaticInfoPlist() != null) {
      control.addImmutableSourceFile(bundling.getAutomaticInfoPlist().getExecPathString());
    }

    if (bundling.getPrimaryBundleId() != null) {
      control.setPrimaryBundleId(bundling.getPrimaryBundleId());
    }

    if (bundling.getFallbackBundleId() != null) {
      control.setFallbackBundleId(bundling.getFallbackBundleId());
    }

    if (bundling.variableSubstitutions() != null) {
      control.putAllVariableSubstitutionMap(bundling.variableSubstitutions());
    }

    control.setExecutableName(bundling.getName());

    return control.build();
  }
}
