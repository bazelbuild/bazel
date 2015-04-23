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


import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;

/**
 * A sequence of xib source files. Each {@code .xib} file can be compiled to a {@code .nib} file or
 * directory. Because it might be a directory, we always use zip files to store the output and use
 * the {@code actooloribtoolzip} utility to run ibtool and zip the output.
 */
public final class XibFiles extends IterableWrapper<Artifact> {
  public XibFiles(Iterable<Artifact> artifacts) {
    super(artifacts);
  }

  /**
   * Returns a sequence where each element of this sequence is converted to the file which contains
   * the compiled contents of the xib.
   */
  public ImmutableList<Artifact> compiledZips(IntermediateArtifacts intermediateArtifacts) {
    ImmutableList.Builder<Artifact> zips = new ImmutableList.Builder<>();
    for (Artifact xib : this) {
      zips.add(intermediateArtifacts.compiledXibFileZip(xib));
    }
    return zips.build();
  }
}
