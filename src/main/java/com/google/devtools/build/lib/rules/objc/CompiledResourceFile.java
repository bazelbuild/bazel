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

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;

/**
 * Represents a strings file.
 */
public class CompiledResourceFile {
  private final Artifact original;
  private final BundleableFile bundled;

  private CompiledResourceFile(Artifact original, BundleableFile bundled) {
    this.original = Preconditions.checkNotNull(original);
    this.bundled = Preconditions.checkNotNull(bundled);
  }

  /**
   * The checked-in version of the bundled file.
   */
  public Artifact getOriginal() {
    return original;
  }

  public BundleableFile getBundled() {
    return bundled;
  }

  public static final Function<CompiledResourceFile, BundleableFile> TO_BUNDLED =
      new Function<CompiledResourceFile, BundleableFile>() {
        @Override
        public BundleableFile apply(CompiledResourceFile input) {
          return input.bundled;
        }
      };

  /**
   * Given a sequence of artifacts corresponding to {@code .strings} files, returns a sequence of
   * the same length of instances of this class. The value returned by {@link #getBundled()} of each
   * instance will be the plist file in binary form.
   */
  public static Iterable<CompiledResourceFile> fromStringsFiles(
      IntermediateArtifacts intermediateArtifacts, Iterable<Artifact> strings) {
    ImmutableList.Builder<CompiledResourceFile> result = new ImmutableList.Builder<>();
    for (Artifact originalFile : strings) {
      Artifact binaryFile = intermediateArtifacts.convertedStringsFile(originalFile);
      result.add(new CompiledResourceFile(
          originalFile,
          new BundleableFile(
              binaryFile, BundleableFile.flatBundlePath(originalFile.getExecPath()))));
    }
    return result.build();
  }
}
