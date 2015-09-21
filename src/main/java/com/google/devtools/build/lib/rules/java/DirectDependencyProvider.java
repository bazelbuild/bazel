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
package com.google.devtools.build.lib.rules.java;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

/**
 * A provider that returns the direct dependencies of a target. Used for strict dependency
 * checking.
 */
@Immutable
public final class DirectDependencyProvider implements TransitiveInfoProvider {

  private final ImmutableList<Dependency> strictDependencies;

  public DirectDependencyProvider(Iterable<Dependency> strictDependencies) {
    this.strictDependencies = ImmutableList.copyOf(strictDependencies);
  }

  /**
   * @returns the direct (strict) dependencies of this provider. All symbols that are directly
   * reachable from the sources of the provider should be available in one these artifacts.
   */
  public Iterable<Dependency> getStrictDependencies() {
    return strictDependencies;
  }

  /**
   * A pair of label and its generated list of artifacts.
   */
  public static class Dependency {
    private final Label label;

    // TODO(bazel-team): change this to Artifacts
    private final Iterable<String> fileExecPaths;

    public Dependency(Label label, Iterable<String> fileExecPaths) {
      this.label = label;
      this.fileExecPaths = fileExecPaths;
    }

    public Label getLabel() {
      return label;
    }

    public Iterable<String> getDependencyOutputs() {
      return fileExecPaths;
    }
  }
}
