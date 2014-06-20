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

package com.google.devtools.build.lib.view;

import java.util.Objects;

/**
 * Contains options which control the set of artifacts to build for top-level targets.
 */
public final class TopLevelArtifactContext {

  public static final TopLevelArtifactContext DEFAULT = new TopLevelArtifactContext(
      "", /*compileOnly=*/false, /*compilationPrerequisitesOnly*/false,
      /*runTestsExclusively=*/false);

  private final boolean compileOnly;
  private final boolean compilationPrerequisitesOnly;
  private final boolean runTestsExclusively;
  private final String buildCommand;

  public TopLevelArtifactContext(String buildCommand, boolean compileOnly,
      boolean compilationPrerequisitesOnly, boolean runTestsExclusively) {
    this.buildCommand = buildCommand;
    this.compileOnly = compileOnly;
    this.compilationPrerequisitesOnly = compilationPrerequisitesOnly;
    this.runTestsExclusively = runTestsExclusively;
  }

  /** Returns the build command as a string. */
  public String buildCommand() {
    return buildCommand;
  }

  /** Returns the value of the --compile_only flag. */
  public boolean compileOnly() {
    return compileOnly;
  }

  /** Returns the value of the --compilation_prerequisites_only flag. */
  public boolean compilationPrerequisitesOnly() {
    return compilationPrerequisitesOnly;
  }

  /** Whether to run tests in exclusive mode. */
  public boolean runTestsExclusively() {
    return runTestsExclusively;
  }

  // TopLevelArtifactContexts are stored in maps in BuildView,
  // so equals() and hashCode() need to work.
  @Override
  public boolean equals(Object other) {
    if (other instanceof TopLevelArtifactContext) {
      TopLevelArtifactContext otherContext = (TopLevelArtifactContext) other;
      return buildCommand.equals(otherContext.buildCommand) &&
             compileOnly == otherContext.compileOnly &&
             runTestsExclusively == otherContext.runTestsExclusively;
    } else {
      return false;
    }
  }

  @Override
  public int hashCode() {
    return Objects.hash(buildCommand, compileOnly, runTestsExclusively);
  }
}
