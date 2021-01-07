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

package com.google.devtools.build.lib.rules.cpp;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.ExtraLinkTimeLibraryApi;

/**
 * An extra library to include in a link. The actual library is built at link time.
 *
 * <p>This can be used for non-C++ inputs to a C++ link. A class that implements this interface will
 * support transitively gathering all inputs from link dependencies, and then combine them all
 * together into a set of C++ libraries.
 *
 * <p>Any implementations must be immutable (and therefore thread-safe), because this is passed
 * between rules and accessed in a multi-threaded context.
 */
public interface ExtraLinkTimeLibrary extends ExtraLinkTimeLibraryApi {

  /** Output of {@link #buildLibraries}. Pair of libraries to link and runtime libraries. */
  class BuildLibraryOutput {
    public NestedSet<CcLinkingContext.LinkerInput> linkerInputs;
    public NestedSet<Artifact> runtimeLibraries;

    public BuildLibraryOutput(
        NestedSet<CcLinkingContext.LinkerInput> linkerInputs,
        NestedSet<Artifact> runtimeLibraries) {
      this.linkerInputs = linkerInputs;
      this.runtimeLibraries = runtimeLibraries;
    }

    public NestedSet<CcLinkingContext.LinkerInput> getLinkerInputs() {
      return linkerInputs;
    }

    public NestedSet<Artifact> getRuntimeLibraries() {
      return runtimeLibraries;
    }
  }

  /**
   * Build and return the LinkerInput inputs to pass to the C++ linker and the associated runtime
   * libraries.
   */
  BuildLibraryOutput buildLibraries(
      RuleContext context, boolean staticMode, boolean forDynamicLibrary)
      throws InterruptedException, RuleErrorException;

  /**
   * Get a new Builder for this ExtraLinkTimeLibrary class.  This acts
   * like a static method, in that the result does not depend on the
   * current state of the object, and the new Builder starts out
   * empty.
   */
  Builder getBuilder();

  /**
   * The Builder interface builds an ExtraLinkTimeLibrary.
   */
  public interface Builder {
    /**
     * Add the inputs associated with another instance of the same
     * underlying ExtraLinkTimeLibrary type.
     */
    void addTransitive(ExtraLinkTimeLibrary dep);

    /**
     * Build the ExtraLinkTimeLibrary based on the inputs.
     */
    ExtraLinkTimeLibrary build();
  }
}
