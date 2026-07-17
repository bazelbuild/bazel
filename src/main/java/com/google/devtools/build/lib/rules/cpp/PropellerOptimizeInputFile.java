// Copyright 2020 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.StructImpl;
import net.starlark.java.eval.EvalException;

/** Value object reused by propeller configurations that has two artifacts. */
@Immutable
public final class PropellerOptimizeInputFile {
  private final StructImpl propellerOptimiseInputFile;

  public PropellerOptimizeInputFile(StructImpl propellerOptimiseInputFile) {
    this.propellerOptimiseInputFile = propellerOptimiseInputFile;
  }

  public Artifact getCcArtifact() throws EvalException {
    return propellerOptimiseInputFile.getNoneableValue("cc_profile", Artifact.class);
  }

  public Artifact getLdArtifact() throws EvalException {
    return propellerOptimiseInputFile.getNoneableValue("ld_profile", Artifact.class);
  }
}
