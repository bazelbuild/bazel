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
package com.google.devtools.build.lib.rules.cpp;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.CppModuleMapApi;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkThread;

/** Structure for C++ module maps. Stores the name of the module and a .cppmap artifact. */
@Immutable
public final class CppModuleMap implements CppModuleMapApi<Artifact> {
  public static final String SEPARATE_MODULE_SUFFIX = ".sep";

  // NOTE: If you add a field here, you'll likely need to update CppModuleMapAction.computeKey().
  private final Artifact artifact;
  private final String name;

  public CppModuleMap(Artifact artifact, String name) {
    this.artifact = artifact;
    this.name = name;
  }

  public Artifact getArtifact() {
    return artifact;
  }

  @Override
  public Artifact getArtifactForStarlark(StarlarkThread thread) throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return artifact;
  }

  public String getName() {
    return name;
  }

  @Override
  public String getNameForStarlark(StarlarkThread thread) throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return name;
  }

  @Override
  public int hashCode() {
    // It would be incorrect for two CppModuleMap instances in the same build graph to have the same
    // artifact but different names or umbrella headers. Since Artifacts' hash codes are cached, use
    // only it for efficiency.
    return artifact.hashCode();
  }

  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    }
    if (other instanceof CppModuleMap that) {
      return artifact.equals(that.artifact) && name.equals(that.name);
    }
    return false;
  }

  @Override
  public String toString() {
    return name + "@" + artifact;
  }
}
