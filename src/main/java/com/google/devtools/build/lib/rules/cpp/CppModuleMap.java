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
import com.google.devtools.build.lib.packages.StarlarkInfo;
import net.starlark.java.eval.EvalException;

/** Structure for C++ module maps. Stores the name of the module and a .cppmap artifact. */
@Immutable
public final class CppModuleMap {
  private final StarlarkInfo moduleMap;

  public CppModuleMap(StarlarkInfo moduleMap) {
    this.moduleMap = moduleMap;
  }

  public Artifact getArtifact() {
    try {
      return moduleMap.getValue("file", Artifact.class);
    } catch (EvalException e) {
      throw new IllegalStateException(e);
    }
  }

  public String getName() {
    try {
      return moduleMap.getValue("name", String.class);
    } catch (EvalException e) {
      throw new IllegalStateException(e);
    }
  }
}
