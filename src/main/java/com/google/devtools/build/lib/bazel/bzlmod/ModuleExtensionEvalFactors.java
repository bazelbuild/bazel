// Copyright 2023 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.bzlmod;

import com.google.auto.value.AutoValue;
import com.ryanharter.auto.value.gson.GenerateTypeAdapter;

/**
 * This object holds the evaluation factors for module extensions in the lockfile, such as the
 * operating system and architecture it depends on. If an extension has no dependencies in this
 * regard, the object remains empty
 */
@AutoValue
@GenerateTypeAdapter
public abstract class ModuleExtensionEvalFactors {

  /** Returns the OS this extension is evaluated on, or empty if it doesn't depend on the os */
  public abstract String getOs();

  /**
   * Returns the architecture this extension is evaluated on, or empty if it doesn't depend on the
   * architecture
   */
  public abstract String getArch();

  public boolean isEmpty() {
    return getOs().isEmpty() && getArch().isEmpty();
  }

  public boolean hasSameDependenciesAs(ModuleExtensionEvalFactors other) {
    return getOs().isEmpty() == other.getOs().isEmpty()
        && getArch().isEmpty() == other.getArch().isEmpty();
  }

  public static ModuleExtensionEvalFactors create(String os, String arch) {
    return new AutoValue_ModuleExtensionEvalFactors(os, arch);
  }
}
