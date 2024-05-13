// Copyright 2021 The Bazel Authors. All rights reserved.
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
//

package com.google.devtools.build.lib.bazel.bzlmod;

import com.google.auto.value.AutoValue;
import com.ryanharter.auto.value.gson.GenerateTypeAdapter;

/**
 * An abridged version of a {@link Module}, with a reduced set of information available, used for
 * module extension resolution.
 */
@AutoValue
@GenerateTypeAdapter
public abstract class AbridgedModule {
  public abstract String getName();

  public abstract Version getVersion();

  public abstract ModuleKey getKey();

  public static AbridgedModule from(Module module) {
    return new AutoValue_AbridgedModule(module.getName(), module.getVersion(), module.getKey());
  }
}
