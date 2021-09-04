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
import com.google.common.collect.ImmutableList;

/**
 * Specifies that the module should still come from a registry, but multiple versions of it should
 * be allowed to coexist.
 */
@AutoValue
public abstract class MultipleVersionOverride implements RegistryOverride {

  public static MultipleVersionOverride create(ImmutableList<Version> versions, String registry) {
    return new AutoValue_MultipleVersionOverride(versions, registry);
  }

  /** The versions of this module that should coexist. */
  public abstract ImmutableList<Version> getVersions();

  @Override
  public abstract String getRegistry();
}
