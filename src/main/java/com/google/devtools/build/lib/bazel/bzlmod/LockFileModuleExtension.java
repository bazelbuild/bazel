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

package com.google.devtools.build.lib.bazel.bzlmod;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;
import com.ryanharter.auto.value.gson.GenerateTypeAdapter;

/**
 * This object serves as a container for the transitive digest (obtained from transitive .bzl files)
 * and the generated repositories from evaluating a module extension. Its purpose is to store this
 * information within the lockfile.
 */
@AutoValue
@GenerateTypeAdapter
public abstract class LockFileModuleExtension implements Postable {

  public static Builder builder() {
    return new AutoValue_LockFileModuleExtension.Builder()
        // TODO(salmasamy) can be removed when updating lockfile version
        .setEnvVariables(ImmutableMap.of())
        .setAccumulatedFileDigests(ImmutableMap.of());
  }

  @SuppressWarnings("mutable")
  public abstract byte[] getBzlTransitiveDigest();

  public abstract ImmutableMap<Label, String> getAccumulatedFileDigests();

  public abstract ImmutableMap<String, String> getEnvVariables();

  public abstract ImmutableMap<String, RepoSpec> getGeneratedRepoSpecs();

  public abstract Builder toBuilder();

  /** Builder type for {@link LockFileModuleExtension}. */
  @AutoValue.Builder
  public abstract static class Builder {

    public abstract Builder setBzlTransitiveDigest(byte[] digest);

    public abstract Builder setAccumulatedFileDigests(ImmutableMap<Label, String> value);

    public abstract Builder setEnvVariables(ImmutableMap<String, String> value);

    public abstract Builder setGeneratedRepoSpecs(ImmutableMap<String, RepoSpec> value);

    public abstract LockFileModuleExtension build();
  }
}
