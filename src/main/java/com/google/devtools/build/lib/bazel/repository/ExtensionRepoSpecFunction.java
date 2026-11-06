// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.repository;

import com.google.devtools.build.lib.bazel.bzlmod.ModuleExtensionId;
import com.google.devtools.build.lib.bazel.bzlmod.RepoSpec;
import com.google.devtools.build.lib.bazel.bzlmod.SingleExtensionValue;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import javax.annotation.Nullable;

/** Looks up a single repository spec in a validated module extension's generated specs. */
final class ExtensionRepoSpecFunction implements SkyFunction {
  @AutoCodec
  record Key(ModuleExtensionId extensionId, RepositoryName repositoryName) implements SkyKey {
    private static final SkyKeyInterner<Key> interner = SkyKey.newInterner();

    @AutoCodec.Instantiator
    static Key create(ModuleExtensionId extensionId, RepositoryName repositoryName) {
      return interner.intern(new Key(extensionId, repositoryName));
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.EXTENSION_REPO_SPEC;
    }

    @Override
    public SkyKeyInterner<Key> getSkyKeyInterner() {
      return interner;
    }
  }

  sealed interface Value extends SkyValue {}

  @AutoCodec
  record Found(String internalName, RepoSpec spec) implements Value {}

  enum NotFound implements Value {
    INSTANCE
  }

  @Nullable
  @Override
  public Value compute(SkyKey skyKey, Environment env) throws InterruptedException {
    var key = (Key) skyKey;
    var extension =
        (SingleExtensionValue) env.getValue(SingleExtensionValue.key(key.extensionId()));
    if (extension == null) {
      return null;
    }
    String internalName = extension.canonicalRepoNameToInternalNames().get(key.repositoryName());
    return internalName == null
        ? NotFound.INSTANCE
        : new Found(internalName, extension.generatedRepoSpecs().get(internalName));
  }
}
