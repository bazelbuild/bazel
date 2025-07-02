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

package com.google.devtools.build.lib.bazel.repository;

import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.NotComparableSkyValue;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;

/**
 * The result of {@link RepoDefinitionFunction}, holding a repository rule instance.
 *
 * <p>This has to be a {@link NotComparableSkyValue} for a very subtle reason. Two {@link
 * RepoDefinitionValue}s can compare equal if, for example, the .bzl file containing the repo rule
 * hasn't changed across a Bazel invocation, and the attributes stay the same. However, this doesn't
 * mean that the two definitions are actually equivalent, because certain information the repo rule
 * has access to (notably, the repo mapping applicable to the .bzl file) is *not* encoded in the
 * {@link RepoRule} object. In the particular case of the repo mapping (usable by the Starlark
 * {@code Label()} function), the repo rule's impl function essentially closes over it, but the
 * {@link net.starlark.java.eval.StarlarkCallable} object stored in {@link RepoRule} does *not*
 * compare unequal if only its containing .bzl's repo mapping is different.
 *
 * <p>Certainly, we can fix this by somehow making {@link RepoRule} store all the information it
 * could technically close over, and use that to influence its {@code equals} method; but we can't
 * easily guarantee the exhaustiveness of this (it's just very subtle). Instead, we declare {@link
 * RepoDefinitionValue} to be a {@link NotComparableSkyValue}, which inherits the condition that we
 * used to have when repo definitions were stored as {@code Rule}s in {@code Package}s; and no
 * {@code Package}s compare equal, ever.
 *
 * <p>This means that we're relying on the repo marker files to be the ultimate "change pruners" of
 * this SkyValue.
 */
public sealed interface RepoDefinitionValue extends NotComparableSkyValue {
  SkyFunctionName REPO_DEFINITION = SkyFunctionName.createHermetic("REPO_DEFINITION");

  RepoDefinitionValue NOT_FOUND = new NotFound();

  /** No repo found with the given name. */
  @AutoCodec
  record NotFound() implements RepoDefinitionValue {}

  /** A repo with the given name is found. */
  @AutoCodec
  record Found(RepoDefinition repoDefinition) implements RepoDefinitionValue {}

  static Key key(RepositoryName repositoryName) {
    return Key.create(repositoryName);
  }

  /** Key type for {@link RepoDefinitionValue}. */
  @AutoCodec
  class Key extends AbstractSkyKey<RepositoryName> {
    private static final SkyKeyInterner<Key> interner = SkyKey.newInterner();

    private Key(RepositoryName arg) {
      super(arg);
    }

    @VisibleForSerialization
    @AutoCodec.Instantiator
    static Key create(RepositoryName arg) {
      return interner.intern(new Key(arg));
    }

    @Override
    public SkyFunctionName functionName() {
      return REPO_DEFINITION;
    }

    @Override
    public SkyKeyInterner<Key> getSkyKeyInterner() {
      return interner;
    }
  }
}
