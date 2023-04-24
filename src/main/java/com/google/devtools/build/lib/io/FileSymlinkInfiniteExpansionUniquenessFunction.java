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
package com.google.devtools.build.lib.io;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;

/**
 * A {@link com.google.devtools.build.skyframe.SkyFunction} that has the side effect of reporting a
 * file symlink expansion error exactly once. This is achieved by forcing the same value key for two
 * logically equivalent expansion errors (e.g. ['a' -> 'b' -> 'c' -> 'a/nope'] and ['b' -> 'c' ->
 * 'a' -> 'a/nope']), and letting Skyframe do its magic.
 */
public class FileSymlinkInfiniteExpansionUniquenessFunction
    extends AbstractFileChainUniquenessFunction {
  public static final SkyFunctionName NAME =
      SkyFunctionName.createHermetic("FILE_SYMLINK_INFINITE_EXPANSION_UNIQUENESS");

  public static SkyKey key(ImmutableList<RootedPath> cycle) {
    return Key.create(AbstractFileChainUniquenessFunction.canonicalize(cycle));
  }

  @AutoCodec.VisibleForSerialization
  @AutoCodec
  static class Key extends AbstractSkyKey.WithCachedHashCode<ImmutableList<RootedPath>> {
    private static final SkyKeyInterner<Key> interner = SkyKey.newInterner();

    private Key(ImmutableList<RootedPath> arg) {
      super(arg);
    }

    @AutoCodec.VisibleForSerialization
    @AutoCodec.Instantiator
    static Key create(ImmutableList<RootedPath> arg) {
      return interner.intern(new Key(arg));
    }

    @Override
    public SkyFunctionName functionName() {
      return NAME;
    }

    @Override
    public SkyKeyInterner<Key> getSkyKeyInterner() {
      return interner;
    }
  }

  @Override
  protected String elementToString(RootedPath elt) {
    return elt.asPath().toString();
  }

  @Override
  protected String getConciseDescription() {
    return "infinite symlink expansion";
  }

  @Override
  protected String getHeaderMessage() {
    return "[start of symlink chain]";
  }

  @Override
  protected String getFooterMessage() {
    return "[end of symlink chain]";
  }
}

