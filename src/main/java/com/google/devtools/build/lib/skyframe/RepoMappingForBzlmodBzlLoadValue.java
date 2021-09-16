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

package com.google.devtools.build.lib.skyframe;

import com.google.auto.value.AutoValue;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyValue;

/** The result of {@link RepoMappingForBzlmodBzlLoadFunction}. */
@AutoValue
public abstract class RepoMappingForBzlmodBzlLoadValue implements SkyValue {
  public abstract RepositoryMapping getRepoMapping();

  public static RepoMappingForBzlmodBzlLoadValue create(RepositoryMapping repoMapping) {
    return new AutoValue_RepoMappingForBzlmodBzlLoadValue(repoMapping);
  }

  public static Key key(RepositoryName arg) {
    return Key.create(arg);
  }

  @AutoCodec
  static class Key extends AbstractSkyKey<RepositoryName> {
    private static final Interner<Key> interner = BlazeInterners.newWeakInterner();

    private Key(RepositoryName arg) {
      super(arg);
    }

    @AutoCodec.Instantiator
    static Key create(RepositoryName arg) {
      return interner.intern(new Key(arg));
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.REPO_MAPPING_FOR_BZLMOD_BZL_LOAD;
    }
  }
}
