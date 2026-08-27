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
package com.google.devtools.build.lib.skyframe;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FileContentsProxy;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator.AspectKey;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.StallableSkykey;
import java.util.Collection;
import java.util.Objects;
import javax.annotation.Nullable;

/** The value of an AspectCompletion. */
public class AspectCompletionValue implements SkyValue {
  @SerializationConstant
  static final AspectCompletionValue INSTANCE = new AspectCompletionValue(null);

  @Nullable private final ImmutableMap<Artifact, FileContentsProxy> materializedOutputs;

  private AspectCompletionValue(
      @Nullable ImmutableMap<Artifact, FileContentsProxy> materializedOutputs) {
    this.materializedOutputs = materializedOutputs;
  }

  public static AspectCompletionValue create(
      @Nullable ImmutableMap<Artifact, FileContentsProxy> materializedOutputs) {
    return materializedOutputs == null || materializedOutputs.isEmpty()
        ? INSTANCE
        : new AspectCompletionValue(materializedOutputs);
  }

  /** See {@link TargetCompletionValue#getMaterializedOutputs}. */
  @Nullable
  public ImmutableMap<Artifact, FileContentsProxy> getMaterializedOutputs() {
    return materializedOutputs;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (!(o instanceof AspectCompletionValue that)) {
      return false;
    }
    return Objects.equals(materializedOutputs, that.materializedOutputs);
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(materializedOutputs);
  }

  public static Iterable<SkyKey> keys(Collection<AspectKey> keys, TopLevelArtifactContext ctx) {
    return Iterables.transform(keys, k -> AspectCompletionKey.create(k, ctx));
  }

  /** The key of an AspectCompletionValue. */
  @AutoValue
  public abstract static class AspectCompletionKey
      implements TopLevelActionLookupKeyWrapper, StallableSkykey {
    public static AspectCompletionKey create(
        AspectKey aspectKey, TopLevelArtifactContext topLevelArtifactContext) {
      return new AutoValue_AspectCompletionValue_AspectCompletionKey(
          topLevelArtifactContext, aspectKey);
    }

    @Override
    public abstract AspectKey actionLookupKey();

    @Override
    public final SkyFunctionName functionName() {
      return SkyFunctions.ASPECT_COMPLETION;
    }

    @Override
    public final boolean valueIsShareable() {
      return false;
    }
  }
}
