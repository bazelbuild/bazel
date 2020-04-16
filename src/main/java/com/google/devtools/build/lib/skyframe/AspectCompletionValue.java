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
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.lib.skyframe.AspectValueKey.AspectKey;
import com.google.devtools.build.lib.skyframe.CompletionFunction.TopLevelActionLookupKey;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Collection;

/**
 * The value of an AspectCompletion. Currently this just stores an Aspect.
 */
public class AspectCompletionValue implements SkyValue {
  @AutoCodec static final AspectCompletionValue INSTANCE = new AspectCompletionValue();

  private AspectCompletionValue() {}

  public static Iterable<SkyKey> keys(
      Collection<AspectValue> targets, final TopLevelArtifactContext ctx) {
    return Iterables.transform(
        targets, aspectValue -> AspectCompletionKey.create(aspectValue.getKey(), ctx));
  }

  /** The key of an AspectCompletionValue. */
  @AutoValue
  public abstract static class AspectCompletionKey implements TopLevelActionLookupKey {
    public static AspectCompletionKey create(
        AspectKey aspectKey, TopLevelArtifactContext topLevelArtifactContext) {
      return new AutoValue_AspectCompletionValue_AspectCompletionKey(
          topLevelArtifactContext, aspectKey);
    }

    @Override
    public abstract AspectKey actionLookupKey();

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.ASPECT_COMPLETION;
    }
  }
}
