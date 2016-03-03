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

import com.google.common.base.Function;
import com.google.common.collect.Iterables;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.util.Collection;

/**
 * The value of an AspectCompletion. Currently this just stores an Aspect.
 */
public class AspectCompletionValue implements SkyValue {
  private final AspectValue aspectValue;

  AspectCompletionValue(AspectValue aspectValue) {
    this.aspectValue = aspectValue;
  }

  public AspectValue getAspectValue() {
    return aspectValue;
  }

  public static Iterable<SkyKey> keys(Collection<AspectValue> targets) {
    return Iterables.transform(
        targets,
        new Function<AspectValue, SkyKey>() {
          @Override
          public SkyKey apply(AspectValue aspectValue) {
            return SkyKey.create(SkyFunctions.ASPECT_COMPLETION, aspectValue.getKey());
          }
        });
  }
}
