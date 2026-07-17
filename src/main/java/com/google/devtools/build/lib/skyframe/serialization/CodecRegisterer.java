// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe.serialization;

import com.google.common.collect.ImmutableList;

/**
 * Custom registration behavior for {@link ObjectCodec}.
 *
 * <p>This class should only be needed for low-level codecs like collections and probably shouldn't
 * be used in normal application code.
 *
 * <p>Instances of this class are discovered automatically by {@link CodecScanner} and used for
 * custom registration of codec instances. This can be useful when a codec requires constructor
 * parameters, e.g. because it is used to serialize different classes (see for example {@link
 * MapEntryCodec}).
 *
 * <p>Implementations must have a class name ending in {@code CodecRegisterer} to be recognized by
 * {@link CodecScanner}. There must also be a parameterless constructor, or else {@link
 * CodecScanner} will throw an exception.
 *
 * <p>The existence of a {@link CodecRegisterer} does not prevent automatic registration of an
 * {@link ObjectCodec} class. If only manual registration is desired, the codec class should
 * override {@link ObjectCodec#autoRegister} if it otherwise qualifies (note that lack of a
 * parameterless constructor disqualifies a codec from automatic registration).
 */
public interface CodecRegisterer {
  ImmutableList<ObjectCodec<?>> getCodecsToRegister();
}
