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

import java.util.Collections;

/**
 * Custom registration behavior for a codec.
 *
 * <p>This class should only be needed for low-level codecs like collections and probably shouldn't
 * be used in normal application code.
 *
 * <p>Instances of this class are discovered automatically by {@link CodecScanner} and used for
 * custom registration of a codec. This can be useful when a single codec is used for multiple
 * classes or when the class that is being serialized has a hidden type, e.g., {@link
 * com.google.common.collect.RegularImmutableList}.
 *
 * <p>If a {@link CodecRegisterer} definition exists, the codec will only be registered by the
 * {@link CodecRegisterer#register} method. Otherwise, an attempt will be made to register the codec
 * to its generic parameter type.
 *
 * <p>Implementations must have a default constructor.
 *
 * <p>Inheriting {@link CodecRegisterer} through a superclass is illegal. It must be directly
 * implemented. Also, the generic parameter of {@link CodecRegisterer} must be reified.
 *
 * <p>Constraint violations will cause exceptions to be raised from {@link CodecScanner}.
 */
public interface CodecRegisterer<T extends ObjectCodec<?>> {

  default Iterable<? extends ObjectCodec<?>> getCodecsToRegister() {
    return Collections.emptyList();
  }
}
