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

import com.google.common.base.Supplier;
import com.google.common.base.Suppliers;
import java.io.IOException;

/**
 * A lazy, automatically populated registry.
 *
 * <p>Must not be accessed by any {@link CodecRegisterer} or {@link ObjectCodec} constructors or
 * static initializers.
 */
public class AutoRegistry {

  private static final Supplier<ObjectCodecRegistry> SUPPLIER =
      Suppliers.memoize(AutoRegistry::create);

  public static ObjectCodecRegistry get() {
    return SUPPLIER.get();
  }

  private static ObjectCodecRegistry create() {
    try {
      return CodecScanner.initializeCodecRegistry("com.google.devtools.build")
          .setAllowDefaultCodec(false)
          .build();
    } catch (IOException | ReflectiveOperationException e) {
      throw new IllegalStateException(e);
    }
  }
}
