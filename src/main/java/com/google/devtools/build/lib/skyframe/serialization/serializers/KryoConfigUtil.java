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

package com.google.devtools.build.lib.skyframe.serialization.serializers;

import com.esotericsoftware.kryo.Kryo;
import com.google.common.collect.Ordering;
import java.util.Collections;

/** Utility for Kryo configuration. */
public class KryoConfigUtil {
  private KryoConfigUtil() {}

  /** Creates a {@link Kryo} instance with extensions defined in this package. */
  public static Kryo create() {
    Kryo kryo = new Kryo(new CanonicalReferenceResolver());
    registerSerializers(kryo);
    return kryo;
  }

  /** Registers Serializers defined in this package. */
  private static void registerSerializers(Kryo kryo) {
    kryo.register(Ordering.natural().getClass());
    kryo.register(Collections.reverseOrder().getClass());

    ClassSerializer.registerSerializers(kryo);
    HashCodeSerializer.registerSerializers(kryo);
    ImmutableListSerializer.registerSerializers(kryo);
    ImmutableMapSerializer.registerSerializers(kryo);
    ImmutableMultimapSerializer.registerSerializers(kryo);
    ImmutableSetSerializer.registerSerializers(kryo);
    ImmutableSortedMapSerializer.registerSerializers(kryo);
    ImmutableSortedSetSerializer.registerSerializers(kryo);
    MapEntrySerializer.registerSerializers(kryo);
    MultimapSerializer.registerSerializers(kryo);
    PatternSerializer.registerSerializers(kryo);
    ReverseListSerializer.registerSerializers(kryo);
    OptionalSerializer.registerSerializers(kryo);
    UnmodifiableListSerializer.registerSerializers(kryo);
    UnmodifiableMapSerializer.registerSerializers(kryo);
    UnmodifiableNavigableSetSerializer.registerSerializers(kryo);
  }
}
