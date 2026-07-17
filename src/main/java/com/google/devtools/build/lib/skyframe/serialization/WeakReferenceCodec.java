// Copyright 2024 The Bazel Authors. All rights reserved.
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

import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.lang.ref.WeakReference;

/**
 * A codec for weak references.
 *
 * <p>We must always be prepared for a weak reference to suddenly vanish, so simply not serializing
 * the referenced object works.
 */
@SuppressWarnings({"rawtypes"})
public final class WeakReferenceCodec extends AsyncObjectCodec<WeakReference> {

  @Override
  public WeakReference deserializeAsync(
      AsyncDeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    WeakReference result = new WeakReference<>(null);
    context.registerInitialValue(result);
    return result;
  }

  @Override
  public Class<WeakReference> getEncodedClass() {
    return WeakReference.class;
  }

  @Override
  public void serialize(SerializationContext context, WeakReference obj, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    // We don't need to serialize anything; the referenced object is simply discarded since weak
    // references are only used for caching.
  }
}
