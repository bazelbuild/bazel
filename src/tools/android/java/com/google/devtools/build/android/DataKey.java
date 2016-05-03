// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android;

import java.io.IOException;
import java.io.OutputStream;

/**
 * A general interface for resource and asset keys.
 *
 * Resource and Assets are merged on the basis of a key value:
 *
 * For Resources, this is the fully qualified name, consisting of the resource package, name, type,
 * and qualifiers.
 *
 * For Assets, it is the asset path from the assets directory.
 */
public interface DataKey {
  /**
   * Writes the Key and the value size to a stream.
   *
   * @param output The destination stream to serialize the key.
   * @param valueSize The size, in bytes, of the serialized output for this key. The value size can
   * be used for calculating offsets of the value in the stream.
   */
  void serializeTo(OutputStream output, int valueSize) throws IOException;

  /**
   * Returns a human readable string representation of the key.
   */
  String toPrettyString();
}
