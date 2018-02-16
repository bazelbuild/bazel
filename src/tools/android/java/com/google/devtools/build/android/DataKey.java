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

/**
 * A general interface for resource and asset keys.
 *
 * <p>Resource and Assets are merged on the basis of a key value:
 *
 * <p>For Resources, this is the fully qualified name, consisting of the resource package, name,
 * type, and qualifiers.
 *
 * <p>For Assets, it is the asset path from the assets directory.
 */
public interface DataKey extends Comparable<DataKey>, Writeable {

  /** Returns a human readable string representation of the key. */
  String toPrettyString();

  /** Defines a total ordering on the different key types to assist in compareTo operations. */
  enum KeyType {
    ASSET_PATH,
    FULL_QUALIFIED_NAME
  }

  KeyType getKeyType();
}
