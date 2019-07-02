// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.repository.downloader;

import com.google.common.hash.HashCode;
import com.google.devtools.build.lib.bazel.repository.cache.RepositoryCache.KeyType;

/** The content checksum for an HTTP download, which knows its own type. */
public class Checksum {
  private final KeyType keyType;
  private final HashCode hashCode;

  private Checksum(KeyType keyType, HashCode hashCode) {
    this.keyType = keyType;
    this.hashCode = hashCode;
  }

  /** Constructs a new Checksum for a given key type and hash, in hex format. */
  public static Checksum fromString(KeyType keyType, String hash) {
    if (!keyType.isValid(hash)) {
      throw new IllegalArgumentException("Invalid " + keyType + " checksum '" + hash + "'");
    }
    return new Checksum(keyType, HashCode.fromString(hash));
  }

  @Override
  public String toString() {
    return hashCode.toString();
  }

  public HashCode getHashCode() {
    return hashCode;
  }

  public KeyType getKeyType() {
    return keyType;
  }
}
