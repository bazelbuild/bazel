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
package com.google.devtools.build.lib.actions;

import com.google.protobuf.ByteString;
import java.io.Serializable;

/** A marker interface for objects which can return a byte[] digest. */
public interface HasDigest extends Serializable {
  byte[] getDigest();

  public static final HasDigest EMPTY = new ByteStringDigest(new byte[] {});

  /** An immutable wrapper around a byte[] digest. */
  public static class ByteStringDigest implements HasDigest {
    private final ByteString bytes;

    public ByteStringDigest(byte[] bytes) {
      this.bytes = ByteString.copyFrom(bytes);
    }

    @Override
    public byte[] getDigest() {
      return bytes.toByteArray();
    }

    @Override
    public boolean equals(Object other) {
      if (other instanceof ByteStringDigest) {
        return bytes.equals(((ByteStringDigest) other).bytes);
      }
      return false;
    }

    @Override
    public int hashCode() {
      return bytes.hashCode();
    }
  }
}
