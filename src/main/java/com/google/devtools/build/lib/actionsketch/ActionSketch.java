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

package com.google.devtools.build.lib.actionsketch;

import com.google.auto.value.AutoValue;
import com.google.protobuf.ByteString;
import java.math.BigInteger;
import java.nio.ByteBuffer;
import javax.annotation.Nullable;

/**
 * An {@link ActionSketch} encodes a transitive hash of an action sufficient to associate with it
 * the result of executing the action. Therefore, this must include a hash on some upper bound of
 * all transitively consumed input files.
 */
@AutoValue
public abstract class ActionSketch {
  public static final int BIGINTEGER_ENCODED_LENGTH = /*length=*/ 1 + /*payload=*/ 39;
  public static final int MAX_BYTES = /*hashes=*/ 2 * BIGINTEGER_ENCODED_LENGTH;

  private static final ActionSketch NULL_SKETCH =
      ActionSketch.builder()
          .setTransitiveSourceHash(null)
          .autoBuild();

  @Nullable
  public abstract HashAndVersion transitiveSourceHash();

  public static Builder builder() {
    return new AutoValue_ActionSketch.Builder();
  }

  public abstract Builder toBuilder();

  /** A builder for {@link ActionSketch}. */
  @AutoValue.Builder
  public abstract static class Builder {
    public abstract Builder setTransitiveSourceHash(HashAndVersion transitiveSourceHash);

    @Nullable
    abstract HashAndVersion transitiveSourceHash();

    abstract ActionSketch autoBuild();

    public final ActionSketch build() {
      return transitiveSourceHash() == null ? NULL_SKETCH : autoBuild();
    }
  }

  public final ByteString toBytes() {
    ByteBuffer buffer = ByteBuffer.allocate(MAX_BYTES);
    writeTo(buffer);
    return ByteString.copyFrom(buffer.array(), 0, buffer.position());
  }

  public final void writeTo(ByteBuffer buffer) {
    writeNextValue(transitiveSourceHash(), buffer);
  }

  private static void writeNextValue(HashAndVersion value, ByteBuffer buffer) {
    if (value == null) {
      buffer.put((byte) -1);
    } else {
      byte[] bytes = value.hash().toByteArray();
      buffer.put((byte) bytes.length).put(bytes).putLong(value.version());
    }
  }

  public static ActionSketch fromBytes(ByteString inputBytes) {
    return fromByteBuffer(inputBytes.asReadOnlyByteBuffer());
  }

  public static ActionSketch fromByteBuffer(ByteBuffer buffer) {
    return builder().setTransitiveSourceHash(readNextHashAndVersion(buffer)).build();
  }

  @Nullable
  private static HashAndVersion readNextHashAndVersion(ByteBuffer buffer) {
    byte length = buffer.get();
    if (length < 0) {
      return null;
    }
    byte[] val = new byte[length];
    buffer.get(val);
    long version = buffer.getLong();
    return HashAndVersion.create(new BigInteger(val), version);
  }
}
