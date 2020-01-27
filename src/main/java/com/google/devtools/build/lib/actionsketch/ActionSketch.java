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
import com.google.devtools.build.skyframe.SkyValue;
import com.google.protobuf.ByteString;
import java.math.BigInteger;
import java.nio.ByteBuffer;
import javax.annotation.Nullable;

/**
 * An {@link ActionSketch} encodes a transitive hash of an action sufficient to associate with it
 * the result of executing the action. Therefore, this must include a hash on some upper bound of
 * all transitively consumed input files, as well as a transitive hash of all action keys.
 */
@AutoValue
public abstract class ActionSketch implements SkyValue {
  public static final int BIGINTEGER_ENCODED_LENGTH = /*length=*/ 1 + /*payload=*/ 39;
  public static final int MAX_BYTES = /*hashes=*/ 2 * BIGINTEGER_ENCODED_LENGTH;

  private static final ActionSketch NULL_SKETCH =
      ActionSketch.builder()
          .setTransitiveSourceHash(null)
          .setTransitiveActionLookupHash(null)
          .autoBuild();

  @Nullable
  public abstract BigInteger transitiveSourceHash();

  @Nullable
  public abstract BigInteger transitiveActionLookupHash();

  public static Builder builder() {
    return new AutoValue_ActionSketch.Builder();
  }

  public abstract Builder toBuilder();

  /** A builder for {@link ActionSketch}. */
  @AutoValue.Builder
  public abstract static class Builder {
    public abstract Builder setTransitiveSourceHash(BigInteger transitiveSourceHash);

    public abstract Builder setTransitiveActionLookupHash(BigInteger transitiveActionLookupHash);

    @Nullable
    abstract BigInteger transitiveSourceHash();

    @Nullable
    abstract BigInteger transitiveActionLookupHash();

    abstract ActionSketch autoBuild();

    public final ActionSketch build() {
      return transitiveSourceHash() == null && transitiveActionLookupHash() == null
          ? NULL_SKETCH
          : autoBuild();
    }
  }

  public final ByteString toBytes() {
    ByteBuffer buffer = ByteBuffer.allocate(MAX_BYTES);
    writeTo(buffer);
    return ByteString.copyFrom(buffer.array(), 0, buffer.position());
  }

  public final void writeTo(ByteBuffer buffer) {
    writeNextValue(transitiveSourceHash(), buffer);
    writeNextValue(transitiveActionLookupHash(), buffer);
  }

  public static void writeNextValue(@Nullable BigInteger value, ByteBuffer buffer) {
    if (value == null) {
      buffer.put((byte) -1);
    } else {
      byte[] bytes = value.toByteArray();
      buffer.put((byte) bytes.length).put(bytes);
    }
  }

  public static ActionSketch fromBytes(ByteString inputBytes) {
    return fromByteBuffer(inputBytes.asReadOnlyByteBuffer());
  }

  public static ActionSketch fromByteBuffer(ByteBuffer buffer) {
    Builder builder =
        builder()
            .setTransitiveSourceHash(readNextValue(buffer))
            .setTransitiveActionLookupHash(readNextValue(buffer));
    return builder.build();
  }

  @Nullable
  public static BigInteger readNextValue(ByteBuffer buffer) {
    byte length = buffer.get();
    if (length < 0) {
      return null;
    }
    byte[] val = new byte[length];
    buffer.get(val);
    return new BigInteger(val);
  }
}
