// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe.serialization.testutils;

import static com.google.common.util.concurrent.Uninterruptibles.getUninterruptibly;

import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.skyframe.serialization.AutoRegistry;
import com.google.devtools.build.lib.skyframe.serialization.DeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.FingerprintValueService;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecRegistry;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.skyframe.serialization.SkyframeDependencyException;
import com.google.devtools.build.lib.skyframe.serialization.SkyframeLookupContinuation;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.state.EnvironmentForUtilities;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.concurrent.ExecutionException;
import javax.annotation.Nullable;

/** Helpers for round tripping in serialization tests. */
public class RoundTripping {

  private RoundTripping() {}

  /** Serialize a value to a new byte array. */
  public static <T> byte[] toBytes(SerializationContext context, ObjectCodec<T> codec, T value)
      throws IOException, SerializationException {
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    CodedOutputStream codedOut = CodedOutputStream.newInstance(bytes);
    codec.serialize(context, value, codedOut);
    codedOut.flush();
    return bytes.toByteArray();
  }

  public static <T> ByteString toBytes(SerializationContext serializationContext, T value)
      throws IOException, SerializationException {
    ByteString.Output output = ByteString.newOutput();
    CodedOutputStream codedOut = CodedOutputStream.newInstance(output);
    serializationContext.serialize(value, codedOut);
    codedOut.flush();
    return output.toByteString();
  }

  public static Object fromBytes(DeserializationContext deserializationContext, ByteString bytes)
      throws IOException, SerializationException {
    return deserializationContext.deserialize(bytes.newCodedInput());
  }

  /** Deserialize a value from a byte array. */
  public static <T> T fromBytes(DeserializationContext context, ObjectCodec<T> codec, byte[] bytes)
      throws SerializationException, IOException {
    return codec.deserialize(context, CodedInputStream.newInstance(bytes));
  }

  public static <T> T roundTrip(T value, ObjectCodecRegistry registry)
      throws IOException, SerializationException {
    return roundTrip(value, new ObjectCodecs(registry));
  }

  public static <T> T roundTrip(T value, ImmutableClassToInstanceMap<Object> dependencies)
      throws IOException, SerializationException {
    return roundTrip(value, new ObjectCodecs(dependencies));
  }

  public static <T> T roundTrip(T value) throws IOException, SerializationException {
    return roundTrip(value, new ObjectCodecs());
  }

  private static <T> T roundTrip(T value, ObjectCodecs codecs) throws SerializationException {
    @SuppressWarnings("unchecked")
    T result = (T) codecs.deserialize(codecs.serialize(value));
    return result;
  }

  public static ByteString toBytesMemoized(Object original, ObjectCodecRegistry registry)
      throws IOException, SerializationException {
    return new ObjectCodecs(registry).serializeMemoized(original);
  }

  public static Object fromBytesMemoized(ByteString bytes, ObjectCodecRegistry registry)
      throws SerializationException {
    return new ObjectCodecs(registry).deserializeMemoized(bytes);
  }

  public static Object fromBytesWithSkyframe(
      ObjectCodecs codecs,
      FingerprintValueService fingerprintValueService,
      EnvironmentForUtilities.ResultProvider resultProvider,
      ByteString data)
      throws SerializationException, SkyframeDependencyException, MissingResultException {
    Object result = codecs.deserializeWithSkyframe(fingerprintValueService, data);
    if (result instanceof ListenableFuture<?> futureContinuation) {
      SkyframeLookupContinuation continuation;
      try {
        continuation = (SkyframeLookupContinuation) getUninterruptibly(futureContinuation);
      } catch (ExecutionException e) {
        throw new SerializationException("waiting for remote values", e.getCause());
      }
      var recordingResultProvider = new KeyRecordingResultProvider(resultProvider);
      ListenableFuture<?> futureValue;
      try {
        futureValue = continuation.process(new EnvironmentForUtilities(recordingResultProvider));
      } catch (InterruptedException e) {
        // Formally, an InterruptedException may occur when interacting with a LookupEnvironment,
        // but the EnvironmentForUtilities never throws it.
        throw new AssertionError("unexpected InterruptedException", e);
      }
      if (futureValue == null) {
        throw new MissingResultException(recordingResultProvider.formatRecordedSkyKeys());
      }
      try {
        return getUninterruptibly(futureValue);
      } catch (ExecutionException e) {
        throw new SerializationException("waiting for bookkeeping and shared values", e.getCause());
      }
    }
    return result;
  }

  /**
   * Thrown if the {@code resultProvider} passed to {@link #fromBytesWithSkyframe} is missing
   * values.
   */
  public static class MissingResultException extends Exception {
    private MissingResultException(String message) {
      super(message);
    }
  }

  @SuppressWarnings("unchecked")
  public static <T> T roundTripMemoized(T original, ObjectCodecRegistry registry)
      throws IOException, SerializationException {
    ObjectCodecs codecs = new ObjectCodecs(registry);
    return (T) codecs.deserializeMemoized(codecs.serializeMemoized(original));
  }

  public static <T> T roundTripMemoized(T original, ObjectCodec<?>... codecs)
      throws IOException, SerializationException {
    ObjectCodecRegistry.Builder builder = AutoRegistry.get().getBuilder();
    for (ObjectCodec<?> codec : codecs) {
      builder.add(codec);
    }
    return roundTripMemoized(original, builder.build());
  }

  private static class KeyRecordingResultProvider
      implements EnvironmentForUtilities.ResultProvider {
    private final EnvironmentForUtilities.ResultProvider delegate;
    private final ArrayList<SkyKey> presentKeys = new ArrayList<>();
    private final ArrayList<SkyKey> missingKeys = new ArrayList<>();

    private KeyRecordingResultProvider(EnvironmentForUtilities.ResultProvider delegate) {
      this.delegate = delegate;
    }

    @Override
    @Nullable
    public Object getValueOrException(SkyKey key) {
      Object result = delegate.getValueOrException(key);
      if (result == null) {
        missingKeys.add(key);
      } else {
        presentKeys.add(key);
      }
      return result;
    }

    public String formatRecordedSkyKeys() {
      StringBuilder builder = new StringBuilder("successfully looked up=");
      formatSkyKeys(presentKeys, builder);
      builder.append(", missing=");
      formatSkyKeys(missingKeys, builder);
      return builder.toString();
    }

    private static void formatSkyKeys(Iterable<SkyKey> keys, StringBuilder builder) {
      builder.append('[');
      boolean isFirst = true;
      for (SkyKey key : keys) {
        if (isFirst) {
          isFirst = false;
        } else {
          builder.append(", ");
        }
        // Explicitly includes the type because many SkyKey types have String representations where
        // this is unclear.
        builder.append(key).append('<').append(key.getClass().getName()).append('>');
      }
      builder.append(']');
    }
  }
}
