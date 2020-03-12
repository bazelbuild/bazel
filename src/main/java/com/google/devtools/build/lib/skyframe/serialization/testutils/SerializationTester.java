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

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;

import com.google.common.base.Preconditions;
import com.google.common.base.Stopwatch;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.skyframe.serialization.AutoRegistry;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecRegistry;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Map;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Utility for testing serialization of given subjects.
 *
 * <p>Differs from {@link ObjectCodecTester} in that this uses the context to perform serialization
 * deserialization instead of a specific codec.
 */
public class SerializationTester {
  public static final int DEFAULT_JUNK_INPUTS = 20;
  public static final int JUNK_LENGTH_UPPER_BOUND = 20;

  private static final Logger logger = Logger.getLogger(SerializationTester.class.getName());

  /** Interface for testing successful deserialization of an object. */
  @FunctionalInterface
  public interface VerificationFunction<T> {
    /**
     * Verifies that the original object was sufficiently serialized/deserialized.
     *
     * <p><i>Must</i> throw an exception on failure.
     */
    void verifyDeserialized(T original, T deserialized) throws Exception;
  }

  private final ImmutableList<?> subjects;
  private final ImmutableMap.Builder<Class<?>, Object> dependenciesBuilder;
  private final ArrayList<ObjectCodec<?>> additionalCodecs = new ArrayList<>();
  private boolean memoize;
  private boolean allowFutureBlocking;
  private ObjectCodecs objectCodecs;

  @SuppressWarnings("rawtypes")
  private VerificationFunction verificationFunction =
      (original, deserialized) -> assertThat(deserialized).isEqualTo(original);

  private int repetitions = 1;

  public SerializationTester(Object... subjects) {
    this(ImmutableList.copyOf(subjects));
  }

  public SerializationTester(ImmutableList<?> subjects) {
    Preconditions.checkArgument(!subjects.isEmpty());
    this.subjects = subjects;
    this.dependenciesBuilder = ImmutableMap.builder();
  }

  public <D> SerializationTester addDependency(Class<? super D> type, D dependency) {
    dependenciesBuilder.put(type, dependency);
    return this;
  }

  public SerializationTester addDependencies(Map<Class<?>, Object> dependencies) {
    dependenciesBuilder.putAll(dependencies);
    return this;
  }

  public SerializationTester addCodec(ObjectCodec<?> codec) {
    additionalCodecs.add(codec);
    return this;
  }

  public SerializationTester makeMemoizing() {
    this.memoize = true;
    return this;
  }

  public SerializationTester makeMemoizingAndAllowFutureBlocking(boolean allowFutureBlocking) {
    makeMemoizing();
    this.allowFutureBlocking = allowFutureBlocking;
    return this;
  }

  public SerializationTester setObjectCodecs(ObjectCodecs objectCodecs) {
    this.objectCodecs = objectCodecs;
    return this;
  }

  @SuppressWarnings("rawtypes")
  public <T> SerializationTester setVerificationFunction(
      VerificationFunction<T> verificationFunction) {
    this.verificationFunction = verificationFunction;
    return this;
  }

  /** Sets the number of times to repeat serialization and deserialization. */
  public SerializationTester setRepetitions(int repetitions) {
    this.repetitions = repetitions;
    return this;
  }

  public void runTests() throws Exception {
    ObjectCodecs codecs = this.objectCodecs == null ? createObjectCodecs() : this.objectCodecs;
    testSerializeDeserialize(codecs);
    testStableSerialization(codecs);
    testDeserializeJunkData(codecs);
  }

  private ObjectCodecs createObjectCodecs() {
    ObjectCodecRegistry registry = AutoRegistry.get();
    ImmutableMap<Class<?>, Object> dependencies = dependenciesBuilder.build();
    ObjectCodecRegistry.Builder registryBuilder = registry.getBuilder();
    for (Object val : dependencies.values()) {
      registryBuilder.addReferenceConstant(val);
    }
    for (ObjectCodec<?> codec : additionalCodecs) {
      registryBuilder.add(codec);
    }
    return new ObjectCodecs(registryBuilder.build(), dependencies);
  }

  private ByteString serialize(Object subject, ObjectCodecs codecs) throws SerializationException {
    if (memoize) {
      if (allowFutureBlocking) {
        return codecs.serializeMemoizedAndBlocking(subject).getObject();
      } else {
        return codecs.serializeMemoized(subject);
      }
    } else {
      return codecs.serialize(subject);
    }
  }

  private Object deserialize(ByteString serialized, ObjectCodecs codecs)
      throws SerializationException, IOException {
    if (memoize) {
      return codecs.deserializeMemoized(serialized);
    } else {
      return codecs.deserialize(serialized);
    }
  }

  /** Runs serialization/deserialization tests. */
  @SuppressWarnings("unchecked")
  private void testSerializeDeserialize(ObjectCodecs codecs) throws Exception {
    Stopwatch timer = Stopwatch.createStarted();
    int totalBytes = 0;
    for (int i = 0; i < repetitions; ++i) {
      for (Object subject : subjects) {
        ByteString serialized = serialize(subject, codecs);
        totalBytes += serialized.size();
        Object deserialized = deserialize(serialized, codecs);
        verificationFunction.verifyDeserialized(subject, deserialized);
      }
    }
    logger.log(
        Level.INFO,
        subjects.get(0).getClass().getSimpleName()
            + " total serialized bytes = "
            + totalBytes
            + ", "
            + timer);
  }

  /** Runs serialized bytes stability tests. */
  private void testStableSerialization(ObjectCodecs codecs)
      throws SerializationException, IOException {
    for (Object subject : subjects) {
      ByteString serialized = serialize(subject, codecs);
      Object deserialized = deserialize(serialized, codecs);
      ByteString reserialized = serialize(deserialized, codecs);
      assertThat(reserialized).isEqualTo(serialized);
    }
  }

  /** Runs junk-data recognition tests. */
  private void testDeserializeJunkData(ObjectCodecs codecs) throws IOException {
    Random rng = new Random(0);
    for (int i = 0; i < DEFAULT_JUNK_INPUTS; ++i) {
      byte[] junkData = new byte[rng.nextInt(JUNK_LENGTH_UPPER_BOUND)];
      rng.nextBytes(junkData);
      try {
        deserialize(ByteString.copyFrom(junkData), codecs);
        // OK. Junk string was coincidentally parsed.
      } catch (SerializationException | InvalidProtocolBufferException e) {
        // OK. Deserialization of junk failed.
        return;
      }
    }
    assertWithMessage("all junk was parsed successfully").fail();
  }
}
