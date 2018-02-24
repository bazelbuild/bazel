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
import static com.google.common.truth.Truth.assert_;

import com.google.common.base.Preconditions;
import com.google.common.base.Stopwatch;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.skyframe.serialization.AutoRegistry;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.protobuf.ByteString;
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

  private final ImmutableList<Object> subjects;
  private final ImmutableMap.Builder<Class<?>, Object> dependenciesBuilder;

  @SuppressWarnings("rawtypes")
  private VerificationFunction verificationFunction =
      (original, deserialized) -> assertThat(deserialized).isEqualTo(original);

  private int repetitions = 1;

  public SerializationTester(Object... subjects) {
    Preconditions.checkArgument(subjects.length > 0);
    this.subjects = ImmutableList.copyOf(subjects);
    this.dependenciesBuilder = ImmutableMap.builder();
  }

  public <D> SerializationTester addDependency(Class<? super D> type, D dependency) {
    dependenciesBuilder.put(type, dependency);
    return this;
  }

  @SuppressWarnings("rawtypes")
  public SerializationTester setVerificationFunction(VerificationFunction verificationFunction) {
    this.verificationFunction = verificationFunction;
    return this;
  }

  /** Sets the number of times to repeat serialization and deserialization. */
  public SerializationTester setRepetitions(int repetitions) {
    this.repetitions = repetitions;
    return this;
  }

  public void runTests() throws Exception {
    ObjectCodecs codecs = new ObjectCodecs(AutoRegistry.get(), dependenciesBuilder.build());
    testSerializeDeserialize(codecs);
    testStableSerialization(codecs);
    testDeserializeJunkData(codecs);
  }

  /** Runs serialization/deserialization tests. */
  @SuppressWarnings("unchecked")
  private void testSerializeDeserialize(ObjectCodecs codecs) throws Exception {
    Stopwatch timer = Stopwatch.createStarted();
    int totalBytes = 0;
    for (int i = 0; i < repetitions; ++i) {
      for (Object subject : subjects) {
        ByteString serialized = codecs.serialize(subject);
        totalBytes += serialized.size();
        Object deserialized = codecs.deserialize(serialized);
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
  private void testStableSerialization(ObjectCodecs codecs) throws SerializationException {
    for (Object subject : subjects) {
      ByteString serialized = codecs.serialize(subject);
      Object deserialized = codecs.deserialize(serialized);
      ByteString reserialized = codecs.serialize(deserialized);
      assertThat(reserialized).isEqualTo(serialized);
    }
  }

  /** Runs junk-data recognition tests. */
  private static void testDeserializeJunkData(ObjectCodecs codecs) {
    Random rng = new Random(0);
    for (int i = 0; i < DEFAULT_JUNK_INPUTS; ++i) {
      byte[] junkData = new byte[rng.nextInt(JUNK_LENGTH_UPPER_BOUND)];
      rng.nextBytes(junkData);
      try {
        codecs.deserialize(ByteString.copyFrom(junkData));
        // OK. Junk string was coincidentally parsed.
      } catch (SerializationException e) {
        // OK. Deserialization of junk failed.
        return;
      }
    }
    assert_().fail("all junk was parsed successfully");
  }
}
