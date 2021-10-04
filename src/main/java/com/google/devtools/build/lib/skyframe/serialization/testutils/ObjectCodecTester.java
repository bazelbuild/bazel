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
import static org.junit.Assert.fail;

import com.google.common.base.Preconditions;
import com.google.common.base.Stopwatch;
import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.common.collect.ImmutableList;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.skyframe.serialization.DeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.protobuf.CodedInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;

/** Utility for testing {@link ObjectCodec} instances. */
public class ObjectCodecTester<T> {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  /** Interface for testing successful deserialization of an object. */
  @FunctionalInterface
  public interface VerificationFunction<T> {
    /**
     * Verify whether or not the original object was sufficiently serialized/deserialized. Typically
     * this will be some sort of assertion.
     *
     * @throws Exception on verification failure
     */
    void verifyDeserialized(T original, T deserialized) throws Exception;
  }

  /**
   * Create an {@link ObjectCodecTester.Builder} for the supplied instance. See
   * {@link ObjectCodecTester.Builder} for details.
   */
  public static <T> ObjectCodecTester.Builder<T> newBuilder(ObjectCodec<T> toTest) {
    return new ObjectCodecTester.Builder<>(toTest);
  }

  private final ObjectCodec<T> underTest;
  private final ImmutableList<T> subjects;
  private final SerializationContext writeContext;
  private final DeserializationContext readContext;
  private final boolean skipBadDataTest;
  private final VerificationFunction<T> verificationFunction;
  private final int repetitions;

  private ObjectCodecTester(
      ObjectCodec<T> underTest,
      ImmutableList<T> subjects,
      SerializationContext writeContext,
      DeserializationContext readContext,
      boolean skipBadDataTest,
      VerificationFunction<T> verificationFunction,
      int repetitions) {
    this.underTest = underTest;
    Preconditions.checkState(!subjects.isEmpty(), "No subjects provided");
    this.subjects = subjects;
    this.writeContext = writeContext;
    this.readContext = readContext;
    this.skipBadDataTest = skipBadDataTest;
    this.verificationFunction = verificationFunction;
    this.repetitions = repetitions;
  }

  private void runTests() throws Exception {
    testSerializeDeserialize();
    testStableSerialization();
    if (!skipBadDataTest) {
      testDeserializeJunkData();
    }
  }

  /** Runs serialization/deserialization tests. */
  void testSerializeDeserialize() throws Exception {
    Stopwatch timer = Stopwatch.createStarted();
    int totalBytes = 0;
    for (int i = 0; i < repetitions; ++i) {
      for (T subject : subjects) {
        byte[] serialized = toBytes(subject);
        totalBytes += serialized.length;
        T deserialized = fromBytes(serialized);
        verificationFunction.verifyDeserialized(subject, deserialized);
      }
    }
    logger.atInfo().log(
        "%s total serialized bytes = %d, %s",
        underTest.getEncodedClass().getSimpleName(), totalBytes, timer);
  }

  /** Runs serialized bytes stability tests. */
  void testStableSerialization() throws Exception {
    for (T subject : subjects) {
      byte[] serialized = toBytes(subject);
      T deserialized = fromBytes(serialized);
      byte[] reserialized = toBytes(deserialized);
      assertThat(reserialized).isEqualTo(serialized);
    }
  }

  /** Runs junk-data recognition tests. */
  void testDeserializeJunkData() {
    try {
      underTest.deserialize(
          readContext, CodedInputStream.newInstance("junk".getBytes(StandardCharsets.UTF_8)));
      fail("Expected exception");
    } catch (SerializationException | IOException e) {
      // Expected.
    }
  }

  private T fromBytes(byte[] bytes) throws SerializationException, IOException {
    return TestUtils.fromBytes(readContext, underTest, bytes);
  }

  private byte[] toBytes(T subject) throws IOException, SerializationException {
    return TestUtils.toBytes(writeContext, underTest, subject);
  }

  /** Builder for {@link ObjectCodecTester}. */
  public static class Builder<T> {
    private final ObjectCodec<T> underTest;
    private final ImmutableList.Builder<T> subjectsBuilder = ImmutableList.builder();
    private final ImmutableClassToInstanceMap.Builder<Object> dependenciesBuilder =
        ImmutableClassToInstanceMap.builder();
    private boolean skipBadDataTest = false;
    private VerificationFunction<T> verificationFunction =
        (original, deserialized) -> assertThat(deserialized).isEqualTo(original);
    int repetitions = 1;

    private Builder(ObjectCodec<T> underTest) {
      this.underTest = underTest;
    }

    /** Add subjects to be tested for serialization/deserialization. */
    @SafeVarargs
    public final Builder<T> addSubjects(T... subjects) {
      return addSubjects(ImmutableList.copyOf(subjects));
    }

    /** Add subjects to be tested for serialization/deserialization. */
    public Builder<T> addSubjects(ImmutableList<T> subjects) {
      subjectsBuilder.addAll(subjects);
      return this;
    }

    /** Add subjects to be tested for serialization/deserialization. */
    public final <D> Builder<T> addDependency(Class<? super D> type, D dependency) {
      dependenciesBuilder.put(type, dependency);
      return this;
    }

    /**
     * Skip tests that check for the ability to detect bad data. This may be useful for simpler
     * codecs which don't do any error verification.
     */
    public Builder<T> skipBadDataTest() {
      this.skipBadDataTest = true;
      return this;
    }

    /**
     * Sets {@link ObjectCodecTester.VerificationFunction} for verifying deserialization. Default
     * is simple equality assertion, a custom version may be provided for more, or less, detailed
     * checks.
     */
    public Builder<T> verificationFunction(VerificationFunction<T> verificationFunction) {
      this.verificationFunction = Preconditions.checkNotNull(verificationFunction);
      return this;
    }

    public Builder<T> setRepetitions(int repetitions) {
      this.repetitions = repetitions;
      return this;
    }

    /** Captures the state of this builder and run all associated tests. */
    public void buildAndRunTests() throws Exception {
      build().runTests();
    }

    /**
     * Creates a new {@link ObjectCodecTester} from this builder. Exposed to allow running tests
     * individually.
     */
    ObjectCodecTester<T> build() {
      ImmutableClassToInstanceMap<Object> dependencies = dependenciesBuilder.build();
      return new ObjectCodecTester<>(
          underTest,
          subjectsBuilder.build(),
          new SerializationContext(dependencies),
          new DeserializationContext(dependencies),
          skipBadDataTest,
          verificationFunction,
          repetitions);
    }
  }
}
