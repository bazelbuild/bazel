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
import static com.google.devtools.build.lib.skyframe.serialization.FutureHelpers.waitForSerializationFuture;

import com.google.common.base.Preconditions;
import com.google.common.base.Stopwatch;
import com.google.common.collect.ClassToInstanceMap;
import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.common.collect.ImmutableList;
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.skyframe.serialization.AutoRegistry;
import com.google.devtools.build.lib.skyframe.serialization.FingerprintValueService;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecRegistry;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.skyframe.serialization.SerializationResult;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.google.protobuf.ByteString;
import java.util.ArrayList;
import java.util.Random;
import javax.annotation.Nullable;

/**
 * Utility for testing serialization of given subjects.
 *
 * <p>Differs from {@link ObjectCodecTester} in that this uses the context to perform serialization
 * deserialization instead of a specific codec.
 */
public class SerializationTester {
  public static final int DEFAULT_JUNK_INPUTS = 20;
  public static final int JUNK_LENGTH_UPPER_BOUND = 20;

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

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
  private final ImmutableClassToInstanceMap.Builder<Object> dependenciesBuilder =
      ImmutableClassToInstanceMap.builder();
  private final ArrayList<ObjectCodec<?>> additionalCodecs = new ArrayList<>();
  private boolean memoize;
  private boolean allowFutureBlocking;
  private ObjectCodecs objectCodecs;

  // TODO: b/297857068 - consider splitting out a builder to cleanly separate this state
  @Nullable // lazily initialized
  private FingerprintValueService fingerprintValueService;

  private boolean exerciseDeserializationInKeyValueStore = true;

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
  }

  @CanIgnoreReturnValue
  public <D> SerializationTester addDependency(Class<? super D> type, D dependency) {
    dependenciesBuilder.put(type, dependency);
    return this;
  }

  @CanIgnoreReturnValue
  public SerializationTester addDependencies(ClassToInstanceMap<?> dependencies) {
    dependenciesBuilder.putAll(dependencies);
    return this;
  }

  @CanIgnoreReturnValue
  public SerializationTester addCodec(ObjectCodec<?> codec) {
    additionalCodecs.add(codec);
    return this;
  }

  @CanIgnoreReturnValue
  public SerializationTester makeMemoizing() {
    this.memoize = true;
    return this;
  }

  @CanIgnoreReturnValue
  public SerializationTester makeMemoizingAndAllowFutureBlocking(boolean allowFutureBlocking) {
    makeMemoizing();
    this.allowFutureBlocking = allowFutureBlocking;
    return this;
  }

  @CanIgnoreReturnValue
  public SerializationTester setObjectCodecs(ObjectCodecs objectCodecs) {
    this.objectCodecs = objectCodecs;
    return this;
  }

  @CanIgnoreReturnValue
  public <T> SerializationTester setVerificationFunction(
      VerificationFunction<T> verificationFunction) {
    this.verificationFunction = verificationFunction;
    return this;
  }

  /** Sets the number of times to repeat serialization and deserialization. */
  @CanIgnoreReturnValue
  public SerializationTester setRepetitions(int repetitions) {
    this.repetitions = repetitions;
    return this;
  }

  @CanIgnoreReturnValue
  public SerializationTester setExerciseDeserializationInKeyValueStore(
      boolean exerciseDeserializationInKeyValueStore) {
    this.exerciseDeserializationInKeyValueStore = exerciseDeserializationInKeyValueStore;
    return this;
  }

  private void runTests(boolean verifyStableSerialization) throws Exception {
    ObjectCodecs codecs = this.objectCodecs == null ? createObjectCodecs() : this.objectCodecs;
    testSerializeDeserialize(codecs);
    fingerprintValueService = null;
    if (verifyStableSerialization) {
      testStableSerialization(codecs);
      fingerprintValueService = null;
    }
    testDeserializeJunkData(codecs);
    fingerprintValueService = null;
  }

  public void runTests() throws Exception {
    runTests(true);
  }

  /**
   * Runs serialization tests without checking for stable serialization ({@code
   * serialize(deserialize(serialize(x))) == serialize(x)}). Call {@link #runTests()}} instead if
   * possible.
   *
   * <p>To be used only when serialization is not stable for good reasons: please understand the
   * cause before using this. Typically unstable serialization is the result of non-determinism in
   * your underlying objects, which can cause problems throughout Blaze by harming incrementality.
   * Only if you are sure that the non-determinism in your objects is not detectable in its public
   * interface or behavior (including {@code equals} if implemented) should you use this instead of
   * {@link #runTests()}.
   */
  public void runTestsWithoutStableSerializationCheck() throws Exception {
    runTests(false);
  }

  private ObjectCodecs createObjectCodecs() {
    ObjectCodecRegistry registry = AutoRegistry.get();
    ImmutableClassToInstanceMap<Object> dependencies = dependenciesBuilder.build();
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
    if (!memoize) {
      return codecs.serialize(subject);
    }
    if (!allowFutureBlocking) {
      return codecs.serializeMemoized(subject);
    }
    SerializationResult<ByteString> result =
        codecs.serializeMemoizedAndBlocking(getFingerprintValueService(), subject);
    ListenableFuture<Void> writeFuture = result.getFutureToBlockWritesOn();
    if (writeFuture != null) {
      var unused = waitForSerializationFuture(writeFuture);
    }
    return result.getObject();
  }

  private Object deserialize(ByteString serialized, ObjectCodecs codecs)
      throws SerializationException {
    if (!memoize) {
      return codecs.deserialize(serialized);
    }
    return allowFutureBlocking
        ? codecs.deserializeMemoizedAndBlocking(getFingerprintValueService(), serialized)
        : codecs.deserializeMemoized(serialized);
  }

  private FingerprintValueService getFingerprintValueService() {
    if (fingerprintValueService == null) {
      fingerprintValueService =
          FingerprintValueService.createForTesting(exerciseDeserializationInKeyValueStore);
    }
    return fingerprintValueService;
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
    logger.atInfo().log(
        "%s total serialized bytes = %d, %s",
        subjects.get(0).getClass().getSimpleName(), totalBytes, timer);
  }

  /** Runs serialized bytes stability tests. */
  private void testStableSerialization(ObjectCodecs codecs) throws SerializationException {
    for (Object subject : subjects) {
      ByteString serialized = serialize(subject, codecs);
      Object deserialized = deserialize(serialized, codecs);
      ByteString reserialized = serialize(deserialized, codecs);
      assertThat(reserialized).isEqualTo(serialized);
    }
  }

  /** Runs junk-data recognition tests. */
  private void testDeserializeJunkData(ObjectCodecs codecs) {
    Random rng = new Random(0);
    for (int i = 0; i < DEFAULT_JUNK_INPUTS; ++i) {
      byte[] junkData = new byte[rng.nextInt(JUNK_LENGTH_UPPER_BOUND)];
      rng.nextBytes(junkData);
      try {
        var unused = deserialize(ByteString.copyFrom(junkData), codecs);
        // OK. Junk string was coincidentally parsed.
      } catch (SerializationException e) {
        // OK. Deserialization of junk failed.
        return;
      }
    }
    assertWithMessage("all junk was parsed successfully").fail();
  }
}
