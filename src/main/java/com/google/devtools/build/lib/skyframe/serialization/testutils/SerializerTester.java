// Copyright 2018 The Bazel Authors. All rights reserved.
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

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.KryoException;
import com.esotericsoftware.kryo.Serializer;
import com.esotericsoftware.kryo.io.UnsafeInput;
import com.esotericsoftware.kryo.io.UnsafeOutput;
import com.google.common.base.Preconditions;
import com.google.common.base.Stopwatch;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.skyframe.serialization.serializers.KryoConfigUtil;
import java.io.ByteArrayOutputStream;
import java.util.Random;
import java.util.function.Consumer;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.objenesis.instantiator.ObjectInstantiator;

/** Utility for testing {@link Serializer} instances. */
public class SerializerTester<SubjectT, SerializerT extends SubjectT> {
  public static final int DEFAULT_JUNK_INPUTS = 20;
  public static final int JUNK_LENGTH_UPPER_BOUND = 20;

  private static final Logger logger = Logger.getLogger(SerializerTester.class.getName());

  /** Interface for testing successful deserialization of an object. */
  @FunctionalInterface
  public interface VerificationFunction<T> {
    /**
     * Verifies whether or not the original object was sufficiently serialized/deserialized.
     *
     * @throws Exception on verification failure
     */
    void verifyDeserialized(T original, T deserialized) throws Exception;
  }

  /**
   * Creates a {@link SerializerTester.Builder} for the given class.
   *
   * <p>See {@link SerializerTester.Builder} for details.
   */
  public static <T> SerializerTester.Builder<T, T> newBuilder(Class<T> type) {
    return new SerializerTester.Builder<>(type);
  }

  private final Class<SerializerT> type;
  private final Kryo kryo;
  private final ImmutableList<SubjectT> subjects;
  private final VerificationFunction<SubjectT> verificationFunction;

  private SerializerTester(
      Class<SerializerT> type,
      Kryo kryo,
      ImmutableList<SubjectT> subjects,
      VerificationFunction<SubjectT> verificationFunction) {
    this.type = type;
    this.kryo = kryo;
    Preconditions.checkState(!subjects.isEmpty(), "No subjects provided");
    this.subjects = subjects;
    this.verificationFunction = verificationFunction;
  }

  private void runTests() throws Exception {
    testSerializeDeserialize();
    testStableSerialization();
    testDeserializeJunkData();
  }

  /** Runs serialization/deserialization tests. */
  void testSerializeDeserialize() throws Exception {
    Stopwatch timer = Stopwatch.createStarted();
    int totalBytes = 0;
    for (SubjectT subject : subjects) {
      byte[] serialized = toBytes(subject);
      totalBytes += serialized.length;
      SubjectT deserialized = fromBytes(serialized);
      verificationFunction.verifyDeserialized(subject, deserialized);
    }
    logger.log(
        Level.INFO,
        type.getSimpleName() + " total serialized bytes = " + totalBytes + ", " + timer);
  }

  /** Runs serialized bytes stability tests. */
  void testStableSerialization() throws Exception {
    for (SubjectT subject : subjects) {
      byte[] serialized = toBytes(subject);
      SubjectT deserialized = fromBytes(serialized);
      byte[] reserialized = toBytes(deserialized);
      assertThat(reserialized).isEqualTo(serialized);
    }
  }

  /**
   * Junk tolerance test.
   *
   * <p>Verifies that the Serializer only throws KryoException or IndexOutOfBoundsException.
   *
   * <p>TODO(shahan): Allowing IndexOutOfBoundsException and NegativeArraySizeException here is not
   * ideal, but Kryo itself encodes lengths in the stream and seeking to random lengths triggers
   * this.
   */
  void testDeserializeJunkData() {
    Random rng = new Random(0);
    int numFailures = 0;
    for (int i = 0; i < DEFAULT_JUNK_INPUTS; ++i) {
      byte[] junkData = new byte[rng.nextInt(JUNK_LENGTH_UPPER_BOUND)];
      rng.nextBytes(junkData);
      try {
        UnsafeInput input = new UnsafeInput(junkData);
        kryo.readObject(input, type);
        // OK. Junk string was coincidentally parsed.
      } catch (IndexOutOfBoundsException | NegativeArraySizeException | KryoException e) {
        // OK. Deserialization of junk failed.
        ++numFailures;
      }
    }
    assertThat(numFailures).isAtLeast(1);
  }

  private SubjectT fromBytes(byte[] bytes) {
    return kryo.readObject(new UnsafeInput(bytes), type);
  }

  private byte[] toBytes(SubjectT subject) {
    ByteArrayOutputStream byteOut = new ByteArrayOutputStream();
    UnsafeOutput out = new UnsafeOutput(byteOut);
    kryo.writeObject(out, subject);
    out.flush();
    return byteOut.toByteArray();
  }

  /** Builder for {@link SerializerTester}. */
  public static class Builder<SubjectT, SerializerT extends SubjectT> {
    private final Class<SerializerT> type;
    private final Kryo kryo;
    private final ImmutableList.Builder<SubjectT> subjectsBuilder = ImmutableList.builder();
    private VerificationFunction<SubjectT> verificationFunction =
        (original, deserialized) -> assertThat(deserialized).isEqualTo(original);

    private Builder(Class<SerializerT> type) {
      this.type = type;
      this.kryo = KryoConfigUtil.create();
      kryo.setRegistrationRequired(true);
    }

    public Builder(Class<SubjectT> unusedSubjectType, Class<SerializerT> type) {
      this(type);
    }

    public Builder<SubjectT, SerializerT> registerSerializer(Serializer<SerializerT> serializer) {
      kryo.register(type, serializer);
      return this;
    }

    public <X> Builder<SubjectT, SerializerT> register(Class<X> type, Serializer<X> serializer) {
      kryo.register(type, serializer);
      return this;
    }

    public <X> Builder<SubjectT, SerializerT> register(Class<X> type) {
      kryo.register(type);
      return this;
    }

    public <X> Builder<SubjectT, SerializerT> register(
        Class<X> type, ObjectInstantiator instantiator) {
      kryo.register(type).setInstantiator(instantiator);
      return this;
    }

    /**
     * Hands a {@link Kryo} instance to the visitor.
     *
     * @param visitor usually a reference to a {@code registerSerializers} method
     */
    public Builder<SubjectT, SerializerT> visitKryo(Consumer<Kryo> visitor) {
      visitor.accept(kryo);
      return this;
    }

    /** Adds subjects to be tested for serialization/deserialization. */
    @SafeVarargs
    public final Builder<SubjectT, SerializerT> addSubjects(
        @SuppressWarnings("unchecked") SubjectT... subjects) {
      return addSubjects(ImmutableList.copyOf(subjects));
    }

    /** Adds subjects to be tested for serialization/deserialization. */
    public Builder<SubjectT, SerializerT> addSubjects(ImmutableList<SubjectT> subjects) {
      subjectsBuilder.addAll(subjects);
      return this;
    }

    /**
     * Sets {@link SerializerTester.VerificationFunction} for verifying deserialization.
     *
     * <p>Default is simple equality assertion, a custom version may be provided for more, or less,
     * detailed checks.
     */
    public Builder<SubjectT, SerializerT> setVerificationFunction(
        VerificationFunction<SubjectT> verificationFunction) {
      this.verificationFunction = Preconditions.checkNotNull(verificationFunction);
      return this;
    }

    public Builder<SubjectT, SerializerT> setRegistrationRequired(boolean isRequired) {
      kryo.setRegistrationRequired(isRequired);
      return this;
    }

    /** Captures the state of this builder and runs all associated tests. */
    public void buildAndRunTests() throws Exception {
      build().runTests();
    }

    /** Creates a new {@link SerializerTester} from this builder. */
    private SerializerTester<SubjectT, SerializerT> build() {
      return new SerializerTester<>(type, kryo, subjectsBuilder.build(), verificationFunction);
    }
  }
}
