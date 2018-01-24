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
import com.google.common.collect.ImmutableList;
import java.io.ByteArrayOutputStream;
import java.util.Random;
import org.objenesis.instantiator.ObjectInstantiator;

/** Utility for testing {@link Serializer} instances. */
public class SerializerTester<T> {
  public static final int DEFAULT_JUNK_INPUTS = 20;
  public static final int JUNK_LENGTH_UPPER_BOUND = 20;

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
  public static <T> SerializerTester.Builder<T> newBuilder(Class<T> type) {
    return new SerializerTester.Builder<>(type);
  }

  private final Class<T> type;
  private final Kryo kryo;
  private final ImmutableList<T> subjects;
  private final VerificationFunction<T> verificationFunction;

  private SerializerTester(
      Class<T> type,
      Kryo kryo,
      ImmutableList<T> subjects,
      VerificationFunction<T> verificationFunction) {
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
    for (T subject : subjects) {
      byte[] serialized = toBytes(subject);
      T deserialized = fromBytes(serialized);
      verificationFunction.verifyDeserialized(subject, deserialized);
    }
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

  /**
   * Junk tolerance test.
   *
   * <p>Verifies that the Serializer only throws KryoException or IndexOutOfBoundsException.
   *
   * <p>TODO(shahan): Allowing IndexOutOfBoundsException here is not ideal, but Kryo itself encodes
   * lengths in the stream and seeking to random lengths triggers this.
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
      } catch (IndexOutOfBoundsException | KryoException e) {
        // OK. Deserialization of junk failed.
        ++numFailures;
      }
    }
    assertThat(numFailures).isAtLeast(1);
  }

  private T fromBytes(byte[] bytes) {
    return kryo.readObject(new UnsafeInput(bytes), type);
  }

  private byte[] toBytes(T subject) {
    ByteArrayOutputStream byteOut = new ByteArrayOutputStream();
    UnsafeOutput out = new UnsafeOutput(byteOut);
    kryo.writeObject(out, subject);
    out.flush();
    return byteOut.toByteArray();
  }

  /** Builder for {@link SerializerTester}. */
  public static class Builder<T> {
    private final Class<T> type;
    private final Kryo kryo;
    private final ImmutableList.Builder<T> subjectsBuilder = ImmutableList.builder();
    private VerificationFunction<T> verificationFunction =
        (original, deserialized) -> assertThat(deserialized).isEqualTo(original);

    private Builder(Class<T> type) {
      this.type = type;
      this.kryo = new Kryo();
      kryo.setRegistrationRequired(true);
    }

    public <X> Builder<T> register(Class<X> type, Serializer<X> serializer) {
      kryo.register(type, serializer);
      return this;
    }

    public <X> Builder<T> register(Class<X> type) {
      kryo.register(type);
      return this;
    }

    public <X> Builder<T> register(Class<X> type, ObjectInstantiator instantiator) {
      kryo.register(type).setInstantiator(instantiator);
      return this;
    }

    /** Adds subjects to be tested for serialization/deserialization. */
    @SafeVarargs
    public final Builder<T> addSubjects(@SuppressWarnings("unchecked") T... subjects) {
      return addSubjects(ImmutableList.copyOf(subjects));
    }

    /** Adds subjects to be tested for serialization/deserialization. */
    public Builder<T> addSubjects(ImmutableList<T> subjects) {
      subjectsBuilder.addAll(subjects);
      return this;
    }

    /**
     * Sets {@link SerializerTester.VerificationFunction} for verifying deserialization.
     *
     * <p>Default is simple equality assertion, a custom version may be provided for more, or less,
     * detailed checks.
     */
    public Builder<T> setVerificationFunction(VerificationFunction<T> verificationFunction) {
      this.verificationFunction = Preconditions.checkNotNull(verificationFunction);
      return this;
    }

    public Builder<T> setRegistrationRequired(boolean isRequired) {
      kryo.setRegistrationRequired(isRequired);
      return this;
    }

    /** Captures the state of this builder and runs all associated tests. */
    public void buildAndRunTests() throws Exception {
      build().runTests();
    }

    /** Creates a new {@link SerializerTester} from this builder. */
    private SerializerTester<T> build() {
      return new SerializerTester<>(type, kryo, subjectsBuilder.build(), verificationFunction);
    }
  }
}
