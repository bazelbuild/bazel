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
import com.google.devtools.build.lib.skyframe.serialization.DeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Random;
import java.util.function.Supplier;
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
  private Supplier<SerializationContext> writeContextFactory =
      () -> new SerializationContext(ImmutableMap.of());
  private Supplier<DeserializationContext> readContextFactory =
      () -> new DeserializationContext(ImmutableMap.of());

  @SuppressWarnings("rawtypes")
  private VerificationFunction verificationFunction =
      (original, deserialized) -> assertThat(deserialized).isEqualTo(original);

  public SerializationTester(Object... subjects) {
    Preconditions.checkArgument(subjects.length > 0);
    this.subjects = ImmutableList.copyOf(subjects);
  }

  public SerializationTester setWriteContextFactory(
      Supplier<SerializationContext> writeContextFactory) {
    this.writeContextFactory = writeContextFactory;
    return this;
  }

  public SerializationTester setReadContextFactory(
      Supplier<DeserializationContext> readContextFactory) {
    this.readContextFactory = readContextFactory;
    return this;
  }

  @SuppressWarnings("rawtypes")
  public SerializationTester setVerificationFunction(VerificationFunction verificationFunction) {
    this.verificationFunction = verificationFunction;
    return this;
  }

  public void runTests() throws Exception {
    testSerializeDeserialize();
    testStableSerialization();
    testDeserializeJunkData();
  }

  /** Runs serialization/deserialization tests. */
  private void testSerializeDeserialize() throws Exception {
    Stopwatch timer = Stopwatch.createStarted();
    int totalBytes = 0;
    for (Object subject : subjects) {
      byte[] serialized = toBytes(subject);
      totalBytes += serialized.length;
      Object deserialized = fromBytes(serialized);
      verificationFunction.verifyDeserialized(subject, deserialized);
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
  private void testStableSerialization() throws IOException, SerializationException {
    for (Object subject : subjects) {
      byte[] serialized = toBytes(subject);
      Object deserialized = fromBytes(serialized);
      byte[] reserialized = toBytes(deserialized);
      assertThat(reserialized).isEqualTo(serialized);
    }
  }

  /** Runs junk-data recognition tests. */
  private void testDeserializeJunkData() {
    Random rng = new Random(0);
    for (int i = 0; i < DEFAULT_JUNK_INPUTS; ++i) {
      byte[] junkData = new byte[rng.nextInt(JUNK_LENGTH_UPPER_BOUND)];
      rng.nextBytes(junkData);
      try {
        readContextFactory.get().deserialize(CodedInputStream.newInstance(junkData));
        // OK. Junk string was coincidentally parsed.
      } catch (IOException | SerializationException e) {
        // OK. Deserialization of junk failed.
        return;
      }
    }
    assert_().fail("all junk was parsed successfully");
  }

  private Object fromBytes(byte[] bytes) throws IOException, SerializationException {
    return readContextFactory.get().deserialize(CodedInputStream.newInstance(bytes));
  }

  private byte[] toBytes(Object subject) throws IOException, SerializationException {
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    CodedOutputStream codedOut = CodedOutputStream.newInstance(bytes);
    writeContextFactory.get().serialize(subject, codedOut);
    codedOut.flush();
    return bytes.toByteArray();
  }
}
