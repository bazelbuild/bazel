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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.skyframe.serialization.DeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import java.io.IOException;
import javax.annotation.Nullable;
import org.junit.Before;
import org.junit.Test;

/**
 * Base class for {@link ObjectCodec} tests. This is a slim wrapper around {@link ObjectCodecTester}
 * and exists mostly to support existing tests.
 */
public abstract class AbstractObjectCodecTest<T> {

  @Nullable protected ObjectCodec<T> underTest;
  @Nullable protected ImmutableList<T> subjects;
  private ObjectCodecTester<T> objectCodecTester;

  /** Construct with the given codec and subjects. */
  protected AbstractObjectCodecTest(ObjectCodec<T> underTest, T... subjects) {
    this.underTest = underTest;
    this.subjects = ImmutableList.copyOf(subjects);
  }

  /**
   * Construct without a codec and subjects. They must be set in the subclass's constructor instead.
   *
   * <p>This is useful if the logic for creating the codec and/or subjects is non-trivial. Using
   * this super constructor, the logic can be placed in the subclass's constructor; whereas if using
   * the above super constructor, the logic must be factored into a static method.
   */
  protected AbstractObjectCodecTest() {}

  @Before
  public void initialize() {
    Preconditions.checkNotNull(underTest);
    Preconditions.checkNotNull(subjects);
    objectCodecTester = ObjectCodecTester.newBuilder(underTest)
        .verificationFunction(
            (original, deserialized) -> this.verifyDeserialization(deserialized, original))
        .addSubjects(subjects)
        .build();
  }

  @Test
  public void testSuccessfulSerializationDeserialization() throws Exception {
    objectCodecTester.testSerializeDeserialize();
  }

  @Test
  public void testSerializationRoundTripBytes() throws Exception {
    objectCodecTester.testStableSerialization();
  }

  @Test
  public void testDeserializeBadDataThrowsSerializationException() {
    objectCodecTester.testDeserializeJunkData();
  }

  protected T fromBytes(DeserializationContext context, byte[] bytes)
      throws SerializationException, IOException {
    return RoundTripping.fromBytes(context, underTest, bytes);
  }

  /** Serialize subject using the {@link ObjectCodec} under test. */
  protected byte[] toBytes(SerializationContext context, T subject)
      throws IOException, SerializationException {
    return RoundTripping.toBytes(context, underTest, subject);
  }

  protected void verifyDeserialization(T deserialized, T subject) {
    assertThat(deserialized).isEqualTo(subject);
  }
}
