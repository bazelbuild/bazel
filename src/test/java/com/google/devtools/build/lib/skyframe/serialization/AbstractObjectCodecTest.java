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

package com.google.devtools.build.lib.skyframe.serialization;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.protobuf.CodedInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import javax.annotation.Nullable;
import org.junit.Before;
import org.junit.Test;

/** Common ObjectCodec tests. */
public abstract class AbstractObjectCodecTest<T> {

  @Nullable protected ObjectCodec<T> underTest;
  @Nullable protected ImmutableList<T> subjects;

  /**
   * Override to false to skip testDeserializeBadDataThrowsSerializationException(). Codecs that
   * cannot distinguish good and bad data should do this.
   */
  protected boolean shouldTestDeserializeBadData = true;

  /** Construct with the given codec and subjects. */
  @SafeVarargs
  protected AbstractObjectCodecTest(
      ObjectCodec<T> underTest, T... subjects) {
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
  public void checkInitialized() {
    Preconditions.checkNotNull(underTest);
    Preconditions.checkNotNull(subjects);
  }

  @Test
  public void testSuccessfulSerializationDeserialization() throws Exception {
    for (T subject : subjects) {
      byte[] serialized = toBytes(subject);
      Object deserialized = fromBytes(serialized);
      verifyDeserialization(deserialized, subject);
    }
  }

  @Test
  public void testSerializationRoundTripBytes() throws Exception {
    for (T subject : subjects) {
      byte[] serialized = toBytes(subject);
      T deserialized = fromBytes(serialized);
      byte[] reserialized = toBytes(deserialized);
      assertThat(reserialized).isEqualTo(serialized);
    }
  }

  @Test
  public void testDeserializeBadDataThrowsSerializationException() {
    if (!shouldTestDeserializeBadData) {
      return;
    }
    try {
      underTest.deserialize(CodedInputStream.newInstance("junk".getBytes(StandardCharsets.UTF_8)));
      fail("Expected exception");
    } catch (SerializationException | IOException e) {
      // Expected.
    }
  }

  protected T fromBytes(byte[] bytes) throws SerializationException, IOException {
    return TestUtils.fromBytes(underTest, bytes);
  }

  /** Serialize subject using the {@link ObjectCodec} under test. */
  protected byte[] toBytes(T subject) throws IOException, SerializationException {
    return TestUtils.toBytes(underTest, subject);
  }

  protected void verifyDeserialization(Object deserialized, T subject) {
    assertThat(deserialized).isEqualTo(subject);
  }
}
