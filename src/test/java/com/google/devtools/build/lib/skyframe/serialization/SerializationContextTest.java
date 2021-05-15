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

package com.google.devtools.build.lib.skyframe.serialization;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.when;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec.MemoizationStrategy;
import com.google.devtools.build.lib.skyframe.serialization.testutils.TestUtils;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.ArgumentMatchers;
import org.mockito.Mockito;

/** Tests for {@link SerializationContext}. */
@RunWith(JUnit4.class)
public class SerializationContextTest {
  @Test
  public void nullSerialize() throws IOException, SerializationException {
    ObjectCodecRegistry registry = Mockito.mock(ObjectCodecRegistry.class);
    CodedOutputStream codedOutputStream = Mockito.mock(CodedOutputStream.class);
    SerializationContext serializationContext =
        new SerializationContext(registry, ImmutableClassToInstanceMap.of());
    serializationContext.serialize(null, codedOutputStream);
    Mockito.verify(codedOutputStream).writeSInt32NoTag(0);
    Mockito.verifyZeroInteractions(registry);
  }

  @Test
  public void constantSerialize() throws IOException, SerializationException {
    ObjectCodecRegistry registry = Mockito.mock(ObjectCodecRegistry.class);
    when(registry.maybeGetTagForConstant(ArgumentMatchers.any())).thenReturn(1);
    CodedOutputStream codedOutputStream = Mockito.mock(CodedOutputStream.class);
    SerializationContext serializationContext =
        new SerializationContext(registry, ImmutableClassToInstanceMap.of());
    Object constant = new Object();
    serializationContext.serialize(constant, codedOutputStream);
    Mockito.verify(codedOutputStream).writeSInt32NoTag(1);
    Mockito.verify(registry).maybeGetTagForConstant(constant);
  }

  @Test
  public void descriptorSerialize() throws SerializationException, IOException {
    ObjectCodecRegistry.CodecDescriptor codecDescriptor =
        Mockito.mock(ObjectCodecRegistry.CodecDescriptor.class);
    when(codecDescriptor.getTag()).thenReturn(1);
    ObjectCodecRegistry registry = Mockito.mock(ObjectCodecRegistry.class);
    when(registry.maybeGetTagForConstant(ArgumentMatchers.any())).thenReturn(null);
    when(registry.getCodecDescriptorForObject("string")).thenReturn(codecDescriptor);
    CodedOutputStream codedOutputStream = Mockito.mock(CodedOutputStream.class);
    SerializationContext underTest =
        new SerializationContext(registry, ImmutableClassToInstanceMap.of());
    underTest.serialize("string", codedOutputStream);
    Mockito.verify(codedOutputStream).writeSInt32NoTag(1);
    Mockito.verify(registry).maybeGetTagForConstant("string");
    Mockito.verify(registry).getCodecDescriptorForObject("string");
    Mockito.verify(codecDescriptor).getTag();
    Mockito.verify(codecDescriptor).serialize(underTest, "string", codedOutputStream);
  }

  @Test
  public void memoizingSerialize_null() throws IOException, SerializationException {
    ObjectCodecRegistry registry = Mockito.mock(ObjectCodecRegistry.class);
    CodedOutputStream codedOutputStream = Mockito.mock(CodedOutputStream.class);
    SerializationContext serializationContext =
        new SerializationContext(registry, ImmutableClassToInstanceMap.of());
    serializationContext.getMemoizingContext().serialize(null, codedOutputStream);
    Mockito.verify(codedOutputStream).writeSInt32NoTag(0);
    Mockito.verifyZeroInteractions(registry);
  }

  @Test
  public void memoizingSerialize_constant() throws IOException, SerializationException {
    ObjectCodecRegistry registry = Mockito.mock(ObjectCodecRegistry.class);
    when(registry.maybeGetTagForConstant(ArgumentMatchers.any())).thenReturn(1);
    CodedOutputStream codedOutputStream = Mockito.mock(CodedOutputStream.class);
    SerializationContext serializationContext =
        new SerializationContext(registry, ImmutableClassToInstanceMap.of());
    Object constant = new Object();
    serializationContext.getMemoizingContext().serialize(constant, codedOutputStream);
    Mockito.verify(codedOutputStream).writeSInt32NoTag(1);
    Mockito.verify(registry).maybeGetTagForConstant(constant);
  }

  @Test
  public void memoizingSerialize_descriptor() throws SerializationException, IOException {
    @SuppressWarnings("unchecked")
    ObjectCodec<Object> codec = Mockito.mock(ObjectCodec.class);
    when(codec.getStrategy()).thenReturn(MemoizationStrategy.MEMOIZE_AFTER);
    ObjectCodecRegistry.CodecDescriptor codecDescriptor =
        Mockito.mock(ObjectCodecRegistry.CodecDescriptor.class);
    when(codecDescriptor.getTag()).thenReturn(1);
    doReturn(codec).when(codecDescriptor).getCodec();
    ObjectCodecRegistry registry = Mockito.mock(ObjectCodecRegistry.class);
    when(registry.maybeGetTagForConstant(ArgumentMatchers.any())).thenReturn(null);
    when(registry.getCodecDescriptorForObject("string")).thenReturn(codecDescriptor);
    CodedOutputStream codedOutputStream = Mockito.mock(CodedOutputStream.class);
    SerializationContext underTest =
        new SerializationContext(registry, ImmutableClassToInstanceMap.of()).getMemoizingContext();
    underTest.serialize("string", codedOutputStream);
    Mockito.verify(codedOutputStream).writeSInt32NoTag(1);
    Mockito.verify(registry).maybeGetTagForConstant("string");
    Mockito.verify(registry).getCodecDescriptorForObject("string");
    Mockito.verify(codecDescriptor).getTag();
    Mockito.verify(codecDescriptor).getCodec();
    Mockito.verify(codec).serialize(underTest, "string", codedOutputStream);
  }

  @Test
  public void startMemoizingIsIdempotent() throws IOException, SerializationException {
    ObjectCodecRegistry registry =
        ObjectCodecRegistry.newBuilder()
            .add(new CodecMemoizing())
            .add(new CalledOnlyOnce())
            .build();

    String repeated = "repeated string";
    ImmutableList<Object> obj = ImmutableList.of(ImmutableList.of(repeated, repeated), repeated);
    assertThat(TestUtils.roundTrip(obj, registry)).isEqualTo(obj);
  }

  @Test
  public void explicitlyAllowedClassCheck() throws SerializationException {
    SerializationContext underTest =
        new SerializationContext(
                ObjectCodecRegistry.newBuilder().build(), ImmutableClassToInstanceMap.of())
            .getMemoizingContext();
    underTest.addExplicitlyAllowedClass(String.class);
    underTest.checkClassExplicitlyAllowed(String.class, "str");
    assertThrows(
        SerializationException.class,
        () -> underTest.checkClassExplicitlyAllowed(Integer.class, 0));
    // Explicitly registered classes do not carry over to a new context.
    assertThrows(
        SerializationException.class,
        () -> underTest.getNewMemoizingContext().checkClassExplicitlyAllowed(String.class, "str"));
  }

  @Test
  public void explicitlyAllowedClassCheckFailsIfNotMemoizing() {
    SerializationContext underTest =
        new SerializationContext(
            ObjectCodecRegistry.newBuilder().build(), ImmutableClassToInstanceMap.of());
    assertThrows(
        SerializationException.class, () -> underTest.addExplicitlyAllowedClass(String.class));
  }

  private static class CodecMemoizing implements ObjectCodec<ImmutableList<Object>> {
    @SuppressWarnings("unchecked")
    @Override
    public Class<ImmutableList<Object>> getEncodedClass() {
      return (Class<ImmutableList<Object>>) (Class<?>) ImmutableList.class;
    }

    @Override
    public void serialize(
        SerializationContext context, ImmutableList<Object> obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      context = context.getMemoizingContext();
      codedOut.writeInt32NoTag(obj.size());
      for (Object item : obj) {
        context.serialize(item, codedOut);
      }
    }

    @Override
    public ImmutableList<Object> deserialize(
        DeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      context = context.getMemoizingContext();
      int size = codedIn.readInt32();
      ImmutableList.Builder<Object> builder = ImmutableList.builder();
      for (int i = 0; i < size; i++) {
        builder.add(context.<Object>deserialize(codedIn));
      }
      return builder.build();
    }
  }

  private static class CalledOnlyOnce implements ObjectCodec<String> {
    private final AtomicBoolean serializationCalled = new AtomicBoolean(false);
    private final AtomicBoolean deserializationCalled = new AtomicBoolean(false);

    @Override
    public Class<String> getEncodedClass() {
      return String.class;
    }

    @Override
    public void serialize(SerializationContext context, String obj, CodedOutputStream codedOut)
        throws IOException {
      Preconditions.checkState(!serializationCalled.getAndSet(true));
      codedOut.writeStringNoTag(obj);
    }

    @Override
    public String deserialize(DeserializationContext context, CodedInputStream codedIn)
        throws IOException {
      Preconditions.checkState(!deserializationCalled.getAndSet(true));
      return codedIn.readString();
    }
  }

  @Test
  public void mismatchMemoizingRoundtrip() {
    ArrayList<Object> repeatedObject = new ArrayList<>();
    repeatedObject.add(null);
    repeatedObject.add(null);
    ArrayList<Object> container = new ArrayList<>();
    container.add(repeatedObject);
    ArrayList<Object> toSerialize = new ArrayList<>();
    toSerialize.add(repeatedObject);
    toSerialize.add(container);
    assertThrows(
        Exception.class,
        () ->
            TestUtils.roundTrip(
                toSerialize,
                ObjectCodecRegistry.newBuilder()
                    .add(new BadCodecOnlyMemoizesWhenDeserializing())
                    .build()));
  }

  private static class BadCodecOnlyMemoizesWhenDeserializing implements ObjectCodec<ArrayList<?>> {
    @SuppressWarnings("unchecked")
    @Override
    public Class<ArrayList<?>> getEncodedClass() {
      return (Class<ArrayList<?>>) (Class<?>) ArrayList.class;
    }

    @Override
    public void serialize(
        SerializationContext context, ArrayList<?> obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      codedOut.writeInt32NoTag(obj.size());
      for (Object item : obj) {
        context.serialize(item, codedOut);
      }
    }

    @Override
    public ArrayList<?> deserialize(DeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      context = context.getMemoizingContext();
      int size = codedIn.readInt32();
      ArrayList<?> result = new ArrayList<>();
      for (int i = 0; i < size; i++) {
        result.add(context.deserialize(codedIn));
      }
      return result;
    }
  }
}
