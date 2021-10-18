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

import com.google.auto.value.AutoValue;
import com.google.common.collect.ClassToInstanceMap;
import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.skyframe.serialization.testutils.TestUtils;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link SerializationContext}. */
@RunWith(TestParameterInjector.class)
public final class SerializationContextTest {

  private static final Object CONSTANT = new Object();

  @AutoValue
  abstract static class Example {
    abstract String getDataToSerialize();

    static Example withData(String data) {
      return new AutoValue_SerializationContextTest_Example(data);
    }
  }

  private final class ExampleCodec implements ObjectCodec<Example> {

    @Override
    public Class<Example> getEncodedClass() {
      return Example.class;
    }

    @Override
    public void serialize(SerializationContext context, Example obj, CodedOutputStream codedOut)
        throws IOException {
      exampleCodecSerializeCalls++;
      codedOut.writeStringNoTag(obj.getDataToSerialize());
    }

    @Override
    public Example deserialize(DeserializationContext context, CodedInputStream codedIn)
        throws IOException {
      exampleCodecDeserializeCalls++;
      return Example.withData(codedIn.readString());
    }
  }

  private final ObjectCodecRegistry registry =
      ObjectCodecRegistry.newBuilder()
          .addReferenceConstant(CONSTANT)
          .add(new ExampleCodec())
          .build();

  private int exampleCodecSerializeCalls = 0;
  private int exampleCodecDeserializeCalls = 0;

  @Test
  public void nullSerialize(@TestParameter boolean memoize) throws Exception {
    SerializationContext context =
        new SerializationContext(registry, ImmutableClassToInstanceMap.of());
    if (memoize) {
      context = context.getMemoizingContext();
    }
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    CodedOutputStream codedOut = CodedOutputStream.newInstance(bytes);

    context.serialize(null, codedOut);
    codedOut.flush();

    CodedInputStream codedIn = CodedInputStream.newInstance(bytes.toByteArray());
    assertThat(codedIn.readSInt32()).isEqualTo(0);
    assertThat(codedIn.isAtEnd()).isTrue();
  }

  @Test
  public void constantSerialize(@TestParameter boolean memoize) throws Exception {
    SerializationContext context =
        new SerializationContext(registry, ImmutableClassToInstanceMap.of());
    if (memoize) {
      context = context.getMemoizingContext();
    }
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    CodedOutputStream codedOut = CodedOutputStream.newInstance(bytes);

    context.serialize(CONSTANT, codedOut);
    codedOut.flush();

    CodedInputStream codedIn = CodedInputStream.newInstance(bytes.toByteArray());
    assertThat(codedIn.readSInt32()).isEqualTo(registry.maybeGetTagForConstant(CONSTANT));
    assertThat(codedIn.isAtEnd()).isTrue();
  }

  @Test
  public void descriptorSerialize() throws SerializationException, IOException {
    Example obj = Example.withData("data");
    SerializationContext context =
        new SerializationContext(registry, ImmutableClassToInstanceMap.of());
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    CodedOutputStream codedOut = CodedOutputStream.newInstance(bytes);

    context.serialize(obj, codedOut);
    codedOut.flush();

    CodedInputStream codedIn = CodedInputStream.newInstance(bytes.toByteArray());
    assertThat(codedIn.readSInt32()).isEqualTo(registry.getCodecDescriptorForObject(obj).getTag());
    assertThat(codedIn.readString()).isEqualTo(obj.getDataToSerialize());
    assertThat(codedIn.isAtEnd()).isTrue();
  }

  @Test
  public void descriptorSerialize_memoizing() throws SerializationException, IOException {
    Example obj = Example.withData("data");
    SerializationContext context =
        new SerializationContext(registry, ImmutableClassToInstanceMap.of()).getMemoizingContext();
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    CodedOutputStream codedOut = CodedOutputStream.newInstance(bytes);

    context.serialize(obj, codedOut);
    context.serialize(obj, codedOut);
    codedOut.flush();

    CodedInputStream codedIn = CodedInputStream.newInstance(bytes.toByteArray());
    assertThat(codedIn.readSInt32()).isEqualTo(registry.getCodecDescriptorForObject(obj).getTag());
    assertThat(codedIn.readString()).isEqualTo(obj.getDataToSerialize());
    assertThat(codedIn.isAtEnd()).isFalse();
    assertThat(exampleCodecSerializeCalls).isEqualTo(1);
  }

  @Test
  public void startMemoizingIsIdempotent() throws Exception {
    ObjectCodecRegistry registry =
        ObjectCodecRegistry.newBuilder()
            .add(new ExampleCodec())
            .add(new MemoizingImmutableListCodec())
            .build();
    Example obj = Example.withData("data");
    ImmutableList<Object> repetitiveList = ImmutableList.of(ImmutableList.of(obj, obj), obj);

    assertThat(TestUtils.roundTrip(repetitiveList, registry)).isEqualTo(repetitiveList);
    assertThat(exampleCodecSerializeCalls).isEqualTo(1);
    assertThat(exampleCodecDeserializeCalls).isEqualTo(1);
  }

  @Test
  public void explicitlyAllowedClassCheck() throws SerializationException {
    SerializationContext context =
        new SerializationContext(registry, ImmutableClassToInstanceMap.of()).getMemoizingContext();
    context.addExplicitlyAllowedClass(String.class);
    context.checkClassExplicitlyAllowed(String.class, "str");
    assertThrows(
        SerializationException.class, () -> context.checkClassExplicitlyAllowed(Integer.class, 0));
    // Explicitly registered classes do not carry over to a new context.
    assertThrows(
        SerializationException.class,
        () -> context.getNewMemoizingContext().checkClassExplicitlyAllowed(String.class, "str"));
  }

  @Test
  public void explicitlyAllowedClassCheckFailsIfNotMemoizing() {
    SerializationContext context =
        new SerializationContext(registry, ImmutableClassToInstanceMap.of());
    assertThrows(
        SerializationException.class, () -> context.addExplicitlyAllowedClass(String.class));
  }

  private static final class MemoizingImmutableListCodec
      implements ObjectCodec<ImmutableList<Object>> {
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

  @Test
  public void mismatchMemoizingRoundtrip() throws Exception {
    ObjectCodecRegistry registry =
        ObjectCodecRegistry.newBuilder().add(new BadCodecOnlyMemoizesWhenDeserializing()).build();
    ArrayList<Object> repeatedObject = new ArrayList<>();
    repeatedObject.add(null);
    repeatedObject.add(null);
    ArrayList<Object> container = new ArrayList<>();
    container.add(repeatedObject);
    ArrayList<Object> toSerialize = new ArrayList<>();
    toSerialize.add(repeatedObject);
    toSerialize.add(container);
    assertThrows(RuntimeException.class, () -> TestUtils.roundTrip(toSerialize, registry));
  }

  private static final class BadCodecOnlyMemoizesWhenDeserializing
      implements ObjectCodec<ArrayList<?>> {
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

  @Test
  public void getDependency() {
    SerializationContext context =
        new SerializationContext(registry, ImmutableClassToInstanceMap.of(String.class, "abc"));
    assertThat(context.getDependency(String.class)).isEqualTo("abc");
  }

  @Test
  public void getDependency_notPresent() {
    SerializationContext context =
        new SerializationContext(registry, ImmutableClassToInstanceMap.of());
    Exception e =
        assertThrows(NullPointerException.class, () -> context.getDependency(String.class));
    assertThat(e).hasMessageThat().contains("Missing dependency of type " + String.class);
  }

  @Test
  public void dependencyOverrides_alreadyPresent() {
    SerializationContext context =
        new SerializationContext(registry, ImmutableClassToInstanceMap.of(String.class, "abc"));
    SerializationContext overridden =
        context.withDependencyOverrides(ImmutableClassToInstanceMap.of(String.class, "xyz"));
    assertThat(overridden.getDependency(String.class)).isEqualTo("xyz");
  }

  @Test
  public void dependencyOverrides_new() {
    SerializationContext context =
        new SerializationContext(registry, ImmutableClassToInstanceMap.of(String.class, "abc"));
    SerializationContext overridden =
        context.withDependencyOverrides(ImmutableClassToInstanceMap.of(Integer.class, 1));
    assertThat(overridden.getDependency(Integer.class)).isEqualTo(1);
  }

  @Test
  public void dependencyOverrides_unchanged() {
    SerializationContext context =
        new SerializationContext(registry, ImmutableClassToInstanceMap.of(String.class, "abc"));
    SerializationContext overridden =
        context.withDependencyOverrides(ImmutableClassToInstanceMap.of(Integer.class, 1));
    assertThat(overridden.getDependency(String.class)).isEqualTo("abc");
  }

  @Test
  public void dependencyOverrides_disallowedOnMemoizingContext() {
    SerializationContext context =
        new SerializationContext(registry, ImmutableClassToInstanceMap.of());
    SerializationContext memoizing = context.getMemoizingContext();
    ClassToInstanceMap<?> overrides = ImmutableClassToInstanceMap.of(Integer.class, 1);
    assertThrows(IllegalStateException.class, () -> memoizing.withDependencyOverrides(overrides));
  }
}
