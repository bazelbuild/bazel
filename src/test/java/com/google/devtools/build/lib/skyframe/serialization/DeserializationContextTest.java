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
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec.MemoizationStrategy;
import com.google.protobuf.CodedInputStream;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link DeserializationContext}. */
@RunWith(JUnit4.class)
public final class DeserializationContextTest {

  @Test
  public void nullDeserialize() throws Exception {
    ObjectCodecRegistry registry = mock(ObjectCodecRegistry.class);
    CodedInputStream codedInputStream = mock(CodedInputStream.class);
    when(codedInputStream.readSInt32()).thenReturn(0);
    DeserializationContext deserializationContext =
        new ImmutableDeserializationContext(registry, ImmutableClassToInstanceMap.of());
    assertThat((Object) deserializationContext.deserialize(codedInputStream)).isNull();
    verify(codedInputStream).readSInt32();
    verifyNoInteractions(registry);
  }

  @Test
  public void constantDeserialize() throws Exception {
    ObjectCodecRegistry registry = mock(ObjectCodecRegistry.class);
    Object constant = new Object();
    when(registry.maybeGetConstantByTag(1)).thenReturn(constant);
    CodedInputStream codedInputStream = mock(CodedInputStream.class);
    when(codedInputStream.readSInt32()).thenReturn(1);
    DeserializationContext deserializationContext =
        new ImmutableDeserializationContext(registry, ImmutableClassToInstanceMap.of());
    assertThat((Object) deserializationContext.deserialize(codedInputStream))
        .isSameInstanceAs(constant);
    verify(codedInputStream).readSInt32();
    verify(registry).maybeGetConstantByTag(1);
  }

  @Test
  public void memoizingDeserialize_null() throws SerializationException, IOException {
    ObjectCodecRegistry registry = mock(ObjectCodecRegistry.class);
    CodedInputStream codedInputStream = mock(CodedInputStream.class);
    ObjectCodecs codecs = new ObjectCodecs(registry, ImmutableClassToInstanceMap.of());
    when(codedInputStream.readSInt32()).thenReturn(0);
    assertThat(
            (Object)
                codecs.getMemoizingDeserializationContextForTesting().deserialize(codedInputStream))
        .isEqualTo(null);
    verify(codedInputStream).readSInt32();
    verifyNoInteractions(registry);
  }

  @Test
  public void memoizingDeserialize_constant() throws SerializationException, IOException {
    Object constant = new Object();
    ObjectCodecRegistry registry = mock(ObjectCodecRegistry.class);
    when(registry.maybeGetConstantByTag(1)).thenReturn(constant);
    CodedInputStream codedInputStream = mock(CodedInputStream.class);
    ObjectCodecs codecs = new ObjectCodecs(registry, ImmutableClassToInstanceMap.of());
    when(codedInputStream.readSInt32()).thenReturn(1);
    assertThat(
            (Object)
                codecs.getMemoizingDeserializationContextForTesting().deserialize(codedInputStream))
        .isEqualTo(constant);
    verify(codedInputStream).readSInt32();
    verify(registry).maybeGetConstantByTag(1);
  }

  @Test
  public void memoizingDeserialize_codec() throws SerializationException, IOException {
    Object returned = new Object();
    @SuppressWarnings("unchecked")
    ObjectCodec<Object> codec = mock(ObjectCodec.class);
    when(codec.getStrategy()).thenReturn(MemoizationStrategy.MEMOIZE_AFTER);
    when(codec.getEncodedClass()).thenAnswer(unused -> Object.class);
    when(codec.additionalEncodedClasses()).thenReturn(ImmutableSet.of());
    when(codec.safeCast(any())).thenAnswer(invocation -> invocation.getArgument(0));
    ObjectCodecRegistry.CodecDescriptor codecDescriptor =
        new ObjectCodecRegistry.CodecDescriptor(/* tag= */ 1, codec);
    ObjectCodecRegistry registry = mock(ObjectCodecRegistry.class);
    when(registry.getCodecDescriptorByTag(1)).thenReturn(codecDescriptor);
    CodedInputStream codedInputStream = mock(CodedInputStream.class);
    DeserializationContext deserializationContext =
        new ObjectCodecs(registry).getMemoizingDeserializationContextForTesting();
    when(codec.deserialize(deserializationContext, codedInputStream)).thenReturn(returned);
    when(codedInputStream.readSInt32()).thenReturn(1);
    assertThat((Object) deserializationContext.deserialize(codedInputStream)).isEqualTo(returned);
    verify(codedInputStream).readSInt32();
    verify(registry).maybeGetConstantByTag(1);
    verify(registry).getCodecDescriptorByTag(1);
    verify(codec).deserialize(deserializationContext, codedInputStream);
  }

  @Test
  public void getDependency() {
    DeserializationContext context =
        new ImmutableDeserializationContext(
            mock(ObjectCodecRegistry.class), ImmutableClassToInstanceMap.of(String.class, "abc"));
    assertThat(context.getDependency(String.class)).isEqualTo("abc");
  }

  @Test
  public void getDependency_notPresent() {
    DeserializationContext context =
        new ImmutableDeserializationContext(
            mock(ObjectCodecRegistry.class), ImmutableClassToInstanceMap.of());
    Exception e =
        assertThrows(NullPointerException.class, () -> context.getDependency(String.class));
    assertThat(e).hasMessageThat().contains("Missing dependency of type " + String.class);
  }

  @Test
  public void dependencyOverrides_alreadyPresent() {
    ObjectCodecs codecs =
        new ObjectCodecs(
            mock(ObjectCodecRegistry.class), ImmutableClassToInstanceMap.of(String.class, "abc"));
    DeserializationContext overridden =
        codecs
            .withDependencyOverridesForTesting(ImmutableClassToInstanceMap.of(String.class, "xyz"))
            .getDeserializationContextForTesting();
    assertThat(overridden.getDependency(String.class)).isEqualTo("xyz");
  }

  @Test
  public void dependencyOverrides_new() {
    ObjectCodecs codecs =
        new ObjectCodecs(
            mock(ObjectCodecRegistry.class), ImmutableClassToInstanceMap.of(String.class, "abc"));
    DeserializationContext overridden =
        codecs
            .withDependencyOverridesForTesting(ImmutableClassToInstanceMap.of(Integer.class, 1))
            .getDeserializationContextForTesting();
    assertThat(overridden.getDependency(Integer.class)).isEqualTo(1);
  }

  @Test
  public void dependencyOverrides_unchanged() {
    ObjectCodecs codecs =
        new ObjectCodecs(
            mock(ObjectCodecRegistry.class), ImmutableClassToInstanceMap.of(String.class, "abc"));
    DeserializationContext overridden =
        codecs
            .withDependencyOverridesForTesting(ImmutableClassToInstanceMap.of(Integer.class, 1))
            .getDeserializationContextForTesting();
    assertThat(overridden.getDependency(String.class)).isEqualTo("abc");
  }
}
