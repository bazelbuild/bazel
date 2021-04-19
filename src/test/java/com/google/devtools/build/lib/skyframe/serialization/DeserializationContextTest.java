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
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec.MemoizationStrategy;
import com.google.protobuf.CodedInputStream;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;

/** Tests for {@link DeserializationContext}. */
@RunWith(JUnit4.class)
public class DeserializationContextTest {
  @Test
  public void nullDeserialize() throws Exception {
    ObjectCodecRegistry registry = Mockito.mock(ObjectCodecRegistry.class);
    CodedInputStream codedInputStream = Mockito.mock(CodedInputStream.class);
    when(codedInputStream.readSInt32()).thenReturn(0);
    DeserializationContext deserializationContext =
        new DeserializationContext(registry, ImmutableClassToInstanceMap.of());
    assertThat((Object) deserializationContext.deserialize(codedInputStream)).isNull();
    Mockito.verify(codedInputStream).readSInt32();
    Mockito.verifyZeroInteractions(registry);
  }

  @Test
  public void constantDeserialize() throws Exception {
    ObjectCodecRegistry registry = Mockito.mock(ObjectCodecRegistry.class);
    Object constant = new Object();
    when(registry.maybeGetConstantByTag(1)).thenReturn(constant);
    CodedInputStream codedInputStream = Mockito.mock(CodedInputStream.class);
    when(codedInputStream.readSInt32()).thenReturn(1);
    DeserializationContext deserializationContext =
        new DeserializationContext(registry, ImmutableClassToInstanceMap.of());
    assertThat((Object) deserializationContext.deserialize(codedInputStream))
        .isSameInstanceAs(constant);
    Mockito.verify(codedInputStream).readSInt32();
    Mockito.verify(registry).maybeGetConstantByTag(1);
  }

  @Test
  public void descriptorDeserialize() throws Exception {
    ObjectCodecRegistry.CodecDescriptor codecDescriptor =
        Mockito.mock(ObjectCodecRegistry.CodecDescriptor.class);
    ObjectCodecRegistry registry = Mockito.mock(ObjectCodecRegistry.class);
    when(registry.getCodecDescriptorByTag(1)).thenReturn(codecDescriptor);
    CodedInputStream codedInputStream = Mockito.mock(CodedInputStream.class);
    when(codedInputStream.readSInt32()).thenReturn(1);
    DeserializationContext deserializationContext =
        new DeserializationContext(registry, ImmutableClassToInstanceMap.of());
    Object returnValue = new Object();
    when(codecDescriptor.deserialize(deserializationContext, codedInputStream))
        .thenReturn(returnValue);
    assertThat((Object) deserializationContext.deserialize(codedInputStream))
        .isSameInstanceAs(returnValue);
    Mockito.verify(codedInputStream).readSInt32();
    Mockito.verify(registry).getCodecDescriptorByTag(1);
    Mockito.verify(codecDescriptor).deserialize(deserializationContext, codedInputStream);
  }

  @Test
  public void memoizingDeserialize_null() throws SerializationException, IOException {
    ObjectCodecRegistry registry = Mockito.mock(ObjectCodecRegistry.class);
    CodedInputStream codedInputStream = Mockito.mock(CodedInputStream.class);
    DeserializationContext deserializationContext =
        new DeserializationContext(registry, ImmutableClassToInstanceMap.of());
    when(codedInputStream.readSInt32()).thenReturn(0);
    assertThat((Object) deserializationContext.getMemoizingContext().deserialize(codedInputStream))
        .isEqualTo(null);
    Mockito.verify(codedInputStream).readSInt32();
    Mockito.verifyZeroInteractions(registry);
  }

  @Test
  public void memoizingDeserialize_constant() throws SerializationException, IOException {
    Object constant = new Object();
    ObjectCodecRegistry registry = Mockito.mock(ObjectCodecRegistry.class);
    when(registry.maybeGetConstantByTag(1)).thenReturn(constant);
    CodedInputStream codedInputStream = Mockito.mock(CodedInputStream.class);
    DeserializationContext deserializationContext =
        new DeserializationContext(registry, ImmutableClassToInstanceMap.of());
    when(codedInputStream.readSInt32()).thenReturn(1);
    assertThat((Object) deserializationContext.getMemoizingContext().deserialize(codedInputStream))
        .isEqualTo(constant);
    Mockito.verify(codedInputStream).readSInt32();
    Mockito.verify(registry).maybeGetConstantByTag(1);
  }

  @Test
  public void memoizingDeserialize_codec() throws SerializationException, IOException {
    Object returned = new Object();
    @SuppressWarnings("unchecked")
    ObjectCodec<Object> codec = Mockito.mock(ObjectCodec.class);
    when(codec.getStrategy()).thenReturn(MemoizationStrategy.MEMOIZE_AFTER);
    when(codec.getEncodedClass()).thenAnswer(unused -> Object.class);
    when(codec.additionalEncodedClasses()).thenReturn(ImmutableList.of());
    ObjectCodecRegistry.CodecDescriptor codecDescriptor =
        Mockito.mock(ObjectCodecRegistry.CodecDescriptor.class);
    doReturn(codec).when(codecDescriptor).getCodec();
    ObjectCodecRegistry registry = Mockito.mock(ObjectCodecRegistry.class);
    when(registry.getCodecDescriptorByTag(1)).thenReturn(codecDescriptor);
    CodedInputStream codedInputStream = Mockito.mock(CodedInputStream.class);
    DeserializationContext deserializationContext =
        new DeserializationContext(registry, ImmutableClassToInstanceMap.of())
            .getMemoizingContext();
    when(codec.deserialize(deserializationContext, codedInputStream)).thenReturn(returned);
    when(codedInputStream.readSInt32()).thenReturn(1);
    assertThat((Object) deserializationContext.deserialize(codedInputStream)).isEqualTo(returned);
    Mockito.verify(codedInputStream).readSInt32();
    Mockito.verify(registry).maybeGetConstantByTag(1);
    Mockito.verify(registry).getCodecDescriptorByTag(1);
    Mockito.verify(codecDescriptor).getCodec();
    Mockito.verify(codec).deserialize(deserializationContext, codedInputStream);
  }
}
