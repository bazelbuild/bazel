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

import com.google.common.collect.ImmutableMap;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;

/** Tests for {@link SerializationContext}. */
@RunWith(JUnit4.class)
public class SerializationContextTest {
  @Test
  public void nullSerialize() throws IOException, SerializationException {
    ObjectCodecRegistry registry = Mockito.mock(ObjectCodecRegistry.class);
    CodedOutputStream codedOutputStream = Mockito.mock(CodedOutputStream.class);
    SerializationContext serializationContext =
        new SerializationContext(registry, ImmutableMap.of());
    serializationContext.serialize(null, codedOutputStream);
    Mockito.verify(codedOutputStream).writeSInt32NoTag(0);
    Mockito.verifyZeroInteractions(registry);
  }

  @Test
  public void constantSerialize() throws IOException, SerializationException {
    ObjectCodecRegistry registry = Mockito.mock(ObjectCodecRegistry.class);
    when(registry.maybeGetTagForConstant(Mockito.anyObject())).thenReturn(1);
    CodedOutputStream codedOutputStream = Mockito.mock(CodedOutputStream.class);
    SerializationContext serializationContext =
        new SerializationContext(registry, ImmutableMap.of());
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
    when(registry.maybeGetTagForConstant(Mockito.anyObject())).thenReturn(null);
    when(registry.getCodecDescriptor(String.class)).thenReturn(codecDescriptor);
    CodedOutputStream codedOutputStream = Mockito.mock(CodedOutputStream.class);
    SerializationContext underTest = new SerializationContext(registry, ImmutableMap.of());
    underTest.serialize("string", codedOutputStream);
    Mockito.verify(codedOutputStream).writeSInt32NoTag(1);
    Mockito.verify(registry).maybeGetTagForConstant("string");
    Mockito.verify(registry).getCodecDescriptor(String.class);
    Mockito.verify(codecDescriptor).getTag();
    Mockito.verify(codecDescriptor).serialize(underTest, "string", codedOutputStream);
  }

  @Test
  public void memoizingSerialize_null() throws IOException, SerializationException {
    ObjectCodecRegistry registry = Mockito.mock(ObjectCodecRegistry.class);
    CodedOutputStream codedOutputStream = Mockito.mock(CodedOutputStream.class);
    SerializationContext serializationContext =
        new SerializationContext(registry, ImmutableMap.of());
    assertThat(serializationContext.recordAndMaybeGetMemoizingCodec(null, codedOutputStream))
        .isNull();
    Mockito.verify(codedOutputStream).writeSInt32NoTag(0);
    Mockito.verifyZeroInteractions(registry);
  }

  @Test
  public void memoizingSerialize_constant() throws IOException, SerializationException {
    ObjectCodecRegistry registry = Mockito.mock(ObjectCodecRegistry.class);
    when(registry.maybeGetTagForConstant(Mockito.anyObject())).thenReturn(1);
    CodedOutputStream codedOutputStream = Mockito.mock(CodedOutputStream.class);
    SerializationContext serializationContext =
        new SerializationContext(registry, ImmutableMap.of());
    Object constant = new Object();
    assertThat(serializationContext.recordAndMaybeGetMemoizingCodec(constant, codedOutputStream))
        .isNull();
    Mockito.verify(codedOutputStream).writeSInt32NoTag(1);
    Mockito.verify(registry).maybeGetTagForConstant(constant);
  }

  @Test
  public void memoizingSerialize_memoize() throws SerializationException, IOException {
    @SuppressWarnings("unchecked")
    Memoizer.MemoizingCodec<Object> memoizingCodec = Mockito.mock(Memoizer.MemoizingCodec.class);
    @SuppressWarnings("unchecked")
    ObjectCodecRegistry.MemoizingCodecDescriptor<Object> memoizingCodecDescriptor =
        Mockito.mock(ObjectCodecRegistry.MemoizingCodecDescriptor.class);
    when(memoizingCodecDescriptor.getTag()).thenReturn(1);
    doReturn(memoizingCodec).when(memoizingCodecDescriptor).getMemoizingCodec();
    ObjectCodecRegistry registry = Mockito.mock(ObjectCodecRegistry.class);
    when(registry.maybeGetTagForConstant(Mockito.anyObject())).thenReturn(null);
    doReturn(memoizingCodecDescriptor).when(registry).getMemoizingCodecDescriptor(String.class);
    CodedOutputStream codedOutputStream = Mockito.mock(CodedOutputStream.class);
    SerializationContext underTest = new SerializationContext(registry, ImmutableMap.of());
    assertThat(underTest.recordAndMaybeGetMemoizingCodec("string", codedOutputStream))
        .isSameAs(memoizingCodec);
    Mockito.verify(codedOutputStream).writeSInt32NoTag(1);
    Mockito.verify(registry).maybeGetTagForConstant("string");
    Mockito.verify(registry).getMemoizingCodecDescriptor(String.class);
    Mockito.verifyNoMoreInteractions(registry);
  }

  @Test
  public void memoizingSerialize_descriptor() throws SerializationException, IOException {
    @SuppressWarnings("unchecked")
    ObjectCodec<Object> codec = Mockito.mock(ObjectCodec.class);
    ObjectCodecRegistry.CodecDescriptor codecDescriptor =
        Mockito.mock(ObjectCodecRegistry.CodecDescriptor.class);
    when(codecDescriptor.getTag()).thenReturn(1);
    doReturn(codec).when(codecDescriptor).getCodec();
    ObjectCodecRegistry registry = Mockito.mock(ObjectCodecRegistry.class);
    when(registry.maybeGetTagForConstant(Mockito.anyObject())).thenReturn(null);
    when(registry.getMemoizingCodecDescriptor(Mockito.anyObject())).thenReturn(null);
    when(registry.getCodecDescriptor(String.class)).thenReturn(codecDescriptor);
    CodedOutputStream codedOutputStream = Mockito.mock(CodedOutputStream.class);
    SerializationContext underTest = new SerializationContext(registry, ImmutableMap.of());
    underTest
        .recordAndMaybeGetMemoizingCodec("string", codedOutputStream)
        .serializePayload(underTest, "string", codedOutputStream, null);
    Mockito.verify(codedOutputStream).writeSInt32NoTag(1);
    Mockito.verify(registry).maybeGetTagForConstant("string");
    Mockito.verify(registry).getMemoizingCodecDescriptor(String.class);
    Mockito.verify(registry).getCodecDescriptor(String.class);
    Mockito.verify(codecDescriptor).getTag();
    Mockito.verify(codecDescriptor).getCodec();
    Mockito.verify(codec).serialize(underTest, "string", codedOutputStream);
  }
}
