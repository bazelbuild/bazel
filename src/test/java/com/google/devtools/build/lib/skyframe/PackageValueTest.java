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

package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;
import static org.mockito.Matchers.any;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageCodecDependencies;
import com.google.devtools.build.lib.packages.PackageCodecDependencies.SimplePackageCodecDependencies;
import com.google.devtools.build.lib.packages.PackageDeserializationException;
import com.google.devtools.build.lib.packages.PackageDeserializerInterface;
import com.google.devtools.build.lib.skyframe.serialization.DeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.protobuf.CodedInputStream;
import java.io.IOException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.ArgumentCaptor;

/** Basic tests for {@link PackageValue}. */
@RunWith(JUnit4.class)
public class PackageValueTest {

  private PackageDeserializerInterface mockDeserializer;
  private ObjectCodec<PackageValue> underTest;
  SimplePackageCodecDependencies codecDeps;

  @Before
  public void setUp() {
    this.mockDeserializer = mock(PackageDeserializerInterface.class);
    this.underTest = PackageValue.CODEC;
    this.codecDeps = new SimplePackageCodecDependencies(null, mockDeserializer);
  }

  @Test
  public void testDeserializationIsDelegatedToPackageDeserializer()
      throws SerializationException, IOException, PackageDeserializationException,
          InterruptedException {
    // Mock because all we need is to verify that we're properly delegating to Package deserializer.
    Package mockPackage = mock(Package.class);

    when(mockDeserializer.deserialize(ArgumentCaptor.forClass(CodedInputStream.class).capture()))
        .thenReturn(mockPackage);

    CodedInputStream codedIn = CodedInputStream.newInstance(new byte[] {1, 2, 3, 4});
    PackageValue result =
        underTest.deserialize(
            new DeserializationContext(ImmutableMap.of(PackageCodecDependencies.class, codecDeps)),
            codedIn);

    assertThat(result.getPackage()).isSameAs(mockPackage);
  }

  @Test
  public void testInterruptedExceptionRaisesIllegalStateException() throws Exception {
    InterruptedException staged = new InterruptedException("Stop that!");
    doThrow(staged).when(mockDeserializer).deserialize(any(CodedInputStream.class));

    try {
      underTest.deserialize(
          new DeserializationContext(ImmutableMap.of(PackageCodecDependencies.class, codecDeps)),
          CodedInputStream.newInstance(new byte[] {1, 2, 3, 4}));
      fail("Expected exception");
    } catch (IllegalStateException e) {
      assertThat(e)
          .hasMessageThat()
          .isEqualTo("Unexpected InterruptedException during Package deserialization");
      assertThat(e).hasCauseThat().isSameAs(staged);
    }
  }
}
