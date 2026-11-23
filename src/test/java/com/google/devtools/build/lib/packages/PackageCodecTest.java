// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.packages;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.devtools.build.lib.packages.PackageCodecDependencies.SimplePackageCodecDependencies;
import com.google.devtools.build.lib.skyframe.PackageValue;
import com.google.devtools.build.lib.skyframe.serialization.DeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.ImmutableDeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.protobuf.CodedInputStream;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Basic tests for {@link PackageValue}. */
@RunWith(JUnit4.class)
public class PackageCodecTest {

  private final PackageSerializerInterface mockPackageSerializer =
      mock(PackageSerializerInterface.class);
  private final ObjectCodec<Package> underTest = new Package.PackageCodec();
  private final SimplePackageCodecDependencies codecDeps =
      new SimplePackageCodecDependencies(mockPackageSerializer);

  @Test
  public void testDeserializationIsDelegatedToPackageDeserializer()
      throws SerializationException, IOException, InterruptedException {
    // Mock because all we need is to verify that we're properly delegating to Package deserializer.
    Package mockPackage = mock(Package.class);

    when(mockPackageSerializer.deserialize(
        any(DeserializationContext.class), any(CodedInputStream.class))).thenReturn(mockPackage);

    CodedInputStream codedIn = CodedInputStream.newInstance(new byte[] {1, 2, 3, 4});
    Package result =
        underTest.deserialize(
            new ImmutableDeserializationContext(
                ImmutableClassToInstanceMap.of(PackageCodecDependencies.class, codecDeps)),
            codedIn);

    assertThat(result).isSameInstanceAs(mockPackage);
  }
}
