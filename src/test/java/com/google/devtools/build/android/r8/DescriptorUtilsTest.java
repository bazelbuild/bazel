// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.r8;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for DexFileMerger. */
@RunWith(JUnit4.class)
public class DescriptorUtilsTest {

  @Test
  public void isClassDescriptor() throws Exception {
    assertThat(DescriptorUtils.isClassDescriptor("La/b/I$-CC;")).isTrue();
    assertThat(DescriptorUtils.isClassDescriptor("La/b/I;")).isTrue();
    assertThat(DescriptorUtils.isClassDescriptor("a/b/I$-CC")).isFalse();
    assertThat(DescriptorUtils.isClassDescriptor("a/b/I")).isFalse();
    assertThat(DescriptorUtils.isClassDescriptor("a.b.I$-CC")).isFalse();
    assertThat(DescriptorUtils.isClassDescriptor("a.b.I")).isFalse();
  }

  @Test
  public void isBinaryName() throws Exception {
    assertThat(DescriptorUtils.isBinaryName("a/b/I$-CC")).isTrue();
    assertThat(DescriptorUtils.isBinaryName("a/b/I")).isTrue();
    assertThat(DescriptorUtils.isBinaryName("La/b/I$-CC;")).isFalse();
    assertThat(DescriptorUtils.isBinaryName("La/b/I;")).isFalse();
    assertThat(DescriptorUtils.isBinaryName("a.b.I$-CC")).isFalse();
    assertThat(DescriptorUtils.isBinaryName("a.b.I")).isFalse();
  }

  @Test
  public void isCompanionClassBinaryName() throws Exception {
    assertThat(DescriptorUtils.isCompanionClassBinaryName("a/b/I$-CC")).isTrue();
    assertThat(DescriptorUtils.isCompanionClassBinaryName("a/b/I")).isFalse();
    assertThrows(
        IllegalArgumentException.class,
        () -> DescriptorUtils.isCompanionClassBinaryName("La/b/I$-CC;"));
    assertThrows(
        IllegalArgumentException.class,
        () -> DescriptorUtils.isCompanionClassBinaryName("La/b/I;"));
  }

  @Test
  public void descriptorToBinaryName() throws Exception {
    assertThat(DescriptorUtils.descriptorToBinaryName("LA;")).isEqualTo("A");
    assertThat(DescriptorUtils.descriptorToBinaryName("Ljava/lang/Object;"))
        .isEqualTo("java/lang/Object");
    assertThrows(IllegalArgumentException.class, () -> DescriptorUtils.descriptorToBinaryName("A"));
    assertThrows(
        IllegalArgumentException.class,
        () -> DescriptorUtils.descriptorToBinaryName("java/lang/Object"));
  }

  @Test
  public void descriptorToClassFileName() throws Exception {
    assertThat(DescriptorUtils.descriptorToClassFileName("LA;")).isEqualTo("A.class");
    assertThat(DescriptorUtils.descriptorToClassFileName("Ljava/lang/Object;"))
        .isEqualTo("java/lang/Object.class");
    assertThrows(
        IllegalArgumentException.class, () -> DescriptorUtils.descriptorToClassFileName("A"));
    assertThrows(
        IllegalArgumentException.class,
        () -> DescriptorUtils.descriptorToClassFileName("java/lang/Object"));
  }

  @Test
  public void classToBinaryName() throws Exception {
    assertThat(DescriptorUtils.classToBinaryName(Object.class)).isEqualTo("java/lang/Object");
  }
}
