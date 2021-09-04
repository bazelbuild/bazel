// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.cpp;

import static com.google.common.truth.Truth.assertThat;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Test for {@link CppFileTypes}.
 */
@RunWith(JUnit4.class)
public class CppFileTypesTest {

  @Test
  public void testTwoDotExtensions() {
    assertThat(CppFileTypes.OBJECT_FILE.matches("test.o")).isTrue();
    assertThat(CppFileTypes.PIC_OBJECT_FILE.matches("test.pic.o")).isTrue();
    assertThat(CppFileTypes.OBJECT_FILE.matches("test.pic.o")).isFalse();
  }

  @Test
  public void testRlib() {
    assertThat(CppFileTypes.RUST_RLIB.matches("foo.a")).isFalse();
    assertThat(CppFileTypes.RUST_RLIB.matches("foo.rlib")).isTrue();
  }

  @Test
  public void testVersionedSharedLibraries() {
    assertThat(CppFileTypes.SHARED_LIBRARY.matches("somelibrary.so")).isTrue();
    assertThat(CppFileTypes.VERSIONED_SHARED_LIBRARY.matches("somelibrary.so.2")).isTrue();
    assertThat(CppFileTypes.VERSIONED_SHARED_LIBRARY.matches("somelibrary.so.20")).isTrue();
    assertThat(CppFileTypes.VERSIONED_SHARED_LIBRARY.matches("somelibrary.so.20.2")).isTrue();
    assertThat(CppFileTypes.VERSIONED_SHARED_LIBRARY.matches("a/somelibrary.so.2")).isTrue();
    assertThat(CppFileTypes.VERSIONED_SHARED_LIBRARY.matches("somelibrary.so.e")).isFalse();
    assertThat(CppFileTypes.VERSIONED_SHARED_LIBRARY.matches("somelibrary.so.2e")).isTrue();
    assertThat(CppFileTypes.VERSIONED_SHARED_LIBRARY.matches("somelibrary.so.e2")).isFalse();
    assertThat(CppFileTypes.VERSIONED_SHARED_LIBRARY.matches("somelibrary.so.20.e2")).isFalse();
    assertThat(CppFileTypes.VERSIONED_SHARED_LIBRARY.matches("somelibrary.a.2")).isFalse();
    assertThat(CppFileTypes.VERSIONED_SHARED_LIBRARY.matches("somelibrary.so.2$")).isFalse();
    assertThat(CppFileTypes.VERSIONED_SHARED_LIBRARY.matches("somelibrary.so.1a_b2")).isTrue();
    assertThat(CppFileTypes.VERSIONED_SHARED_LIBRARY.matches("libA.so.gen.empty.def")).isFalse();
    assertThat(CppFileTypes.VERSIONED_SHARED_LIBRARY.matches("libA.so.if.exp")).isFalse();
    assertThat(CppFileTypes.VERSIONED_SHARED_LIBRARY.matches("libA.so.if.lib")).isFalse();
  }
}
