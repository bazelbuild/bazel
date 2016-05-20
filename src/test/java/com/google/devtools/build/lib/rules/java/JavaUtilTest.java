// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.java;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.vfs.PathFragment;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link JavaUtil} methods.
 */
@RunWith(JUnit4.class)
public class JavaUtilTest {

  private String getRootPath(String path) {
    return JavaUtil.getJavaRoot(new PathFragment(path)).getPathString();
  }

  @Test
  public void testGetJavaRoot() {
    assertThat(
        JavaUtil.getJavaRoot(new PathFragment("path/without/Java/or/Javatests/or/Src/lowercase")))
        .isNull();
    assertThat(getRootPath("java/com/google/common/case")).isEqualTo("java");
    assertThat(getRootPath("javatests/com/google/common/case")).isEqualTo("javatests");
    assertThat(getRootPath("project/java")).isEqualTo("project/java");
    assertThat(getRootPath("project/java/anything")).isEqualTo("project/java");
    assertThat(getRootPath("project/javatests")).isEqualTo("project/javatests");
    assertThat(getRootPath("project/javatests/anything")).isEqualTo("project/javatests");
    assertThat(getRootPath("third_party/java_src/project/src/main/java/foo"))
        .isEqualTo("third_party/java_src/project/src/main/java");
    assertThat(getRootPath("third_party/java_src/project/src/test/java/foo"))
        .isEqualTo("third_party/java_src/project/src/test/java");
    assertThat(getRootPath("third_party/java_src/project/javatests/foo"))
        .isEqualTo("third_party/java_src/project/javatests");
  }

  @Test
  public void testGetJavaPackageName() {
    assertThat(JavaUtil.getJavaPackageName(new PathFragment("java/com/google/foo/FooModule.java")))
        .isEqualTo("com.google.foo");
    assertThat(JavaUtil.getJavaPackageName(new PathFragment("org/foo/FooUtil.java")))
        .isEqualTo("org.foo");
  }

  @Test
  public void testGetJavaFullClassname() {
    assertThat(
        JavaUtil.getJavaFullClassname(new PathFragment("java/com/google/foo/FooModule.java")))
        .isEqualTo("com.google.foo.FooModule.java");
    assertThat(JavaUtil.getJavaFullClassname(new PathFragment("org/foo/FooUtil.java"))).isNull();
  }

  @Test
  public void testGetJavaPath() {
    assertThat(
        JavaUtil.getJavaPath(
            new PathFragment("java/com/google/foo/FooModule.java")).getPathString())
        .isEqualTo("com/google/foo/FooModule.java");
    assertThat(JavaUtil.getJavaPath(new PathFragment("org/foo/FooUtil.java"))).isNull();
  }
}
