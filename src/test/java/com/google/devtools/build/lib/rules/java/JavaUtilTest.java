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
    return JavaUtil.getJavaRoot(PathFragment.create(path)).getPathString();
  }

  @Test
  public void testGetJavaRoot() {
    assertThat(
        JavaUtil.getJavaRoot(
            PathFragment.create("path/without/Java/or/Javatests/or/Src/lowercase")))
        .isNull();
    assertThat(getRootPath("java/com/google/common/case")).isEqualTo("java");
    assertThat(getRootPath("javatests/com/google/common/case")).isEqualTo("javatests");
    assertThat(getRootPath("src/com/myproject/util")).isEqualTo("src");
    assertThat(getRootPath("testsrc/com/myproject/util")).isEqualTo("testsrc");
    assertThat(getRootPath("project/java")).isEqualTo("project/java");
    assertThat(getRootPath("project/java/anything")).isEqualTo("project/java");
    assertThat(getRootPath("project/javatests")).isEqualTo("project/javatests");
    assertThat(getRootPath("project/javatests/anything")).isEqualTo("project/javatests");
    assertThat(getRootPath("project/src")).isEqualTo("project/src");
    assertThat(getRootPath("project/testsrc")).isEqualTo("project/testsrc");
    assertThat(getRootPath("project/src/anything")).isEqualTo("project/src");
    assertThat(getRootPath("project/testsrc/anything")).isEqualTo("project/testsrc");
    assertThat(getRootPath("third_party/java_src/project/src/main/java/foo"))
        .isEqualTo("third_party/java_src/project/src/main/java");
    assertThat(getRootPath("third_party/java_src/project/src/test/java/foo"))
        .isEqualTo("third_party/java_src/project/src/test/java");
    assertThat(getRootPath("third_party/java_src/project/src/main/resources/foo"))
        .isEqualTo("third_party/java_src/project/src/main/resources");
    assertThat(getRootPath("third_party/java_src/project/src/test/resources/foo"))
        .isEqualTo("third_party/java_src/project/src/test/resources");
    assertThat(getRootPath("third_party/java_src/project/javatests/foo"))
        .isEqualTo("third_party/java_src/project/javatests");

    // Cases covering nested /src/ directories.
    assertThat(getRootPath("java/com/google/project/module/src/com"))
        .isEqualTo("java/com/google/project/module/src");
    assertThat(getRootPath("java/com/google/project/module/src/org"))
        .isEqualTo("java/com/google/project/module/src");
    assertThat(getRootPath("java/com/google/project/module/src/net"))
        .isEqualTo("java/com/google/project/module/src");
    assertThat(getRootPath("java/com/google/project/module/src/main/java"))
        .isEqualTo("java/com/google/project/module/src/main/java");
    assertThat(getRootPath("java/com/google/project/module/src/test/java"))
        .isEqualTo("java/com/google/project/module/src/test/java");
    assertThat(getRootPath("javatests/com/google/project/src/com"))
        .isEqualTo("javatests/com/google/project/src");
    assertThat(getRootPath("src/com/google/project/src/main/java"))
        .isEqualTo("src/com/google/project/src/main/java");
    assertThat(getRootPath("java/com/google/project/module/src/somethingelse"))
        .isEqualTo("java");
    assertThat(getRootPath("java/com/google/project/module/testsrc/somethingelse"))
        .isEqualTo("java");
    assertThat(getRootPath("java/com/google/project/module/src/foo/java"))
        .isEqualTo("java");
    assertThat(getRootPath("java/com/google/project/module/testsrc/foo/java"))
        .isEqualTo("java");
    assertThat(getRootPath("java/com/google/project/module/src/main/com"))
        .isEqualTo("java");
    assertThat(getRootPath("java/com/google/project/module/src/test/org"))
        .isEqualTo("java");
    assertThat(getRootPath("java/com/google/project/module/src/java/com"))
        .isEqualTo("java");
    assertThat(getRootPath("foo/java/com/google/project/src/com"))
        .isEqualTo("foo/java");
    assertThat(getRootPath("src/com/google/java/javac"))
        .isEqualTo("src");
    assertThat(getRootPath("testsrc/com/google/java/javac"))
        .isEqualTo("testsrc");

    assertThat(getRootPath("src/java_tools/buildjar/javatests/com"))
        .isEqualTo("src/java_tools/buildjar/javatests");
    assertThat(getRootPath("third_party/project/src/java_tools/buildjar/javatests/com"))
        .isEqualTo("third_party/project/src/java_tools/buildjar/javatests");
    assertThat(getRootPath("third_party/project/src/java_tools/buildjar/java/net"))
        .isEqualTo("third_party/project/src/java_tools/buildjar/java");
    assertThat(getRootPath("src/java_tools/buildjar/javatests/foo"))
        .isEqualTo("src");
    assertThat(getRootPath("src/tools/workspace/src/test/java/foo"))
        .isEqualTo("src/tools/workspace/src/test/java");
    assertThat(getRootPath("foo/src/tools/workspace/src/test/java/foo"))
        .isEqualTo("foo/src/tools/workspace/src/test/java");
  }

  @Test
  public void testGetJavaPackageName() {
    assertThat(JavaUtil.getJavaPackageName(
        PathFragment.create("java/com/google/foo/FooModule.java"))).isEqualTo("com.google.foo");
    assertThat(JavaUtil.getJavaPackageName(PathFragment.create("org/foo/FooUtil.java")))
        .isEqualTo("org.foo");
  }

  @Test
  public void testGetJavaFullClassname() {
    assertThat(
        JavaUtil.getJavaFullClassname(PathFragment.create("java/com/google/foo/FooModule.java")))
        .isEqualTo("com.google.foo.FooModule.java");
    assertThat(JavaUtil.getJavaFullClassname(PathFragment.create("org/foo/FooUtil.java"))).isNull();
  }

  @Test
  public void testGetJavaPath() {
    assertThat(
        JavaUtil.getJavaPath(
            PathFragment.create("java/com/google/foo/FooModule.java")).getPathString())
        .isEqualTo("com/google/foo/FooModule.java");
    assertThat(JavaUtil.getJavaPath(PathFragment.create("org/foo/FooUtil.java"))).isNull();
  }
}
