// Copyright 2016 The Bazel Authors. All Rights Reserved.
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

package com.google.devtools.build.lib.syntax;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkInterfaceUtils;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import java.lang.reflect.Method;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 *  Test Skylark interface annotations and utilities.
 */
@RunWith(JUnit4.class)
public class SkylarkInterfaceUtilsTest {

  /** MockClassA */
  @SkylarkModule(name = "MockClassA", doc = "MockClassA")
  public static class MockClassA {
    @SkylarkCallable(doc = "MockClassA#foo")
    public void foo() {}
    @SkylarkCallable(doc = "MockClassA#bar")
    public void bar() {}
    public void baz() {}
  }

  /** MockInterfaceB1 */
  @SkylarkModule(name = "MockInterfaceB1", doc = "MockInterfaceB1")
  public static interface MockInterfaceB1 {
    @SkylarkCallable(doc = "MockInterfaceB1#foo")
    void foo();
    @SkylarkCallable(doc = "MockInterfaceB1#bar")
    void bar();
    @SkylarkCallable(doc = "MockInterfaceB1#baz")
    void baz();
  }

  /** MockInterfaceB2 */
  @SkylarkModule(name = "MockInterfaceB2", doc = "MockInterfaceB2")
  public static interface MockInterfaceB2 {
    @SkylarkCallable(doc = "MockInterfaceB2#baz")
    void baz();
    @SkylarkCallable(doc = "MockInterfaceB2#qux")
    void qux();
  }

  /** MockClassC */
  @SkylarkModule(name = "MockClassC", doc = "MockClassC")
  public static class MockClassC extends MockClassA implements MockInterfaceB1, MockInterfaceB2 {
    @Override
    @SkylarkCallable(doc = "MockClassC#foo")
    public void foo() {}
    @Override
    public void bar() {}
    @Override
    public void baz() {}
    @Override
    public void qux() {}
  }

  /** MockClassD */
  public static class MockClassD extends MockClassC {
    @Override
    @SkylarkCallable(doc = "MockClassD#foo")
    public void foo() {}
  }

  @Test
  public void testGetSkylarkCallableBasic() throws Exception {
    // Normal case. Ensure two-arg form is consistent with one-arg form.
    Method method = MockClassA.class.getMethod("foo");
    SkylarkCallable ann = SkylarkInterfaceUtils.getSkylarkCallable(method);
    assertThat(ann).isNotNull();
    assertThat(ann.doc()).isEqualTo("MockClassA#foo");

    SkylarkCallable ann2 = SkylarkInterfaceUtils.getSkylarkCallable(MockClassA.class, method);
    assertThat(ann2).isEqualTo(ann);
  }

  @Test
  public void testGetSkylarkCallableSubclass() throws Exception {
    // Subclass's annotation is used.
    Method method = MockClassC.class.getMethod("foo");
    SkylarkCallable ann = SkylarkInterfaceUtils.getSkylarkCallable(method);
    assertThat(ann).isNotNull();
    assertThat(ann.doc()).isEqualTo("MockClassC#foo");
  }

  @Test
  public void testGetSkylarkCallableSubclassNoSubannotation() throws Exception {
    // Falls back on superclass's annotation. Superclass takes precedence over interface.
    Method method = MockClassC.class.getMethod("bar");
    SkylarkCallable ann = SkylarkInterfaceUtils.getSkylarkCallable(method);
    assertThat(ann).isNotNull();
    assertThat(ann.doc()).isEqualTo("MockClassA#bar");
  }

  @Test
  public void testGetSkylarkCallableTwoargForm() throws Exception {
    // Ensure that when passing superclass in directly, we bypass subclass's annotation.
    Method method = MockClassC.class.getMethod("foo");
    SkylarkCallable ann = SkylarkInterfaceUtils.getSkylarkCallable(MockClassA.class, method);
    assertThat(ann).isNotNull();
    assertThat(ann.doc()).isEqualTo("MockClassA#foo");
  }

  @Test
  public void testGetSkylarkCallableNotFound() throws Exception {
    // Null result when no annotation present...
    Method method = MockClassA.class.getMethod("baz");
    SkylarkCallable ann = SkylarkInterfaceUtils.getSkylarkCallable(method);
    assertThat(ann).isNull();

    // ... including when it's only present in a subclass that was bypassed.
    method = MockClassC.class.getMethod("baz");
    ann = SkylarkInterfaceUtils.getSkylarkCallable(MockClassA.class, method);
    assertThat(ann).isNull();
  }

  @Test
  public void testGetSkylarkCallableInterface() throws Exception {
    // Search through parent interfaces. First interface takes priority.
    Method method = MockClassC.class.getMethod("baz");
    SkylarkCallable ann = SkylarkInterfaceUtils.getSkylarkCallable(method);
    assertThat(ann).isNotNull();
    assertThat(ann.doc()).isEqualTo("MockInterfaceB1#baz");

    // Make sure both are still traversed.
    method = MockClassC.class.getMethod("qux");
    ann = SkylarkInterfaceUtils.getSkylarkCallable(method);
    assertThat(ann).isNotNull();
    assertThat(ann.doc()).isEqualTo("MockInterfaceB2#qux");
  }

  @Test
  public void testGetSkylarkCallableIgnoreNonModules() throws Exception {
    // Don't return SkylarkCallable annotations in classes and interfaces
    // not marked @SkylarkModule.
    Method method = MockClassD.class.getMethod("foo");
    SkylarkCallable ann = SkylarkInterfaceUtils.getSkylarkCallable(method);
    assertThat(ann).isNotNull();
    assertThat(ann.doc()).isEqualTo("MockClassC#foo");
  }
}
