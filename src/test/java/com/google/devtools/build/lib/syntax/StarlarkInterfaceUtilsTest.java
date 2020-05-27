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
import static org.junit.Assert.assertThrows;

import java.lang.reflect.Method;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkInterfaceUtils;
import net.starlark.java.annot.StarlarkMethod;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test Starlark interface annotations and utilities. */
@RunWith(JUnit4.class)
public class StarlarkInterfaceUtilsTest {

  /** MockClassA */
  @StarlarkBuiltin(name = "MockClassA", doc = "MockClassA")
  public static class MockClassA implements StarlarkValue {
    @StarlarkMethod(name = "foo", doc = "MockClassA#foo")
    public void foo() {}

    @StarlarkMethod(name = "bar", doc = "MockClassA#bar")
    public void bar() {}

    public void baz() {}
  }

  /** MockInterfaceB1 */
  @StarlarkBuiltin(name = "MockInterfaceB1", doc = "MockInterfaceB1")
  public static interface MockInterfaceB1 extends StarlarkValue {
    @StarlarkMethod(name = "foo", doc = "MockInterfaceB1#foo")
    void foo();

    @StarlarkMethod(name = "bar", doc = "MockInterfaceB1#bar")
    void bar();

    @StarlarkMethod(name = "baz", doc = "MockInterfaceB1#baz")
    void baz();
  }

  /** MockInterfaceB2 */
  @StarlarkBuiltin(name = "MockInterfaceB2", doc = "MockInterfaceB2")
  public static interface MockInterfaceB2 extends StarlarkValue {
    @StarlarkMethod(name = "baz", doc = "MockInterfaceB2#baz")
    void baz();

    @StarlarkMethod(name = "qux", doc = "MockInterfaceB2#qux")
    void qux();
  }

  /** MockClassC */
  @StarlarkBuiltin(name = "MockClassC", doc = "MockClassC")
  public static class MockClassC extends MockClassA implements MockInterfaceB1, MockInterfaceB2 {
    @Override
    @StarlarkMethod(name = "foo", doc = "MockClassC#foo")
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
    @StarlarkMethod(name = "foo", doc = "MockClassD#foo")
    public void foo() {}
  }

  /**
   * A mock class that implements two unrelated module interfaces. This is invalid as the skylark
   * type of such an object is ambiguous.
   */
  public static class ImplementsTwoUnrelatedInterfaceModules
      implements MockInterfaceB1, MockInterfaceB2 {
    @Override
    public void foo() {}
    @Override
    public void bar() {}
    @Override
    public void baz() {}
    @Override
    public void qux() {}
  }

  /** ClassAModule test class */
  @StarlarkBuiltin(name = "ClassAModule", doc = "ClassAModule")
  public static class ClassAModule implements StarlarkValue {}

  /** ExtendsClassA test class */
  public static class ExtendsClassA extends ClassAModule {}

  /** InterfaceBModule test interface */
  @StarlarkBuiltin(name = "InterfaceBModule", doc = "InterfaceBModule")
  public static interface InterfaceBModule extends StarlarkValue {}

  /** ExtendsInterfaceB test interface */
  public static interface ExtendsInterfaceB extends InterfaceBModule {}

  /**
   * A mock class which has two transitive superclasses ({@link ClassAModule} and {@link
   * InterfaceBModule})) which are unrelated modules. This is invalid as the Starlark type of such
   * an object is ambiguous.
   *
   * <p>In other words: AmbiguousClass -> ClassAModule AmbiguousClass -> InterfaceBModule ... but
   * ClassAModule and InterfaceBModule have no relation.
   */
  public static class AmbiguousClass extends ExtendsClassA implements ExtendsInterfaceB {}

  /** SubclassOfBoth test interface */
  @StarlarkBuiltin(name = "SubclassOfBoth", doc = "SubclassOfBoth")
  public static class SubclassOfBoth extends ExtendsClassA implements ExtendsInterfaceB {}

  /**
   * A mock class similar to {@link AmbiugousClass} in that it has two separate superclass-paths to
   * Starlark modules, but is resolvable.
   *
   * <p>Concretely: UnambiguousClass -> SubclassOfBoth UnambiguousClass -> InterfaceBModule
   * SubclassOfBoth -> InterfaceBModule
   *
   * <p>... so UnambiguousClass is of type SubclassOfBoth.
   */
  public static class UnambiguousClass extends SubclassOfBoth implements ExtendsInterfaceB {}

  /** MockClassZ */
  public static class MockClassZ {
  }

  // The tests for getStarlarkBuiltin() double as tests for getParentWithStarlarkBuiltin(),
  // since they share an implementation.

  @Test
  public void testGetStarlarkBuiltinBasic() throws Exception {
    // Normal case.
    StarlarkBuiltin ann = StarlarkInterfaceUtils.getStarlarkBuiltin(MockClassA.class);
    Class<?> cls = StarlarkInterfaceUtils.getParentWithStarlarkBuiltin(MockClassA.class);
    assertThat(ann).isNotNull();
    assertThat(ann.doc()).isEqualTo("MockClassA");
    assertThat(cls).isNotNull();
    assertThat(cls).isEqualTo(MockClassA.class);
  }

  @Test
  public void testGetStarlarkBuiltinSubclass() throws Exception {
    // Subclass's annotation is used.
    StarlarkBuiltin ann = StarlarkInterfaceUtils.getStarlarkBuiltin(MockClassC.class);
    Class<?> cls = StarlarkInterfaceUtils.getParentWithStarlarkBuiltin(MockClassC.class);
    assertThat(ann).isNotNull();
    assertThat(ann.doc()).isEqualTo("MockClassC");
    assertThat(cls).isNotNull();
    assertThat(cls).isEqualTo(MockClassC.class);
  }

  @Test
  public void testGetStarlarkBuiltinSubclassNoSubannotation() throws Exception {
    // Falls back on superclass's annotation.
    StarlarkBuiltin ann = StarlarkInterfaceUtils.getStarlarkBuiltin(MockClassD.class);
    Class<?> cls = StarlarkInterfaceUtils.getParentWithStarlarkBuiltin(MockClassD.class);
    assertThat(ann).isNotNull();
    assertThat(ann.doc()).isEqualTo("MockClassC");
    assertThat(cls).isNotNull();
    assertThat(cls).isEqualTo(MockClassC.class);
  }

  @Test
  public void testGetStarlarkBuiltinNotFound() throws Exception {
    // Doesn't exist.
    StarlarkBuiltin ann = StarlarkInterfaceUtils.getStarlarkBuiltin(MockClassZ.class);
    Class<?> cls = StarlarkInterfaceUtils.getParentWithStarlarkBuiltin(MockClassZ.class);
    assertThat(ann).isNull();
    assertThat(cls).isNull();
  }

  @Test
  public void testGetStarlarkBuiltinAmbiguous() throws Exception {
    assertThrows(IllegalArgumentException.class,
        () -> StarlarkInterfaceUtils.getStarlarkBuiltin(ImplementsTwoUnrelatedInterfaceModules.class));
  }

  @Test
  public void testGetStarlarkBuiltinTransitivelyAmbiguous() throws Exception {
    assertThrows(IllegalArgumentException.class,
        () -> StarlarkInterfaceUtils.getStarlarkBuiltin(AmbiguousClass.class));
  }

  @Test
  public void testGetStarlarkBuiltinUnambiguousComplex() throws Exception {
    assertThat(StarlarkInterfaceUtils.getStarlarkBuiltin(SubclassOfBoth.class))
        .isEqualTo(SubclassOfBoth.class.getAnnotation(StarlarkBuiltin.class));

    assertThat(StarlarkInterfaceUtils.getStarlarkBuiltin(UnambiguousClass.class))
        .isEqualTo(SubclassOfBoth.class.getAnnotation(StarlarkBuiltin.class));
  }

  @Test
  public void testGetStarlarkCallableBasic() throws Exception {
    // Normal case. Ensure two-arg form is consistent with one-arg form.
    Method method = MockClassA.class.getMethod("foo");
    StarlarkMethod ann = StarlarkInterfaceUtils.getStarlarkMethod(method);
    assertThat(ann).isNotNull();
    assertThat(ann.doc()).isEqualTo("MockClassA#foo");

    StarlarkMethod ann2 = StarlarkInterfaceUtils.getStarlarkMethod(MockClassA.class, method);
    assertThat(ann2).isEqualTo(ann);
  }

  @Test
  public void testGetStarlarkCallableSubclass() throws Exception {
    // Subclass's annotation is used.
    Method method = MockClassC.class.getMethod("foo");
    StarlarkMethod ann = StarlarkInterfaceUtils.getStarlarkMethod(method);
    assertThat(ann).isNotNull();
    assertThat(ann.doc()).isEqualTo("MockClassC#foo");
  }

  @Test
  public void testGetStarlarkCallableSubclassNoSubannotation() throws Exception {
    // Falls back on superclass's annotation. Superclass takes precedence over interface.
    Method method = MockClassC.class.getMethod("bar");
    StarlarkMethod ann = StarlarkInterfaceUtils.getStarlarkMethod(method);
    assertThat(ann).isNotNull();
    assertThat(ann.doc()).isEqualTo("MockClassA#bar");
  }

  @Test
  public void testGetStarlarkCallableTwoargForm() throws Exception {
    // Ensure that when passing superclass in directly, we bypass subclass's annotation.
    Method method = MockClassC.class.getMethod("foo");
    StarlarkMethod ann = StarlarkInterfaceUtils.getStarlarkMethod(MockClassA.class, method);
    assertThat(ann).isNotNull();
    assertThat(ann.doc()).isEqualTo("MockClassA#foo");
  }

  @Test
  public void testGetStarlarkCallableNotFound() throws Exception {
    // Null result when no annotation present...
    Method method = MockClassA.class.getMethod("baz");
    StarlarkMethod ann = StarlarkInterfaceUtils.getStarlarkMethod(method);
    assertThat(ann).isNull();

    // ... including when it's only present in a subclass that was bypassed...
    method = MockClassC.class.getMethod("baz");
    ann = StarlarkInterfaceUtils.getStarlarkMethod(MockClassA.class, method);
    assertThat(ann).isNull();

    // ... or when the method itself is only in the subclass that was bypassed.
    method = MockClassC.class.getMethod("qux");
    ann = StarlarkInterfaceUtils.getStarlarkMethod(MockClassA.class, method);
    assertThat(ann).isNull();
  }

  @Test
  public void testGetStarlarkCallableInterface() throws Exception {
    // Search through parent interfaces. First interface takes priority.
    Method method = MockClassC.class.getMethod("baz");
    StarlarkMethod ann = StarlarkInterfaceUtils.getStarlarkMethod(method);
    assertThat(ann).isNotNull();
    assertThat(ann.doc()).isEqualTo("MockInterfaceB1#baz");

    // Make sure both are still traversed.
    method = MockClassC.class.getMethod("qux");
    ann = StarlarkInterfaceUtils.getStarlarkMethod(method);
    assertThat(ann).isNotNull();
    assertThat(ann.doc()).isEqualTo("MockInterfaceB2#qux");
  }
}
