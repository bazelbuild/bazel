// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.desugar;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.android.desugar.io.BitFlags;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.objectweb.asm.ClassReader;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;

@RunWith(JUnit4.class)
public class Java7CompatibilityTest {

  @Test
  public void testJava7CompatibleInterface() throws Exception {
    ClassReader reader = new ClassReader(ExtendsDefault.class.getName());
    ClassTester tester = new ClassTester();
    reader.accept(new Java7Compatibility(tester, null, null), 0);
    assertThat(tester.version).isEqualTo(Opcodes.V1_7);
    assertThat(tester.bridgeMethods).isEqualTo(0); // make sure we strip bridge methods
    assertThat(tester.clinitMethods).isEqualTo(1); // make sure we don't strip <clinit>
  }

  @Test
  public void testDefaultMethodFails() throws Exception {
    ClassReader reader = new ClassReader(WithDefault.class.getName());
    IllegalArgumentException expected =
        assertThrows(
            IllegalArgumentException.class,
            () -> reader.accept(new Java7Compatibility(null, null, null), 0));
    assertThat(expected).hasMessageThat().contains("getVersion()I");
  }

  /**
   * Tests that a class implementing interfaces with bridge methods redeclares those bridges. This
   * is behavior of javac that we rely on.
   */
  @Test
  public void testConcreteClassRedeclaresBridges() throws Exception {
    ClassReader reader = new ClassReader(Impl.class.getName());
    ClassTester tester = new ClassTester();
    reader.accept(new Java7Compatibility(tester, null, null), 0);
    assertThat(tester.version).isEqualTo(Opcodes.V1_7);
    assertThat(tester.bridgeMethods).isEqualTo(2);
  }

  private static class ClassTester extends ClassVisitor {

    int version;
    int bridgeMethods;
    int clinitMethods;

    private ClassTester() {
      super(Opcodes.ASM8, null);
    }

    @Override
    public void visit(
        int version,
        int access,
        String name,
        String signature,
        String superName,
        String[] interfaces) {
      this.version = version;
      super.visit(version, access, name, signature, superName, interfaces);
    }

    @Override
    public MethodVisitor visitMethod(
        int access, String name, String desc, String signature, String[] exceptions) {
      if (BitFlags.isSet(access, Opcodes.ACC_BRIDGE)) {
        ++bridgeMethods;
      }
      if ("<clinit>".equals(name)) {
        ++clinitMethods;
      }
      return super.visitMethod(access, name, desc, signature, exceptions);
    }
  }

  interface WithDefault<T> {
    default int getVersion() {
      return 18;
    }

    T get();
  }

  // Javac will generate a default bridge method "Object get()" that Java7Compatibility will remove
  interface ExtendsDefault<T extends Number> extends WithDefault<T> {
    public static final Integer X = Integer.valueOf(37);

    String name();

    @Override
    T get();
  }

  // Javac will generate 2 bridge methods that we *don't* want to remove
  static class Impl implements ExtendsDefault<Integer> {
    @Override
    public Integer get() {
      return X;
    }

    @Override
    public String name() {
      return "test";
    }
  }
}
