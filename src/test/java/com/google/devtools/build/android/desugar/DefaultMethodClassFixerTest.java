// Copyright 2017 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.android.desugar.DefaultMethodClassFixer.SubtypeComparator.INSTANCE;

import com.google.common.collect.ImmutableList;
import com.google.common.io.Closer;
import com.google.devtools.build.android.desugar.io.CoreLibraryRewriter;
import com.google.devtools.build.android.desugar.io.HeaderClassLoader;
import com.google.devtools.build.android.desugar.io.IndexedInputs;
import com.google.devtools.build.android.desugar.io.ThrowingClassLoader;
import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.TreeSet;
import java.util.concurrent.Callable;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.objectweb.asm.ClassReader;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.tree.AbstractInsnNode;
import org.objectweb.asm.tree.InsnNode;
import org.objectweb.asm.tree.MethodInsnNode;
import org.objectweb.asm.tree.MethodNode;

/** Unit Test for {@link DefaultMethodClassFixer} */
@RunWith(JUnit4.class)
public class DefaultMethodClassFixerTest {

  private ClassReaderFactory classpathReader;
  private ClassReaderFactory bootclassPath;
  private ClassLoader classLoader;
  private Closer closer;

  @Before
  public void setup() throws IOException {
    closer = Closer.create();
    CoreLibraryRewriter rewriter = new CoreLibraryRewriter("");

    IndexedInputs indexedInputs =
        toIndexedInputs(closer, System.getProperty("DefaultMethodClassFixerTest.input"));
    IndexedInputs indexedClasspath =
        toIndexedInputs(closer, System.getProperty("DefaultMethodClassFixerTest.classpath"));
    IndexedInputs indexedBootclasspath =
        toIndexedInputs(closer, System.getProperty("DefaultMethodClassFixerTest.bootclasspath"));

    bootclassPath = new ClassReaderFactory(indexedBootclasspath, rewriter);
    IndexedInputs indexedClasspathAndInputFiles = indexedClasspath.withParent(indexedInputs);
    classpathReader = new ClassReaderFactory(indexedClasspathAndInputFiles, rewriter);
    ClassLoader bootclassloader =
        new HeaderClassLoader(indexedBootclasspath, rewriter, new ThrowingClassLoader());
    classLoader = new HeaderClassLoader(indexedClasspathAndInputFiles, rewriter, bootclassloader);
  }

  @After
  public void teardown() throws IOException {
    closer.close();
  }

  private static IndexedInputs toIndexedInputs(Closer closer, String stringPathList)
      throws IOException {
    final List<Path> pathList = readPathListFromString(stringPathList);
    return new IndexedInputs(Desugar.toRegisteredInputFileProvider(closer, pathList));
  }

  private static List<Path> readPathListFromString(String pathList) {
    return Arrays.stream(checkNotNull(pathList).split(File.pathSeparator))
        .map(Paths::get)
        .collect(ImmutableList.toImmutableList());
  }

  private byte[] desugar(String classname) {
    ClassReader reader = classpathReader.readIfKnown(classname);
    return desugar(reader);
  }

  private byte[] desugar(ClassReader reader) {
    ClassWriter writer = new ClassWriter(ClassWriter.COMPUTE_MAXS);
    DefaultMethodClassFixer fixer =
        new DefaultMethodClassFixer(
            writer,
            /*useGeneratedBaseClasses=*/ false,
            classpathReader,
            DependencyCollector.NoWriteCollectors.FAIL_ON_MISSING,
            /*coreLibrarySupport=*/ null,
            bootclassPath,
            classLoader);
    reader.accept(fixer, 0);
    return writer.toByteArray();
  }

  private byte[] desugar(byte[] classContent) {
    ClassReader reader = new ClassReader(classContent);
    return desugar(reader);
  }

  @Test
  public void testDesugaringDirectImplementation() {
    byte[] desugaredClass =
        desugar(
            ("com.google.devtools.build.android.desugar.testdata.java8."
                    + "DefaultInterfaceMethodWithStaticInitializer$TestInterfaceSetOne$C")
                .replace('.', '/'));
    checkClinitForDefaultInterfaceMethodWithStaticInitializerTestInterfaceSetOneC(desugaredClass);

    byte[] desugaredClassAgain = desugar(desugaredClass);
    checkClinitForDefaultInterfaceMethodWithStaticInitializerTestInterfaceSetOneC(
        desugaredClassAgain);

    desugaredClassAgain = desugar(desugaredClassAgain);
    checkClinitForDefaultInterfaceMethodWithStaticInitializerTestInterfaceSetOneC(
        desugar(desugaredClassAgain));
  }

  private void checkClinitForDefaultInterfaceMethodWithStaticInitializerTestInterfaceSetOneC(
      byte[] classContent) {
    ClassReader reader = new ClassReader(classContent);
    reader.accept(
        new ClassVisitor(Opcodes.ASM8) {

          class ClinitMethod extends MethodNode {

            public ClinitMethod(
                int access, String name, String desc, String signature, String[] exceptions) {
              super(Opcodes.ASM8, access, name, desc, signature, exceptions);
            }
          }

          private ClinitMethod clinit;

          @Override
          public MethodVisitor visitMethod(
              int access, String name, String desc, String signature, String[] exceptions) {
            if ("<clinit>".equals(name)) {
              assertThat(clinit).isNull();
              clinit = new ClinitMethod(access, name, desc, signature, exceptions);
              return clinit;
            }
            return super.visitMethod(access, name, desc, signature, exceptions);
          }

          @Override
          public void visitEnd() {
            assertThat(clinit).isNotNull();
            assertThat(clinit.instructions.size()).isEqualTo(3);
            AbstractInsnNode instruction = clinit.instructions.getFirst();
            {
              assertThat(instruction).isInstanceOf(MethodInsnNode.class);
              MethodInsnNode field = (MethodInsnNode) instruction;
              assertThat(field.owner)
                  .isEqualTo(
                      "com/google/devtools/build/android/desugar/testdata/java8/"
                          + "DefaultInterfaceMethodWithStaticInitializer"
                          + "$TestInterfaceSetOne$I1$$CC");
              assertThat(field.name)
                  .isEqualTo(InterfaceDesugaring.COMPANION_METHOD_TO_TRIGGER_INTERFACE_CLINIT_NAME);
              assertThat(field.desc)
                  .isEqualTo(InterfaceDesugaring.COMPANION_METHOD_TO_TRIGGER_INTERFACE_CLINIT_DESC);
            }
            {
              instruction = instruction.getNext();
              assertThat(instruction).isInstanceOf(MethodInsnNode.class);
              MethodInsnNode field = (MethodInsnNode) instruction;
              assertThat(field.owner)
                  .isEqualTo(
                      "com/google/devtools/build/android/desugar/testdata/java8/"
                          + "DefaultInterfaceMethodWithStaticInitializer"
                          + "$TestInterfaceSetOne$I2$$CC");
              assertThat(field.name)
                  .isEqualTo(InterfaceDesugaring.COMPANION_METHOD_TO_TRIGGER_INTERFACE_CLINIT_NAME);
              assertThat(field.desc)
                  .isEqualTo(InterfaceDesugaring.COMPANION_METHOD_TO_TRIGGER_INTERFACE_CLINIT_DESC);
            }
            {
              instruction = instruction.getNext();
              assertThat(instruction).isInstanceOf(InsnNode.class);
              assertThat(instruction.getOpcode()).isEqualTo(Opcodes.RETURN);
            }
          }
        },
        0);
  }

  @Test
  public void testInterfaceComparator() {
    assertThat(INSTANCE.compare(Runnable.class, Runnable.class)).isEqualTo(0);
    assertThat(INSTANCE.compare(Runnable.class, MyRunnable1.class)).isEqualTo(1);
    assertThat(INSTANCE.compare(MyRunnable2.class, Runnable.class)).isEqualTo(-1);
    assertThat(INSTANCE.compare(MyRunnable3.class, Runnable.class)).isEqualTo(-1);
    assertThat(INSTANCE.compare(MyRunnable1.class, MyRunnable3.class)).isEqualTo(1);
    assertThat(INSTANCE.compare(MyRunnable3.class, MyRunnable2.class)).isEqualTo(-1);
    assertThat(INSTANCE.compare(MyRunnable2.class, MyRunnable1.class)).isGreaterThan(0);
    assertThat(INSTANCE.compare(Runnable.class, Serializable.class)).isGreaterThan(0);
    assertThat(INSTANCE.compare(Serializable.class, Runnable.class)).isLessThan(0);

    TreeSet<Class<?>> orderedSet = new TreeSet<>(INSTANCE);
    orderedSet.add(Serializable.class);
    orderedSet.add(Runnable.class);
    orderedSet.add(MyRunnable2.class);
    orderedSet.add(Callable.class);
    orderedSet.add(Serializable.class);
    orderedSet.add(MyRunnable1.class);
    orderedSet.add(MyRunnable3.class);
    assertThat(orderedSet)
        .containsExactly(
            MyRunnable3.class, // subtype before supertype(s)
            MyRunnable1.class,
            MyRunnable2.class,
            Serializable.class, // java... comes textually after com.google...
            Runnable.class,
            Callable.class)
        .inOrder();
  }

  private static interface MyRunnable1 extends Runnable {}

  private static interface MyRunnable2 extends Runnable {}

  private static interface MyRunnable3 extends MyRunnable1, MyRunnable2 {}
}
