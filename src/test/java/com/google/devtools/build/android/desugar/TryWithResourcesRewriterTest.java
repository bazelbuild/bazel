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

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.android.desugar.runtime.ThrowableExtensionTestUtility.getStrategyClassName;
import static com.google.devtools.build.android.desugar.runtime.ThrowableExtensionTestUtility.getTwrStrategyClassNameSpecifiedInSystemProperty;
import static com.google.devtools.build.android.desugar.runtime.ThrowableExtensionTestUtility.isMimicStrategy;
import static com.google.devtools.build.android.desugar.runtime.ThrowableExtensionTestUtility.isNullStrategy;
import static com.google.devtools.build.android.desugar.runtime.ThrowableExtensionTestUtility.isReuseStrategy;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.fail;
import static org.objectweb.asm.ClassWriter.COMPUTE_MAXS;
import static org.objectweb.asm.Opcodes.ASM8;
import static org.objectweb.asm.Opcodes.INVOKESTATIC;
import static org.objectweb.asm.Opcodes.INVOKEVIRTUAL;

import com.google.devtools.build.android.desugar.io.BitFlags;
import com.google.devtools.build.android.desugar.runtime.ThrowableExtension;
import com.google.devtools.build.android.desugar.testdata.ClassUsingTryWithResources;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.objectweb.asm.ClassReader;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;

/** This is the unit test for {@link TryWithResourcesRewriter} */
@RunWith(JUnit4.class)
public class TryWithResourcesRewriterTest {

  private final DesugaringClassLoader classLoader =
      new DesugaringClassLoader(ClassUsingTryWithResources.class.getName());
  private Class<?> desugaredClass;

  @Before
  public void setup() {
    try {
      desugaredClass = classLoader.findClass(ClassUsingTryWithResources.class.getName());
    } catch (ClassNotFoundException e) {
      throw new AssertionError(e);
    }
  }

  @Test
  public void testMethodsAreDesugared() {
    // verify whether the desugared class is indeed desugared.
    DesugaredThrowableMethodCallCounter origCounter =
        countDesugaredThrowableMethodCalls(ClassUsingTryWithResources.class);
    DesugaredThrowableMethodCallCounter desugaredCounter =
        countDesugaredThrowableMethodCalls(classLoader.classContent, classLoader);
    /**
     * In java9, javac creates a helper method {@code $closeResource(Throwable, AutoCloseable)
     * to close resources. So, the following number 3 is highly dependant on the version of javac.
     */
    assertThat(hasAutoCloseable(classLoader.classContent)).isFalse();
    assertThat(classLoader.numOfTryWithResourcesInvoked.intValue()).isAtLeast(2);
    assertThat(classLoader.visitedExceptionTypes)
        .containsExactly(
            "java/lang/Exception", "java/lang/Throwable", "java/io/UnsupportedEncodingException");
    assertDesugaringBehavior(origCounter, desugaredCounter);
  }

  @Test
  public void testCheckSuppressedExceptionsReturningEmptySuppressedExceptions() {
    {
      Throwable[] suppressed = ClassUsingTryWithResources.checkSuppressedExceptions(false);
      assertThat(suppressed).isEmpty();
    }
    try {
      Throwable[] suppressed =
          (Throwable[])
              desugaredClass
                  .getMethod("checkSuppressedExceptions", boolean.class)
                  .invoke(null, Boolean.FALSE);
      assertThat(suppressed).isEmpty();
    } catch (Exception e) {
      e.printStackTrace();
      throw new AssertionError(e);
    }
  }

  @Test
  public void testPrintStackTraceOfCaughtException() {
    {
      String trace = ClassUsingTryWithResources.printStackTraceOfCaughtException();
      assertThat(trace.toLowerCase()).contains("suppressed");
    }
    try {
      String trace =
          (String) desugaredClass.getMethod("printStackTraceOfCaughtException").invoke(null);

      if (isMimicStrategy()) {
        assertThat(trace.toLowerCase()).contains("suppressed");
      } else if (isReuseStrategy()) {
        assertThat(trace.toLowerCase()).contains("suppressed");
      } else if (isNullStrategy()) {
        assertThat(trace.toLowerCase()).doesNotContain("suppressed");
      } else {
        fail("unexpected desugaring strategy " + ThrowableExtension.getStrategy());
      }
    } catch (Exception e) {
      e.printStackTrace();
      throw new AssertionError(e);
    }
  }

  @Test
  public void testCheckSuppressedExceptionReturningOneSuppressedException() {
    {
      Throwable[] suppressed = ClassUsingTryWithResources.checkSuppressedExceptions(true);
      assertThat(suppressed).hasLength(1);
    }
    try {
      Throwable[] suppressed =
          (Throwable[])
              desugaredClass
                  .getMethod("checkSuppressedExceptions", boolean.class)
                  .invoke(null, Boolean.TRUE);

      if (isMimicStrategy()) {
        assertThat(suppressed).hasLength(1);
      } else if (isReuseStrategy()) {
        assertThat(suppressed).hasLength(1);
      } else if (isNullStrategy()) {
        assertThat(suppressed).isEmpty();
      } else {
        fail("unexpected desugaring strategy " + ThrowableExtension.getStrategy());
      }
    } catch (Exception e) {
      e.printStackTrace();
      throw new AssertionError(e);
    }
  }

  @Test
  public void testSimpleTryWithResources() throws Throwable {
    {
      RuntimeException expected =
          assertThrows(
              RuntimeException.class, () -> ClassUsingTryWithResources.simpleTryWithResources());
      assertThat(expected.getClass()).isEqualTo(RuntimeException.class);
      assertThat(expected.getSuppressed()).hasLength(1);
      assertThat(expected.getSuppressed()[0].getClass()).isEqualTo(IOException.class);
    }

    try {
      InvocationTargetException e =
          assertThrows(
              InvocationTargetException.class,
              () -> desugaredClass.getMethod("simpleTryWithResources").invoke(null));
      throw e.getCause();
    } catch (RuntimeException expected) {
      String expectedStrategyName = getTwrStrategyClassNameSpecifiedInSystemProperty();
      assertThat(getStrategyClassName()).isEqualTo(expectedStrategyName);
      if (isMimicStrategy()) {
        assertThat(expected.getSuppressed()).isEmpty();
        assertThat(ThrowableExtension.getSuppressed(expected)).hasLength(1);
        assertThat(ThrowableExtension.getSuppressed(expected)[0].getClass())
            .isEqualTo(IOException.class);
      } else if (isReuseStrategy()) {
        assertThat(expected.getSuppressed()).hasLength(1);
        assertThat(expected.getSuppressed()[0].getClass()).isEqualTo(IOException.class);
        assertThat(ThrowableExtension.getSuppressed(expected)[0].getClass())
            .isEqualTo(IOException.class);
      } else if (isNullStrategy()) {
        assertThat(expected.getSuppressed()).isEmpty();
        assertThat(ThrowableExtension.getSuppressed(expected)).isEmpty();
      } else {
        fail("unexpected desugaring strategy " + ThrowableExtension.getStrategy());
      }
    }
  }

  private static void assertDesugaringBehavior(
      DesugaredThrowableMethodCallCounter orig, DesugaredThrowableMethodCallCounter desugared) {
    assertThat(desugared.countThrowableGetSuppressed()).isEqualTo(orig.countExtGetSuppressed());
    assertThat(desugared.countThrowableAddSuppressed()).isEqualTo(orig.countExtAddSuppressed());
    assertThat(desugared.countThrowablePrintStackTrace()).isEqualTo(orig.countExtPrintStackTrace());
    assertThat(desugared.countThrowablePrintStackTracePrintStream())
        .isEqualTo(orig.countExtPrintStackTracePrintStream());
    assertThat(desugared.countThrowablePrintStackTracePrintWriter())
        .isEqualTo(orig.countExtPrintStackTracePrintWriter());

    assertThat(orig.countThrowableGetSuppressed()).isEqualTo(desugared.countExtGetSuppressed());
    // $closeResource may be specialized into multiple versions.
    assertThat(orig.countThrowableAddSuppressed()).isAtMost(desugared.countExtAddSuppressed());
    assertThat(orig.countThrowablePrintStackTrace()).isEqualTo(desugared.countExtPrintStackTrace());
    assertThat(orig.countThrowablePrintStackTracePrintStream())
        .isEqualTo(desugared.countExtPrintStackTracePrintStream());
    assertThat(orig.countThrowablePrintStackTracePrintWriter())
        .isEqualTo(desugared.countExtPrintStackTracePrintWriter());

    if (orig.getSyntheticCloseResourceCount() > 0) {
      // Depending on the specific javac version, $closeResource(Throwable, AutoCloseable) may not
      // be there.
      assertThat(orig.getSyntheticCloseResourceCount()).isEqualTo(1);
      assertThat(desugared.getSyntheticCloseResourceCount()).isAtLeast(1);
    }
    assertThat(desugared.countThrowablePrintStackTracePrintStream()).isEqualTo(0);
    assertThat(desugared.countThrowablePrintStackTracePrintStream()).isEqualTo(0);
    assertThat(desugared.countThrowablePrintStackTracePrintWriter()).isEqualTo(0);
    assertThat(desugared.countThrowableAddSuppressed()).isEqualTo(0);
    assertThat(desugared.countThrowableGetSuppressed()).isEqualTo(0);
  }

  private static DesugaredThrowableMethodCallCounter countDesugaredThrowableMethodCalls(
      Class<?> klass) {
    try {
      ClassReader reader = new ClassReader(klass.getName());
      DesugaredThrowableMethodCallCounter counter =
          new DesugaredThrowableMethodCallCounter(klass.getClassLoader());
      reader.accept(counter, 0);
      return counter;
    } catch (IOException e) {
      e.printStackTrace();
      fail(e.toString());
      return null;
    }
  }

  private static DesugaredThrowableMethodCallCounter countDesugaredThrowableMethodCalls(
      byte[] content, ClassLoader loader) {
    ClassReader reader = new ClassReader(content);
    DesugaredThrowableMethodCallCounter counter = new DesugaredThrowableMethodCallCounter(loader);
    reader.accept(counter, 0);
    return counter;
  }

  /** Check whether java.lang.AutoCloseable is used as arguments of any method. */
  private static boolean hasAutoCloseable(byte[] classContent) {
    ClassReader reader = new ClassReader(classContent);
    final AtomicInteger counter = new AtomicInteger();
    ClassVisitor visitor =
        new ClassVisitor(Opcodes.ASM8) {
          @Override
          public MethodVisitor visitMethod(
              int access, String name, String desc, String signature, String[] exceptions) {
            for (Type argumentType : Type.getArgumentTypes(desc)) {
              if ("Ljava/lang/AutoCloseable;".equals(argumentType.getDescriptor())) {
                counter.incrementAndGet();
              }
            }
            return null;
          }
        };
    reader.accept(visitor, 0);
    return counter.get() > 0;
  }

  private static class DesugaredThrowableMethodCallCounter extends ClassVisitor {
    private final ClassLoader classLoader;
    private final Map<String, AtomicInteger> counterMap;
    private int syntheticCloseResourceCount;

    public DesugaredThrowableMethodCallCounter(ClassLoader loader) {
      super(ASM8);
      classLoader = loader;
      counterMap = new HashMap<>();
      TryWithResourcesRewriter.TARGET_METHODS
          .entries()
          .forEach(entry -> counterMap.put(entry.getKey() + entry.getValue(), new AtomicInteger()));
      TryWithResourcesRewriter.TARGET_METHODS
          .entries()
          .forEach(
              entry ->
                  counterMap.put(
                      entry.getKey()
                          + TryWithResourcesRewriter.METHOD_DESC_MAP.get(entry.getValue()),
                      new AtomicInteger()));
    }

    @Override
    public MethodVisitor visitMethod(
        int access, String name, String desc, String signature, String[] exceptions) {
      if (BitFlags.isSet(access, Opcodes.ACC_SYNTHETIC | Opcodes.ACC_STATIC)
          && name.equals("$closeResource")
          && Type.getArgumentTypes(desc).length == 2
          && Type.getArgumentTypes(desc)[0].getDescriptor().equals("Ljava/lang/Throwable;")) {
        ++syntheticCloseResourceCount;
      }
      return new InvokeCounter();
    }

    private class InvokeCounter extends MethodVisitor {

      public InvokeCounter() {
        super(ASM8);
      }

      private boolean isAssignableToThrowable(String owner) {
        try {
          Class<?> ownerClass = classLoader.loadClass(owner.replace('/', '.'));
          return Throwable.class.isAssignableFrom(ownerClass);
        } catch (ClassNotFoundException e) {
          throw new AssertionError(e);
        }
      }

      @Override
      public void visitMethodInsn(int opcode, String owner, String name, String desc, boolean itf) {
        String signature = name + desc;
        if ((opcode == INVOKEVIRTUAL && isAssignableToThrowable(owner))
            || (opcode == INVOKESTATIC
                && Type.getInternalName(ThrowableExtension.class).equals(owner))) {
          AtomicInteger counter = counterMap.get(signature);
          if (counter == null) {
            return;
          }
          counter.incrementAndGet();
        }
      }
    }

    public int getSyntheticCloseResourceCount() {
      return syntheticCloseResourceCount;
    }

    public int countThrowableAddSuppressed() {
      return counterMap.get("addSuppressed(Ljava/lang/Throwable;)V").get();
    }

    public int countThrowableGetSuppressed() {
      return counterMap.get("getSuppressed()[Ljava/lang/Throwable;").get();
    }

    public int countThrowablePrintStackTrace() {
      return counterMap.get("printStackTrace()V").get();
    }

    public int countThrowablePrintStackTracePrintStream() {
      return counterMap.get("printStackTrace(Ljava/io/PrintStream;)V").get();
    }

    public int countThrowablePrintStackTracePrintWriter() {
      return counterMap.get("printStackTrace(Ljava/io/PrintWriter;)V").get();
    }

    public int countExtAddSuppressed() {
      return counterMap.get("addSuppressed(Ljava/lang/Throwable;Ljava/lang/Throwable;)V").get();
    }

    public int countExtGetSuppressed() {
      return counterMap.get("getSuppressed(Ljava/lang/Throwable;)[Ljava/lang/Throwable;").get();
    }

    public int countExtPrintStackTrace() {
      return counterMap.get("printStackTrace(Ljava/lang/Throwable;)V").get();
    }

    public int countExtPrintStackTracePrintStream() {
      return counterMap.get("printStackTrace(Ljava/lang/Throwable;Ljava/io/PrintStream;)V").get();
    }

    public int countExtPrintStackTracePrintWriter() {
      return counterMap.get("printStackTrace(Ljava/lang/Throwable;Ljava/io/PrintWriter;)V").get();
    }
  }

  private static class DesugaringClassLoader extends ClassLoader {

    private final String targetedClassName;
    private Class<?> klass;
    private byte[] classContent;
    private final Set<String> visitedExceptionTypes = new HashSet<>();
    private final AtomicInteger numOfTryWithResourcesInvoked = new AtomicInteger();

    public DesugaringClassLoader(String targetedClassName) {
      super(DesugaringClassLoader.class.getClassLoader());
      this.targetedClassName = targetedClassName;
    }

    @Override
    protected Class<?> findClass(String name) throws ClassNotFoundException {
      if (name.equals(targetedClassName)) {
        if (klass != null) {
          return klass;
        }
        // desugar the class, and return the desugared one.
        classContent = desugarTryWithResources(name);
        klass = defineClass(name, classContent, 0, classContent.length);
        return klass;
      } else {
        return super.findClass(name);
      }
    }

    private byte[] desugarTryWithResources(String className) {
      try {
        ClassReader reader = new ClassReader(className);
        CloseResourceMethodScanner scanner = new CloseResourceMethodScanner();
        reader.accept(scanner, ClassReader.SKIP_DEBUG);
        ClassWriter writer = new ClassWriter(reader, COMPUTE_MAXS);
        TryWithResourcesRewriter rewriter =
            new TryWithResourcesRewriter(
                writer,
                TryWithResourcesRewriterTest.class.getClassLoader(),
                visitedExceptionTypes,
                numOfTryWithResourcesInvoked,
                scanner.hasCloseResourceMethod());
        reader.accept(rewriter, 0);
        return writer.toByteArray();
      } catch (IOException e) {
        fail(e.toString());
        return null; // suppress compiler error.
      }
    }
  }
}
