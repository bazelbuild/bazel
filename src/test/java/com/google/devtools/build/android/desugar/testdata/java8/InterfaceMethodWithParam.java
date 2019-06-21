package com.google.devtools.build.android.desugar.testdata.java8;

import java.lang.annotation.Documented;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
import java.lang.reflect.Method;

/** Interface for testing the parameters of static and default methods. */
public interface InterfaceMethodWithParam {

  /** For testing the annotations on the parameters of interface static and default methods */
  @Documented
  @Retention(RetentionPolicy.RUNTIME)
  @Target({ElementType.PARAMETER, ElementType.METHOD})
  @interface Foo {
    String value() default "default-attr";
  }

  /** For testing the annotations on the parameters of interface static and default methods */
  @Documented
  @Retention(RetentionPolicy.RUNTIME)
  @Target({ElementType.TYPE_USE})
  @interface TyFoo {}

  static long simpleComputeStatic(int x) {
    return (long) x;
  }

  default long simpleComputeDefault(int x) {
    return (long) x;
  }

  /**
   * @param v4897b02fddeda3bb31bc15b3cad0f6febc61508 a cryptic parameter name for checking whether
   *     the name is available at run-time. The name is from $ echo "desugar-static" | sha1sum
   * @return The reflection representation of the method itself.
   */
  @Foo("custom-attr-value-1")
  @TyFoo
  static Method inspectCompanionMethodOfStaticMethod(
      @Foo @TyFoo String v4897b02fddeda3bb31bc15b3cad0f6febc61508) throws Exception {
    return TestHelper.getEnclosingRuntimeMethod(
        new Throwable(v4897b02fddeda3bb31bc15b3cad0f6febc61508));
  }

  /**
   * @param v12525d61e4b10b3e27bc280dd61e56728e3e8c27 A cryptic parameter name for checking whether
   *     the name is available at run-time. The name is from $ echo "desugar-default" | sha1sum
   * @return The reflection representation of the method itself.
   */
  @Foo("custom-attr-value-2")
  @TyFoo
  default Method inspectCompanionMethodOfDefaultMethod(
      @Foo @TyFoo String v12525d61e4b10b3e27bc280dd61e56728e3e8c27) throws Exception {
    return TestHelper.getEnclosingRuntimeMethod(
        new Throwable(v12525d61e4b10b3e27bc280dd61e56728e3e8c27));
  }

  /** A concrete class that implements an interface with static methods and default methods. */
  final class Concrete implements InterfaceMethodWithParam {}

  /** Test cases for the invocations of interface static methods and default methods. */
  final class MethodInvocations {

    public static long simpleComputeStatic(int x) {
      return InterfaceMethodWithParam.simpleComputeStatic(x);
    }

    public static long simpleComputeDefault(int x) {
      Concrete concrete = new Concrete();
      return concrete.simpleComputeDefault(x);
    }

    public static Method inspectDesugaredDefaultMethod() throws Exception {
      return InterfaceMethodWithParam.class.getDeclaredMethod(
          "inspectCompanionMethodOfDefaultMethod", String.class);
    }

    public static Method inspectCompanionOfStaticMethod() throws Exception {
      return InterfaceMethodWithParam.inspectCompanionMethodOfStaticMethod("random-input");
    }

    public static Method inspectCompanionMethodOfDefaultMethod() throws Exception {
      Concrete concrete = new Concrete();
      return concrete.inspectCompanionMethodOfDefaultMethod("random-input");
    }
  }

  /** A helper class that interface methods with parameters. */
  final class TestHelper {

    /** Returns the runtime-invoked method that encloses {@param enclosedThrowable}. */
    static Method getEnclosingRuntimeMethod(Throwable enclosedThrowable)
        throws ClassNotFoundException, NoSuchMethodException {
      StackTraceElement stackTraceElement = enclosedThrowable.getStackTrace()[0];
      String methodName = stackTraceElement.getMethodName();
      String className = stackTraceElement.getClassName();
      for (Method method : Class.forName(className).getDeclaredMethods()) {
        if (methodName.equals(method.getName())) {
          return method;
        }
      }
      throw new NoSuchMethodException();
    }
  }
}
