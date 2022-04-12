package com.google.devtools.build.android.desugar.testdata.java8;

import com.google.devtools.build.android.desugar.testdata.java8.InterfaceMethodWithParam.Foo;
import com.google.devtools.build.android.desugar.testdata.java8.InterfaceMethodWithParam.TyFoo;

/** A testing interface subject to the inspection its compiled bytecode. */
public interface InterfaceMethodParameters {

  @Foo
  static @TyFoo int simpleComputeStatic(int x, @Foo @TyFoo int y) {
    return x + y;
  }

  @Foo
  default @TyFoo int simpleComputeDefault(int x, @Foo @TyFoo int y) {
    return x + y;
  }
}
