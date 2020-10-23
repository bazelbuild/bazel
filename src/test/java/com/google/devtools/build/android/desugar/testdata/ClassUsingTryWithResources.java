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
package com.google.devtools.build.android.desugar.testdata;

import static java.nio.charset.StandardCharsets.UTF_8;

import java.io.ByteArrayOutputStream;
import java.io.Closeable;
import java.io.IOException;
import java.io.PrintStream;
import java.io.UnsupportedEncodingException;

/**
 * This is a test subject for {@link
 * com.google.devtools.build.android.desugar.TryWithResourcesRewriter}
 */
public class ClassUsingTryWithResources {

  /**
   * A simple resource, which always throws an exception when being closed.
   *
   * <p>Note that we need to implement java.io.Closeable instead of java.lang.AutoCloseable, because
   * AutoCloseable is not available below API 19
   *
   * <p>java9 will emit $closeResource(Throwable, AutoCloseable) for the following class.
   */
  public static class SimpleResource implements Closeable {

    public void call(boolean throwException) {
      if (throwException) {
        throw new RuntimeException("exception in call()");
      }
    }

    @Override
    public void close() throws IOException {
      throw new IOException("exception in close().");
    }
  }

  /** A resource inheriting the close() method from its parent. */
  public static class InheritanceResource extends SimpleResource {}

  /** This method will always throw {@link java.lang.Exception}. */
  public static void simpleTryWithResources() throws Exception {
    // Throwable.addSuppressed(Throwable) should be called in the following block.
    try (SimpleResource resource = new SimpleResource()) {
      resource.call(true);
    }
  }

  /** A simple resource type for testing try-with-resources with multiple resources. */
  public static class SimpleCloseable implements Closeable {

    /** This method tests a method with method reference as resources. */
    public void multipleTryWithResources() throws Exception {
      try (Closeable resource1 = new SimpleResource();
          Closeable resource2 = this::close;
          Closeable resource3 = new SimpleResource()) {}
    }

    @Override
    public void close() throws IOException {
      throw new IOException("exception in close().");
    }
  }

  public static void multipleTryWithResources() throws Exception {
    SimpleCloseable resource = new SimpleCloseable();
    resource.multipleTryWithResources();
  }

  /**
   * This method useds {@link InheritanceResource}, which inherits all methods from {@link
   * SimpleResource}.
   */
  public static void inheritanceTryWithResources() throws Exception {
    // Throwable.addSuppressed(Throwable) should be called in the following block.
    try (InheritanceResource resource = new InheritanceResource()) {
      resource.call(true);
    }
  }

  public static Throwable[] checkSuppressedExceptions(boolean throwException) {
    // Throwable.addSuppressed(Throwable) should be called in the following block.
    try (SimpleResource resource = new SimpleResource()) {
      resource.call(throwException);
    } catch (Exception e) {
      return e.getSuppressed(); // getSuppressed() is called.
    }
    return new Throwable[0];
  }

  public static String printStackTraceOfCaughtException() {
    try {
      simpleTryWithResources();
    } catch (Exception e) {
      PrintStream err = System.err;
      ByteArrayOutputStream stream = new ByteArrayOutputStream();
      try {
        System.setErr(new PrintStream(stream, true, "utf-8"));
        e.printStackTrace();
      } catch (UnsupportedEncodingException e1) {
        throw new AssertionError(e1);
      } finally {
        System.setErr(err);
      }
      return new String(stream.toByteArray(), UTF_8);
    }
    return "";
  }
}
