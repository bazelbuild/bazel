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
package com.google.devtools.build.android.desugar.runtime;

import java.io.PrintStream;
import java.io.PrintWriter;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.WeakHashMap;

/**
 * This is an extension class for java.lang.Throwable. It emulates the methods
 * addSuppressed(Throwable) and getSuppressed(), so the language feature try-with-resources can be
 * used on Android devices whose API level is below 19.
 *
 * <p>Note that the Desugar should avoid desugaring this class.
 */
public class ThrowableExtension {

  static final AbstractDesugaringStrategy STRATEGY;
  /**
   * This property allows users to change the desugared behavior of try-with-resources at runtime.
   * If its value is {@code true}, then {@link MimicDesugaringStrategy} will NOT be used, and {@link
   * NullDesugaringStrategy} is used instead.
   *
   * <p>Note: this property is ONLY used when the API level on the device is below 19.
   */
  public static final String SYSTEM_PROPERTY_TWR_DISABLE_MIMIC =
      "com.google.devtools.build.android.desugar.runtime.twr_disable_mimic";

  static {
    AbstractDesugaringStrategy strategy;
    try {
      Integer apiLevel = readApiLevelFromBuildVersion();
      if (apiLevel != null && apiLevel.intValue() >= 19) {
        strategy = new ReuseDesugaringStrategy();
      } else if (useMimicStrategy()) {
        strategy = new MimicDesugaringStrategy();
      } else {
        strategy = new NullDesugaringStrategy();
      }
    } catch (Throwable e) {
      // This catchall block is intentionally created to avoid anything unexpected, so that
      // the desugared app will continue running in case of exceptions.
      System.err.println(
          "An error has occured when initializing the try-with-resources desuguring strategy. "
              + "The default strategy "
              + NullDesugaringStrategy.class.getName()
              + "will be used. The error is: ");
      e.printStackTrace(System.err);
      strategy = new NullDesugaringStrategy();
    }
    STRATEGY = strategy;
  }

  public static AbstractDesugaringStrategy getStrategy() {
    return STRATEGY;
  }

  public static void addSuppressed(Throwable receiver, Throwable suppressed) {
    STRATEGY.addSuppressed(receiver, suppressed);
  }

  public static Throwable[] getSuppressed(Throwable receiver) {
    return STRATEGY.getSuppressed(receiver);
  }

  public static void printStackTrace(Throwable receiver) {
    STRATEGY.printStackTrace(receiver);
  }

  public static void printStackTrace(Throwable receiver, PrintWriter writer) {
    STRATEGY.printStackTrace(receiver, writer);
  }

  public static void printStackTrace(Throwable receiver, PrintStream stream) {
    STRATEGY.printStackTrace(receiver, stream);
  }

  private static boolean useMimicStrategy() {
    return !Boolean.getBoolean(SYSTEM_PROPERTY_TWR_DISABLE_MIMIC);
  }

  private static final String ANDROID_OS_BUILD_VERSION = "android.os.Build$VERSION";

  /**
   * Get the API level from {@link android.os.Build.VERSION} via reflection. The reason to use
   * relection is to avoid dependency on {@link android.os.Build.VERSION}. The advantage of doing
   * this is that even when you desugar a jar twice, and Desugars sees this class, there is no need
   * to put {@link android.os.Build.VERSION} on the classpath.
   *
   * <p>Another reason of doing this is that it does not introduce any additional dependency into
   * the input jars.
   *
   * @return The API level of the current device. If it is {@code null}, then it means there was an
   *     exception.
   */
  private static Integer readApiLevelFromBuildVersion() {
    try {
      Class<?> buildVersionClass = Class.forName(ANDROID_OS_BUILD_VERSION);
      Field field = buildVersionClass.getField("SDK_INT");
      return (Integer) field.get(null);
    } catch (Exception e) {
      System.err.println(
          "Failed to retrieve value from "
              + ANDROID_OS_BUILD_VERSION
              + ".SDK_INT due to the following exception.");
      e.printStackTrace(System.err);
      return null;
    }
  }

  /**
   * The strategy to desugar try-with-resources statements. A strategy handles the behavior of an
   * exception in terms of suppressed exceptions and stack trace printing.
   */
  abstract static class AbstractDesugaringStrategy {

    protected static final Throwable[] EMPTY_THROWABLE_ARRAY = new Throwable[0];

    public abstract void addSuppressed(Throwable receiver, Throwable suppressed);

    public abstract Throwable[] getSuppressed(Throwable receiver);

    public abstract void printStackTrace(Throwable receiver);

    public abstract void printStackTrace(Throwable receiver, PrintStream stream);

    public abstract void printStackTrace(Throwable receiver, PrintWriter writer);
  }

  /** This strategy just delegates all the method calls to java.lang.Throwable. */
  static class ReuseDesugaringStrategy extends AbstractDesugaringStrategy {

    @Override
    public void addSuppressed(Throwable receiver, Throwable suppressed) {
      receiver.addSuppressed(suppressed);
    }

    @Override
    public Throwable[] getSuppressed(Throwable receiver) {
      return receiver.getSuppressed();
    }

    @Override
    public void printStackTrace(Throwable receiver) {
      receiver.printStackTrace();
    }

    @Override
    public void printStackTrace(Throwable receiver, PrintStream stream) {
      receiver.printStackTrace(stream);
    }

    @Override
    public void printStackTrace(Throwable receiver, PrintWriter writer) {
      receiver.printStackTrace(writer);
    }
  }

  /** This strategy mimics the behavior of suppressed exceptions with a map. */
  static class MimicDesugaringStrategy extends AbstractDesugaringStrategy {

    public static final String SUPPRESSED_PREFIX = "Suppressed: ";
    private final WeakHashMap<Throwable, List<Throwable>> map = new WeakHashMap<>();

    /**
     * Suppress an exception. If the exception to be suppressed is {@receiver} or {@null}, an
     * exception will be thrown.
     *
     * @param receiver
     * @param suppressed
     */
    @Override
    public void addSuppressed(Throwable receiver, Throwable suppressed) {
      if (suppressed == receiver) {
        throw new IllegalArgumentException("Self suppression is not allowed.", suppressed);
      }
      if (suppressed == null) {
        throw new NullPointerException("The suppressed exception cannot be null.");
      }
      synchronized (this) {
        List<Throwable> list = map.get(receiver);
        if (list == null) {
          list = new ArrayList<>(1);
          map.put(receiver, list);
        }
        list.add(suppressed);
      }
    }

    @Override
    public synchronized Throwable[] getSuppressed(Throwable receiver) {
      List<Throwable> list = map.get(receiver);
      if (list == null || list.isEmpty()) {
        return EMPTY_THROWABLE_ARRAY;
      }
      return list.toArray(new Throwable[0]);
    }

    /**
     * Print the stack trace for the parameter {@code receiver}. Note that it is deliberate to NOT
     * reuse the implementation {@code MimicDesugaringStrategy.printStackTrace(Throwable,
     * PrintStream)}, because we are not sure whether the developer prints the stack trace to a
     * different stream other than System.err. Therefore, it is a caveat that the stack traces of
     * {@code receiver} and its suppressed exceptions are printed in two different streams.
     */
    @Override
    public synchronized void printStackTrace(Throwable receiver) {
      receiver.printStackTrace();
      for (Throwable suppressed : getSuppressed(receiver)) {
        System.err.print(SUPPRESSED_PREFIX);
        suppressed.printStackTrace();
      }
    }

    @Override
    public synchronized void printStackTrace(Throwable receiver, PrintStream stream) {
      receiver.printStackTrace(stream);
      for (Throwable suppressed : getSuppressed(receiver)) {
        stream.print(SUPPRESSED_PREFIX);
        suppressed.printStackTrace(stream);
      }
    }

    @Override
    public synchronized void printStackTrace(Throwable receiver, PrintWriter writer) {
      receiver.printStackTrace(writer);
      for (Throwable suppressed : getSuppressed(receiver)) {
        writer.print(SUPPRESSED_PREFIX);
        suppressed.printStackTrace(writer);
      }
    }
  }

  /** This strategy ignores all suppressed exceptions, which is how retrolambda does. */
  static class NullDesugaringStrategy extends AbstractDesugaringStrategy {

    @Override
    public void addSuppressed(Throwable receiver, Throwable suppressed) {
      // Do nothing. The suppressed exception is discarded.
    }

    @Override
    public Throwable[] getSuppressed(Throwable receiver) {
      return EMPTY_THROWABLE_ARRAY;
    }

    @Override
    public void printStackTrace(Throwable receiver) {
      receiver.printStackTrace();
    }

    @Override
    public void printStackTrace(Throwable receiver, PrintStream stream) {
      receiver.printStackTrace(stream);
    }

    @Override
    public void printStackTrace(Throwable receiver, PrintWriter writer) {
      receiver.printStackTrace(writer);
    }
  }
}
