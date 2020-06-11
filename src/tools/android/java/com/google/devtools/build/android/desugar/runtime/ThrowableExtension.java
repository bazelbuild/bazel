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

import java.io.Closeable;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.lang.ref.Reference;
import java.lang.ref.ReferenceQueue;
import java.lang.ref.WeakReference;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.List;
import java.util.Vector;
import java.util.concurrent.ConcurrentHashMap;

/**
 * This is an extension class for java.lang.Throwable. It emulates the methods
 * addSuppressed(Throwable) and getSuppressed(), so the language feature try-with-resources can be
 * used on Android devices whose API level is below 19.
 *
 * <p>Note that the Desugar should avoid desugaring this class.
 */
public final class ThrowableExtension {

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

  // Visible for testing.
  static final int API_LEVEL;

  static {
    AbstractDesugaringStrategy strategy;
    Integer apiLevel = null;
    try {
      apiLevel = readApiLevelFromBuildVersion();
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
          "An error has occurred when initializing the try-with-resources desuguring strategy. "
              + "The default strategy "
              + NullDesugaringStrategy.class.getName()
              + "will be used. The error is: ");
      e.printStackTrace(System.err);
      strategy = new NullDesugaringStrategy();
    }
    STRATEGY = strategy;
    API_LEVEL = apiLevel == null ? 1 : apiLevel.intValue();
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

  public static void closeResource(Throwable throwable, Object resource) throws Throwable {
    if (resource == null) {
      return;
    }
    try {
      if (API_LEVEL >= 19) {
        ((AutoCloseable) resource).close();
      } else {
        if (resource instanceof Closeable) {
          ((Closeable) resource).close();
        } else {
          try {
            Method method = resource.getClass().getMethod("close");
            method.invoke(resource);
          } catch (NoSuchMethodException | SecurityException e) {
            throw new AssertionError(resource.getClass() + " does not have a close() method.", e);
          } catch (IllegalAccessException
              | IllegalArgumentException
              | ExceptionInInitializerError e) {
            throw new AssertionError("Fail to call close() on " + resource.getClass(), e);
          } catch (InvocationTargetException e) {
            // Exception occurs during the invocation to the close method. The cause is the real
            // exception.
            Throwable cause = e.getCause();
            throw cause;
          }
        }
      }
    } catch (Throwable e) {
      if (throwable != null) {
        addSuppressed(throwable, e);
        throw throwable;
      } else {
        throw e;
      }
    }
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
  static final class ReuseDesugaringStrategy extends AbstractDesugaringStrategy {

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
  static final class MimicDesugaringStrategy extends AbstractDesugaringStrategy {

    static final String SUPPRESSED_PREFIX = "Suppressed: ";
    private final ConcurrentWeakIdentityHashMap map = new ConcurrentWeakIdentityHashMap();

    /**
     * Suppress an exception. If the exception to be suppressed is {@receiver} or {@null}, an
     * exception will be thrown.
     */
    @Override
    public void addSuppressed(Throwable receiver, Throwable suppressed) {
      if (suppressed == receiver) {
        throw new IllegalArgumentException("Self suppression is not allowed.", suppressed);
      }
      if (suppressed == null) {
        throw new NullPointerException("The suppressed exception cannot be null.");
      }
      // The returned list is a synchrnozed list.
      map.get(receiver, /*createOnAbsence=*/ true).add(suppressed);
    }

    @Override
    public Throwable[] getSuppressed(Throwable receiver) {
      List<Throwable> list = map.get(receiver, /*createOnAbsence=*/ false);
      if (list == null || list.isEmpty()) {
        return EMPTY_THROWABLE_ARRAY;
      }
      return list.toArray(EMPTY_THROWABLE_ARRAY);
    }

    /**
     * Print the stack trace for the parameter {@code receiver}. Note that it is deliberate to NOT
     * reuse the implementation {@code MimicDesugaringStrategy.printStackTrace(Throwable,
     * PrintStream)}, because we are not sure whether the developer prints the stack trace to a
     * different stream other than System.err. Therefore, it is a caveat that the stack traces of
     * {@code receiver} and its suppressed exceptions are printed in two different streams.
     */
    @Override
    public void printStackTrace(Throwable receiver) {
      receiver.printStackTrace();
      List<Throwable> suppressedList = map.get(receiver, /*createOnAbsence=*/ false);
      if (suppressedList == null) {
        return;
      }
      synchronized (suppressedList) {
        for (Throwable suppressed : suppressedList) {
          System.err.print(SUPPRESSED_PREFIX);
          suppressed.printStackTrace();
        }
      }
    }

    @Override
    public void printStackTrace(Throwable receiver, PrintStream stream) {
      receiver.printStackTrace(stream);
      List<Throwable> suppressedList = map.get(receiver, /*createOnAbsence=*/ false);
      if (suppressedList == null) {
        return;
      }
      synchronized (suppressedList) {
        for (Throwable suppressed : suppressedList) {
          stream.print(SUPPRESSED_PREFIX);
          suppressed.printStackTrace(stream);
        }
      }
    }

    @Override
    public void printStackTrace(Throwable receiver, PrintWriter writer) {
      receiver.printStackTrace(writer);
      List<Throwable> suppressedList = map.get(receiver, /*createOnAbsence=*/ false);
      if (suppressedList == null) {
        return;
      }
      synchronized (suppressedList) {
        for (Throwable suppressed : suppressedList) {
          writer.print(SUPPRESSED_PREFIX);
          suppressed.printStackTrace(writer);
        }
      }
    }
  }

  /** A hash map, that is concurrent, weak-key, and identity-hashing. */
  static final class ConcurrentWeakIdentityHashMap {

    private final ConcurrentHashMap<WeakKey, List<Throwable>> map =
        new ConcurrentHashMap<>(16, 0.75f, 10);
    private final ReferenceQueue<Throwable> referenceQueue = new ReferenceQueue<>();

    /**
     * @param throwable, the key to retrieve or create associated list.
     * @param createOnAbsence {@code true} to create a new list if there is no value for the key.
     * @return the associated value with the given {@code throwable}. If {@code createOnAbsence} is
     *     {@code true}, the returned value will be non-null. Otherwise, it can be {@code null}
     */
    public List<Throwable> get(Throwable throwable, boolean createOnAbsence) {
      deleteEmptyKeys();
      WeakKey keyForQuery = new WeakKey(throwable, null);
      List<Throwable> list = map.get(keyForQuery);
      if (!createOnAbsence) {
        return list;
      }
      if (list != null) {
        return list;
      }
      List<Throwable> newValue = new Vector<>(2);
      list = map.putIfAbsent(new WeakKey(throwable, referenceQueue), newValue);
      return list == null ? newValue : list;
    }

    /** For testing-purpose */
    int size() {
      return map.size();
    }

    void deleteEmptyKeys() {
      // The ReferenceQueue.poll() is thread-safe.
      for (Reference<?> key = referenceQueue.poll(); key != null; key = referenceQueue.poll()) {
        map.remove(key);
      }
    }

    private static final class WeakKey extends WeakReference<Throwable> {

      /**
       * The hash code is used later to retrieve the entry, of which the key is the current weak
       * key. If the referent is marked for garbage collection and is set to null, we are still able
       * to locate the entry.
       */
      private final int hash;

      public WeakKey(Throwable referent, ReferenceQueue<Throwable> q) {
        super(referent, q);
        if (referent == null) {
          throw new NullPointerException("The referent cannot be null");
        }
        hash = System.identityHashCode(referent);
      }

      @Override
      public int hashCode() {
        return hash;
      }

      @Override
      public boolean equals(Object obj) {
        if (obj == null || obj.getClass() != getClass()) {
          return false;
        }
        if (this == obj) {
          return true;
        }
        WeakKey other = (WeakKey) obj;
        // Note that, after the referent is garbage collected, then the referent will be null.
        // And the equality test still holds.
        return this.hash == other.hash && this.get() == other.get();
      }
    }
  }

  /** This strategy ignores all suppressed exceptions, which is how retrolambda does. */
  static final class NullDesugaringStrategy extends AbstractDesugaringStrategy {

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
