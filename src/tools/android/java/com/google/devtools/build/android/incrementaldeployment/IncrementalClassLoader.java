// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.android.incrementaldeployment;

import android.util.Log;
import dalvik.system.BaseDexClassLoader;
import java.io.File;
import java.lang.reflect.Field;
import java.util.List;

/**
 * A class loader that loads classes from any .dex file in a particular directory on the SD card.
 *
 * <p>Used to implement incremental deployment to Android phones.
 */
public class IncrementalClassLoader extends ClassLoader {
  private final DelegateClassLoader delegateClassLoader;

  public IncrementalClassLoader(ClassLoader original,
      String packageName, File codeCacheDir, String nativeLibDir, List<String> dexes) {
    super(original.getParent());

    // TODO(bazel-team): For some mysterious reason, we need to use two class loaders so that
    // everything works correctly. Investigate why that is the case so that the code can be
    // simplified.
    delegateClassLoader = createDelegateClassLoader(codeCacheDir, nativeLibDir, dexes, original);
  }

  @Override
  public Class<?> findClass(String className) throws ClassNotFoundException {
    return delegateClassLoader.findClass(className);
  }

  /**
   * A class loader whose only purpose is to make {@code findClass()} public.
   */
  private static class DelegateClassLoader extends BaseDexClassLoader {
    private DelegateClassLoader(
        String dexPath, File optimizedDirectory, String libraryPath, ClassLoader parent) {
      super(dexPath, optimizedDirectory, libraryPath, parent);
    }

    @Override
    public Class<?> findClass(String name) throws ClassNotFoundException {
      return super.findClass(name);
    }
  }

  private static DelegateClassLoader createDelegateClassLoader(
      File codeCacheDir, String nativeLibDir, List<String> dexes, ClassLoader original) {
    StringBuilder pathBuilder = new StringBuilder();
    boolean first = true;
    for (String dex : dexes) {
      if (first) {
        first = false;
      } else {
        pathBuilder.append(File.pathSeparator);
      }

      pathBuilder.append(dex);
    }

    Log.v("IncrementalClassLoader", "Incremental dex path is " + pathBuilder);
    Log.v("IncrementalClassLoader", "Native lib dir is " + nativeLibDir);
    return new DelegateClassLoader(pathBuilder.toString(), codeCacheDir,
        nativeLibDir, original);
  }

  private static void setParent(ClassLoader classLoader, ClassLoader newParent) {
    try {
      Field parent = ClassLoader.class.getDeclaredField("parent");
      parent.setAccessible(true);
      parent.set(classLoader, newParent);
    } catch (IllegalArgumentException | IllegalAccessException | NoSuchFieldException e) {
      throw new RuntimeException(e);
    }
  }

  public static void inject(
      ClassLoader classLoader, String packageName, File codeCacheDir,
      String nativeLibDir, List<String> dexes) {
    IncrementalClassLoader incrementalClassLoader =
        new IncrementalClassLoader(classLoader, packageName, codeCacheDir, nativeLibDir, dexes);
    setParent(classLoader, incrementalClassLoader);
  }
}
