// Copyright 2020 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.runner;

import com.google.common.hash.HashCode;
import java.net.URL;
import java.net.URLClassLoader;

/**
 * A custom classloader used by the persistent test runner.
 *
 * <p>Each classloader stores the combined hash code for the loaded jars.
 */
final class PersistentTestRunnerClassLoader extends URLClassLoader {

  private final HashCode checksum;
  private PersistentTestRunnerClassLoader child;

  public PersistentTestRunnerClassLoader(URL[] urls, ClassLoader parent, HashCode checksum) {
    super(urls, parent);
    this.checksum = checksum;
  }

  void setChild(PersistentTestRunnerClassLoader child) {
    this.child = child;
  }

  HashCode getChecksum() {
    return checksum;
  }

  /**
   * Loads the class with the specified name and resolves it if required.
   *
   * <p>If the classloader has a child: check if the class was already loaded by the child if the
   * current classloader did not succeed in loading the class.
   *
   * <p>If the classloader doesn't have a child: use the default class loading logic.
   */
  @Override
  public Class<?> loadClass(String name, boolean resolve) throws ClassNotFoundException {
    if (child == null) {
      return super.loadClass(name, resolve);
    }

    synchronized (this.getClassLoadingLock(name)) {
      Class<?> result;
      try {
        result = super.loadClass(name, resolve);
      } catch (ClassNotFoundException e) {
        result = child.findLoadedClass(name);
      }
      if (result == null) {
        throw new ClassNotFoundException("Could not find " + name);
      }
      return result;
    }
  }
}
