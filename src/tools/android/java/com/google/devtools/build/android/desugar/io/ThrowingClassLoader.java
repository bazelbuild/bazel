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
package com.google.devtools.build.android.desugar.io;

/** Class loader that throws whenever it can, for use the parent of a class loader hierarchy. */
public class ThrowingClassLoader extends ClassLoader {
  @Override
  protected Class<?> loadClass(String name, boolean resolve) throws ClassNotFoundException {
    if (name.startsWith("java.")) {
      // Use system class loader for java. classes, since ClassLoader.defineClass gets
      // grumpy when those don't come from the standard place.
      return super.loadClass(name, resolve);
    }
    throw new ClassNotFoundException();
  }
}
