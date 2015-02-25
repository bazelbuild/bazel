// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.util;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.io.ByteStreams;

import java.io.IOException;
import java.io.InputStream;

/**
 * A little utility to load resources (property files) from jars or
 * the classpath. Recommended for longer texts that do not fit nicely into
 * a piece of Java code - e.g. a template for a lengthy email.
 */
public final class ResourceFileLoader {

  private ResourceFileLoader() {}

  /**
   * Loads a text resource that is located in a directory on the Java classpath that
   * corresponds to the package of <code>relativeToClass</code> using UTF8 encoding.
   * E.g.
   * <code>loadResource(Class.forName("com.google.foo.Foo", "bar.txt"))</code>
   * will look for <code>com/google/foo/bar.txt</code> in the classpath.
   */
  public static String loadResource(Class<?> relativeToClass, String resourceName)
      throws IOException {
    ClassLoader loader = relativeToClass.getClassLoader();
    // TODO(bazel-team): use relativeToClass.getPackage().getName().
    String className = relativeToClass.getName();
    String packageName = className.substring(0, className.lastIndexOf('.'));
    String path = packageName.replace('.', '/');
    String resource = path + '/' + resourceName;
    InputStream stream = loader.getResourceAsStream(resource);
    if (stream == null) {
      throw new IOException(resourceName + " not found.");
    }
    try {
      return new String(ByteStreams.toByteArray(stream), UTF_8);
    } finally {
      stream.close();
    }
  }
}
