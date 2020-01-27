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

package com.google.devtools.build.docgen;

import org.apache.velocity.app.VelocityEngine;
import org.apache.velocity.runtime.resource.loader.ClasspathResourceLoader;
import org.apache.velocity.runtime.resource.loader.JarResourceLoader;

/**
 * Utility class used for creating pages to be generated using VelcityEngine.
 */
final class TemplateEngine {
  private TemplateEngine() {}

  /**
   * Returns a new {@link Page} using the given .vm template file path. The template file
   * path must be the resource path for the .vm file in the JAR since the VelocityEngine
   * is configured to load .vm files from JAR resources.
   */
  public static Page newPage(String template) {
    VelocityEngine engine = new VelocityEngine();
    engine.setProperty("resource.loader", "classpath, jar");
    engine.setProperty("classpath.resource.loader.class", ClasspathResourceLoader.class.getName());
    engine.setProperty("jar.resource.loader.class", JarResourceLoader.class.getName());
    engine.setProperty("input.encoding", "UTF-8");
    engine.setProperty("output.encoding", "UTF-8");
    engine.setProperty("directive.set.null.allowed", true);
    engine.setProperty("parser.pool.size", 3);
    engine.setProperty("runtime.references.strict", true);
    return new Page(engine, template);
  }
}
