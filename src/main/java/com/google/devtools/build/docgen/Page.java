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

import static java.nio.charset.StandardCharsets.UTF_8;

import java.io.File;
import java.io.IOException;
import java.io.StringWriter;
import java.io.Writer;
import java.nio.file.Files;
import org.apache.velocity.VelocityContext;
import org.apache.velocity.app.VelocityEngine;
import org.apache.velocity.exception.MethodInvocationException;
import org.apache.velocity.exception.ParseErrorException;
import org.apache.velocity.exception.ResourceNotFoundException;

/**
 * Class that represents a page to be generated using the {@link TemplateEngine}.
 */
class Page {
  private final VelocityEngine engine;
  private final VelocityContext context;
  private final String template;

  /**
   * Creates a new Page instance using the reference to the VelocityEngine and the .vm
   * template file path.
   */
  public Page(VelocityEngine engine, String template) {
    this.engine = engine;
    this.template = template;
    this.context = new VelocityContext();
  }

  /**
   * Sets a Velocity variable in the template with the given value.
   */
  public void add(String var, Object value) {
    context.put(var, value);
  }

  /**
   * Renders the template and writes the output to the given file.
   *
   * Strips all trailing whitespace before writing to file.
   */
  public void write(File outputFile) throws IOException {
    StringWriter stringWriter = new StringWriter();
    try {
      engine.mergeTemplate(template, "UTF-8", context, stringWriter);
    } catch (ResourceNotFoundException|ParseErrorException|MethodInvocationException e) {
      throw new IOException(e);
    }
    stringWriter.close();

    String[] lines = stringWriter.toString().split(System.getProperty("line.separator"));
    try (Writer fileWriter = Files.newBufferedWriter(outputFile.toPath(), UTF_8)) {
      for (String line : lines) {
        // Strip trailing whitespace then append newline before writing to file.
        fileWriter.write(line.replaceFirst("\\s+$", "") + "\n");
      }
    }
  }
}
