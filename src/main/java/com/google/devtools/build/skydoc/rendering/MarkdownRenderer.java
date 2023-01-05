// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.skydoc.rendering;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.AspectInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.ModuleInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.ProviderInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.RuleInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.StarlarkFunctionInfo;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.io.StringWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.apache.velocity.VelocityContext;
import org.apache.velocity.app.VelocityEngine;
import org.apache.velocity.exception.MethodInvocationException;
import org.apache.velocity.exception.ParseErrorException;
import org.apache.velocity.exception.ResourceNotFoundException;
import org.apache.velocity.runtime.resource.loader.ClasspathResourceLoader;
import org.apache.velocity.runtime.resource.loader.JarResourceLoader;

/** Produces skydoc output in markdown form. */
public class MarkdownRenderer {
  // TODO(kendalllane): Refactor MarkdownRenderer to take in something other than filepaths.
  private final String headerTemplateFilename;
  private final String ruleTemplateFilename;
  private final String providerTemplateFilename;
  private final String functionTemplateFilename;
  private final String aspectTemplateFilename;

  private final VelocityEngine velocityEngine;

  public MarkdownRenderer(
      String headerTemplate,
      String ruleTemplate,
      String providerTemplate,
      String functionTemplate,
      String aspectTemplate) {
    this.headerTemplateFilename = headerTemplate;
    this.ruleTemplateFilename = ruleTemplate;
    this.providerTemplateFilename = providerTemplate;
    this.functionTemplateFilename = functionTemplate;
    this.aspectTemplateFilename = aspectTemplate;

    this.velocityEngine = new VelocityEngine();
    velocityEngine.setProperty("resource.loader", "classpath, jar");
    velocityEngine.setProperty(
        "classpath.resource.loader.class", ClasspathResourceLoader.class.getName());
    velocityEngine.setProperty("jar.resource.loader.class", JarResourceLoader.class.getName());
    velocityEngine.setProperty("input.encoding", "UTF-8");
    velocityEngine.setProperty("output.encoding", "UTF-8");
    velocityEngine.setProperty("runtime.references.strict", true);

    // Ensure formatting is the same on Velocity 1.7 and 2.x.
    velocityEngine.setProperty("parser.space_gobbling", "bc");
  }

  /**
   * Returns a markdown header string that should appear at the top of Stardoc's output, providing a
   * summary for the input Starlark module.
   */
  public String renderMarkdownHeader(ModuleInfo moduleInfo) throws IOException {
    VelocityContext context = new VelocityContext();
    context.put("util", new MarkdownUtil());
    context.put("moduleDocstring", moduleInfo.getModuleDocstring());

    StringWriter stringWriter = new StringWriter();
    Reader reader = readerFromPath(headerTemplateFilename);
    try {
      velocityEngine.evaluate(context, stringWriter, headerTemplateFilename, reader);
    } catch (ResourceNotFoundException | ParseErrorException | MethodInvocationException e) {
      throw new IOException(e);
    }
    return stringWriter.toString();
  }

  /**
   * Returns a markdown rendering of rule documentation for the given rule information object with
   * the given rule name.
   */
  public String render(String ruleName, RuleInfo ruleInfo) throws IOException {
    VelocityContext context = new VelocityContext();
    context.put("util", new MarkdownUtil());
    context.put("ruleName", ruleName);
    context.put("ruleInfo", ruleInfo);

    StringWriter stringWriter = new StringWriter();
    Reader reader = readerFromPath(ruleTemplateFilename);
    try {
      velocityEngine.evaluate(context, stringWriter, ruleTemplateFilename, reader);
    } catch (ResourceNotFoundException | ParseErrorException | MethodInvocationException e) {
      throw new IOException(e);
    }
    return stringWriter.toString();
  }

  /**
   * Returns a markdown rendering of provider documentation for the given provider information
   * object with the given name.
   */
  public String render(String providerName, ProviderInfo providerInfo) throws IOException {
    VelocityContext context = new VelocityContext();
    context.put("util", new MarkdownUtil());
    context.put("providerName", providerName);
    context.put("providerInfo", providerInfo);

    StringWriter stringWriter = new StringWriter();
    Reader reader = readerFromPath(providerTemplateFilename);
    try {
      velocityEngine.evaluate(context, stringWriter, providerTemplateFilename, reader);
    } catch (ResourceNotFoundException | ParseErrorException | MethodInvocationException e) {
      throw new IOException(e);
    }
    return stringWriter.toString();
  }

  /**
   * Returns a markdown rendering of a user-defined function's documentation for the function info
   * object.
   */
  public String render(StarlarkFunctionInfo functionInfo) throws IOException {
    VelocityContext context = new VelocityContext();
    context.put("util", new MarkdownUtil());
    context.put("funcInfo", functionInfo);

    StringWriter stringWriter = new StringWriter();
    Reader reader = readerFromPath(functionTemplateFilename);
    try {
      velocityEngine.evaluate(context, stringWriter, functionTemplateFilename, reader);
    } catch (ResourceNotFoundException | ParseErrorException | MethodInvocationException e) {
      throw new IOException(e);
    }
    return stringWriter.toString();
  }

  /**
   * Returns a markdown rendering of aspect documentation for the given aspect information object
   * with the given aspect name.
   */
  public String render(String aspectName, AspectInfo aspectInfo) throws IOException {
    VelocityContext context = new VelocityContext();
    context.put("util", new MarkdownUtil());
    context.put("aspectName", aspectName);
    context.put("aspectInfo", aspectInfo);

    StringWriter stringWriter = new StringWriter();
    Reader reader = readerFromPath(aspectTemplateFilename);
    try {
      velocityEngine.evaluate(context, stringWriter, aspectTemplateFilename, reader);
    } catch (ResourceNotFoundException | ParseErrorException | MethodInvocationException e) {
      throw new IOException(e);
    }
    return stringWriter.toString();
  }
  /**
   * Returns a reader from the given path.
   *
   * @param filePath The given path, either a filesystem path or a java Resource
   */
  private static Reader readerFromPath(String filePath) throws IOException {
    if (Files.exists(Paths.get(filePath))) {
      Path path = Paths.get(filePath);
      return Files.newBufferedReader(path);
    }

    InputStream inputStream = MarkdownRenderer.class.getClassLoader().getResourceAsStream(filePath);
    if (inputStream == null) {
      throw new FileNotFoundException(filePath + " was not found as a resource.");
    }
    return new InputStreamReader(inputStream, UTF_8);
  }
}
