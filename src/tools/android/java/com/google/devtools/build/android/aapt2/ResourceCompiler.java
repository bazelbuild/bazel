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

package com.google.devtools.build.android.aapt2;

import com.android.SdkConstants;
import com.android.builder.core.VariantType;
import com.android.repository.Revision;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableList.Builder;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.devtools.build.android.AaptCommandBuilder;
import com.google.devtools.build.android.AndroidDataSerializer;
import com.google.devtools.build.android.DataResourceXml;
import com.google.devtools.build.android.FullyQualifiedName;
import com.google.devtools.build.android.FullyQualifiedName.Factory;
import com.google.devtools.build.android.FullyQualifiedName.VirtualType;
import com.google.devtools.build.android.XmlResourceValues;
import com.google.devtools.build.android.xml.Namespaces;
import com.google.devtools.build.android.xml.ResourcesAttribute;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.logging.Logger;
import javax.xml.namespace.QName;
import javax.xml.stream.XMLEventReader;
import javax.xml.stream.XMLInputFactory;
import javax.xml.stream.events.Attribute;
import javax.xml.stream.events.StartElement;

/** Invokes aapt2 to compile resources. */
public class ResourceCompiler {
  static class CompileError extends Aapt2Exception {

    protected CompileError(Throwable e) {
      super(e);
    }

    private CompileError() {
      super();
    }

    public static CompileError of(List<Throwable> compilationErrors) {
      final CompileError compileError = new CompileError();
      compilationErrors.forEach(compileError::addSuppressed);
      return compileError;
    }
  }

  private static final Logger logger = Logger.getLogger(ResourceCompiler.class.getName());

  private final CompilingVisitor compilingVisitor;

  private static class CompileTask implements Callable<List<Path>> {

    private final Path file;
    private final Path compiledResourcesOut;
    private final Path aapt2;
    private final Revision buildToolsVersion;

    private CompileTask(
        Path file,
        Path compiledResourcesOut,
        Path aapt2,
        Revision buildToolsVersion) {
      this.file = file;
      this.compiledResourcesOut = compiledResourcesOut;
      this.aapt2 = aapt2;
      this.buildToolsVersion = buildToolsVersion;
    }

    @Override
    public List<Path> call() throws Exception {
      logger.fine(
          new AaptCommandBuilder(aapt2)
              .forBuildToolsVersion(buildToolsVersion)
              .forVariantType(VariantType.LIBRARY)
              .add("compile")
              .add("-v")
              .add("--legacy")
              .add("-o", compiledResourcesOut.toString())
              .add(file.toString())
              .execute("Compiling " + file));

      String type = file.getParent().getFileName().toString();
      String filename = file.getFileName().toString();

      List<Path> results = new ArrayList<>();
      if (type.startsWith("values")) {
        filename =
            (filename.indexOf('.') != -1 ? filename.substring(0, filename.indexOf('.')) : filename)
                + ".arsc";

        XMLEventReader xmlEventReader = null;
        try {
          // aapt2 compile strips out namespaces and attributes from the resources tag.
          // Read them here separately and package them with the other flat files.
          xmlEventReader =
              XMLInputFactory.newInstance()
                  .createXMLEventReader(new FileInputStream(file.toString()));

          StartElement rootElement = xmlEventReader.nextTag().asStartElement();
          Iterator<Attribute> attributeIterator =
              XmlResourceValues.iterateAttributesFrom(rootElement);

          if (attributeIterator.hasNext()) {
            results.add(
                createAttributesProto(type, filename, attributeIterator));
          }
        } finally {
          if (xmlEventReader != null) {
            xmlEventReader.close();
          }
        }
      }

      final Path compiledResourcePath =
          compiledResourcesOut.resolve(type + "_" + filename + ".flat");
      Preconditions.checkArgument(
          Files.exists(compiledResourcePath),
          "%s does not exists after aapt2 ran.",
          compiledResourcePath);
      results.add(compiledResourcePath);
      return results;
    }

    private Path createAttributesProto(
        String type,
        String filename,
        Iterator<Attribute> attributeIterator)
        throws IOException {

      AndroidDataSerializer serializer = AndroidDataSerializer.create();
      final Path resourcesAttributesPath =
          compiledResourcesOut.resolve(type + "_" + filename + ".attributes");

      while (attributeIterator.hasNext()) {
        Attribute attribute = attributeIterator.next();
        String namespaceUri = attribute.getName().getNamespaceURI();
        String localPart = attribute.getName().getLocalPart();
        String prefix = attribute.getName().getPrefix();
        QName qName = new QName(namespaceUri, localPart, prefix);

        Namespaces namespaces = Namespaces.from(qName);
        String attributeName =
            namespaceUri.isEmpty()
                ? localPart
                : prefix + ":" + localPart;

        final String[] dirNameAndQualifiers = type.split(SdkConstants.RES_QUALIFIER_SEP);
        Factory fqnFactory = Factory.fromDirectoryName(dirNameAndQualifiers);
        FullyQualifiedName fqn =
            fqnFactory.create(VirtualType.RESOURCES_ATTRIBUTE, qName.toString());
        ResourcesAttribute resourceAttribute =
            ResourcesAttribute.of(fqn, attributeName, attribute.getValue());
        DataResourceXml resource =
            DataResourceXml.createWithNamespaces(file, resourceAttribute, namespaces);

        serializer.queueForSerialization(fqn, resource);
      }

      serializer.flushTo(resourcesAttributesPath);
      return resourcesAttributesPath;
    }

    @Override
    public String toString() {
      return "ResourceCompiler.CompileTask(" + file + ")";
    }
  }

  private static class CompilingVisitor extends SimpleFileVisitor<Path> {

    private final ListeningExecutorService executorService;
    private final Path compiledResources;
    private final List<ListenableFuture<List<Path>>> tasks = new ArrayList<>();
    private final Path aapt2;
    private final Revision buildToolsVersion;

    public CompilingVisitor(
        ListeningExecutorService executorService,
        Path compiledResources,
        Path aapt2,
        Revision buildToolsVersion) {
      this.executorService = executorService;
      this.compiledResources = compiledResources;
      this.aapt2 = aapt2;
      this.buildToolsVersion = buildToolsVersion;
    }

    @Override
    public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
      // Ignore directories and "hidden" files that start with .
      if (!Files.isDirectory(file) && !file.getFileName().toString().startsWith(".")) {
        tasks.add(
            executorService.submit(
                new CompileTask(
                    file,
                    // Creates a relative output path based on the input path under the
                    // compiledResources path.
                    Files.createDirectories(
                        compiledResources.resolve(
                            (file.isAbsolute() ? file.getRoot().relativize(file) : file)
                                .getParent()
                                .getParent())),
                    aapt2,
                    buildToolsVersion)));
      }
      return super.visitFile(file, attrs);
    }

    List<Path> getCompiledArtifacts() throws InterruptedException, ExecutionException {
      Builder<Path> builder = ImmutableList.builder();
      List<Throwable> compilationErrors = new ArrayList<>();
      for (ListenableFuture<List<Path>> task : tasks) {
        try {
          builder.addAll(task.get());
        } catch (InterruptedException | ExecutionException e) {
          compilationErrors.add(Optional.ofNullable(e.getCause()).orElse(e));
        }
      }
      if (compilationErrors.isEmpty()) {
        return builder.build();
      }
      throw CompileError.of(compilationErrors);
    }
  }

  /** Creates a new {@link ResourceCompiler}. */
  public static ResourceCompiler create(
      ListeningExecutorService executorService,
      Path compiledResources,
      Path aapt2,
      Revision buildToolsVersion) {
    return new ResourceCompiler(
        new CompilingVisitor(executorService, compiledResources, aapt2, buildToolsVersion));
  }

  private ResourceCompiler(CompilingVisitor compilingVisitor) {
    this.compilingVisitor = compilingVisitor;
  }

  /** Adds a task to compile the directory using aapt2. */
  public void queueDirectoryForCompilation(Path resource) throws IOException {
    Files.walkFileTree(resource, compilingVisitor);
  }

  /** Returns all paths of the aapt2 compiled resources. */
  public List<Path> getCompiledArtifacts() throws InterruptedException, ExecutionException {
    return compilingVisitor.getCompiledArtifacts();
  }
}
