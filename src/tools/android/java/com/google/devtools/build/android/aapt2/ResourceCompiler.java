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
import com.android.resources.ResourceFolderType;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.devtools.build.android.AaptCommandBuilder;
import com.google.devtools.build.android.AndroidDataSerializer;
import com.google.devtools.build.android.DataResourceXml;
import com.google.devtools.build.android.FullyQualifiedName;
import com.google.devtools.build.android.FullyQualifiedName.Factory;
import com.google.devtools.build.android.FullyQualifiedName.Qualifiers;
import com.google.devtools.build.android.FullyQualifiedName.VirtualType;
import com.google.devtools.build.android.ResourceProcessorBusyBox;
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
import java.util.Arrays;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.logging.Logger;
import javax.xml.namespace.QName;
import javax.xml.stream.XMLEventReader;
import javax.xml.stream.XMLInputFactory;
import javax.xml.stream.XMLStreamConstants;
import javax.xml.stream.events.Attribute;
import javax.xml.stream.events.StartElement;
import javax.xml.stream.events.XMLEvent;

/** Invokes aapt2 to compile resources. */
public class ResourceCompiler {

  /** Types of compiled resources. */
  public enum CompiledType {
    NORMAL(null),
    GENERATED("generated"),
    DEFAULT("default");

    private final String prefix;

    CompiledType(String prefix) {
      this.prefix = prefix;
    }

    boolean prefixes(String filename) {
      return prefix != null && filename.startsWith(prefix);
    }

    public String asPrefix() {
      return prefix;
    }

    public String asComment() {
      return prefix;
    }

    public String prefix(String path) {
      return prefix + "/" + path;
    }
  }

  public static CompiledType getCompiledType(String fileName) {
    return Arrays.stream(CompiledType.values())
        .filter(t -> t.prefixes(fileName))
        .findFirst()
        .orElse(CompiledType.NORMAL);
  }

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

  // https://android-review.googlesource.com/c/platform/frameworks/base/+/1202901
  public static final boolean USE_VISIBILITY_FROM_AAPT2 =
      ResourceProcessorBusyBox.getProperty("use_visibility_from_aapt2");

  private final CompilingVisitor compilingVisitor;

  private static class CompileTask implements Callable<List<Path>> {

    private final Path file;
    private final Path compiledResourcesOut;
    private final Path aapt2;
    private final Revision buildToolsVersion;
    private final Optional<Path> generatedResourcesOut;

    private CompileTask(
        Path file,
        Path compiledResourcesOut,
        Path aapt2,
        Revision buildToolsVersion,
        Optional<Path> generatedResourcesOut) {
      this.file = file;
      this.compiledResourcesOut = compiledResourcesOut;
      this.aapt2 = aapt2;
      this.buildToolsVersion = buildToolsVersion;
      this.generatedResourcesOut = generatedResourcesOut;
    }

    @Override
    public List<Path> call() throws Exception {
      final String directoryName = file.getParent().getFileName().toString();
      final Qualifiers qualifiers = Qualifiers.parseFrom(directoryName);
      final ResourceFolderType resourceFolderType = qualifiers.asFolderType();
      if (resourceFolderType == null) {
        throw new CompileError(
            new IllegalArgumentException("Unexpected resource folder for file: " + file));
      }

      final String filename =
          interpolateAapt2Filename(resourceFolderType, file.getFileName().toString());
      final List<Path> results = new ArrayList<>();
      if (resourceFolderType.equals(ResourceFolderType.VALUES)
          || (resourceFolderType.equals(ResourceFolderType.RAW)
              && file.getFileName().toString().endsWith(".xml"))) {
        extractAttributes(directoryName, filename, results);
      }

      if (qualifiers.containDefaultLocale()
          && resourceFolderType.equals(ResourceFolderType.VALUES)) {
        compile(
            directoryName,
            filename,
            results,
            compiledResourcesOut.resolve(CompiledType.DEFAULT.asPrefix()),
            file,
            false);
        // aapt2 only generates pseudo locales for the default locale.
        generatedResourcesOut.ifPresent(
            out -> compile(directoryName, filename, results, out, file, true));
      } else {
        compile(directoryName, filename, results, compiledResourcesOut, file, false);
      }
      return results;
    }

    static String interpolateAapt2Filename(ResourceFolderType resourceFolderType, String filename) {
      // res/<not values>/foo.bar -> foo.bar
      if (!resourceFolderType.equals(ResourceFolderType.VALUES)) {
        return filename;
      }

      int periodIndex = filename.indexOf('.');

      // res/values/foo -> foo.arsc
      if (periodIndex == -1) {
        return filename + ".arsc";
      }

      // res/values/foo.bar.baz -> throw error.
      if (filename.lastIndexOf('.') != periodIndex) {
        throw new CompileError(
            new IllegalArgumentException(
                "aapt2 does not support compiling resource xmls with multiple periods in the "
                    + "filename: "
                    + filename));
      }

      // res/values/foo.xml -> foo.arsc
      return filename.substring(0, periodIndex) + ".arsc";
    }

    private void compile(
        String type,
        String filename,
        List<Path> results,
        Path compileOutRoot,
        Path file,
        boolean generatePseudoLocale) {
      try {
        Path destination = CompilingVisitor.destinationPath(file, compileOutRoot);
        final Path compiledResourcePath = destination.resolve(type + "_" + filename + ".flat");

        logger.fine(
            new AaptCommandBuilder(aapt2)
                .forBuildToolsVersion(buildToolsVersion)
                .forVariantType(VariantType.LIBRARY)
                .add("compile")
                .add("-v")
                .add("--legacy")
                .when(USE_VISIBILITY_FROM_AAPT2)
                .thenAdd("--preserve-visibility-of-styleables")
                .when(generatePseudoLocale)
                .thenAdd("--pseudo-localize")
                .add("-o", destination.toString())
                .add(file.toString())
                .execute("Compiling " + file));

        Preconditions.checkArgument(
            Files.exists(compiledResourcePath),
            "%s does not exists after aapt2 ran.",
            compiledResourcePath);
        results.add(compiledResourcePath);
      } catch (IOException e) {
        throw new CompileError(e);
      }
    }

    private void extractAttributes(String type, String filename, List<Path> results)
        throws Exception {
      XMLEventReader xmlEventReader = null;
      try {
        // aapt2 compile strips out namespaces and attributes from the resources tag.
        // Read them here separately and package them with the other flat files.
        xmlEventReader =
            XMLInputFactory.newInstance()
                .createXMLEventReader(new FileInputStream(file.toString()));

        // Iterate through the XML until we find a start element.
        // This should mimic xmlEventReader.nextTag() except that it also skips DTD elements.
        StartElement rootElement = null;
        while (xmlEventReader.hasNext()) {
          XMLEvent event = xmlEventReader.nextEvent();
          if (event.getEventType() != XMLStreamConstants.COMMENT
              && event.getEventType() != XMLStreamConstants.DTD
              && event.getEventType() != XMLStreamConstants.PROCESSING_INSTRUCTION
              && event.getEventType() != XMLStreamConstants.SPACE
              && event.getEventType() != XMLStreamConstants.START_DOCUMENT) {

            // If the event should not be skipped, try parsing it as a start element here.
            // If the event is not a start element, an appropriate exception will be thrown.
            rootElement = event.asStartElement();
            break;
          }
        }

        if (rootElement == null) {
          throw new Exception("No start element found in resource XML file: " + file.toString());
        }

        Iterator<Attribute> attributeIterator =
            XmlResourceValues.iterateAttributesFrom(rootElement);

        if (attributeIterator.hasNext()) {
          results.add(createAttributesProto(type, filename, attributeIterator));
        }
      } finally {
        if (xmlEventReader != null) {
          xmlEventReader.close();
        }
      }
    }

    private Path createAttributesProto(
        String type, String filename, Iterator<Attribute> attributeIterator) throws IOException {

      AndroidDataSerializer serializer = AndroidDataSerializer.create();
      final Path resourcesAttributesPath =
          CompilingVisitor.destinationPath(file, compiledResourcesOut)
              .resolve(type + "_" + filename + CompiledResources.ATTRIBUTES_FILE_EXTENSION);

      Preconditions.checkArgument(!Files.exists(resourcesAttributesPath),
          "%s was already created for another resource.", resourcesAttributesPath);

      while (attributeIterator.hasNext()) {
        Attribute attribute = attributeIterator.next();
        String namespaceUri = attribute.getName().getNamespaceURI();
        String localPart = attribute.getName().getLocalPart();
        String prefix = attribute.getName().getPrefix();
        QName qName = new QName(namespaceUri, localPart, prefix);

        Namespaces namespaces = Namespaces.from(qName);
        String attributeName = namespaceUri.isEmpty() ? localPart : prefix + ":" + localPart;

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
    private final Path compiledResourcesOut;
    private final Set<Path> pathToProcessed = new LinkedHashSet<>();
    private final Path aapt2;
    private final Revision buildToolsVersion;
    private final Optional<Path> generatedResourcesOut;

    public CompilingVisitor(
        ListeningExecutorService executorService,
        Path compiledResourcesOut,
        Path aapt2,
        Revision buildToolsVersion,
        Optional<Path> generatedResourcesOut) {
      this.executorService = executorService;
      this.compiledResourcesOut = compiledResourcesOut;
      this.aapt2 = aapt2;
      this.buildToolsVersion = buildToolsVersion;
      this.generatedResourcesOut = generatedResourcesOut;
    }

    @Override
    public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
      // Ignore directories and "hidden" files that start with .
      if (!Files.isDirectory(file) && !file.getFileName().toString().startsWith(".")) {
        pathToProcessed.add(file);
      }
      return super.visitFile(file, attrs);
    }

    public static Path destinationPath(Path file, Path compiledResourcesOut) {
      // Creates a relative output path based on the input path under the
      // compiledResources path.
      try {
        return Files.createDirectories(
            compiledResourcesOut.resolve(
                (file.isAbsolute() ? file.getRoot().relativize(file) : file)
                    .getParent()
                    .getParent()));
      } catch (IOException e) {
        throw new CompileError(e);
      }
    }

    List<Path> getCompiledArtifacts() {
      generatedResourcesOut.ifPresent(
          out -> {
            try {
              Files.createDirectories(out);
            } catch (IOException e) {
              throw new CompileError(e);
            }
          });

      List<ListenableFuture<List<Path>>> tasks = new ArrayList<>();
      for (Path uncompiled : pathToProcessed) {
        tasks.add(
            executorService.submit(
                new CompileTask(
                    uncompiled,
                    compiledResourcesOut,
                    aapt2,
                    buildToolsVersion,
                    generatedResourcesOut)));
      }

      ImmutableList.Builder<Path> compiled = ImmutableList.builder();
      ImmutableList.Builder<Path> generated = ImmutableList.builder();
      List<Throwable> compilationErrors = new ArrayList<>();
      for (ListenableFuture<List<Path>> task : tasks) {
        try {
          // Split the generated and non-generated resources into different collections.
          // This allows the generated files to be placed first in the compile order,
          // ensuring that the generated locale (en-XA and ar-XB) can be overwritten by
          // user provided versions for those locales, as aapt2 will take the last value for
          // a configuration when linking.
          task.get()
              .forEach(
                  path -> {
                    if (generatedResourcesOut.map(path::startsWith).orElse(false)) {
                      generated.add(path);
                    } else {
                      compiled.add(path);
                    }
                  });
        } catch (InterruptedException | ExecutionException e) {
          compilationErrors.add(e.getCause() != null ? e.getCause() : e);
        }
      }
      generated.addAll(compiled.build());
      if (compilationErrors.isEmpty()) {
        // ensure that the generated files are before the normal files.
        return generated.build();
      }
      throw CompileError.of(compilationErrors);
    }
  }

  /** Creates a new {@link ResourceCompiler}. */
  public static ResourceCompiler create(
      ListeningExecutorService executorService,
      Path compiledResources,
      Path aapt2,
      Revision buildToolsVersion,
      boolean generatePseudoLocale) {

    return new ResourceCompiler(
        new CompilingVisitor(
            executorService,
            compiledResources,
            aapt2,
            buildToolsVersion,
            generatePseudoLocale
                ? Optional.of(compiledResources.resolve(CompiledType.GENERATED.asPrefix()))
                : Optional.empty()));
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
