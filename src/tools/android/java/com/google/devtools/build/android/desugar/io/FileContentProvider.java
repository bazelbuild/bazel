// Copyright 2019 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkState;
import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.ImmutableListMultimap.toImmutableListMultimap;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.common.collect.Iterables;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.android.desugar.langmodel.ClassName;
import java.io.ByteArrayInputStream;
import java.io.IOError;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.util.Collection;
import java.util.Objects;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import javax.inject.Provider;
import org.objectweb.asm.ClassReader;
import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.commons.ClassRemapper;
import org.objectweb.asm.commons.Remapper;

/**
 * A provider of an input stream with a file path label. The struct can be used to index byte code
 * files in a jar file, and serve as the reading source for the ASM library's class reader {@link
 * org.objectweb.asm.ClassReader}.
 */
public final class FileContentProvider<S extends InputStream> implements Provider<S> {

  private final String binaryPathName;
  private final Provider<S> inputStreamProvider;

  public FileContentProvider(String inArchiveBinaryPathName, Provider<S> inputStreamProvider) {
    checkState(
        !inArchiveBinaryPathName.startsWith("/"),
        "Expect inArchiveBinaryPathName is relative: (%s)",
        inArchiveBinaryPathName);
    this.binaryPathName = inArchiveBinaryPathName;
    this.inputStreamProvider = inputStreamProvider;
  }

  public static FileContentProvider<InputStream> fromBytes(
      String inArchiveBinaryPathName, byte[] bytes) {
    return new FileContentProvider<>(
        inArchiveBinaryPathName, () -> new ByteArrayInputStream(bytes));
  }

  public static FileContentProvider<InputStream> fromJarItem(JarItem jarItem) {
    return new FileContentProvider<>(jarItem.jarEntry().getName(), jarItem::getInputStream);
  }

  public static ImmutableListMultimap<String, FileContentProvider<InputStream>> fromJars(
      Collection<Path> jars, Predicate<JarItem> jarItemFilter) {
    return jars.stream()
        .map(Path::toFile)
        .map(JarItem::newJarFile)
        .flatMap(JarItem::jarItemStream)
        .filter(jarItemFilter)
        .map(FileContentProvider::fromJarItem)
        .collect(
            toImmutableListMultimap(FileContentProvider::getBinaryPathName, content -> content));
  }

  public static ImmutableList<FileContentProvider<InputStream>> fromJarsWithFiFoResolution(
      Collection<Path> jars, Predicate<JarItem> jarItemFilter) {
    ImmutableListMultimap<String, FileContentProvider<InputStream>> raw =
        fromJars(jars, jarItemFilter);
    ImmutableList<FileContentProvider<InputStream>> contents =
        raw.asMap().values().stream()
            .map(values -> Iterables.getFirst(values, null))
            .filter(Objects::nonNull)
            .collect(toImmutableList());
    return contents;
  }

  public String getBinaryPathName() {
    return binaryPathName;
  }

  @Override
  public S get() {
    return inputStreamProvider.get();
  }

  public boolean isClassFile() {
    return binaryPathName.endsWith(".class") && !binaryPathName.startsWith("META-INF/");
  }

  public final ImmutableSetMultimap<Predicate<ClassName>, ClassName> findReferencedTypes(
      ImmutableSet<Predicate<ClassName>> typeFilters) {
    ImmutableSetMultimap.Builder<ClassName, Predicate<ClassName>> collectedTypes =
        ImmutableSetMultimap.builder();
    // Takes an advantage of hit-all-referenced-types ASM Remapper to perform type collection.
    try (S inputStream = get()) {
      ClassReader cr = new ClassReader(inputStream);
      cr.accept(
          new ClassRemapper(
              new ClassWriter(ClassWriter.COMPUTE_FRAMES),
              new Remapper() {
                @Override
                public String map(String internalName) {
                  ClassName className = ClassName.create(internalName);
                  collectedTypes.putAll(
                      className,
                      typeFilters.stream()
                          .filter(className::acceptTypeFilter)
                          .collect(Collectors.toList()));
                  return super.map(internalName);
                }
              }),
          /* parsingOptions= */ 0);
    } catch (IOException e) {
      throw new IOError(e);
    }
    return collectedTypes.build().inverse();
  }

  public final ImmutableSet<ClassName> findReferencedTypes(Predicate<ClassName> typeFilter) {
    return findReferencedTypes(ImmutableSet.of(typeFilter)).get(typeFilter);
  }

  public void sink(OutputFileProvider outputFileProvider) {
    try (S input = get()) {
      outputFileProvider.write(getBinaryPathName(), ByteStreams.toByteArray(input));
    } catch (IOException e) {
      throw new IOError(e);
    }
  }

  @Override
  public String toString() {
    return String.format("Binary Path: (%s)", binaryPathName);
  }
}
