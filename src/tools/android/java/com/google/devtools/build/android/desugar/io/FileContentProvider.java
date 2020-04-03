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

import com.google.common.collect.ImmutableSet;
import com.google.common.io.Resources;
import com.google.devtools.build.android.desugar.langmodel.ClassName;
import java.io.IOError;
import java.io.IOException;
import java.io.InputStream;
import java.util.function.Predicate;
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

  public static FileContentProvider<? extends InputStream> fromResources(ClassName className) {
    return new FileContentProvider<>(
        className.classFilePathName(),
        () -> {
          try {
            return Resources.getResource(className.classFilePathName()).openStream();
          } catch (IOException e) {
            throw new IOError(e);
          }
        });
  }

  public String getBinaryPathName() {
    return binaryPathName;
  }

  @Override
  public S get() {
    return inputStreamProvider.get();
  }

  public boolean isClassFile() {
    return binaryPathName.endsWith(".class");
  }

  public final ImmutableSet<ClassName> findReferencedTypes(Predicate<ClassName> typeFilter) {
    checkState(
        typeFilter.test(ClassName.create(stripRequiredStringEnd(binaryPathName, ".class"))),
        "Expected the initial class itself to satisfy the type filter. Actual: (%s)",
        this);
    ImmutableSet.Builder<ClassName> collectedTypes = ImmutableSet.builder();
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
                  if (typeFilter.test(className)) {
                    collectedTypes.add(className);
                  }
                  return super.map(internalName);
                }
              }),
          /* parsingOptions= */ 0);
    } catch (IOException e) {
      throw new IOError(e);
    }
    return collectedTypes.build();
  }

  @Override
  public String toString() {
    return String.format("Binary Path: (%s)", binaryPathName);
  }

  private static String stripRequiredStringEnd(String text, String trailingText) {
    checkState(
        text.endsWith(trailingText), "Expected %s to end with %s to strip", text, trailingText);
    return text.substring(0, text.length() - trailingText.length());
  }
}
