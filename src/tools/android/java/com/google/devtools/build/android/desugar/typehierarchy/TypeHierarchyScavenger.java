/*
 * Copyright 2020 The Bazel Authors. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.devtools.build.android.desugar.typehierarchy;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.android.desugar.io.FileContentProvider;
import com.google.devtools.build.android.desugar.typehierarchy.TypeHierarchy.TypeHierarchyBuilder;
import java.io.IOError;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.util.Collection;
import java.util.stream.Collectors;
import org.objectweb.asm.ClassReader;
import org.objectweb.asm.Opcodes;

/** The public APs that collects type hierarchy information from IO operations. */
public class TypeHierarchyScavenger {

  public static TypeHierarchy analyze(
      ImmutableList<FileContentProvider<InputStream>> inputProviders,
      boolean requireTypeResolutionComplete) {
    TypeHierarchyBuilder typeHierarchyBuilder =
        TypeHierarchy.builder().setRequireTypeResolutionComplete(requireTypeResolutionComplete);
    for (FileContentProvider<? extends InputStream> contentProvider : inputProviders) {
      if (contentProvider.isClassFile()) {
        try (InputStream inputStream = contentProvider.get()) {
          ClassReader cr = new ClassReader(inputStream);
          TypeHierarchyClassVisitor cv =
              new TypeHierarchyClassVisitor(
                  Opcodes.ASM8, contentProvider.getBinaryPathName(), typeHierarchyBuilder, null);
          cr.accept(cv, ClassReader.SKIP_CODE | ClassReader.SKIP_DEBUG);
        } catch (IOException e) {
          throw new IOError(e);
        }
      }
    }
    return typeHierarchyBuilder.build();
  }

  public static TypeHierarchy analyze(
      Collection<Path> jarPaths, boolean requireTypeResolutionComplete) {
    ImmutableList<FileContentProvider<InputStream>> fileContentProviders =
        FileContentProvider.fromJarsWithFiFoResolution(
            jarPaths.stream()
                .filter(path -> path.toString().endsWith(".jar"))
                .collect(Collectors.toList()),
            jarItem ->
                jarItem.jarEntry().getName().endsWith(".class")
                    && !jarItem.jarEntry().getName().startsWith("META-INF/"));
    return analyze(fileContentProviders, requireTypeResolutionComplete);
  }

  private TypeHierarchyScavenger() {}
}
