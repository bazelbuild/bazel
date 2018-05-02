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
package com.google.devtools.build.android.resources;

import com.android.builder.core.VariantConfiguration;
import com.android.builder.dependency.SymbolFileProvider;
import com.android.resources.ResourceType;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.Multimap;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningExecutorService;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.logging.Logger;
import javax.annotation.Nullable;

/** Encapsulates the logic for loading and writing resource symbols. */
public class ResourceSymbols {
  private static final Logger logger = Logger.getLogger(ResourceSymbols.class.getCanonicalName());

  /** Task to load and parse R.txt symbols */
  private static final class SymbolLoadingTask implements Callable<ResourceSymbols> {

    private final Path rTxtSymbols;

    SymbolLoadingTask(Path symbolFile) {
      this.rTxtSymbols = symbolFile;
    }

    @Override
    public ResourceSymbols call() throws Exception {
      List<String> lines = Files.readAllLines(rTxtSymbols, StandardCharsets.UTF_8);

      Map<ResourceType, Map<String, FieldInitializer>> initializers =
          new EnumMap<>(ResourceType.class);

      for (int lineIndex = 1; lineIndex <= lines.size(); lineIndex++) {
        String line = null;
        try {
          line = lines.get(lineIndex - 1);

          // format is "<type> <class> <name> <value>"
          // don't want to split on space as value could contain spaces.
          int pos = line.indexOf(' ');
          String type = line.substring(0, pos);
          int pos2 = line.indexOf(' ', pos + 1);
          String className = line.substring(pos + 1, pos2);
          int pos3 = line.indexOf(' ', pos2 + 1);
          String name = line.substring(pos2 + 1, pos3);
          String value = line.substring(pos3 + 1);

          final ResourceType resourceType = ResourceType.getEnum(className);
          final Map<String, FieldInitializer> fields;
          if (initializers.containsKey(resourceType)) {
            fields = initializers.get(resourceType);
          } else {
            fields = new TreeMap<>();
            initializers.put(resourceType, fields);
          }
          if ("int".equals(type)) {
            fields.put(name, IntFieldInitializer.of(value));
          } else {
            fields.put(name, IntArrayFieldInitializer.of(value));
          }
        } catch (IndexOutOfBoundsException e) {
          String s =
              String.format(
                  "File format error reading %s\tline %d: '%s'",
                  rTxtSymbols.toString(), lineIndex, line);
          logger.severe(s);
          throw new IOException(s, e);
        }
      }
      return ResourceSymbols.from(FieldInitializers.copyOf(initializers));
    }
  }

  private static final class PackageParsingTask implements Callable<String> {

    private final File manifest;

    PackageParsingTask(File manifest) {
      this.manifest = manifest;
    }

    @Override
    public String call() throws Exception {
      return VariantConfiguration.getManifestPackage(manifest);
    }
  }

  /**
   * Loads the SymbolTables from a list of SymbolFileProviders.
   *
   * @param dependencies The full set of library symbols to load.
   * @param executor The executor use during loading.
   * @param packageToExclude A string package to elide if it exists in the providers.
   * @return A list of loading {@link ResourceSymbols} instances.
   * @throws ExecutionException
   * @throws InterruptedException when there is an error loading the symbols.
   */
  public static Multimap<String, ListenableFuture<ResourceSymbols>> loadFrom(
      Iterable<? extends SymbolFileProvider> dependencies,
      ListeningExecutorService executor,
      @Nullable String packageToExclude)
      throws InterruptedException, ExecutionException {
    Map<SymbolFileProvider, ListenableFuture<String>> providerToPackage = new HashMap<>();
    for (SymbolFileProvider dependency : dependencies) {
      providerToPackage.put(
          dependency, executor.submit(new PackageParsingTask(dependency.getManifest())));
    }
    Multimap<String, ListenableFuture<ResourceSymbols>> packageToTable = HashMultimap.create();
    for (Map.Entry<SymbolFileProvider, ListenableFuture<String>> entry :
        providerToPackage.entrySet()) {
      File symbolFile = entry.getKey().getSymbolFile();
      if (!Objects.equals(entry.getValue().get(), packageToExclude)) {
        packageToTable.put(entry.getValue().get(), load(symbolFile.toPath(), executor));
      }
    }
    return packageToTable;
  }

  public static ResourceSymbols from(FieldInitializers fieldInitializers) {
    return new ResourceSymbols(fieldInitializers);
  }

  public static ResourceSymbols merge(Collection<ResourceSymbols> symbolTables) {
    List<FieldInitializers> fieldInitializers = new ArrayList<>(symbolTables.size());
    for (ResourceSymbols symbolTableProvider : symbolTables) {
      fieldInitializers.add(symbolTableProvider.asInitializers());
    }
    return from(FieldInitializers.mergedFrom(fieldInitializers));
  }

  /** Read the symbols from the provided symbol file. */
  public static ListenableFuture<ResourceSymbols> load(
      Path primaryRTxt, ListeningExecutorService executorService) {
    return executorService.submit(new SymbolLoadingTask(primaryRTxt));
  }

  private final FieldInitializers values;

  private ResourceSymbols(FieldInitializers fieldInitializers) {
    this.values = fieldInitializers;
  }

  /**
   * Writes the java sources for a given package.
   *
   * @param sourceOut The directory to write the java package structures and sources to.
   * @param packageName The name of the package to write.
   * @param packageSymbols The symbols defined in the given package.
   * @param finalFields
   * @throws IOException when encountering an error during writing.
   */
  public void writeSourcesTo(
      Path sourceOut,
      String packageName,
      Collection<ResourceSymbols> packageSymbols,
      boolean finalFields)
      throws IOException {
    RSourceGenerator.with(sourceOut, asInitializers(), finalFields)
        .write(packageName, merge(packageSymbols).asInitializers());
  }

  public FieldInitializers asInitializers() {
    return values;
  }

  public void writeClassesTo(
      Multimap<String, ResourceSymbols> libMap,
      String appPackageName,
      Path classesOut,
      boolean finalFields)
      throws IOException {
    RClassGenerator classWriter = RClassGenerator.with(classesOut, values, finalFields);
    for (String packageName : libMap.keySet()) {
      classWriter.write(packageName, ResourceSymbols.merge(libMap.get(packageName)).values);
    }
    if (appPackageName != null) {
      // Unlike the R.java generation, we also write the app's R.class file so that the class
      // jar file can be complete (aapt doesn't generate it for us).
      classWriter.write(appPackageName);
    }
  }
}
