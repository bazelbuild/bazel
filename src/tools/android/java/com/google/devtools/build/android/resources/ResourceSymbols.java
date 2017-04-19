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
import com.android.builder.internal.SymbolLoader;
import com.android.builder.internal.SymbolLoader.SymbolEntry;
import com.android.builder.internal.SymbolWriter;
import com.android.resources.ResourceType;
import com.android.utils.ILogger;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.Multimap;
import com.google.common.collect.Table;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningExecutorService;
import java.io.File;
import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.file.Path;
import java.util.Collection;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import javax.annotation.Nullable;

/** 
 * Wraps the {@link SymbolLoader} and {@link SymbolWriter} classes.
 * This provides a unified interface for working with R.txts.
 */
public class ResourceSymbols {
  /** Task to load and parse R.txt symbols */
  private static final class SymbolLoadingTask implements Callable<ResourceSymbols> {

    private final SymbolLoader symbolLoader;

    SymbolLoadingTask(SymbolLoader symbolLoader) {
      this.symbolLoader = symbolLoader;
    }

    @Override
    public ResourceSymbols call() throws Exception {
      symbolLoader.load();
      return ResourceSymbols.wrap(symbolLoader);
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
   * @param iLogger Android logger to use.
   * @param packageToExclude A string package to elide if it exists in the providers.
   * @return A list of loading {@link ResourceSymbols} instances.
   * @throws ExecutionException
   * @throws InterruptedException when there is an error loading the symbols.
   */
  public static Multimap<String, ListenableFuture<ResourceSymbols>> loadFrom(
      Collection<SymbolFileProvider> dependencies,
      ListeningExecutorService executor,
      ILogger iLogger,
      @Nullable String packageToExclude)
      throws InterruptedException, ExecutionException {
    Map<SymbolFileProvider, ListenableFuture<String>> providerToPackage = new HashMap<>();
    for (SymbolFileProvider dependency : dependencies) {
      providerToPackage.put(
          dependency, executor.submit(new PackageParsingTask(dependency.getManifest())));
    }
    Multimap<String, ListenableFuture<ResourceSymbols>> packageToTable = HashMultimap.create();
    for (Entry<SymbolFileProvider, ListenableFuture<String>> entry : providerToPackage.entrySet()) {
      File symbolFile = entry.getKey().getSymbolFile();
      if (!Objects.equals(entry.getValue().get(), packageToExclude)) {
        packageToTable.put(entry.getValue().get(), load(executor, iLogger, symbolFile));
      }
    }
    return packageToTable;
  }

  public static ResourceSymbols merge(Collection<ResourceSymbols> symbolTables) throws IOException {
    final Table<String, String, SymbolEntry> mergedTable = HashBasedTable.create();
    for (ResourceSymbols symbolTableProvider : symbolTables) {
      mergedTable.putAll(symbolTableProvider.asTable());
    }
    try {
      SymbolLoader nullLoader = new SymbolLoader(null, null);
      Field declaredField = SymbolLoader.class.getDeclaredField("mSymbols");
      declaredField.setAccessible(true);
      declaredField.set(nullLoader, mergedTable);
      return wrap(nullLoader);
    } catch (NoSuchFieldException
        | SecurityException
        | IllegalArgumentException
        | IllegalAccessException e) {
      throw new IOException(e);
    }
  }

  /** Read the symbols from the provided symbol file. */
  public static ListenableFuture<ResourceSymbols> load(
      Path primaryRTxt, ListeningExecutorService executorService, ILogger iLogger) {
    return load(executorService, iLogger, primaryRTxt.toFile());
  }

  public static ListenableFuture<ResourceSymbols> load(
      ListeningExecutorService executor, ILogger iLogger, File symbolFile) {
    return executor.submit(new SymbolLoadingTask(new SymbolLoader(symbolFile, iLogger)));
  }

  static ResourceSymbols of(Path rTxt, ILogger logger) {
    return of(rTxt.toFile(), logger);
  }

  public static ResourceSymbols of(File rTxt, ILogger logger) {
    return wrap(new SymbolLoader(rTxt, logger));
  }

  private static ResourceSymbols wrap(SymbolLoader input) {
    return new ResourceSymbols(input);
  }

  private final SymbolLoader symbolLoader;

  private ResourceSymbols(SymbolLoader symbolLoader) {
    this.symbolLoader = symbolLoader;
  }

  public Table<String, String, SymbolEntry> asTable() throws IOException {
    // TODO(bazel-team): remove when we update android_ide_common to a version w/ public visibility
    try {
      Method getSymbols = SymbolLoader.class.getDeclaredMethod("getSymbols");
      getSymbols.setAccessible(true);
      @SuppressWarnings("unchecked")
      Table<String, String, SymbolEntry> result =
          (Table<String, String, SymbolEntry>) getSymbols.invoke(symbolLoader);
      return result;
    } catch (ReflectiveOperationException e) {
      throw new IOException(e);
    }
  }

  /**
   * Writes the java sources for a given package.
   *
   * @param sourceOut The directory to write the java package structures and sources to.
   * @param packageName The name of the package to write.
   * @param packageSymbols The symbols defined in the given package.
   * @throws IOException when encountering an error during writing.
   */
  public void writeTo(
      Path sourceOut, String packageName, Collection<ResourceSymbols> packageSymbols)
      throws IOException {
    SymbolWriter writer = new SymbolWriter(sourceOut.toString(), packageName, symbolLoader);
    for (ResourceSymbols packageSymbol : packageSymbols) {
      writer.addSymbolsToWrite(packageSymbol.symbolLoader);
    }
    writer.write();
  }

  public Map<ResourceType, Set<String>> asFilterMap() throws IOException {
    Map<ResourceType, Set<String>> filter = new EnumMap<>(ResourceType.class);
    Table<String, String, SymbolEntry> symbolTable = asTable();
    for (String typeName : symbolTable.rowKeySet()) {
      Set<String> fields = new HashSet<>();
      for (SymbolEntry symbolEntry : symbolTable.row(typeName).values()) {
        fields.add(symbolEntry.getName());
      }
      if (!fields.isEmpty()) {
        filter.put(ResourceType.getEnum(typeName), fields);
      }
    }
    return filter;
  }
}
