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

import com.android.SdkConstants;
import com.android.builder.core.VariantConfiguration;
import com.android.builder.dependency.SymbolFileProvider;
import com.android.resources.ResourceType;
import com.android.utils.ILogger;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.Multimap;
import com.google.common.collect.Table;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningExecutorService;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.logging.Logger;
import javax.annotation.Nullable;

/** This provides a unified interface for working with R.txt symbols files. */
public class ResourceSymbols {
  private static final Logger logger = Logger.getLogger(ResourceSymbols.class.getCanonicalName());

  /** Represents a resource symbol with a value. */
  // Forked from com.android.builder.internal.SymbolLoader.SymbolEntry.
  static class RTxtSymbolEntry {
    private final String name;
    private final String type;
    private final String value;

    public RTxtSymbolEntry(String name, String type, String value) {
      this.name = name;
      this.type = type;
      this.value = value;
    }

    public String getValue() {
      return value;
    }

    public String getName() {
      return name;
    }

    public String getType() {
      return type;
    }
  }

  /** Task to load and parse R.txt symbols */
  private static final class SymbolLoadingTask implements Callable<ResourceSymbols> {

    private final Path rTxtSymbols;

    SymbolLoadingTask(Path symbolFile) {
      this.rTxtSymbols = symbolFile;
    }

    @Override
    public ResourceSymbols call() throws Exception {
      List<String> lines = Files.readAllLines(rTxtSymbols, StandardCharsets.UTF_8);

      Table<String, String, RTxtSymbolEntry> mSymbols = HashBasedTable.create();

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

          mSymbols.put(className, name, new RTxtSymbolEntry(name, type, value));
        } catch (IndexOutOfBoundsException e) {
          String s =
              String.format(
                  "File format error reading %s\tline %d: '%s'",
                  rTxtSymbols.toString(), lineIndex, line);
          logger.severe(s);
          throw new IOException(s, e);
        }
      }
      return ResourceSymbols.from(mSymbols);
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
        packageToTable.put(entry.getValue().get(), load(symbolFile.toPath(), executor));
      }
    }
    return packageToTable;
  }

  public static ResourceSymbols from(Table<String, String, RTxtSymbolEntry> table) {
    return new ResourceSymbols(table);
  }

  public static ResourceSymbols merge(Collection<ResourceSymbols> symbolTables) {
    final Table<String, String, RTxtSymbolEntry> mergedTable = HashBasedTable.create();
    for (ResourceSymbols symbolTableProvider : symbolTables) {
      mergedTable.putAll(symbolTableProvider.asTable());
    }
    return from(mergedTable);
  }

  /** Read the symbols from the provided symbol file. */
  public static ListenableFuture<ResourceSymbols> load(
      Path primaryRTxt, ListeningExecutorService executorService) {
    return executorService.submit(new SymbolLoadingTask(primaryRTxt));
  }

  private final Table<String, String, RTxtSymbolEntry> values;

  private ResourceSymbols(Table<String, String, RTxtSymbolEntry> values) {
    this.values = values;
  }

  public Table<String, String, RTxtSymbolEntry> asTable() {
    return values;
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
  
    Table<String, String, RTxtSymbolEntry> symbols = HashBasedTable.create();
    for (ResourceSymbols packageSymbol : packageSymbols) {
      symbols.putAll(packageSymbol.asTable());
    }

    Path packageOut = sourceOut.resolve(packageName.replace('.', File.separatorChar));
    Files.createDirectories(packageOut);

    Path file = packageOut.resolve(SdkConstants.FN_RESOURCE_CLASS);

    try (BufferedWriter writer =
        Files.newBufferedWriter(file, StandardCharsets.UTF_8, StandardOpenOption.CREATE_NEW)) {

      writer.write("/* AUTO-GENERATED FILE.  DO NOT MODIFY.\n");
      writer.write(" *\n");
      writer.write(" * This class was automatically generated by the\n");
      writer.write(" * aapt tool from the resource data it found.  It\n");
      writer.write(" * should not be modified by hand.\n");
      writer.write(" */\n");

      writer.write("package ");
      writer.write(packageName);
      writer.write(";\n\npublic final class R {\n");

      Set<String> rowSet = symbols.rowKeySet();
      List<String> rowList = new ArrayList<>(rowSet);
      Collections.sort(rowList);

      for (String row : rowList) {
        writer.write("\tpublic static final class ");
        writer.write(row);
        writer.write(" {\n");

        Map<String, RTxtSymbolEntry> rowMap = symbols.row(row);
        Set<String> symbolSet = rowMap.keySet();
        List<String> symbolList = new ArrayList<>(symbolSet);
        Collections.sort(symbolList);

        for (String symbolName : symbolList) {
          // get the matching SymbolEntry from the values Table.
          RTxtSymbolEntry value = values.get(row, symbolName);
          if (value != null) {
            writer.write("\t\tpublic static final ");
            writer.write(value.getType());
            writer.write(" ");
            writer.write(value.getName());
            writer.write(" = ");
            writer.write(value.getValue());
            writer.write(";\n");
          }
        }

        writer.write("\t}\n");
      }

      writer.write("}\n");
    }
  }

  public Map<ResourceType, Set<String>> asFilterMap() {
    Map<ResourceType, Set<String>> filter = new EnumMap<>(ResourceType.class);
    Table<String, String, RTxtSymbolEntry> symbolTable = asTable();
    for (String typeName : symbolTable.rowKeySet()) {
      Set<String> fields = new HashSet<>();
      for (RTxtSymbolEntry symbolEntry : symbolTable.row(typeName).values()) {
        fields.add(symbolEntry.getName());
      }
      if (!fields.isEmpty()) {
        filter.put(ResourceType.getEnum(typeName), fields);
      }
    }
    return filter;
  }
}
