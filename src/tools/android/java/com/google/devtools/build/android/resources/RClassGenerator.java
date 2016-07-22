// Copyright 2016 The Bazel Authors. All rights reserved.
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
import com.android.builder.internal.SymbolLoader;
import com.android.builder.internal.SymbolLoader.SymbolEntry;
import com.android.resources.ResourceType;
import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import java.io.IOException;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;
import org.objectweb.asm.commons.InstructionAdapter;

/**
 * Writes out bytecode for an R.class directly, rather than go through an R.java and compile. This
 * avoids re-parsing huge R.java files and other time spent in the java compiler (e.g., plugins like
 * ErrorProne). A difference is that this doesn't generate line number tables and other debugging
 * information. Also, the order of the constant pool tends to be different.
 */
public class RClassGenerator {

  private static final int JAVA_VERSION = Opcodes.V1_7;
  private static final String SUPER_CLASS = "java/lang/Object";
  private final Path outFolder;
  private final String packageName;
  private final Map<ResourceType, List<FieldInitializer>> initializers;
  private final boolean finalFields;
  private static final Splitter PACKAGE_SPLITTER = Splitter.on('.');

  /**
   * Create an RClassGenerator given the final binary's symbol values, and a collection of symbols
   * for the given package.
   *
   * @param outFolder base folder to place the output R class files.
   * @param packageName the java package to use for the R class
   * @param values the final symbol values (may include more symbols than needed for this package)
   * @param packageSymbols the symbols in this package
   * @param finalFields true if the fields should be marked final
   */
  public static RClassGenerator fromSymbols(
      Path outFolder,
      String packageName,
      SymbolLoader values,
      Collection<SymbolLoader> packageSymbols,
      boolean finalFields) throws IOException {
    Table<String, String, SymbolEntry> symbolsTable = getAllSymbols(packageSymbols);
    Table<String, String, SymbolEntry> valuesTable = getSymbols(values);
    Map<ResourceType, List<FieldInitializer>> initializers = getInitializers(symbolsTable,
        valuesTable);
    return new RClassGenerator(outFolder, packageName, initializers, finalFields);
  }

  /**
   * Create an RClassGenerator given a collection of initializers.
   *
   * @param outFolder base folder to place the output R class files.
   * @param packageName the java package to use for the R class
   * @param initializers the list of initializers to use for each inner class
   * @param finalFields true if the fields should be marked final
   */
  public RClassGenerator(
      Path outFolder,
      String packageName,
      Map<ResourceType, List<FieldInitializer>> initializers,
      boolean finalFields) {
    this.outFolder = outFolder;
    this.packageName = packageName;
    this.finalFields = finalFields;
    this.initializers = initializers;
  }

  private static Table<String, String, SymbolEntry> getAllSymbols(
      Collection<SymbolLoader> symbolLoaders)
      throws IOException {
    Table<String, String, SymbolEntry> symbols = HashBasedTable.create();
    for (SymbolLoader symbolLoader : symbolLoaders) {
      symbols.putAll(getSymbols(symbolLoader));
    }
    return symbols;
  }


  private static Table<String, String, SymbolEntry> getSymbols(SymbolLoader symbolLoader)
      throws IOException {
    // TODO(bazel-team): remove when we update android_ide_common to a version w/ public visibility
    try {
      Method getSymbols = SymbolLoader.class.getDeclaredMethod("getSymbols");
      getSymbols.setAccessible(true);
      @SuppressWarnings("unchecked")
      Table<String, String, SymbolEntry> result = (Table<String, String, SymbolEntry>)
          getSymbols.invoke(symbolLoader);
      return result;
    } catch (ReflectiveOperationException e) {
      throw new IOException(e);
    }
  }

  /**
   * Convert the {@link SymbolLoader} data, to a map of {@link FieldInitializer}.
   */
  private static Map<ResourceType, List<FieldInitializer>> getInitializers(
      Table<String, String, SymbolEntry> symbols,
      Table<String, String, SymbolEntry> values) {
    Map<ResourceType, List<FieldInitializer>> initializers = new EnumMap<>(ResourceType.class);
    for (String typeName : symbols.rowKeySet()) {
      ResourceType resourceType = ResourceType.getEnum(typeName);
      Preconditions.checkNotNull(resourceType);
      initializers.put(resourceType, getInitializers(typeName, symbols, values));
    }
    return initializers;
  }

  private static List<FieldInitializer> getInitializers(
      String typeName,
      Table<String, String, SymbolEntry> symbols,
      Table<String, String, SymbolEntry> values) {
    Map<String, SymbolEntry> rowMap = symbols.row(typeName);
    Set<String> symbolSet = rowMap.keySet();
    List<String> symbolList = new ArrayList<>(symbolSet);
    Collections.sort(symbolList);
    List<FieldInitializer> initializers = new ArrayList<>();
    for (String symbolName : symbolList) {
      // get the matching SymbolEntry from the values Table.
      SymbolEntry value = values.get(typeName, symbolName);
      Preconditions.checkNotNull(value);
      if (value.getType().equals("int")) {
        initializers.add(IntFieldInitializer.of(value.getName(), value.getValue()));
      } else {
        Preconditions.checkArgument(value.getType().equals("int[]"));
        initializers
            .add(IntArrayFieldInitializer.of(value.getName(), value.getValue()));
      }
    }
    return initializers;
  }

  /**
   * Builds the bytecode and writes out the R.class file, and R$inner.class files.
   */
  public void write() throws IOException {
    Iterable<String> folders = PACKAGE_SPLITTER.split(packageName);
    Path packageDir = outFolder;
    for (String folder : folders) {
      packageDir = packageDir.resolve(folder);
    }
    // At least create the outFolder that was requested. However, if there are no symbols, don't
    // create the R.class and inner class files (no need to have an empty class).
    Files.createDirectories(packageDir);
    if (initializers.isEmpty()) {
      return;
    }
    Path rClassFile = packageDir.resolve(SdkConstants.FN_COMPILED_RESOURCE_CLASS);

    String packageWithSlashes = packageName.replaceAll("\\.", "/");
    String rClassName = packageWithSlashes + "/R";
    ClassWriter classWriter = new ClassWriter(ClassWriter.COMPUTE_MAXS);
    classWriter.visit(
        JAVA_VERSION,
        Opcodes.ACC_PUBLIC | Opcodes.ACC_FINAL | Opcodes.ACC_SUPER,
        rClassName,
        null, /* signature */
        SUPER_CLASS,
        null /* interfaces */);
    classWriter.visitSource(SdkConstants.FN_RESOURCE_CLASS, null);
    writeConstructor(classWriter);

    // Build the R.class w/ the inner classes, then later build the individual R$inner.class.
    for (ResourceType resourceType : initializers.keySet()) {
      String innerClassName = rClassName + "$" + resourceType;
      classWriter.visitInnerClass(
          innerClassName,
          rClassName,
          resourceType.toString(),
          Opcodes.ACC_PUBLIC | Opcodes.ACC_FINAL | Opcodes.ACC_STATIC);
    }
    classWriter.visitEnd();
    Files.write(rClassFile, classWriter.toByteArray());

    // Now generate the R$inner.class files.
    for (Map.Entry<ResourceType, List<FieldInitializer>> entry : initializers.entrySet()) {
      writeInnerClass(entry.getValue(), packageDir, rClassName, entry.getKey().toString());
    }
  }

  private void writeInnerClass(
      List<FieldInitializer> initializers,
      Path packageDir,
      String fullyQualifiedOuterClass,
      String innerClass) throws IOException {
    ClassWriter innerClassWriter = new ClassWriter(ClassWriter.COMPUTE_MAXS);
    String fullyQualifiedInnerClass =
        writeInnerClassHeader(fullyQualifiedOuterClass, innerClass, innerClassWriter);

    List<FieldInitializer> deferredInitializers = new ArrayList<>();
    int fieldAccessLevel = Opcodes.ACC_PUBLIC | Opcodes.ACC_STATIC;
    if (finalFields) {
      fieldAccessLevel |= Opcodes.ACC_FINAL;
    }
    for (FieldInitializer init : initializers) {
      if (init.writeFieldDefinition(innerClassWriter, fieldAccessLevel, finalFields)) {
        deferredInitializers.add(init);
      }
    }
    if (!deferredInitializers.isEmpty()) {
      writeStaticClassInit(innerClassWriter, fullyQualifiedInnerClass, deferredInitializers);
    }

    innerClassWriter.visitEnd();
    Path innerFile = packageDir.resolve("R$" + innerClass + ".class");
    Files.write(innerFile, innerClassWriter.toByteArray());
  }

  private String writeInnerClassHeader(String fullyQualifiedOuterClass, String innerClass,
      ClassWriter innerClassWriter) {
    String fullyQualifiedInnerClass = fullyQualifiedOuterClass + "$" + innerClass;
    innerClassWriter.visit(
        JAVA_VERSION,
        Opcodes.ACC_PUBLIC | Opcodes.ACC_FINAL | Opcodes.ACC_SUPER,
        fullyQualifiedInnerClass,
        null, /* signature */
        SUPER_CLASS,
        null /* interfaces */);
    innerClassWriter.visitSource(SdkConstants.FN_RESOURCE_CLASS, null);
    writeConstructor(innerClassWriter);
    innerClassWriter.visitInnerClass(
        fullyQualifiedInnerClass,
        fullyQualifiedOuterClass,
        innerClass,
        Opcodes.ACC_PUBLIC | Opcodes.ACC_FINAL | Opcodes.ACC_STATIC);
    return fullyQualifiedInnerClass;
  }

  private static void writeConstructor(ClassWriter classWriter) {
    MethodVisitor constructor = classWriter.visitMethod(
        Opcodes.ACC_PUBLIC,
        "<init>",
        "()V",
        null, /* signature */
        null /* exceptions */);
    constructor.visitCode();
    constructor.visitVarInsn(Opcodes.ALOAD, 0);
    constructor.visitMethodInsn(Opcodes.INVOKESPECIAL, SUPER_CLASS, "<init>", "()V", false);
    constructor.visitInsn(Opcodes.RETURN);
    constructor.visitMaxs(1, 1);
    constructor.visitEnd();
  }

  private static void writeStaticClassInit(
      ClassWriter classWriter,
      String className,
      List<FieldInitializer> initializers) {
    MethodVisitor visitor = classWriter.visitMethod(
        Opcodes.ACC_STATIC,
        "<clinit>",
        "()V",
        null,  /* signature */
        null /* exceptions */);
    visitor.visitCode();
    int stackSlotsNeeded = 0;
    InstructionAdapter insts = new InstructionAdapter(visitor);
    for (FieldInitializer fieldInit : initializers) {
      stackSlotsNeeded = Math.max(
          stackSlotsNeeded,
          fieldInit.writeCLInit(insts, className));
    }
    insts.areturn(Type.VOID_TYPE);
    visitor.visitMaxs(stackSlotsNeeded, 0);
    visitor.visitEnd();
  }

}
