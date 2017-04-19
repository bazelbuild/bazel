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
import com.android.builder.internal.SymbolLoader.SymbolEntry;
import com.android.resources.ResourceType;
import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Table;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
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
  private final Map<ResourceType, List<FieldInitializer>> initializers;
  private final boolean finalFields;
  private static final Splitter PACKAGE_SPLITTER = Splitter.on('.');

  /**
   * Create an RClassGenerator initialized with the ResourceSymbols values.
   *
   * @param outFolder base folder to place the output R class files.
   * @param values the final symbol values
   * @param finalFields true if the fields should be marked final
   */
  public static RClassGenerator fromSymbols(
      Path outFolder,
      ResourceSymbols values,
      boolean finalFields)
      throws IOException {
    Table<String, String, SymbolEntry> valuesTable = values.asTable();
    Map<ResourceType, List<FieldInitializer>> initializers =
        getInitializers(valuesTable);
    return new RClassGenerator(outFolder, initializers, finalFields);
  }

  /**
   * Create an RClassGenerator given a collection of initializers.
   *
   * @param outFolder base folder to place the output R class files.
   * @param initializers the list of initializers to use for each inner class
   * @param finalFields true if the fields should be marked final
   */
  public RClassGenerator(
      Path outFolder,
      Map<ResourceType, List<FieldInitializer>> initializers,
      boolean finalFields) {
    this.outFolder = outFolder;
    this.finalFields = finalFields;
    this.initializers = initializers;
  }
  /** Convert the {@link ResourceSymbols} data, to a map of {@link FieldInitializer}. */
  private static Map<ResourceType, List<FieldInitializer>> getInitializers(
      Table<String, String, SymbolEntry> values) {
    Map<ResourceType, List<FieldInitializer>> initializers = new EnumMap<>(ResourceType.class);
    for (String typeName : values.rowKeySet()) {
      ResourceType resourceType = ResourceType.getEnum(typeName);
      Preconditions.checkNotNull(resourceType);
      initializers.put(resourceType, getInitializers(typeName, values));
    }
    return initializers;
  }

  private static List<FieldInitializer> getInitializers(
      String typeName,
      Table<String, String, SymbolEntry> symbols) {
    Map<String, SymbolEntry> rowMap = symbols.row(typeName);
    List<String> symbolList = new ArrayList<>(rowMap.keySet());
    Collections.sort(symbolList);
    List<FieldInitializer> initializers = new ArrayList<>();
    for (String symbolName : symbolList) {
      SymbolEntry value = symbols.get(typeName, symbolName);
      if (value.getType().equals("int")) {
        initializers.add(IntFieldInitializer.of(value.getName(), value.getValue()));
      } else {
        Preconditions.checkArgument(value.getType().equals("int[]"));
        initializers.add(IntArrayFieldInitializer.of(value.getName(), value.getValue()));
      }
    }
    return initializers;
  }

  /**
   * Builds bytecode and writes out R.class file, and R$inner.class files for provided package and
   * symbols
   */
  public void write(String packageName, Map<ResourceType, Set<String>> symbolsToWrite)
      throws IOException {
    Iterable<String> folders = PACKAGE_SPLITTER.split(packageName);
    Path packageDir = outFolder;
    for (String folder : folders) {
      packageDir = packageDir.resolve(folder);
    }
    // At least create the outFolder that was requested. However, if there are no symbols, don't
    // create the R.class and inner class files (no need to have an empty class).
    Files.createDirectories(packageDir);
    Map<ResourceType, List<FieldInitializer>> initializersToWrite =
        filterInitializers(symbolsToWrite);
    if (initializersToWrite.isEmpty()) {
      return;
    }
    Path rClassFile = packageDir.resolve(SdkConstants.FN_COMPILED_RESOURCE_CLASS);

    String packageWithSlashes = packageName.replaceAll("\\.", "/");
    String rClassName = packageWithSlashes.isEmpty() ? "R" : (packageWithSlashes + "/R");
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
    for (ResourceType resourceType : initializersToWrite.keySet()) {
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
    for (Map.Entry<ResourceType, List<FieldInitializer>> entry : initializersToWrite.entrySet()) {
      writeInnerClass(entry.getValue(), packageDir, rClassName, entry.getKey().toString());
    }
  }

  /** Builds bytecode and writes out R.class file, and R$inner.class files for provided package. */
  public void write(String packageName) throws IOException {
    write(packageName, ImmutableMap.<ResourceType, Set<String>>of());
  }
  
  private Map<ResourceType, List<FieldInitializer>> filterInitializers(
      Map<ResourceType, Set<String>> symbolsToWrite) {
    Map<ResourceType, List<FieldInitializer>> initializersToWrite =
        new EnumMap<>(ResourceType.class);
    if (symbolsToWrite.isEmpty()) {
      return initializers;
    }
    for (Entry<ResourceType, Set<String>> entry : symbolsToWrite.entrySet()) {
      List<FieldInitializer> fieldsToWrite = new ArrayList<>();
      // Resource type may be missing if resource overriding eliminates resources at the binary
      // level, which were originally present at the library level.
      if (initializers.containsKey(entry.getKey())) {
        for (FieldInitializer field : initializers.get(entry.getKey())) {
          if (field.nameIsIn(entry.getValue())) {
            fieldsToWrite.add(field);
          }
        }
      }
      if (!fieldsToWrite.isEmpty()) {
        initializersToWrite.put(entry.getKey(), fieldsToWrite);
      }
    }
    return initializersToWrite;
  }

  private void writeInnerClass(
      List<FieldInitializer> initializers,
      Path packageDir,
      String fullyQualifiedOuterClass,
      String innerClass)
      throws IOException {
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

  private String writeInnerClassHeader(
      String fullyQualifiedOuterClass, String innerClass, ClassWriter innerClassWriter) {
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
    MethodVisitor constructor =
        classWriter.visitMethod(
            Opcodes.ACC_PUBLIC, "<init>", "()V", null, /* signature */ null /* exceptions */);
    constructor.visitCode();
    constructor.visitVarInsn(Opcodes.ALOAD, 0);
    constructor.visitMethodInsn(Opcodes.INVOKESPECIAL, SUPER_CLASS, "<init>", "()V", false);
    constructor.visitInsn(Opcodes.RETURN);
    constructor.visitMaxs(1, 1);
    constructor.visitEnd();
  }

  private static void writeStaticClassInit(
      ClassWriter classWriter, String className, List<FieldInitializer> initializers) {
    MethodVisitor visitor =
        classWriter.visitMethod(
            Opcodes.ACC_STATIC, "<clinit>", "()V", null, /* signature */ null /* exceptions */);
    visitor.visitCode();
    int stackSlotsNeeded = 0;
    InstructionAdapter insts = new InstructionAdapter(visitor);
    for (FieldInitializer fieldInit : initializers) {
      stackSlotsNeeded = Math.max(stackSlotsNeeded, fieldInit.writeCLInit(insts, className));
    }
    insts.areturn(Type.VOID_TYPE);
    visitor.visitMaxs(stackSlotsNeeded, 0);
    visitor.visitEnd();
  }
}
