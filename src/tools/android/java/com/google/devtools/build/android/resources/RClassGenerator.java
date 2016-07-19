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

import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Table;
import com.google.common.io.Files;

import com.android.SdkConstants;
import com.android.builder.internal.SymbolLoader;
import com.android.builder.internal.SymbolLoader.SymbolEntry;

import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;
import org.objectweb.asm.commons.InstructionAdapter;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Writes out bytecode for an R.class directly, rather than go through an R.java and compile. This
 * avoids re-parsing huge R.java files and other time spent in the java compiler (e.g., plugins like
 * ErrorProne). A difference is that this doesn't generate line number tables and other debugging
 * information. Also, the order of the constant pool tends to be different.
 */
public class RClassGenerator {

  private static final int JAVA_VERSION = Opcodes.V1_7;
  private static final String SUPER_CLASS = "java/lang/Object";
  private final File outFolder;
  private final String packageName;
  private final List<SymbolLoader> symbolTables = new ArrayList<>();
  private final SymbolLoader symbolValues;
  private final boolean finalFields;

  public RClassGenerator(
      File outFolder,
      String packageName,
      SymbolLoader values,
      boolean finalFields) {
    this.outFolder = outFolder;
    this.packageName = packageName;
    this.symbolValues = values;
    this.finalFields = finalFields;
  }

  public void addSymbolsToWrite(SymbolLoader symbols) {
    symbolTables.add(symbols);
  }

  private Table<String, String, SymbolEntry> getAllSymbols() throws IOException {
    Table<String, String, SymbolEntry> symbols = HashBasedTable.create();
    for (SymbolLoader symbolLoader : symbolTables) {
      symbols.putAll(getSymbols(symbolLoader));
    }
    return symbols;
  }

  private Method symbolsMethod;

  private Table<String, String, SymbolEntry> getSymbols(SymbolLoader symbolLoader)
      throws IOException {
    // TODO(bazel-team): upstream a patch to change the visibility instead of hacking it.
    try {
      if (symbolsMethod == null) {
        Method getSymbols = SymbolLoader.class.getDeclaredMethod("getSymbols");
        getSymbols.setAccessible(true);
        symbolsMethod = getSymbols;
      }
      @SuppressWarnings("unchecked")
      Table<String, String, SymbolEntry> result = (Table<String, String, SymbolEntry>)
          symbolsMethod.invoke(symbolLoader);
      return result;
    } catch (ReflectiveOperationException e) {
      throw new IOException(e);
    }
  }

  /**
   * Builds the bytecode and writes out the R.class file, and R$inner.class files.
   */
  public void write() throws IOException {
    Splitter splitter = Splitter.on('.');
    Iterable<String> folders = splitter.split(packageName);
    File packageDir = outFolder;
    for (String folder : folders) {
      packageDir = new File(packageDir, folder);
    }
    File rClassFile = new File(packageDir, SdkConstants.FN_COMPILED_RESOURCE_CLASS);
    // At least create the outFolder that was requested. However, if there are no symbols, don't
    // create the R.class and inner class files (no need to have an empty class).
    Files.createParentDirs(rClassFile);
    Table<String, String, SymbolEntry> symbols = getAllSymbols();
    if (symbols.isEmpty()) {
      return;
    }

    String packageWithSlashes = packageName.replaceAll("\\.", "/");
    String rClassName = packageWithSlashes + "/R";
    ClassWriter classWriter = new ClassWriter(ClassWriter.COMPUTE_MAXS);
    classWriter
        .visit(JAVA_VERSION, Opcodes.ACC_PUBLIC | Opcodes.ACC_FINAL | Opcodes.ACC_SUPER,
            rClassName, null, SUPER_CLASS, null);
    classWriter.visitSource(SdkConstants.FN_RESOURCE_CLASS, null);
    writeConstructor(classWriter);

    Table<String, String, SymbolEntry> values = getSymbols(symbolValues);
    Set<String> rowSet = symbols.rowKeySet();
    List<String> rowList = new ArrayList<>(rowSet);
    Collections.sort(rowList);

    // Build the R.class w/ the inner classes, then later build the individual R$inner.class.
    for (String row : rowList) {
      String innerClassName = rClassName + "$" + row;
      classWriter.visitInnerClass(innerClassName, rClassName, row,
          Opcodes.ACC_PUBLIC | Opcodes.ACC_FINAL | Opcodes.ACC_STATIC);
    }
    classWriter.visitEnd();
    Files.write(classWriter.toByteArray(), rClassFile);

    // Now generate the R$inner.class files.
    for (String row : rowList) {
      writeInnerClass(symbols, values, packageDir, rClassName, row);
    }
  }

  /**
   * Represents an int or int[] field and its initializer (where initialization is done via code in
   * the static clinit function).
   */
  private interface DeferredInitializer {

    /**
     * Write the code for the initializer via insts.
     *
     * @return the number of stack slots needed for the code.
     */
    int writeCLInit(String className, InstructionAdapter insts);
  }

  private static final class IntArrayDeferredInitializer implements DeferredInitializer {

    private final String fieldName;
    private final ImmutableList<Integer> values;

    IntArrayDeferredInitializer(String fieldName, ImmutableList<Integer> values) {
      this.fieldName = fieldName;
      this.values = values;
    }

    public static DeferredInitializer of(String name, String value) {
      Preconditions.checkArgument(value.startsWith("{ "), "Expected list starting with { ");
      Preconditions.checkArgument(value.endsWith(" }"), "Expected list ending with } ");
      // Check for an empty list, which is "{ }".
      if (value.length() < 4) {
        return new IntArrayDeferredInitializer(name, ImmutableList.<Integer>of());
      }
      ImmutableList.Builder<Integer> intValues = ImmutableList.builder();
      String trimmedValue = value.substring(2, value.length() - 2);
      Iterable<String> valueStrings = Splitter.on(',')
          .trimResults()
          .omitEmptyStrings()
          .split(trimmedValue);
      for (String valueString : valueStrings) {
        intValues.add(Integer.decode(valueString));
      }
      return new IntArrayDeferredInitializer(name, intValues.build());
    }

    @Override
    public int writeCLInit(String className, InstructionAdapter insts) {
      insts.iconst(values.size());
      insts.newarray(Type.INT_TYPE);
      int curIndex = 0;
      for (Integer value : values) {
        insts.dup();
        insts.iconst(curIndex);
        insts.iconst(value);
        insts.astore(Type.INT_TYPE);
        ++curIndex;
      }
      insts.putstatic(className, fieldName, "[I");
      // Needs up to 4 stack slots for: the array ref for the putstatic, the dup of the array ref
      // for the store, the index, and the value to store.
      return 4;
    }
  }

  private static final class IntDeferredInitializer implements DeferredInitializer {

    private final String fieldName;
    private final Integer value;

    IntDeferredInitializer(String fieldName, Integer value) {
      this.fieldName = fieldName;
      this.value = value;
    }

    public static DeferredInitializer of(String name, String value) {
      return new IntDeferredInitializer(name, Integer.decode(value));
    }

    @Override
    public int writeCLInit(String className, InstructionAdapter insts) {
      insts.iconst(value);
      insts.putstatic(className, fieldName, "I");
      // Just needs one stack slot for the iconst.
      return 1;
    }
  }

  private void writeInnerClass(
      Table<String, String, SymbolEntry> symbols,
      Table<String, String, SymbolEntry> values,
      File packageDir,
      String fullyQualifiedOuterClass,
      String innerClass) throws IOException {
    ClassWriter innerClassWriter = new ClassWriter(ClassWriter.COMPUTE_MAXS);
    String fullyQualifiedInnerClass = fullyQualifiedOuterClass + "$" + innerClass;
    innerClassWriter
        .visit(JAVA_VERSION, Opcodes.ACC_PUBLIC | Opcodes.ACC_FINAL | Opcodes.ACC_SUPER,
            fullyQualifiedInnerClass, null, SUPER_CLASS, null);
    innerClassWriter.visitSource("R.java", null);
    writeConstructor(innerClassWriter);
    innerClassWriter.visitInnerClass(
        fullyQualifiedInnerClass, fullyQualifiedOuterClass, innerClass,
        Opcodes.ACC_PUBLIC | Opcodes.ACC_FINAL | Opcodes.ACC_STATIC);

    Map<String, SymbolEntry> rowMap = symbols.row(innerClass);
    Set<String> symbolSet = rowMap.keySet();
    List<String> symbolList = new ArrayList<>(symbolSet);
    Collections.sort(symbolList);
    List<DeferredInitializer> deferredInitializers = new ArrayList<>();
    int fieldAccessLevel = Opcodes.ACC_PUBLIC | Opcodes.ACC_STATIC;
    if (finalFields) {
      fieldAccessLevel |= Opcodes.ACC_FINAL;
    }
    for (String symbolName : symbolList) {
      // get the matching SymbolEntry from the values Table.
      SymbolEntry value = values.get(innerClass, symbolName);
      if (value != null) {
        String desc;
        Object initializer = null;
        if (value.getType().equals("int")) {
          desc = "I";
          if (finalFields) {
            initializer = Integer.decode(value.getValue());
          } else {
            deferredInitializers.add(IntDeferredInitializer.of(value.getName(), value.getValue()));
          }
        } else {
          Preconditions.checkArgument(value.getType().equals("int[]"));
          desc = "[I";
          deferredInitializers
              .add(IntArrayDeferredInitializer.of(value.getName(), value.getValue()));
        }
        innerClassWriter
            .visitField(fieldAccessLevel, value.getName(), desc, null, initializer)
            .visitEnd();
      }
    }

    if (!deferredInitializers.isEmpty()) {
      // build the <clinit> method.
      writeStaticClassInit(innerClassWriter, fullyQualifiedInnerClass, deferredInitializers);
    }

    innerClassWriter.visitEnd();
    File innerFile = new File(packageDir, "R$" + innerClass + ".class");
    Files.write(innerClassWriter.toByteArray(), innerFile);
  }

  private static void writeConstructor(ClassWriter classWriter) {
    MethodVisitor constructor = classWriter.visitMethod(Opcodes.ACC_PUBLIC, "<init>", "()V",
        null, null);
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
      List<DeferredInitializer> deferredInitializers) {
    MethodVisitor visitor = classWriter.visitMethod(Opcodes.ACC_STATIC, "<clinit>", "()V",
        null, null);
    visitor.visitCode();
    int stackSlotsNeeded = 0;
    InstructionAdapter insts = new InstructionAdapter(visitor);
    for (DeferredInitializer fieldInit : deferredInitializers) {
      stackSlotsNeeded = Math.max(stackSlotsNeeded, fieldInit.writeCLInit(className, insts));
    }
    insts.areturn(Type.VOID_TYPE);
    visitor.visitMaxs(stackSlotsNeeded, 0);
    visitor.visitEnd();
  }

}
