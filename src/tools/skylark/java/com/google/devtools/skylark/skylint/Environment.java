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

package com.google.devtools.skylark.skylint;

import com.google.devtools.build.lib.packages.BazelLibrary;
import com.google.devtools.build.lib.skylarkinterface.SkylarkSignature;
import com.google.devtools.build.lib.syntax.ASTNode;
import com.google.devtools.build.lib.syntax.Comment;
import com.google.devtools.build.lib.syntax.Identifier;
import com.google.devtools.build.lib.syntax.MethodLibrary;
import com.google.devtools.skylark.skylint.Environment.NameInfo.Kind;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/** Holds the information about which symbols are in scope during AST traversal. */
public class Environment {
  private final List<LexicalBlock> blocks = new ArrayList<>();
  private final Map<Integer, NameInfo> idToNameInfo = new HashMap<>();
  private int nextId = 0;
  private static final int BUILTINS_INDEX = 0; // index of the block containing builtins
  private static final int GLOBALS_INDEX = 1; // index of the block containing globals

  public static Environment empty() {
    Environment env = new Environment();
    env.blocks.add(new LexicalBlock()); // for builtins
    return env;
  }

  public static Environment defaultBazel() {
    Environment env = empty();
    env.addBuiltin("None");
    env.addBuiltin("True");
    env.addBuiltin("False");
    env.setupFunctions(MethodLibrary.class, BazelLibrary.class);
    return env;
  }

  private void setupFunctions(Class<?>... classes) {
    // Iterate through the skylark functions declared inline within the classes
    // retrieving and storing their names.
    for (Class<?> c : classes) {
      for (Field field : c.getDeclaredFields()) {
        // Skylark functions are defined inline as fields within the class, annotated
        // by @SkylarkSignature.
        SkylarkSignature builtinFuncSignature = field.getAnnotation(SkylarkSignature.class);
        if (builtinFuncSignature != null && builtinFuncSignature.objectType() == Object.class) {
          addBuiltin(builtinFuncSignature.name());
        }
      }
    }
  }

  public void enterBlock() {
    blocks.add(new LexicalBlock());
  }

  public Collection<Integer> exitBlock() {
    if (blocks.size() <= GLOBALS_INDEX) {
      throw new IllegalStateException("no blocks to exit from");
    }
    return blocks.remove(blocks.size() - 1).getAllSymbols();
  }

  public boolean inGlobalBlock() {
    return blocks.size() - 1 == GLOBALS_INDEX;
  }

  public boolean isDefined(String name) {
    return resolveName(name) != null;
  }

  public boolean isDefinedInCurrentScope(String name) {
    return blocks.get(blocks.size() - 1).resolve(name) != null;
  }

  @Nullable
  public NameInfo resolveName(String name) {
    for (int i = blocks.size() - 1; i >= 0; i--) {
      Integer id = blocks.get(i).resolve(name);
      if (id != null) {
        return idToNameInfo.get(id);
      }
    }
    return null;
  }

  public NameInfo resolveExistingName(String name) {
    NameInfo info = resolveName(name);
    if (info == null) {
      throw new IllegalArgumentException("name '" + name + "' doesn't exist");
    }
    return info;
  }

  private void addName(int block, NameInfo nameInfo) {
    NameInfo prev = idToNameInfo.putIfAbsent(nameInfo.id, nameInfo);
    if (prev != null) {
      throw new IllegalStateException("id " + nameInfo.id + " is already used!");
    }
    blocks.get(block).add(nameInfo.name, nameInfo.id);
  }

  private void addBuiltin(String name) {
    addName(BUILTINS_INDEX, createNameInfo(name, new Comment("builtin"), Kind.BUILTIN));
  }

  public void addImported(String name, Identifier node) {
    addName(GLOBALS_INDEX, createNameInfo(name, node, Kind.IMPORTED));
  }

  public void addIdentifier(String name, ASTNode node) {
    Kind kind = blocks.size() - 1 == GLOBALS_INDEX ? Kind.GLOBAL : Kind.LOCAL;
    addName(blocks.size() - 1, createNameInfo(name, node, kind));
  }

  public void addFunction(String name, ASTNode node) {
    addName(GLOBALS_INDEX, createNameInfo(name, node, Kind.FUNCTION));
  }

  public void addParameter(String name, ASTNode param) {
    addName(blocks.size() - 1, createNameInfo(name, param, Kind.PARAMETER));
  }

  private NameInfo createNameInfo(String name, ASTNode definition, Kind kind) {
    return new NameInfo(name, definition, newId(), kind);
  }

  private int newId() {
    int ret = nextId;
    nextId++;
    return ret;
  }

  public Collection<Integer> getNameIdsInCurrentBlock() {
    return getNameIdsInBlock(blocks.size() - 1);
  }

  private Collection<Integer> getNameIdsInBlock(int block) {
    return Collections.unmodifiableCollection(blocks.get(block).nameToId.values());
  }

  public NameInfo getNameInfo(int id) {
    NameInfo info = idToNameInfo.get(id);
    if (info == null) {
      throw new IllegalArgumentException("id " + id + " doesn't exist");
    }
    return info;
  }

  /**
   * Represents a lexical block, e.g. global, function-local or further nested (in a comprehension).
   */
  private static class LexicalBlock {
    private final Map<String, Integer> nameToId = new HashMap<>();

    @Nullable
    private Integer resolve(String name) {
      return nameToId.get(name);
    }

    private void add(String name, int id) {
      Integer entry = nameToId.putIfAbsent(name, id);
      if (entry != null) {
        throw new IllegalArgumentException("name '" + name + "' is already defined");
      }
    }

    public Collection<Integer> getAllSymbols() {
      return nameToId.values();
    }
  }

  /** Holds information about a name/symbol. */
  public static class NameInfo {
    final int id;
    final String name;
    final ASTNode definition;
    final Kind kind;

    /** Kind of definition where the name was declared. */
    public enum Kind {
      BUILTIN,
      IMPORTED,
      GLOBAL,
      FUNCTION,
      PARAMETER,
      LOCAL,
    }

    private NameInfo(String name, ASTNode definition, int id, Kind kind) {
      this.id = id;
      this.name = name;
      this.definition = definition;
      this.kind = kind;
    }
  }
}
