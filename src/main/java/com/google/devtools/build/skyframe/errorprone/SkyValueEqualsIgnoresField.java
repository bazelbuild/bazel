// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.skyframe.errorprone;

import static com.google.errorprone.BugPattern.SeverityLevel.ERROR;
import static com.google.errorprone.matchers.Description.NO_MATCH;

import com.google.errorprone.BugPattern;
import com.google.errorprone.VisitorState;
import com.google.errorprone.bugpatterns.BugChecker;
import com.google.errorprone.bugpatterns.BugChecker.ClassTreeMatcher;
import com.google.errorprone.matchers.Description;
import com.google.errorprone.suppliers.Supplier;
import com.google.errorprone.util.ASTHelpers;
import com.sun.source.tree.ClassTree;
import com.sun.source.tree.IdentifierTree;
import com.sun.source.tree.MemberReferenceTree;
import com.sun.source.tree.MemberSelectTree;
import com.sun.source.tree.MethodInvocationTree;
import com.sun.source.tree.MethodTree;
import com.sun.source.tree.Tree;
import com.sun.source.tree.VariableTree;
import com.sun.source.util.TreeScanner;
import com.sun.tools.javac.code.Symbol;
import com.sun.tools.javac.code.Symbol.ClassSymbol;
import com.sun.tools.javac.code.Symbol.MethodSymbol;
import com.sun.tools.javac.code.Symbol.VarSymbol;
import com.sun.tools.javac.code.Type;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import javax.lang.model.element.ElementKind;
import javax.lang.model.element.Modifier;

/**
 * Flags instance fields of a {@code SkyValue} that its {@code equals} method doesn't read.
 *
 * <p>Skyframe keeps the old value when a recomputed value is equal to it, so consumers keep seeing
 * the old content of such fields. Values that implement {@code NotComparableSkyValue} are never
 * compared, and transient fields and caches are exempt. Other fields that are derived from compared
 * fields can be exempted with {@code @SuppressWarnings("SkyValueEqualsIgnoresField")}.
 */
@BugPattern(
    summary =
        "equals() of a SkyValue ignores this field, so Skyframe change pruning can keep a stale"
            + " value of it",
    severity = ERROR)
public final class SkyValueEqualsIgnoresField extends BugChecker implements ClassTreeMatcher {

  private static final Supplier<Type> SKY_VALUE =
      VisitorState.memoize(
          state -> state.getTypeFromString("com.google.devtools.build.skyframe.SkyValue"));
  private static final Supplier<Type> NOT_COMPARABLE_SKY_VALUE =
      VisitorState.memoize(
          state ->
              state.getTypeFromString("com.google.devtools.build.skyframe.NotComparableSkyValue"));

  private static final String[] CACHE_TYPES = {
    "com.github.benmanes.caffeine.cache.Cache", "com.google.common.cache.Cache",
  };

  @Override
  public Description matchClass(ClassTree tree, VisitorState state) {
    Type skyValue = SKY_VALUE.get(state);
    ClassSymbol classSymbol = ASTHelpers.getSymbol(tree);
    Type notComparableSkyValue = NOT_COMPARABLE_SKY_VALUE.get(state);
    if (skyValue == null
        || classSymbol.isInterface()
        || !ASTHelpers.isSubtype(classSymbol.type, skyValue, state)
        || (notComparableSkyValue != null
            && ASTHelpers.isSubtype(classSymbol.type, notComparableSkyValue, state))) {
      return NO_MATCH;
    }

    Map<MethodSymbol, MethodTree> methods = new HashMap<>();
    Map<VarSymbol, VariableTree> fields = new LinkedHashMap<>();
    MethodTree equals = null;
    for (Tree member : tree.getMembers()) {
      if (member instanceof MethodTree methodTree) {
        MethodSymbol methodSymbol = ASTHelpers.getSymbol(methodTree);
        methods.put(methodSymbol, methodTree);
        if (isEquals(methodSymbol, state)) {
          equals = methodTree;
        }
      } else if (member instanceof VariableTree variableTree) {
        VarSymbol field = ASTHelpers.getSymbol(variableTree);
        if (field.getKind() == ElementKind.FIELD
            && !field.isStatic()
            && !field.getModifiers().contains(Modifier.TRANSIENT)
            && !isCache(field.type, state)
            && !isSuppressedOnField(field)) {
          fields.put(field, variableTree);
        }
      }
    }
    if (fields.isEmpty()) {
      return NO_MATCH;
    }

    Set<Symbol> readSymbols = new HashSet<>();
    if (equals == null) {
      if (!inheritsCustomEquals(classSymbol, state)) {
        return NO_MATCH;
      }
      // An inherited equals can only read the fields of this class through overridden methods.
      for (Map.Entry<MethodSymbol, MethodTree> method : methods.entrySet()) {
        String name = method.getKey().getSimpleName().toString();
        if (!name.equals("hashCode")
            && !name.equals("toString")
            && !ASTHelpers.findSuperMethods(method.getKey(), state.getTypes()).isEmpty()) {
          collectReads(method.getValue(), methods, readSymbols);
        }
      }
    } else if (equals.getBody() != null) {
      collectReads(equals, methods, readSymbols);
    } else {
      return NO_MATCH;
    }

    for (Map.Entry<VarSymbol, VariableTree> field : fields.entrySet()) {
      if (!readSymbols.contains(field.getKey())) {
        state.reportMatch(
            buildDescription(field.getValue())
                .setMessage(
                    String.format(
                        "%s.equals() ignores field '%s'. Skyframe keeps the old value when a"
                            + " recomputed value is equal to it, so consumers can see a stale"
                            + " value of this field.",
                        classSymbol.getSimpleName(), field.getKey().getSimpleName()))
                .build());
      }
    }
    return NO_MATCH;
  }

  private static void collectReads(
      MethodTree equals, Map<MethodSymbol, MethodTree> methods, Set<Symbol> readSymbols) {
    Deque<MethodTree> worklist = new ArrayDeque<>();
    Set<MethodSymbol> visited = new HashSet<>();
    worklist.add(equals);
    visited.add(ASTHelpers.getSymbol(equals));
    while (!worklist.isEmpty()) {
      MethodTree method = worklist.pop();
      if (method.getBody() == null) {
        continue;
      }
      new TreeScanner<Void, Void>() {
        @Override
        public Void visitIdentifier(IdentifierTree node, Void unused) {
          readSymbols.add(ASTHelpers.getSymbol(node));
          return null;
        }

        @Override
        public Void visitMemberSelect(MemberSelectTree node, Void unused) {
          readSymbols.add(ASTHelpers.getSymbol(node));
          return super.visitMemberSelect(node, unused);
        }

        @Override
        public Void visitMethodInvocation(MethodInvocationTree node, Void unused) {
          enqueue(ASTHelpers.getSymbol(node));
          return super.visitMethodInvocation(node, unused);
        }

        @Override
        public Void visitMemberReference(MemberReferenceTree node, Void unused) {
          if (ASTHelpers.getSymbol(node) instanceof MethodSymbol methodSymbol) {
            enqueue(methodSymbol);
          }
          return super.visitMemberReference(node, unused);
        }

        private void enqueue(MethodSymbol methodSymbol) {
          MethodTree callee = methods.get(methodSymbol);
          if (callee != null && visited.add(methodSymbol)) {
            worklist.add(callee);
          }
        }
      }.scan(method.getBody(), null);
    }
  }

  private static boolean isEquals(MethodSymbol method, VisitorState state) {
    return method.getSimpleName().contentEquals("equals")
        && !method.isStatic()
        && method.getParameters().size() == 1
        && ASTHelpers.isSameType(
            method.getParameters().get(0).type, state.getSymtab().objectType, state);
  }

  private static boolean inheritsCustomEquals(ClassSymbol classSymbol, VisitorState state) {
    Type superType = state.getTypes().supertype(classSymbol.type);
    while (superType != null
        && superType.tsym instanceof ClassSymbol superClass
        && !ASTHelpers.isSameType(superType, state.getSymtab().objectType, state)) {
      for (Symbol member : superClass.getEnclosedElements()) {
        if (member instanceof MethodSymbol method
            && isEquals(method, state)
            && !method.getModifiers().contains(Modifier.ABSTRACT)) {
          return true;
        }
      }
      superType = state.getTypes().supertype(superType);
    }
    return false;
  }

  private static boolean isCache(Type type, VisitorState state) {
    for (String cacheType : CACHE_TYPES) {
      Type cache = state.getTypeFromString(cacheType);
      if (cache != null
          && ASTHelpers.isSubtype(
              state.getTypes().erasure(type), state.getTypes().erasure(cache), state)) {
        return true;
      }
    }
    return false;
  }

  private boolean isSuppressedOnField(VarSymbol field) {
    SuppressWarnings suppressWarnings = field.getAnnotation(SuppressWarnings.class);
    return suppressWarnings != null
        && Arrays.asList(suppressWarnings.value()).contains(canonicalName());
  }
}
