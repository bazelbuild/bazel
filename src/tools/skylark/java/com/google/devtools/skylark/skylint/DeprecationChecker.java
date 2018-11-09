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

import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.FunctionDefStatement;
import com.google.devtools.build.lib.syntax.Identifier;
import com.google.devtools.build.lib.syntax.LoadStatement;
import com.google.devtools.build.lib.syntax.StringLiteral;
import com.google.devtools.skylark.skylint.DependencyAnalyzer.DependencyCollector;
import com.google.devtools.skylark.skylint.DocstringUtils.DocstringInfo;
import com.google.devtools.skylark.skylint.Environment.NameInfo;
import com.google.devtools.skylark.skylint.Environment.NameInfo.Kind;
import com.google.devtools.skylark.skylint.Linter.FileFacade;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Checks for usage of deprecated symbols. */
public class DeprecationChecker extends AstVisitorWithNameResolution {
  private static final String DEPRECATED_SYMBOL_CATEGORY = "deprecated-symbol";

  private final List<Issue> issues = new ArrayList<>();
  /** Maps a global function name to its deprecation information, if any. */
  private final Map<String, DeprecatedSymbol> symbolToDeprecation;
  /** Path of the file that is checked. */
  private final Path path;

  private DeprecationChecker(Path path, Map<String, DeprecatedSymbol> symbolToDeprecation) {
    this.path = path;
    this.symbolToDeprecation = symbolToDeprecation;
  }

  public static List<Issue> check(Path path, BuildFileAST ast, FileFacade fileFacade) {
    Map<String, DeprecatedSymbol> symbolToDeprecationWarning =
        DeprecationCollector.getDeprecations(path, fileFacade);
    DeprecationChecker checker = new DeprecationChecker(path, symbolToDeprecationWarning);
    checker.visit(ast);
    return checker.issues;
  }

  @Override
  public void visit(FunctionDefStatement node) {
    // Don't issue deprecation warnings inside of deprecated functions:
    if (!symbolToDeprecation.containsKey(node.getIdentifier().getName())) {
      super.visit(node);
    }
  }

  @Override
  void use(Identifier ident) {
    NameInfo info = env.resolveName(ident.getName());
    if (info == null) {
      return;
    }
    DeprecatedSymbol deprecation = symbolToDeprecation.get(info.name);
    if (deprecation != null && info.kind != Kind.LOCAL && info.kind != Kind.PARAMETER) {
      String originInfo = "";
      if (deprecation.origin != path) {
        originInfo = "(imported from " + deprecation.origin;
        if (!deprecation.originalName.equals(info.name)) {
          originInfo += ", named '" + deprecation.originalName + "' there";
        }
        originInfo += ") ";
      }
      String message =
          "usage of '"
              + info.name
              + "' "
              + originInfo
              + "is deprecated: "
              + deprecation.deprecationMessage;
      issues.add(Issue.create(DEPRECATED_SYMBOL_CATEGORY, message, ident.getLocation()));
    }
  }

  /** Holds information about a deprecated symbol. */
  private static class DeprecatedSymbol {
    final Path origin;
    final String originalName;
    final String deprecationMessage;

    public DeprecatedSymbol(Path origin, String originalName, String deprecationMessage) {
      this.origin = origin;
      this.originalName = originalName;
      this.deprecationMessage = deprecationMessage;
    }
  }

  /** Collects information about deprecated symbols (including dependencies). */
  private static class DeprecationCollector
      implements DependencyCollector<Map<String, DeprecatedSymbol>> {

    /**
     * Returns deprecation information (including dependencies) about symbols in the given file.
     *
     * @param path the path of the file to collect deprecation from
     * @param fileFacade to access files
     * @return a map: symbol name -> deprecation info
     */
    public static Map<String, DeprecatedSymbol> getDeprecations(Path path, FileFacade fileFacade) {
      return new DependencyAnalyzer<>(fileFacade, new DeprecationCollector())
          .collectTransitiveInfo(path);
    }

    @Override
    public Map<String, DeprecatedSymbol> initInfo(Path path) {
      return new LinkedHashMap<>();
    }

    @Override
    public Map<String, DeprecatedSymbol> loadDependency(
        Map<String, DeprecatedSymbol> currentFileInfo,
        LoadStatement stmt,
        Path loadedPath,
        Map<String, DeprecatedSymbol> loadedFileInfo) {
      for (LoadStatement.Binding binding : stmt.getBindings()) {
        String originalName = binding.getOriginalName().getName();
        String alias = binding.getLocalName().getName();
        DeprecatedSymbol originalDeprecation = loadedFileInfo.get(originalName);
        if (originalDeprecation != null) {
          currentFileInfo.put(alias, originalDeprecation);
        }
      }
      return currentFileInfo;
    }

    @Override
    public Map<String, DeprecatedSymbol> collectInfo(
        Path path, BuildFileAST ast, Map<String, DeprecatedSymbol> deprecationInfos) {
      Map<String, StringLiteral> docstrings = DocstringUtils.collectDocstringLiterals(ast);
      for (Map.Entry<String, StringLiteral> entry : docstrings.entrySet()) {
        String symbol = entry.getKey();
        StringLiteral docstring = entry.getValue();
        DocstringInfo info = DocstringUtils.parseDocstring(docstring, new ArrayList<>());
        if (!info.deprecated.isEmpty()) {
          deprecationInfos.put(symbol, new DeprecatedSymbol(path, symbol, info.deprecated));
        }
      }
      return deprecationInfos;
    }
  }
}
