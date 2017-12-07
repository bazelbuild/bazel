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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.LoadStatement;
import com.google.devtools.build.lib.syntax.Statement;
import com.google.devtools.skylark.skylint.Linter.FileFacade;
import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Helps collect information about direct and transitive dependencies of a Skylark file.
 *
 * @param <T> the type of information associated with a file
 */
public class DependencyAnalyzer<T> {
  private final FileFacade fileFacade;
  private final Map<Path, T> pathToInfo = new LinkedHashMap<>();
  private Map<Path, Path> cachedWorkspaceRoot = new LinkedHashMap<>();
  private Map<Path, Path> cachedPackageRoot = new LinkedHashMap<>();
  private DependencyCollector<T> collector;
  private Set<Path> visited = new LinkedHashSet<>();

  private static final ImmutableList<String> BUILD_FILES = ImmutableList.of("BUILD", "BUILD.bazel");
  private static final ImmutableList<String> WORKSPACE_FILE = ImmutableList.of("WORKSPACE");

  /**
   * Creates an instance of DependencyAnalyzer that can be reused for multiple Skylark files.
   *
   * @param fileFacade interface to access file contents
   * @param collector extracts the desired information from a Skylark file
   */
  public DependencyAnalyzer(FileFacade fileFacade, DependencyCollector<T> collector) {
    this.fileFacade = fileFacade;
    this.collector = collector;
  }

  /**
   * Collects information about the given file and its direct and transitive dependencies.
   *
   * <p>The information is cached between calls to this method, so it won't reanalyze the same file
   * twice. This applies even if a file is loaded via different path labels that correspond to the
   * same canonical path.
   *
   * @param path the path of the file to be analyzed
   * @return the information about that file, or null if it can't be read
   */
  @Nullable
  public T collectTransitiveInfo(Path path) {
    path = path.toAbsolutePath();
    if (visited.contains(path)) {
      return pathToInfo.get(path);
    }
    visited.add(path);
    BuildFileAST ast;
    try {
      ast = fileFacade.readAst(path);
    } catch (IOException e) {
      return null;
    }
    T info = collector.initInfo(path);
    for (Statement stmt : ast.getStatements()) {
      if (stmt instanceof LoadStatement) {
        String label = ((LoadStatement) stmt).getImport().getValue();
        Path dep = labelToPath(label, path);
        if (dep == null) {
          continue;
        }
        T depInfo = collectTransitiveInfo(dep);
        if (depInfo == null) {
          continue; // may happen if there's an illegal dependency cycle
        }
        info = collector.loadDependency(info, (LoadStatement) stmt, dep, depInfo);
      }
    }
    info = collector.collectInfo(path, ast, info);
    pathToInfo.put(path, info);
    return info;
  }

  @Nullable
  private Path findAncestorDirectoryContainingAnyOf(Path path, Iterable<String> fileNames) {
    Path dir = path.toAbsolutePath();
    while ((dir = dir.getParent()) != null) {
      for (String fileName : fileNames) {
        if (fileFacade.fileExists(dir.resolve(fileName))) {
          return dir;
        }
      }
    }
    return null;
  }

  /**
   * Resolves the label of a load statement to a path.
   *
   * @param label the import of a load statement
   * @param currentPath the path of the file containing the load statement
   * @return the path corresponding to the label or null if it can't be resolved
   */
  @Nullable
  private Path labelToPath(String label, Path currentPath) {
    if (label.startsWith("@")) {
      // TODO(skylark-team): analyze such dependencies as well
      return null;
    } else if (label.startsWith("//")) {
      Path workspaceRoot = getWorkspaceRoot(currentPath);
      if (workspaceRoot == null) {
        return null;
      }
      label = label.substring(label.startsWith("//:") ? 3 : 2);
      return workspaceRoot.resolve(label.replace(':', '/'));
    } else if (label.startsWith(":")) {
      Path packageRoot = getPackageRoot(currentPath);
      if (packageRoot == null) {
        return null;
      }
      return packageRoot.resolve(label.substring(1));
    } else {
      // otherwise just treat it as a though it started with "//"
      Path workspaceRoot = getWorkspaceRoot(currentPath);
      if (workspaceRoot == null) {
        return null;
      }
      return workspaceRoot.resolve(label.replace(':', '/'));
    }
  }

  @Nullable
  private Path getPackageRoot(Path path) {
    if (!cachedPackageRoot.containsKey(path)) {
      cachedPackageRoot.put(path, findAncestorDirectoryContainingAnyOf(path, BUILD_FILES));
    }
    return cachedPackageRoot.get(path);
  }

  @Nullable
  private Path getWorkspaceRoot(Path path) {
    if (!cachedWorkspaceRoot.containsKey(path)) {
      cachedWorkspaceRoot.put(path, findAncestorDirectoryContainingAnyOf(path, WORKSPACE_FILE));
    }
    return cachedWorkspaceRoot.get(path);
  }

  /**
   * Encapsulates how to produce information about a Skylark file, given its contents and the
   * information about its transitive dependencies.
   *
   * <p>Each of the methods returns new or updated instances of T associated with the file.
   *
   * <p>When analyzing a file, the methods will be invoked in the following order:
   *
   * <ol>
   *   <li>{@link DependencyCollector#initInfo} to return an initial info object
   *   <li>{@link DependencyCollector#loadDependency} is iteratively called for each direct
   *       dependency to get an updated info object that accounts for the info from this dependency,
   *       until all direct dependencies have been processed
   *   <li>{@link DependencyCollector#collectInfo} is called to get a final updated info object that
   *       accounts for the content of the current file.
   * </ol>
   *
   * @param <T> the type of information being collected
   */
  public interface DependencyCollector<T> {

    /**
     * Used to initialize the dependency information when starting analysis of a file.
     *
     * @param path the path of the current file
     * @return the initial information about the current file
     */
    T initInfo(Path path);

    /**
     * Incorporates the information about a dependency in the information about the file.
     *
     * <p>This method is called for every load() statement in the current file.
     *
     * @param currentFileInfo info about the current file so far, may be modified
     * @param stmt the load statement being processed
     * @param loadedPath the path of the dependency that is load()ed
     * @param loadedFileInfo info about the dependency file that is load()ed, must not be modified
     * @return the updated information about the current file
     */
    T loadDependency(T currentFileInfo, LoadStatement stmt, Path loadedPath, T loadedFileInfo);

    /**
     * Collect information about the current file after the load statements have been processed.
     *
     * @param path the path of the current file
     * @param ast the ast of the current file
     * @param info the information about the current file so far, may be modified
     * @return the updated information about the current file
     */
    T collectInfo(Path path, BuildFileAST ast, T info);
  }
}
