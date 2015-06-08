// Copyright 2015 Google Inc. All rights reserved.
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

package com.google.devtools.build.buildjar.javac.plugins.processing;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.buildjar.javac.plugins.BlazeJavaCompilerPlugin;

import java.nio.file.Path;
import java.util.HashSet;

/**
 * A module for information about the compilation's annotation processing.
 */
public class AnnotationProcessingModule {
  
  /**
   * A builder for {@link AnnotationProcessingModule}s.
   */
  public static class Builder {
    private Path sourceGenDir;
    private Path classDir;

    private Builder() {}

    public AnnotationProcessingModule build() {
      return new AnnotationProcessingModule(sourceGenDir, classDir);
    }

    public void setSourceGenDir(Path sourceGenDir) {
      this.sourceGenDir = sourceGenDir.toAbsolutePath();
    }

    public void setClassDir(Path classDir) {
      this.classDir = classDir.toAbsolutePath();
    }
  }

  private final boolean enabled;
  private final Path sourceGenDir;
  private final Path classDir;

  public Path sourceGenDir() {
    return sourceGenDir;
  }

  private AnnotationProcessingModule(Path sourceGenDir, Path classDir) {
    this.sourceGenDir = sourceGenDir;
    this.classDir = classDir;
    this.enabled = sourceGenDir != null && classDir != null;
  }

  public static Builder builder() {
    return new Builder();
  }

  public void registerPlugin(ImmutableList.Builder<BlazeJavaCompilerPlugin> builder) {
    if (enabled) {
      builder.add(new AnnotationProcessingPlugin(this));
    }
  }

  /**
   * The set of prefixes of generated class files.
   */
  private final HashSet<String> pathPrefixes = new HashSet<>();

  /**
   * Record the prefix of a group of generated class files.
   *
   * <p>Prefixes are used to handle generated inner classes. Since
   * e.g. j/c/g/Foo.class and j/c/g/Foo$Inner.class both correspond to the
   * same generated source file, only the prefix "j/c/g/Foo" is recorded.
   */
  void recordPrefix(String pathPrefix) {
    pathPrefixes.add(pathPrefix);
  }

  /** Returns true if the given path is to a generated source file. */
  public boolean isGeneratedSource(Path sourcePath) {
    return sourcePath.toAbsolutePath().startsWith(sourceGenDir);
  }

  /** Returns true if the given path is to a generated class file. */
  public boolean isGeneratedClass(Path path) {
    if (!path.getFileName().toString().endsWith(".class")) {
      return false;
    }
    String prefix = classDir.relativize(path.toAbsolutePath()).toString();
    prefix = prefix.substring(0, prefix.length() - ".class".length());
    int idx = prefix.lastIndexOf('$');
    if (idx > 0) {
      prefix = prefix.substring(0, idx);
    }
    return pathPrefixes.contains(prefix);
  }
}
