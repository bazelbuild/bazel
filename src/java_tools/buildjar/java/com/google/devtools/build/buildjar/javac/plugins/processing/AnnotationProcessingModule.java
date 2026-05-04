// Copyright 2015 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.buildjar.proto.JavaCompilation.CompilationUnit;
import com.google.devtools.build.buildjar.proto.JavaCompilation.Manifest;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** A module for information about the compilation's annotation processing. */
public class AnnotationProcessingModule {

  /** A builder for {@link AnnotationProcessingModule}s. */
  public static class Builder {
    private Path sourceGenDir;
    private Path manifestProto;
    private Path workDir;

    private Builder() {}

    public AnnotationProcessingModule build() {
      return new AnnotationProcessingModule(sourceGenDir, manifestProto, workDir);
    }

    public void setSourceGenDir(Path sourceGenDir) {
      this.sourceGenDir = sourceGenDir;
    }

    public void setManifestProtoPath(Path manifestProto) {
      this.manifestProto = manifestProto.toAbsolutePath();
    }

    public void setWorkDir(Path workDir) {
      this.workDir = workDir;
    }
  }

  private final boolean enabled;
  private final Path sourceGenDir;
  private final Path manifestProto;
  private final Path workDir;

  public boolean isGenerated(Path path) {
    return path.startsWith(sourceGenDir);
  }

  public Path stripSourceRoot(Path path) {
    if (isGenerated(path)) {
      return sourceGenDir.relativize(path);
    }
    // Strip the sandbox working directory prefix to produce deterministic exec-root-relative paths.
    // Without this, multiplex worker sandboxing embeds a non-deterministic slot number
    // (e.g. __sandbox/751/_main/...) in the manifest proto output.
    return stripPrefix(path, workDir);
  }

  /** Returns {@code path} relativized against {@code prefix} if it is under it, else unchanged. */
  private static Path stripPrefix(Path path, Path prefix) {
    return !prefix.toString().isEmpty() && path.startsWith(prefix)
        ? prefix.relativize(path)
        : path;
  }

  private AnnotationProcessingModule(Path sourceGenDir, Path manifestProto, Path workDir) {
    this.sourceGenDir = sourceGenDir;
    this.manifestProto = manifestProto;
    this.workDir = workDir != null ? workDir : Path.of("");
    this.enabled = sourceGenDir != null && manifestProto != null;
  }

  public static Builder builder() {
    return new Builder();
  }

  public void registerPlugin(ImmutableList.Builder<BlazeJavaCompilerPlugin> builder) {
    if (enabled) {
      builder.add(new AnnotationProcessingPlugin(this));
    }
  }

  private final Map<String, CompilationUnit> units = new HashMap<>();

  public void recordUnit(CompilationUnit unit) {
    units.put(unit.getPath(), unit);
  }

  private Manifest buildManifestProto() {
    Manifest.Builder builder = Manifest.newBuilder();

    List<String> keys = new ArrayList<>(units.keySet());
    Collections.sort(keys);
    for (String key : keys) {
      CompilationUnit unit = units.get(key);
      builder.addCompilationUnit(unit);
    }

    return builder.build();
  }

  public void emitManifestProto() throws IOException {
    if (!enabled) {
      return;
    }
    try (OutputStream out = Files.newOutputStream(manifestProto)) {
      buildManifestProto().writeTo(out);
    } catch (IOException ex) {
      throw new IOException("Cannot write manifest to " + manifestProto, ex);
    }
  }
}
