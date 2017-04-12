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
package com.google.devtools.build.lib.analysis;

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.ClassObjectConstructor;
import com.google.devtools.build.lib.packages.NativeClassObjectConstructor;
import com.google.devtools.build.lib.packages.SkylarkClassObject;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

/** DefaultProvider is provided by all targets implicitly and contains all standard fields. */
@Immutable
public final class DefaultProvider extends SkylarkClassObject {

  // Accessors for Skylark
  private static final String DATA_RUNFILES_FIELD = "data_runfiles";
  private static final String DEFAULT_RUNFILES_FIELD = "default_runfiles";
  private static final String FILES_FIELD = "files";
  private static final ImmutableList<String> KEYS =
      ImmutableList.of(
          DATA_RUNFILES_FIELD,
          DEFAULT_RUNFILES_FIELD,
          FILES_FIELD,
          FilesToRunProvider.SKYLARK_NAME);

  private final RunfilesProvider runfilesProvider;
  private final FileProvider fileProvider;
  private final FilesToRunProvider filesToRunProvider;
  private final AtomicReference<SkylarkNestedSet> files = new AtomicReference<>();

  public static final String SKYLARK_NAME = "DefaultInfo";
  public static final ClassObjectConstructor SKYLARK_CONSTRUCTOR =
      new NativeClassObjectConstructor(SKYLARK_NAME) {
        @Override
        protected SkylarkClassObject createInstanceFromSkylark(Object[] args, Location loc) {
          @SuppressWarnings("unchecked")
          Map<String, Object> kwargs = (Map<String, Object>) args[0];
          return new SkylarkClassObject(this, kwargs, loc);
        }
      };

  private DefaultProvider(
      ClassObjectConstructor constructor,
      RunfilesProvider runfilesProvider,
      FileProvider fileProvider,
      FilesToRunProvider filesToRunProvider) {
    // Fields map is not used here to prevent memory regression
    super(constructor, ImmutableMap.<String, Object>of());
    this.runfilesProvider = runfilesProvider;
    this.fileProvider = fileProvider;
    this.filesToRunProvider = filesToRunProvider;
  }

  public static DefaultProvider build(
      RunfilesProvider runfilesProvider,
      FileProvider fileProvider,
      FilesToRunProvider filesToRunProvider) {
    return new DefaultProvider(
        SKYLARK_CONSTRUCTOR, runfilesProvider, fileProvider, filesToRunProvider);
  }

  @Override
  public Object getValue(String name) {
    switch (name) {
      case DATA_RUNFILES_FIELD:
        return (runfilesProvider == null) ? Runfiles.EMPTY : runfilesProvider.getDataRunfiles();
      case DEFAULT_RUNFILES_FIELD:
        return (runfilesProvider == null) ? Runfiles.EMPTY : runfilesProvider.getDefaultRunfiles();
      case FILES_FIELD:
        if (files.get() == null) {
          files.compareAndSet(
              null, SkylarkNestedSet.of(Artifact.class, fileProvider.getFilesToBuild()));
        }
        return files.get();
      case FilesToRunProvider.SKYLARK_NAME:
        return filesToRunProvider;
    }
    return null;
  }

  @Override
  public ImmutableCollection<String> getKeys() {
    return KEYS;
  }
}
