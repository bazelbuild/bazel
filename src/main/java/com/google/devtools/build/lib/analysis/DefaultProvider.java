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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.ClassObjectConstructor;
import com.google.devtools.build.lib.packages.SkylarkClassObject;
import com.google.devtools.build.lib.rules.SkylarkRuleContext;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import java.util.Map;

/** DefaultProvider is provided by all targets implicitly and contains all standard fields. */
@Immutable
public final class DefaultProvider extends SkylarkClassObject {

  // Accessors for Skylark
  private static final String DATA_RUNFILES_FIELD = "data_runfiles";
  private static final String DEFAULT_RUNFILES_FIELD = "default_runfiles";
  private static final String FILES_FIELD = "files";

  private DefaultProvider(ClassObjectConstructor constructor, Map<String, Object> values) {
    super(constructor, values);
  }

  public static DefaultProvider build(
      RunfilesProvider runfilesProvider,
      FileProvider fileProvider,
      FilesToRunProvider filesToRunProvider) {
    ImmutableMap.Builder<String, Object> attrBuilder = new ImmutableMap.Builder<>();
    if (runfilesProvider != null) {
      attrBuilder.put(DATA_RUNFILES_FIELD, runfilesProvider.getDataRunfiles());
      attrBuilder.put(DEFAULT_RUNFILES_FIELD, runfilesProvider.getDefaultRunfiles());
    } else {
      attrBuilder.put(DATA_RUNFILES_FIELD, Runfiles.EMPTY);
      attrBuilder.put(DEFAULT_RUNFILES_FIELD, Runfiles.EMPTY);
    }

    attrBuilder.put(
        FILES_FIELD, SkylarkNestedSet.of(Artifact.class, fileProvider.getFilesToBuild()));
    attrBuilder.put(FilesToRunProvider.SKYLARK_NAME, filesToRunProvider);

    ClassObjectConstructor constructor = SkylarkRuleContext.getDefaultProvider();
    return new DefaultProvider(constructor, attrBuilder.build());
  }
}
