/*
 * Copyright 2020 The Bazel Authors. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *    http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.devtools.build.android.desugar.io;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.android.desugar.langmodel.ClassName;
import java.io.InputStream;
import java.util.Map;

/**
 * Provides class file content based on an in-memory map, which can be used by the desugar tool for
 * processing yet-delivered output files.
 */
@AutoValue
public abstract class MapBasedClassFileProvider implements ClassFileBatchProvider {

  /**
   * An informational tag for this class file provider. No production logic is expected to depend on
   * the tag value.
   */
  abstract String tag();

  abstract ImmutableMap<ClassName, FileContentProvider<InputStream>> fileContents();

  public static MapBasedClassFilesBuilder builder() {
    return new AutoValue_MapBasedClassFileProvider.Builder();
  }

  @Override
  public final FileContentProvider<InputStream> getContent(ClassName className) {
    return fileContents().get(className);
  }

  public final void sink(OutputFileProvider outputFileProvider) {
    fileContents().forEach((className, fileContent) -> fileContent.sink(outputFileProvider));
  }

  /** The bulider for {@link MapBasedClassFileProvider}. */
  @AutoValue.Builder
  public abstract static class MapBasedClassFilesBuilder {

    public abstract MapBasedClassFilesBuilder setTag(String value);

    public abstract MapBasedClassFilesBuilder setFileContents(
        Map<ClassName, FileContentProvider<InputStream>> value);

    public abstract MapBasedClassFileProvider build();
  }
}
