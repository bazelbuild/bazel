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

package com.google.devtools.build.android.ideinfo;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Splitter;
import com.google.devtools.build.android.Converters.PathConverter;
import com.google.devtools.build.lib.ideinfo.androidstudio.PackageManifestOuterClass.ArtifactLocation;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.OptionsParsingException;

import java.nio.file.Path;
import java.util.Iterator;
import java.util.NoSuchElementException;

/**
 * Parses artifact location from comma-separate paths
 */
@VisibleForTesting
public class ArtifactLocationConverter implements Converter<ArtifactLocation> {
  private static final Splitter SPLITTER = Splitter.on(',');
  private static final PathConverter pathConverter = new PathConverter();

  @Override
  public ArtifactLocation convert(String input) throws OptionsParsingException {
    Iterator<String> values = SPLITTER.split(input).iterator();
    try {
      Path rootExecutionPathFragment = pathConverter.convert(values.next());
      Path relPath = pathConverter.convert(values.next());

      // will be removed in a future release -- make it forward-compatible
      String root = values.hasNext() ? pathConverter.convert(values.next()).toString() : "";

      if (values.hasNext()) {
        throw new OptionsParsingException("Expected either 2 or 3 comma-separated paths");
      }

      boolean isSource = rootExecutionPathFragment.toString().isEmpty();
      return ArtifactLocation.newBuilder()
          .setRootPath(root)
          .setRootExecutionPathFragment(rootExecutionPathFragment.toString())
          .setRelativePath(relPath.toString())
          .setIsSource(isSource)
          .build();

    } catch (OptionsParsingException | NoSuchElementException e) {
      throw new OptionsParsingException("Expected either 2 or 3 comma-separated paths", e);
    }
  }

  @Override
  public String getTypeDescription() {
    return "Artifact location parser";
  }

}
