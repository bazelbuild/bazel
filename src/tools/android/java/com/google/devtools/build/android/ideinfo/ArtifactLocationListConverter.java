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

import com.google.common.collect.Lists;
import com.google.devtools.build.lib.ideinfo.androidstudio.PackageManifestOuterClass.ArtifactLocation;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.OptionsParsingException;

import java.util.Collections;
import java.util.List;

/**
 * Parses list of colon-separated artifact locations
 */
public class ArtifactLocationListConverter implements Converter<List<ArtifactLocation>> {
  final ArtifactLocationConverter baseConverter = new ArtifactLocationConverter();

  @Override
  public List<ArtifactLocation> convert(String input) throws OptionsParsingException {
    List<ArtifactLocation> list = Lists.newArrayList();
    for (String piece : input.split(":")) {
      if (!piece.isEmpty()) {
        list.add(baseConverter.convert(piece));
      }
    }
    return Collections.unmodifiableList(list);
  }

  @Override
  public String getTypeDescription() {
    return "a colon-separated list of artifact locations";
  }
}
