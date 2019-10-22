// Copyright 2019 The Bazel Authors. All rights reserved.
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
//

package com.google.devtools.build.lib.bazel.rules.ninja.parser;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.bazel.rules.ninja.file.GenericParsingException;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;

public class NinjaIncludeParser implements NinjaDeclarationParser<PathFragment> {
  public static final NinjaIncludeParser INSTANCE = new NinjaIncludeParser();

  @Override
  public ImmutableSortedSet<NinjaKeyword> getKeywords() {
    return ImmutableSortedSet.of(NinjaKeyword.includeKeyword, NinjaKeyword.subNinja);
  }

  @Override
  public PathFragment parse(List<String> lines) throws GenericParsingException {
    if (lines.size() != 1) {
      throw new GenericParsingException("Wrong include/subninja format: " + String.join("\n", lines));
    }
    String statement = lines.get(0);
    Preconditions.checkArgument(statement.startsWith(NinjaKeyword.includeKeyword.getText())
        || statement.startsWith(NinjaKeyword.subNinja.getText()));
    int spaceIdx = statement.indexOf(' ');
    Preconditions.checkArgument(spaceIdx > 0);
    PathFragment includedPath = PathFragment.create(statement.substring(spaceIdx + 1).trim());
    if (includedPath.isAbsolute()) {
      throw new GenericParsingException("Do not expect absolute files to be included: " + includedPath);
    }

    return includedPath;
  }
}
