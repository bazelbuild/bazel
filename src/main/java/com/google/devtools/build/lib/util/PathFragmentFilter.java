// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.util;

import com.google.common.base.Joiner;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.Converter;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Handles options that specify list of included/excluded directories.
 * Validates whether path is included in that filter.
 *
 * Excluded directories always take precedence over included ones (path depth
 * and order are not important).
 */
public class PathFragmentFilter implements Serializable {
  private final List<PathFragment> inclusions;
  private final List<PathFragment> exclusions;

  /**
   * Converts from a colon-separated list of of paths with optional '-' prefix into the
   * PathFragmentFilter:
   *   [-]path1[,[-]path2]...
   *
   * Order of paths is not important. Empty entries are ignored. '-' marks an excluded path.
   */
  public static class PathFragmentFilterConverter implements Converter<PathFragmentFilter> {

    @Override
    public PathFragmentFilter convert(String input) {
      List<PathFragment> inclusionList = new ArrayList<>();
      List<PathFragment> exclusionList = new ArrayList<>();

      for (String piece : Splitter.on(',').split(input)) {
        if (piece.length() > 1 && piece.startsWith("-")) {
          exclusionList.add(PathFragment.create(piece.substring(1)));
        } else if (piece.length() > 0) {
          inclusionList.add(PathFragment.create(piece));
        }
      }

      // TODO(bazel-team): (2010) Both lists could be optimized not to include unnecessary
      // entries - e.g.  entry 'a/b/c' is not needed if 'a/b' is also specified and 'a/b' on
      // inclusion list is not needed if 'a' or 'a/b' is on exclusion list.
      return new PathFragmentFilter(inclusionList, exclusionList);
    }

    @Override
    public String getTypeDescription() {
      return "a comma-separated list of paths with prefix '-' specifying excluded paths";
    }

  }

  /**
   * Creates new PathFragmentFilter using provided inclusion and exclusion path lists.
   */
  public PathFragmentFilter(List<PathFragment> inclusions, List<PathFragment> exclusions) {
    this.inclusions = ImmutableList.copyOf(inclusions);
    this.exclusions = ImmutableList.copyOf(exclusions);
  }

  /**
   * @return true iff path is included (it is not on the exclusion list and
   *         it is either on the inclusion list or inclusion list is empty).
   */
  public boolean isIncluded(PathFragment path) {
    for (PathFragment excludedPath : exclusions) {
      if (path.startsWith(excludedPath)) {
        return false;
      }
    }
    for (PathFragment includedPath : inclusions) {
      if (path.startsWith(includedPath)) {
        return true;
      }
    }
    return inclusions.isEmpty(); // If inclusion filter is not specified, path is included.
  }

  @Override
  public String toString() {
    List<String> list = new ArrayList<>(inclusions.size() + exclusions.size());
    for (PathFragment path : inclusions) {
      list.add(path.getPathString());
    }
    for (PathFragment path : exclusions) {
      list.add("-" + path.getPathString());
    }
    return Joiner.on(',').join(list);
  }
}
