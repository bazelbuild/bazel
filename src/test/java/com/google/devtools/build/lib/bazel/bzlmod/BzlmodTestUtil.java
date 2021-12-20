// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.bzlmod;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.Attribute;
import net.starlark.java.eval.Dict;
import net.starlark.java.syntax.Location;

/** Utilities for bzlmod tests. */
public final class BzlmodTestUtil {
  private BzlmodTestUtil() {}

  /** Simple wrapper around {@link ModuleKey#create} that takes a string version. */
  public static ModuleKey createModuleKey(String name, String version) {
    try {
      return ModuleKey.create(name, Version.parse(version));
    } catch (Version.ParseException e) {
      throw new IllegalArgumentException(e);
    }
  }

  public static RepositoryMapping createRepositoryMapping(ModuleKey key, String... names) {
    ImmutableMap.Builder<RepositoryName, RepositoryName> mappingBuilder = ImmutableMap.builder();
    for (int i = 0; i < names.length; i += 2) {
      mappingBuilder.put(
          RepositoryName.createUnvalidated(names[i]),
          RepositoryName.createUnvalidated(names[i + 1]));
    }
    return RepositoryMapping.create(mappingBuilder.build(), key.getCanonicalRepoName());
  }

  public static TagClass createTagClass(Attribute... attrs) {
    return TagClass.create(ImmutableList.copyOf(attrs), "doc", Location.BUILTIN);
  }

  /** A builder for {@link Tag} for testing purposes. */
  public static class TestTagBuilder {
    private final Dict.Builder<String, Object> attrValuesBuilder = Dict.builder();
    private final String tagName;

    private TestTagBuilder(String tagName) {
      this.tagName = tagName;
    }

    public TestTagBuilder addAttr(String attrName, Object attrValue) {
      attrValuesBuilder.put(attrName, attrValue);
      return this;
    }

    public Tag build() {
      return Tag.builder()
          .setTagName(tagName)
          .setLocation(Location.BUILTIN)
          .setAttributeValues(attrValuesBuilder.buildImmutable())
          .build();
    }
  }

  public static TestTagBuilder buildTag(String tagName) throws Exception {
    return new TestTagBuilder(tagName);
  }
}
