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

package com.google.devtools.build.lib.bazel.repository.skylark;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.packages.AggregatingAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction.RepositoryFunctionException;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.ClassObject.SkylarkClassObject;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.SkylarkType;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;

import java.io.IOException;

/**
 * Skylark API for the repository_rule's context.
 */
@SkylarkModule(
  name = "repository_ctx",
  doc =
      "The context of the repository rule containing"
          + " helper functions and information about attributes. You get a repository_ctx object"
          + " as an argument to the <code>implementation</code> function when you create a"
          + " repository rule."
)
public class SkylarkRepositoryContext {

  private final Rule rule;
  private final Path outputDirectory;
  private final SkylarkClassObject attrObject;

  /**
   * In native code, private values start with $. In Skylark, private values start with _, because
   * of the grammar.
   */
  private String attributeToSkylark(String oldName) {
    if (!oldName.isEmpty() && (oldName.charAt(0) == '$' || oldName.charAt(0) == ':')) {
      return "_" + oldName.substring(1);
    }
    return oldName;
  }

  /**
   * Create a new context (ctx) object for a skylark repository rule ({@code rule} argument). The
   * environment
   */
  SkylarkRepositoryContext(Rule rule, Path outputDirectory) {
    this.rule = rule;
    this.outputDirectory = outputDirectory;
    AggregatingAttributeMapper attrs = AggregatingAttributeMapper.of(rule);
    ImmutableMap.Builder<String, Object> attrBuilder = new ImmutableMap.Builder<>();
    for (String name : attrs.getAttributeNames()) {
      if (!name.equals("$local")) {
        Type<?> type = attrs.getAttributeType(name);
        Object val = attrs.get(name, type);
        attrBuilder.put(
            attributeToSkylark(name),
            val == null
                ? Runtime.NONE
                // Attribute values should be type safe
                : SkylarkType.convertToSkylark(val, null));
      }
    }
    attrObject = new SkylarkClassObject(attrBuilder.build(), "No such attribute '%s'");
  }

  @SkylarkCallable(
    name = "attr",
    structField = true,
    doc =
        "A struct to access the values of the attributes. The values are provided by "
            + "the user (if not, a default value is used)."
  )
  public SkylarkClassObject getAttr() {
    return attrObject;
  }

  @SkylarkCallable(
    name = "path",
    doc =
        "Returns a path from a string. If the path is relative, it will resolved relative "
            + "to the output directory."
  )
  public SkylarkPath path(String path) {
    PathFragment pathFragment = new PathFragment(path);
    if (pathFragment.isAbsolute()) {
      return new SkylarkPath(outputDirectory.getFileSystem().getPath(path));
    } else {
      return new SkylarkPath(outputDirectory.getRelative(pathFragment));
    }
  }

  @SkylarkCallable(
    name = "symlink",
    doc =
        "Create a symlink on the filesystem, the destination of the symlink should be in the "
            + "output directory."
  )
  public void symlink(SkylarkPath from, SkylarkPath to) throws RepositoryFunctionException {
    try {
      to.path.createSymbolicLink(from.path);
    } catch (IOException e) {
      throw new RepositoryFunctionException(
          new IOException(
              "Could not create symlink from " + from + " to " + to + ": " + e.getMessage(), e),
          Transience.TRANSIENT);
    }
  }

  @Override
  public String toString() {
    return "repository_ctx[" + rule.getLabel() + "]";
  }
}
