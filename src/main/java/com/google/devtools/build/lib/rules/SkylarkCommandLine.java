// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.rules;

import com.google.common.base.Function;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.SkylarkFunction.SimpleSkylarkFunction;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkModule;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.syntax.SkylarkSignature;
import com.google.devtools.build.lib.syntax.SkylarkSignature.Param;

import java.util.Map;

/**
 * A Skylark module class to create memory efficient command lines.
 */
@SkylarkModule(name = "cmd_helper", namespace = true,
    doc = "Module for creating memory efficient command lines.")
public class SkylarkCommandLine {

  @SkylarkSignature(name = "join_paths",
      doc = "Creates a single command line argument joining the paths of a set "
          + "of files on the separator string.",
      objectType = SkylarkCommandLine.class,
      returnType = String.class,
      mandatoryPositionals = {
      @Param(name = "separator", type = String.class, doc = "the separator string to join on"),
      @Param(name = "files", type = SkylarkNestedSet.class, generic1 = Artifact.class,
             doc = "the files to concatenate")})
  private static SimpleSkylarkFunction joinPaths =
      new SimpleSkylarkFunction("join_paths") {
    @Override
    public Object call(Map<String, Object> params, Location loc)
        throws EvalException {
      final String separator = (String) params.get("separator");
      final NestedSet<Artifact> artifacts =
          ((SkylarkNestedSet) params.get("files")).getSet(Artifact.class);
      // TODO(bazel-team): lazy evaluate
      return Artifact.joinExecPaths(separator, artifacts);
    }
  };

  // TODO(bazel-team): this method should support sets of objects and substitute all struct fields.
  @SkylarkSignature(name = "template",
      doc = "Transforms a set of files to a list of strings using the template string.",
      objectType = SkylarkCommandLine.class,
      returnType = SkylarkList.class,
      mandatoryPositionals = {
      @Param(name = "items", type = SkylarkNestedSet.class, generic1 = Artifact.class,
          doc = "The set of structs to transform."),
      @Param(name = "template", type = String.class,
          doc = "The template to use for the transformation, <code>%{path}</code> and "
              + "<code>%{short_path}</code> being substituted with the corresponding fields of each"
              + " file.")})
  private static SimpleSkylarkFunction template = new SimpleSkylarkFunction("template") {
    @Override
    public Object call(Map<String, Object> params, Location loc)
        throws EvalException {
      final String template = (String) params.get("template");
      SkylarkNestedSet items = (SkylarkNestedSet) params.get("items");
      return SkylarkList.lazyList(Iterables.transform(items, new Function<Object, String>() {
        @Override
        public String apply(Object input) {
          Artifact artifact = (Artifact) input;
          return template
              .replace("%{path}", artifact.getExecPathString())
              .replace("%{short_path}", artifact.getRootRelativePathString());
        }
      }), String.class);
    }
  };
}
