// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.repository;

import com.google.common.base.Optional;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.BazelLibrary;
import com.google.devtools.build.lib.packages.BuildFileContainsErrorsException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.ParserInputSource;
import com.google.devtools.build.lib.syntax.SkylarkSemantics;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Value of output hashes for the repositories specified in the resolved file designated for this
 * purpose.
 */
public class ResolvedHashesFunction implements SkyFunction {
  public static final String ORIGINAL_RULE_CLASS = "original_rule_class";
  public static final String ORIGINAL_ATTRIBUTES = "original_attributes";
  public static final String RULE_CLASS = "rule_class";
  public static final String ATTRIBUTES = "attributes";
  public static final String OUTPUT_TREE_HASH = "output_tree_hash";
  public static final String REPOSITORIES = "repositories";

  @Override
  @Nullable
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws InterruptedException, SkyFunctionException {

    Optional<RootedPath> resolvedFile =
        RepositoryDelegatorFunction.RESOLVED_FILE_FOR_VERIFICATION.get(env);
    if (resolvedFile == null) {
      return null;
    }
    if (!resolvedFile.isPresent()) {
      return new ResolvedHashesValue(ImmutableMap.<String, String>of());
    }
    SkylarkSemantics skylarkSemantics = PrecomputedValue.SKYLARK_SEMANTICS.get(env);
    if (skylarkSemantics == null) {
      return null;
    }
    FileValue resolvedFileValue = (FileValue) env.getValue(FileValue.key(resolvedFile.get()));
    if (resolvedFileValue == null) {
      return null;
    }
    try {
      if (!resolvedFileValue.exists()) {
        throw new ResolvedHashesFunctionException(
            new NoSuchThingException(
                "Specified file for resolved hashes '" + resolvedFile.get() + "' not found."));
      } else {
        byte[] bytes =
            FileSystemUtils.readWithKnownFileSize(
                resolvedFile.get().asPath(), resolvedFile.get().asPath().getFileSize());
        BuildFileAST ast =
            BuildFileAST.parseSkylarkFile(
                ParserInputSource.create(bytes, resolvedFile.get().asPath().asFragment()),
                env.getListener());
        if (ast.containsErrors()) {
          throw new ResolvedHashesFunctionException(
              new BuildFileContainsErrorsException(
                  Label.EXTERNAL_PACKAGE_IDENTIFIER,
                  "Failed to parse file resolved file for hash verification"));
        }
        com.google.devtools.build.lib.syntax.Environment resolvedEnvironment;
        try (Mutability mutability = Mutability.create("resolved hashes %s", resolvedFile.get())) {
          resolvedEnvironment =
              com.google.devtools.build.lib.syntax.Environment.builder(mutability)
                  .setSemantics(skylarkSemantics)
                  .setGlobals(BazelLibrary.GLOBALS)
                  .build();
          if (!ast.exec(resolvedEnvironment, env.getListener())) {
            throw new ResolvedHashesFunctionException(
                new BuildFileContainsErrorsException(
                    Label.EXTERNAL_PACKAGE_IDENTIFIER,
                    "Failed to evaluate resolved file for hash verification"));
          }
        }
        Object resolved = resolvedEnvironment.lookup("resolved");
        if (resolved == null) {
          throw new ResolvedHashesFunctionException(
              new BuildFileContainsErrorsException(
                  Label.EXTERNAL_PACKAGE_IDENTIFIER,
                  "Symbol 'resolved' not exported in file for hash verification"));
        }
        if (!(resolved instanceof List)) {
          throw new ResolvedHashesFunctionException(
              new BuildFileContainsErrorsException(
                  Label.EXTERNAL_PACKAGE_IDENTIFIER,
                  "Symbol 'resolved' not a list in file for hash verification"));
        }
        // Collect the hases in a mutable map, to be able to detect duplicates and
        // only take the first entry, following the "maybe pattern" of external repositories,
        // adding a repository only if not already present.
        Map<String, String> hashes = new LinkedHashMap<String, String>();
        for (Object entry : (List) resolved) {
          if (entry instanceof Map) {
            Object repositories = ((Map) entry).get(REPOSITORIES);
            if (repositories instanceof List) {
              for (Object repo : (List) repositories) {
                if (repo instanceof Map) {
                  Object hash = ((Map) repo).get(OUTPUT_TREE_HASH);
                  Object attributes = ((Map) repo).get(ATTRIBUTES);
                  if (attributes instanceof Map) {
                    Object name = ((Map) attributes).get("name");
                    if ((name instanceof String) && (hash instanceof String)) {
                      if (!hashes.containsKey((String) name)) {
                        hashes.put((String) name, (String) hash);
                      }
                    }
                  }
                }
              }
            }
          }
        }
        return new ResolvedHashesValue(ImmutableMap.copyOf(hashes));
      }
    } catch (IOException e) {
      throw new ResolvedHashesFunctionException(e);
    }
  }

  @Override
  @Nullable
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  private static final class ResolvedHashesFunctionException extends SkyFunctionException {
    ResolvedHashesFunctionException(IOException e) {
      super(e, SkyFunctionException.Transience.PERSISTENT);
    }

    ResolvedHashesFunctionException(NoSuchThingException e) {
      super(e, SkyFunctionException.Transience.PERSISTENT);
    }
  }
}
