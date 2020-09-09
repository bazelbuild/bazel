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

package com.google.devtools.build.lib.skyframe;

import com.google.common.hash.HashFunction;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.syntax.FileOptions;
import com.google.devtools.build.lib.syntax.Module;
import com.google.devtools.build.lib.syntax.ParserInput;
import com.google.devtools.build.lib.syntax.Resolver;
import com.google.devtools.build.lib.syntax.StarlarkFile;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * A Skyframe function that reads, parses, and resolves the .bzl file denoted by a Label.
 *
 * <p>Given a {@link Label} referencing a Starlark file, loads it as a syntax tree ({@link
 * StarlarkFile}). The Label must be absolute, and must not reference the special {@code external}
 * package. If the file (or the package containing it) doesn't exist, the function doesn't fail, but
 * instead returns a specific {@code NO_FILE} {@link ASTFileLookupValue}.
 */
// TODO(adonovan): rename to BzlParseAndResolveFunction or (later) BzlCompileFunction.
public class ASTFileLookupFunction implements SkyFunction {

  private final PackageFactory packageFactory;
  private final HashFunction hashFunction;

  public ASTFileLookupFunction(PackageFactory packageFactory, HashFunction hashFunction) {
    this.packageFactory = packageFactory;
    this.hashFunction = hashFunction;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {
    try {
      return computeInline(
          (ASTFileLookupValue.Key) skyKey.argument(), env, packageFactory, hashFunction);
    } catch (ASTLookupFailedException e) {
      throw new ASTLookupFunctionException(e);
    }
  }

  static ASTFileLookupValue computeInline(
      ASTFileLookupValue.Key key,
      Environment env,
      PackageFactory packageFactory,
      HashFunction digestHashFunction)
      throws ASTLookupFailedException, InterruptedException {
    byte[] bytes;
    byte[] digest;
    String inputName;

    if (key.kind == ASTFileLookupValue.Kind.EMPTY_PRELUDE) {
      // Default prelude is empty.
      bytes = new byte[] {};
      digest = null;
      inputName = "<default prelude>";
    } else {

      // Obtain the file.
      RootedPath rootedPath = RootedPath.toRootedPath(key.root, key.label.toPathFragment());
      SkyKey fileSkyKey = FileValue.key(rootedPath);
      FileValue fileValue = null;
      try {
        fileValue = (FileValue) env.getValueOrThrow(fileSkyKey, IOException.class);
      } catch (IOException e) {
        throw new ASTLookupFailedException(e, Transience.PERSISTENT);
      }
      if (fileValue == null) {
        return null;
      }

      if (fileValue.exists()) {
        if (!fileValue.isFile()) {
          return fileValue.isDirectory()
              ? ASTFileLookupValue.noFile("cannot load '%s': is a directory", key.label)
              : ASTFileLookupValue.noFile(
                  "cannot load '%s': not a regular file (dangling link?)", key.label);
        }

        // Read the file.
        Path path = rootedPath.asPath();
        try {
          bytes =
              fileValue.isSpecialFile()
                  ? FileSystemUtils.readContent(path)
                  : FileSystemUtils.readWithKnownFileSize(path, fileValue.getSize());
        } catch (IOException e) {
          throw new ASTLookupFailedException(e, Transience.TRANSIENT);
        }
        digest = fileValue.getDigest(); // may be null
        inputName = path.toString();
      } else {
        if (key.kind == ASTFileLookupValue.Kind.PRELUDE) {
          // A non-existent prelude is fine.
          bytes = new byte[] {};
          digest = null;
          inputName = "<default prelude>";
        } else {
          return ASTFileLookupValue.noFile("cannot load '%s': no such file", key.label);
        }
      }
    }

    // Compute digest if we didn't already get it from a fileValue.
    if (digest == null) {
      digest = digestHashFunction.hashBytes(bytes).asBytes();
    }

    StarlarkSemantics semantics = PrecomputedValue.STARLARK_SEMANTICS.get(env);
    if (semantics == null) {
      return null;
    }

    Map<String, Object> predeclared;
    if (key.kind == ASTFileLookupValue.Kind.BUILTINS) {
      predeclared = packageFactory.getBuiltinsBzlEnv();
    } else {
      // Use the predeclared environment for BUILD-loaded bzl files, ignoring injection. It is not
      // the right env for the actual evaluation of BUILD-loaded bzl files because it doesn't
      // map to the injected symbols. But the names of the symbols are the same, and the names are
      // all we need to do symbol resolution (modulo FlagGuardedValues -- see TODO in
      // PackageFactory.createBuildBzlEnvUsingInjection()).
      //
      // For WORKSPACE-loaded bzl files, the env isn't quite right not because of injection but
      // because the "native" object is different. But A) that will be fixed with #11954, and B) we
      // don't care for the same reason as above.
      predeclared = packageFactory.getUninjectedBuildBzlEnv();
    }

    // We have all deps. Parse, resolve, and return.
    ParserInput input = ParserInput.fromLatin1(bytes, inputName);
    FileOptions options =
        FileOptions.builder()
            // TODO(adonovan): add this, so that loads can normally be truly local.
            // .loadBindsGlobally(key.isPrelude())
            .restrictStringEscapes(
                semantics.getBool(BuildLanguageOptions.INCOMPATIBLE_RESTRICT_STRING_ESCAPES))
            .build();
    StarlarkFile file = StarlarkFile.parse(input, options);
    Module module = Module.withPredeclared(semantics, predeclared);
    Resolver.resolveFile(file, module);
    Event.replayEventsOn(env.getListener(), file.errors()); // TODO(adonovan): fail if !ok()?
    return ASTFileLookupValue.withFile(file, digest);
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  static final class ASTLookupFailedException extends Exception {
    private final Transience transience;

    private ASTLookupFailedException(Exception cause, Transience transience) {
      super(cause.getMessage(), cause);
      this.transience = transience;
    }

    Transience getTransience() {
      return transience;
    }
  }

  private static final class ASTLookupFunctionException extends SkyFunctionException {
    private ASTLookupFunctionException(ASTLookupFailedException cause) {
      super(cause, cause.transience);
    }
  }
}
