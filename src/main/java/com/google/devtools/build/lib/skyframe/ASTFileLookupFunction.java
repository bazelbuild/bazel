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

import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.actions.InconsistentFilesystemException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.syntax.FileOptions;
import com.google.devtools.build.lib.syntax.Module;
import com.google.devtools.build.lib.syntax.ParserInput;
import com.google.devtools.build.lib.syntax.Resolver;
import com.google.devtools.build.lib.syntax.StarlarkFile;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
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

  private final RuleClassProvider ruleClassProvider;
  private final DigestHashFunction digestHashFunction;

  public ASTFileLookupFunction(
      RuleClassProvider ruleClassProvider, DigestHashFunction digestHashFunction) {
    this.ruleClassProvider = ruleClassProvider;
    this.digestHashFunction = digestHashFunction;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {
    try {
      return computeInline(
          (ASTFileLookupValue.Key) skyKey.argument(), env, ruleClassProvider, digestHashFunction);
    } catch (ErrorReadingStarlarkExtensionException e) {
      throw new ASTLookupFunctionException(e, e.getTransience());
    } catch (InconsistentFilesystemException e) {
      throw new ASTLookupFunctionException(e, Transience.PERSISTENT);
    }
  }

  static ASTFileLookupValue computeInline(
      ASTFileLookupValue.Key key,
      Environment env,
      RuleClassProvider ruleClassProvider,
      DigestHashFunction digestHashFunction)
      throws ErrorReadingStarlarkExtensionException, InconsistentFilesystemException,
          InterruptedException {
    // Determine whether the file designated by key.label exists.
    RootedPath rootedPath = RootedPath.toRootedPath(key.root, key.label.toPathFragment());
    SkyKey fileSkyKey = FileValue.key(rootedPath);
    FileValue fileValue = null;
    try {
      fileValue = (FileValue) env.getValueOrThrow(fileSkyKey, IOException.class);
    } catch (IOException e) {
      throw new ErrorReadingStarlarkExtensionException(e, Transience.PERSISTENT);
    }
    if (fileValue == null) {
      return null;
    }
    if (!fileValue.exists()) {
      return ASTFileLookupValue.noFile("cannot load '%s': no such file", key.label);
    }
    if (!fileValue.isFile()) {
      return fileValue.isDirectory()
          ? ASTFileLookupValue.noFile("cannot load '%s': is a directory", key.label)
          : ASTFileLookupValue.noFile(
              "cannot load '%s': not a regular file (dangling link?)", key.label);
    }
    StarlarkSemantics semantics = PrecomputedValue.STARLARK_SEMANTICS.get(env);
    if (semantics == null) {
      return null;
    }

    // Options for scanning, parsing, and resolving a .bzl file (including the prelude).
    FileOptions options =
        FileOptions.builder()
            // TODO(adonovan): add this, so that loads can normally be truly local.
            // .loadBindsGlobally(key.isBUILDPrelude)
            .restrictStringEscapes(semantics.incompatibleRestrictStringEscapes())
            .build();

    // Both the package and the file exist; load and parse the file.
    Path path = rootedPath.asPath();
    StarlarkFile file;
    byte[] digest;
    try {
      byte[] bytes =
          fileValue.isSpecialFile()
              ? FileSystemUtils.readContent(path)
              : FileSystemUtils.readWithKnownFileSize(path, fileValue.getSize());
      digest = getDigestFromFileValueOrFromKnownFileContents(fileValue, bytes, digestHashFunction);
      ParserInput input = ParserInput.fromLatin1(bytes, path.toString());
      file = StarlarkFile.parse(input, options);
    } catch (IOException e) {
      throw new ErrorReadingStarlarkExtensionException(e, Transience.TRANSIENT);
    }

    // resolve (and soon, compile)
    Module module = Module.withPredeclared(semantics, ruleClassProvider.getEnvironment());
    Resolver.resolveFile(file, module);
    Event.replayEventsOn(env.getListener(), file.errors()); // TODO(adonovan): fail if !ok()?

    return ASTFileLookupValue.withFile(file, digest);
  }

  private static byte[] getDigestFromFileValueOrFromKnownFileContents(
      FileValue fileValue, byte[] contents, DigestHashFunction digestHashFunction) {
    byte[] digest = fileValue.getDigest();
    if (digest != null) {
      return digest;
    }
    return digestHashFunction.getHashFunction().hashBytes(contents).asBytes();
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  private static final class ASTLookupFunctionException extends SkyFunctionException {
    private ASTLookupFunctionException(
        ErrorReadingStarlarkExtensionException e, Transience transience) {
      super(e, transience);
    }

    private ASTLookupFunctionException(InconsistentFilesystemException e, Transience transience) {
      super(e, transience);
    }
  }
}
