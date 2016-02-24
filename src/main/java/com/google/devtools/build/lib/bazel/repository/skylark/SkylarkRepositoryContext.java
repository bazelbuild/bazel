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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.AggregatingAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction.RepositoryFunctionException;
import com.google.devtools.build.lib.skyframe.FileSymlinkException;
import com.google.devtools.build.lib.skyframe.FileValue;
import com.google.devtools.build.lib.skyframe.InconsistentFilesystemException;
import com.google.devtools.build.lib.skyframe.PackageLookupValue;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.ClassObject.SkylarkClassObject;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.SkylarkType;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;

import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;

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
  private final SkylarkOS osObject;
  private final Environment env;

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
   * Create a new context (ctx) object for a skylark repository rule ({@code rule} argument).
   */
  SkylarkRepositoryContext(
      Rule rule, Path outputDirectory, Environment environment, Map<String, String> env) {
    this.rule = rule;
    this.outputDirectory = outputDirectory;
    this.env = environment;
    this.osObject = new SkylarkOS(env);
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
        "Returns a path from a string or a label. If the path is relative, it will resolve "
            + "relative to the output directory. If the path is a label, it will resolve to "
            + "the path of the corresponding file. Note that remote repositories are executed "
            + "during the analysis phase and thus cannot depends on a target result (the "
            + "label should point to a non-generated file)."
  )
  public SkylarkPath path(Object path) throws EvalException {
    return getPath("path()", path);
  }

  private SkylarkPath getPath(String method, Object path) throws EvalException {
    if (path instanceof String) {
      PathFragment pathFragment = new PathFragment(path.toString());
      if (pathFragment.isAbsolute()) {
        return new SkylarkPath(outputDirectory.getFileSystem().getPath(path.toString()));
      } else {
        return new SkylarkPath(outputDirectory.getRelative(pathFragment));
      }
    } else if (path instanceof Label) {
      SkylarkPath result = getPathFromLabel((Label) path);
      if (result == null) {
        SkylarkRepositoryFunction.restart();
      }
      return result;
    } else if (path instanceof SkylarkPath) {
      return (SkylarkPath) path;
    } else {
      throw new EvalException(Location.BUILTIN, method + " can only take a string or a label.");
    }
  }

  @SkylarkCallable(
    name = "symlink",
    doc =
        "Create a symlink on the filesystem, the destination of the symlink should be in the "
            + "output directory. <code>from</code> can also be a label to a file."
  )
  public void symlink(Object from, Object to) throws RepositoryFunctionException, EvalException {
    SkylarkPath fromPath = getPath("symlink()", from);
    SkylarkPath toPath = getPath("symlink()", to);
    try {
      checkInOutputDirectory(toPath);
      makeDirectories(toPath.path);
      toPath.path.createSymbolicLink(fromPath.path);
    } catch (IOException e) {
      throw new RepositoryFunctionException(
          new IOException(
              "Could not create symlink from " + fromPath + " to " + toPath + ": " + e.getMessage(),
              e),
          Transience.TRANSIENT);
    }
  }

  private void checkInOutputDirectory(SkylarkPath path) throws RepositoryFunctionException {
    if (!path.path.getPathString().startsWith(outputDirectory.getPathString())) {
      throw new RepositoryFunctionException(
          new IOException("Cannot write outside of the output directory for path " + path),
          Transience.TRANSIENT);
    }
  }

  @SkylarkCallable(name = "file", documented = false)
  public void createFile(Object path) throws RepositoryFunctionException, EvalException {
    createFile(path, "");
  }

  @SkylarkCallable(
    name = "file",
    doc = "Generate a file in the output directory with the provided content"
  )
  public void createFile(Object path, String content)
      throws RepositoryFunctionException, EvalException {
    SkylarkPath p = getPath("file()", path);
    try {
      checkInOutputDirectory(p);
      makeDirectories(p.path);
      try (OutputStream stream = p.path.getOutputStream()) {
        stream.write(content.getBytes(StandardCharsets.UTF_8));
      }
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }
  }

  // Create parent directories for the given path
  private void makeDirectories(Path path) throws IOException {
    if (!path.isRootDirectory()) {
      Path parent = path.getParentDirectory();
      if (!parent.exists()) {
        makeDirectories(path.getParentDirectory());
        parent.createDirectory();
      }
    }
  }

  @SkylarkCallable(
    name = "os",
    structField = true,
    doc = "A struct to access information from the system "
  )
  public SkylarkOS getOS() {
    return osObject;
  }

  @SkylarkCallable(
    name = "execute",
    doc =
        "Executes the command given by the list of arguments. The execution time of the command"
            + " is limited by <code>timeout</code> (in seconds, default 600 seconds). This method"
            + " returns an <code>exec_result</code> structure containing the output of the"
            + " command."
  )
  public SkylarkExecutionResult execute(List<Object> arguments, long timeout) throws EvalException {
    return SkylarkExecutionResult.execute(arguments, timeout / 1000);
  }

  @SkylarkCallable(name = "execute", documented = false)
  public SkylarkExecutionResult execute(List<Object> arguments) throws EvalException {
    return SkylarkExecutionResult.execute(arguments, 600000);
  }

  @SkylarkCallable(
    name = "which",
    doc =
        "Returns the path of the corresponding program or None "
            + "if there is no such program in the path"
  )
  public Object which(String program) throws EvalException {
    if (program.contains("/") || program.contains("\\")) {
      throw new EvalException(
          Location.BUILTIN,
          "Program argument of which() may not contains a / or a \\ ('" + program + "' given)");
    }
    for (String p : pathEnv) {
      PathFragment fragment = new PathFragment(p);
      if (fragment.isAbsolute()) {
        // We ignore relative path as they don't mean much here (relative to where? the workspace
        // root?).
        Path path = outputDirectory.getFileSystem().getPath(fragment).getChild(program);
        try {
          if (path.exists() && path.isExecutable()) {
            return new SkylarkPath(path);
          }
        } catch (IOException e) {
          // IOException when checking executable file means we cannot read the file data so
          // we cannot executes it, swallow the exception.
        }
      }
    }
    return Runtime.NONE;
  }

  // This is non final so that test can overwrite it.
  private static ImmutableList<String> pathEnv = getPathEnvironment();

  @VisibleForTesting
  static void setPathEnvironment(String... pathEnv) {
    SkylarkRepositoryContext.pathEnv = ImmutableList.<String>copyOf(pathEnv);
  }

  private static ImmutableList<String> getPathEnvironment() {
    String pathEnv = System.getenv("PATH");
    if (pathEnv == null) {
      return ImmutableList.of();
    }
    return ImmutableList.copyOf(pathEnv.split(File.pathSeparator));
  }

  @Override
  public String toString() {
    return "repository_ctx[" + rule.getLabel() + "]";
  }

  // Resolve the label given by value into a file path.
  private SkylarkPath getPathFromLabel(Label label) throws EvalException {
    // Look for package.
    SkyKey pkgSkyKey = PackageLookupValue.key(label.getPackageIdentifier());
    PackageLookupValue pkgLookupValue = (PackageLookupValue) env.getValue(pkgSkyKey);
    if (pkgLookupValue == null) {
      return null;
    }
    if (!pkgLookupValue.packageExists()) {
      throw new EvalException(
          Location.BUILTIN, "Unable to load package for " + label + ": not found.");
    }

    // And now for the file
    Path packageRoot = pkgLookupValue.getRoot();
    RootedPath rootedPath = RootedPath.toRootedPath(packageRoot, label.toPathFragment());
    SkyKey fileSkyKey = FileValue.key(rootedPath);
    FileValue fileValue = null;
    try {
      fileValue =
          (FileValue)
              env.getValueOrThrow(
                  fileSkyKey,
                  IOException.class,
                  FileSymlinkException.class,
                  InconsistentFilesystemException.class);
    } catch (IOException | FileSymlinkException | InconsistentFilesystemException e) {
      throw new EvalException(Location.BUILTIN, new IOException(e));
    }

    if (fileValue == null) {
      return null;
    }
    if (!fileValue.isFile()) {
      throw new EvalException(
          Location.BUILTIN, "Not a file: " + rootedPath.asPath().getPathString());
    }

    return new SkylarkPath(rootedPath.asPath());
  }
}
