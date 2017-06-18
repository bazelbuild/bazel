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
import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.bazel.repository.DecompressorDescriptor;
import com.google.devtools.build.lib.bazel.repository.DecompressorValue;
import com.google.devtools.build.lib.bazel.repository.cache.RepositoryCache.KeyType;
import com.google.devtools.build.lib.bazel.repository.downloader.HttpDownloader;
import com.google.devtools.build.lib.bazel.repository.downloader.HttpUtils;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.NativeClassObjectConstructor;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.SkylarkClassObject;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction.RepositoryFunctionException;
import com.google.devtools.build.lib.rules.repository.WorkspaceAttributeMapper;
import com.google.devtools.build.lib.skyframe.FileSymlinkException;
import com.google.devtools.build.lib.skyframe.FileValue;
import com.google.devtools.build.lib.skyframe.InconsistentFilesystemException;
import com.google.devtools.build.lib.skyframe.PackageLookupValue;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.ParamType;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkType;
import com.google.devtools.build.lib.util.StringUtilities;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Skylark API for the repository_rule's context. */
@SkylarkModule(
  name = "repository_ctx",
  category = SkylarkModuleCategory.BUILTIN,
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
  private final HttpDownloader httpDownloader;
  private final Map<String, String> markerData;

  /**
   * Create a new context (repository_ctx) object for a skylark repository rule ({@code rule}
   * argument).
   */
  SkylarkRepositoryContext(Rule rule, Path outputDirectory, Environment environment,
      Map<String, String> env, HttpDownloader httpDownloader, Map<String, String> markerData)
      throws EvalException {
    this.rule = rule;
    this.outputDirectory = outputDirectory;
    this.env = environment;
    this.osObject = new SkylarkOS(env);
    this.httpDownloader = httpDownloader;
    this.markerData = markerData;
    WorkspaceAttributeMapper attrs = WorkspaceAttributeMapper.of(rule);
    ImmutableMap.Builder<String, Object> attrBuilder = new ImmutableMap.Builder<>();
    for (String name : attrs.getAttributeNames()) {
      if (!name.equals("$local")) {
        Object val = attrs.getObject(name);
        attrBuilder.put(
            Attribute.getSkylarkName(name),
            val == null
                ? Runtime.NONE
                // Attribute values should be type safe
                : SkylarkType.convertToSkylark(val, null));
      }
    }
    attrObject = NativeClassObjectConstructor.STRUCT.create(
        attrBuilder.build(), "No such attribute '%s'");
  }

  @SkylarkCallable(
      name = "name",
      structField = true,
      doc = "The name of the external repository created by this rule."
  )
  public String getName() {
    return rule.getName();
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
            + "relative to the repository directory. If the path is a label, it will resolve to "
            + "the path of the corresponding file. Note that remote repositories are executed "
            + "during the analysis phase and thus cannot depends on a target result (the "
            + "label should point to a non-generated file)."
  )
  public SkylarkPath path(Object path) throws EvalException, InterruptedException {
    return getPath("path()", path);
  }

  private SkylarkPath getPath(String method, Object path)
      throws EvalException, InterruptedException {
    if (path instanceof String) {
      PathFragment pathFragment = PathFragment.create(path.toString());
      return new SkylarkPath(pathFragment.isAbsolute()
          ? outputDirectory.getFileSystem().getPath(path.toString())
          : outputDirectory.getRelative(pathFragment));
    } else if (path instanceof Label) {
      return getPathFromLabel((Label) path);
    } else if (path instanceof SkylarkPath) {
      return (SkylarkPath) path;
    } else {
      throw new EvalException(Location.BUILTIN, method + " can only take a string or a label.");
    }
  }

  @SkylarkCallable(
    name = "symlink",
    doc = "Create a symlink on the filesystem.",
    parameters = {
      @Param(
        name = "from",
        allowedTypes = {
          @ParamType(type = String.class),
          @ParamType(type = Label.class),
          @ParamType(type = SkylarkPath.class)
        },
        doc = "path to which the created symlink should point to."
      ),
      @Param(
        name = "to",
        allowedTypes = {
          @ParamType(type = String.class),
          @ParamType(type = Label.class),
          @ParamType(type = SkylarkPath.class)
        },
        doc = "path of the symlink to create, relative to the repository directory."
      ),
    }
  )
  public void symlink(Object from, Object to)
      throws RepositoryFunctionException, EvalException, InterruptedException {
    SkylarkPath fromPath = getPath("symlink()", from);
    SkylarkPath toPath = getPath("symlink()", to);
    try {
      checkInOutputDirectory(toPath);
      makeDirectories(toPath.getPath());
      toPath.getPath().createSymbolicLink(fromPath.getPath());
    } catch (IOException e) {
      throw new RepositoryFunctionException(
          new IOException(
              "Could not create symlink from " + fromPath + " to " + toPath + ": " + e.getMessage(),
              e),
          Transience.TRANSIENT);
    }
  }

  private void checkInOutputDirectory(SkylarkPath path) throws RepositoryFunctionException {
    if (!path.getPath().getPathString().startsWith(outputDirectory.getPathString())) {
      throw new RepositoryFunctionException(
          new EvalException(
              Location.fromFile(path.getPath()),
              "Cannot write outside of the repository directory for path " + path),
          Transience.PERSISTENT);
    }
  }

  @SkylarkCallable(
    name = "file",
    doc = "Generate a file in the repository directory with the provided content.",
    parameters = {
      @Param(
        name = "path",
        allowedTypes = {
          @ParamType(type = String.class),
          @ParamType(type = Label.class),
          @ParamType(type = SkylarkPath.class)
        },
        doc = "path of the file to create, relative to the repository directory."
      ),
      @Param(
        name = "content",
        type = String.class,
        named = true,
        defaultValue = "''",
        doc = "the content of the file to create, empty by default."
      ),
      @Param(
        name = "executable",
        named = true,
        type = Boolean.class,
        defaultValue = "True",
        doc = "set the executable flag on the created file, true by default."
      ),
    }
  )
  public void createFile(Object path, String content, Boolean executable)
      throws RepositoryFunctionException, EvalException, InterruptedException {
    SkylarkPath p = getPath("file()", path);
    try {
      checkInOutputDirectory(p);
      makeDirectories(p.getPath());
      try (OutputStream stream = p.getPath().getOutputStream()) {
        stream.write(content.getBytes(StandardCharsets.UTF_8));
      }
      if (executable) {
        p.getPath().setExecutable(true);
      }
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }
  }

  @SkylarkCallable(
    name = "template",
    doc =
        "Generate a new file using a <code>template</code>. Every occurrence in "
            + "<code>template</code> of a key of <code>substitutions</code> will be replaced by "
            + "the corresponding value. The result is written in <code>path</code>. An optional"
            + "<code>executable</code> argument (default to true) can be set to turn on or off"
            + "the executable bit.",
    parameters = {
      @Param(
        name = "path",
        allowedTypes = {
          @ParamType(type = String.class),
          @ParamType(type = Label.class),
          @ParamType(type = SkylarkPath.class)
        },
        doc = "path of the file to create, relative to the repository directory."
      ),
      @Param(
        name = "template",
        allowedTypes = {
          @ParamType(type = String.class),
          @ParamType(type = Label.class),
          @ParamType(type = SkylarkPath.class)
        },
        doc = "path to the template file."
      ),
      @Param(
        name = "substitutions",
        type = SkylarkDict.class,
        defaultValue = "{}",
        named = true,
        doc = "substitutions to make when expanding the template."
      ),
      @Param(
        name = "executable",
        type = Boolean.class,
        defaultValue = "True",
        named = true,
        doc = "set the executable flag on the created file, true by default."
      ),
    }
  )
  public void createFileFromTemplate(
      Object path, Object template, SkylarkDict<String, String> substitutions, Boolean executable)
      throws RepositoryFunctionException, EvalException, InterruptedException {
    SkylarkPath p = getPath("template()", path);
    SkylarkPath t = getPath("template()", template);
    try {
      checkInOutputDirectory(p);
      makeDirectories(p.getPath());
      String tpl = FileSystemUtils.readContent(t.getPath(), StandardCharsets.UTF_8);
      for (Map.Entry<String, String> substitution : substitutions.entrySet()) {
        tpl =
            StringUtilities.replaceAllLiteral(tpl, substitution.getKey(), substitution.getValue());
      }
      try (OutputStream stream = p.getPath().getOutputStream()) {
        stream.write(tpl.getBytes(StandardCharsets.UTF_8));
      }
      if (executable) {
        p.getPath().setExecutable(true);
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
    doc = "A struct to access information from the system."
  )
  public SkylarkOS getOS() {
    return osObject;
  }

  private void createDirectory(Path directory) throws RepositoryFunctionException {
    try {
      if (!directory.exists()) {
        makeDirectories(directory);
        directory.createDirectory();
      }
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }
  }

  @SkylarkCallable(
      name = "execute",
      doc =
          "Executes the command given by the list of arguments. The execution time of the command"
              + " is limited by <code>timeout</code> (in seconds, default 600 seconds). This method"
              + " returns an <code>exec_result</code> structure containing the output of the"
              + " command. The <code>environment</code> map can be used to override some"
              + " environment variables to be passed to the process.",
      parameters = {
          @Param(
              name = "arguments",
              type = SkylarkList.class,
              doc =
                  "List of arguments, the first element should be the path to the program to "
                      + "execute."
          ),
          @Param(
              name = "timeout",
              type = Integer.class,
              named = true,
              defaultValue = "600",
              doc = "maximum duration of the command in seconds (default is 600 seconds)."
          ),
          @Param(
              name = "environment",
              type = SkylarkDict.class,
              defaultValue = "{}",
              named = true,
              doc = "force some environment variables to be set to be passed to the process."
          ),
          @Param(
              name = "quiet",
              type = Boolean.class,
              defaultValue = "True",
              named = true,
              doc = "If stdout and stderr should be printed to the terminal."
          ),
      }
  )
  public SkylarkExecutionResult execute(
      SkylarkList<Object> arguments, Integer timeout, SkylarkDict<String, String> environment,
      boolean quiet)
      throws EvalException, RepositoryFunctionException {
    createDirectory(outputDirectory);
    return SkylarkExecutionResult.builder(osObject.getEnvironmentVariables())
        .addArguments(arguments)
        .setDirectory(outputDirectory.getPathFile())
        .addEnvironmentVariables(environment)
        .setTimeout(timeout.longValue() * 1000)
        .setQuiet(quiet)
        .execute();
  }

  @SkylarkCallable(
    name = "which",
    doc =
        "Returns the path of the corresponding program or None "
            + "if there is no such program in the path",
    allowReturnNones = true
  )
  public SkylarkPath which(String program) throws EvalException {
    if (program.contains("/") || program.contains("\\")) {
      throw new EvalException(
          Location.BUILTIN,
          "Program argument of which() may not contains a / or a \\ ('" + program + "' given)");
    }
    for (String p : getPathEnvironment()) {
      PathFragment fragment = PathFragment.create(p);
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
    return null;
  }

  @SkylarkCallable(
    name = "download",
    doc = "Download a file to the output path for the provided url.",
    parameters = {
      @Param(
        name = "url",
        allowedTypes = {
          @ParamType(type = String.class),
          @ParamType(type = SkylarkList.class, generic1 = String.class),
        },
        named = true,
        doc = "List of mirror URLs referencing the same file."
      ),
      @Param(
        name = "output",
        allowedTypes = {
          @ParamType(type = String.class),
          @ParamType(type = Label.class),
          @ParamType(type = SkylarkPath.class)
        },
        defaultValue = "''",
        named = true,
        doc = "path to the output file, relative to the repository directory."
      ),
      @Param(
        name = "sha256",
        type = String.class,
        defaultValue = "''",
        named = true,
        doc =
            "the expected SHA-256 hash of the file downloaded."
                + " This must match the SHA-256 hash of the file downloaded. It is a security risk"
                + " to omit the SHA-256 as remote files can change. At best omitting this field"
                + " will make your build non-hermetic. It is optional to make development easier"
                + " but should be set before shipping."
      ),
      @Param(
        name = "executable",
        type = Boolean.class,
        defaultValue = "False",
        named = true,
        doc = "set the executable flag on the created file, false by default."
      ),
    }
  )
  public void download(
      Object url, Object output, String sha256, Boolean executable)
          throws RepositoryFunctionException, EvalException, InterruptedException {
    validateSha256(sha256);
    List<URL> urls = getUrls(url);
    SkylarkPath outputPath = getPath("download()", output);
    try {
      checkInOutputDirectory(outputPath);
      makeDirectories(outputPath.getPath());
      httpDownloader.download(
          urls,
          sha256,
          Optional.<String>absent(),
          outputPath.getPath(),
          env.getListener(),
          osObject.getEnvironmentVariables());
      if (executable) {
        outputPath.getPath().setExecutable(true);
      }
    } catch (InterruptedException e) {
      throw new RepositoryFunctionException(
          new IOException("thread interrupted"), Transience.TRANSIENT);
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }
  }

  @SkylarkCallable(
    name = "download_and_extract",
    doc = "Download a file to the output path for the provided url, and extract it.",
    parameters = {
      @Param(
        name = "url",
        allowedTypes = {
          @ParamType(type = String.class),
          @ParamType(type = SkylarkList.class, generic1 = String.class),
        },
        named = true,
        doc = "List of mirror URLs referencing the same file."
      ),
      @Param(
        name = "output",
        allowedTypes = {
          @ParamType(type = String.class),
          @ParamType(type = Label.class),
          @ParamType(type = SkylarkPath.class)
        },
        defaultValue = "''",
        named = true,
        doc =
            "path to the directory where the archive will be unpacked,"
                + " relative to the repository directory."
      ),
      @Param(
        name = "sha256",
        type = String.class,
        defaultValue = "''",
        named = true,
        doc =
            "the expected SHA-256 hash of the file downloaded."
                + " This must match the SHA-256 hash of the file downloaded. It is a security risk"
                + " to omit the SHA-256 as remote files can change. At best omitting this field"
                + " will make your build non-hermetic. It is optional to make development easier"
                + " but should be set before shipping."
      ),
      @Param(
        name = "type",
        type = String.class,
        defaultValue = "''",
        named = true,
        doc =
            "the archive type of the downloaded file."
                + " By default, the archive type is determined from the file extension of the URL."
                + " If the file has no extension, you can explicitly specify either \"zip\","
                + " \"jar\", \"war\", \"tar.gz\", \"tgz\", \"tar.bz2\", or \"tar.xz\" here."
      ),
      @Param(
        name = "stripPrefix",
        type = String.class,
        defaultValue = "''",
        named = true,
        doc =
            "a directory prefix to strip from the extracted files."
                + "\nMany archives contain a top-level directory that contains all files in the"
                + " archive. Instead of needing to specify this prefix over and over in the"
                + " <code>build_file</code>, this field can be used to strip it from extracted"
                + " files."
      ),
    }
  )
  public void downloadAndExtract(
      Object url, Object output, String sha256, String type, String stripPrefix)
          throws RepositoryFunctionException, InterruptedException, EvalException {
    validateSha256(sha256);
    List<URL> urls = getUrls(url);

    // Download to outputDirectory and delete it after extraction
    SkylarkPath outputPath = getPath("download_and_extract()", output);
    checkInOutputDirectory(outputPath);
    createDirectory(outputPath.getPath());

    Path downloadedPath;
    try {
      downloadedPath =
          httpDownloader.download(
              urls,
              sha256,
              Optional.of(type),
              outputPath.getPath(),
              env.getListener(),
              osObject.getEnvironmentVariables());
    } catch (InterruptedException e) {
      throw new RepositoryFunctionException(
          new IOException("thread interrupted"), Transience.TRANSIENT);
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }
    DecompressorValue.decompress(
        DecompressorDescriptor.builder()
            .setTargetKind(rule.getTargetKind())
            .setTargetName(rule.getName())
            .setArchivePath(downloadedPath)
            .setRepositoryPath(outputPath.getPath())
            .setPrefix(stripPrefix)
            .build());
    try {
      if (downloadedPath.exists()) {
        downloadedPath.delete();
      }
    } catch (IOException e) {
      throw new RepositoryFunctionException(
          new IOException(
              "Couldn't delete temporary file (" + downloadedPath.getPathString() + ")", e),
          Transience.TRANSIENT);
    }
  }

  private static void validateSha256(String sha256) throws RepositoryFunctionException {
    if (!sha256.isEmpty() && !KeyType.SHA256.isValid(sha256)) {
      throw new RepositoryFunctionException(
          new IOException("Invalid SHA256 checksum"), Transience.TRANSIENT);
    }
  }

  private static List<URL> getUrls(Object urlOrList) throws RepositoryFunctionException {
    List<String> urlStrings;
    if (urlOrList instanceof String) {
      urlStrings = ImmutableList.of((String) urlOrList);
    } else {
      @SuppressWarnings("unchecked")
      List<String> list = (List<String>) urlOrList;
      urlStrings = list;
    }
    if (urlStrings.isEmpty()) {
      throw new RepositoryFunctionException(new IOException("urls not set"), Transience.PERSISTENT);
    }
    List<URL> urls = new ArrayList<>();
    for (String urlString : urlStrings) {
      URL url;
      try {
        url = new URL(urlString);
      } catch (MalformedURLException e) {
        throw new RepositoryFunctionException(
            new IOException("Bad URL: " + urlString), Transience.PERSISTENT);
      }
      if (!HttpUtils.isUrlSupportedByDownloader(url)) {
        throw new RepositoryFunctionException(
            new IOException("Unsupported protocol: " + url.getProtocol()), Transience.PERSISTENT);
      }
      urls.add(url);
    }
    return urls;
  }

  // This is just for test to overwrite the path environment
  private static ImmutableList<String> pathEnv = null;

  @VisibleForTesting
  static void setPathEnvironment(String... pathEnv) {
    SkylarkRepositoryContext.pathEnv = ImmutableList.<String>copyOf(pathEnv);
  }

  private ImmutableList<String> getPathEnvironment() {
    if (pathEnv != null) {
      return pathEnv;
    }
    String pathEnviron = osObject.getEnvironmentVariables().get("PATH");
    if (pathEnviron == null) {
      return ImmutableList.of();
    }
    return ImmutableList.copyOf(pathEnviron.split(File.pathSeparator));
  }

  @Override
  public String toString() {
    return "repository_ctx[" + rule.getLabel() + "]";
  }

  private static RootedPath getRootedPathFromLabel(Label label, Environment env)
      throws InterruptedException, EvalException {
    // Look for package.
    if (label.getPackageIdentifier().getRepository().isDefault()) {
      try {
        label = Label.create(label.getPackageIdentifier().makeAbsolute(), label.getName());
      } catch (LabelSyntaxException e) {
        throw new AssertionError(e); // Can't happen because the input label is valid
      }
    }
    SkyKey pkgSkyKey = PackageLookupValue.key(label.getPackageIdentifier());
    PackageLookupValue pkgLookupValue = (PackageLookupValue) env.getValue(pkgSkyKey);
    if (pkgLookupValue == null) {
      throw SkylarkRepositoryFunction.restart();
    }
    if (!pkgLookupValue.packageExists()) {
      throw new EvalException(Location.BUILTIN,
          "Unable to load package for " + label + ": not found.");
    }

    // And now for the file
    Path packageRoot = pkgLookupValue.getRoot();
    return RootedPath.toRootedPath(packageRoot, label.toPathFragment());
  }

  // Resolve the label given by value into a file path.
  private SkylarkPath getPathFromLabel(Label label) throws EvalException, InterruptedException {
    RootedPath rootedPath = getRootedPathFromLabel(label, env);
    SkyKey fileSkyKey = FileValue.key(rootedPath);
    FileValue fileValue = null;
    try {
      fileValue = (FileValue) env.getValueOrThrow(fileSkyKey, IOException.class,
          FileSymlinkException.class, InconsistentFilesystemException.class);
    } catch (IOException | FileSymlinkException | InconsistentFilesystemException e) {
      throw new EvalException(Location.BUILTIN, e);
    }

    if (fileValue == null) {
      throw SkylarkRepositoryFunction.restart();
    }
    if (!fileValue.isFile()) {
      throw new EvalException(Location.BUILTIN,
          "Not a file: " + rootedPath.asPath().getPathString());
    }

    // A label do not contains space so it safe to use as a key.
    markerData.put("FILE:" + label, Integer.toString(fileValue.realFileStateValue().hashCode()));
    return new SkylarkPath(rootedPath.asPath());
  }

  private static boolean verifyLabelMarkerData(String key, String value, Environment env)
      throws InterruptedException {
    Preconditions.checkArgument(key.startsWith("FILE:"));
    try {
      Label label = Label.parseAbsolute(key.substring(5));
      RootedPath rootedPath = getRootedPathFromLabel(label, env);
      SkyKey fileSkyKey = FileValue.key(rootedPath);
      FileValue fileValue = (FileValue) env.getValueOrThrow(fileSkyKey, IOException.class,
          FileSymlinkException.class, InconsistentFilesystemException.class);

      if (fileValue == null || !fileValue.isFile()) {
        return false;
      }

      return Objects.equals(value, Integer.toString(fileValue.realFileStateValue().hashCode()));
    } catch (LabelSyntaxException e) {
      throw new IllegalStateException(
          "Key " + key + " is not a correct file key (should be in form FILE:label)", e);
    } catch (IOException | FileSymlinkException | InconsistentFilesystemException
        | EvalException e) {
      // Consider those exception to be a cause for invalidation
      return false;
    }
  }

  static boolean verifyMarkerDataForFiles(Map<String, String> markerData, Environment env)
      throws InterruptedException {
    for (Map.Entry<String, String> entry : markerData.entrySet()) {
      if (entry.getKey().startsWith("FILE:")) {
        if (!verifyLabelMarkerData(entry.getKey(), entry.getValue(), env)) {
          return false;
        }
      }
    }
    return true;
  }
}
