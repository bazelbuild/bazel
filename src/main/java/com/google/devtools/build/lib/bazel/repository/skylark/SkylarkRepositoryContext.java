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
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.bazel.debug.WorkspaceRuleEvent;
import com.google.devtools.build.lib.bazel.repository.DecompressorDescriptor;
import com.google.devtools.build.lib.bazel.repository.DecompressorValue;
import com.google.devtools.build.lib.bazel.repository.cache.RepositoryCache;
import com.google.devtools.build.lib.bazel.repository.cache.RepositoryCache.KeyType;
import com.google.devtools.build.lib.bazel.repository.downloader.HttpDownloader;
import com.google.devtools.build.lib.bazel.repository.downloader.HttpUtils;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.ExtendedEventHandler.FetchProgress;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.packages.StructProvider;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction.RepositoryFunctionException;
import com.google.devtools.build.lib.rules.repository.WorkspaceAttributeMapper;
import com.google.devtools.build.lib.skylarkbuildapi.repository.SkylarkRepositoryContextApi;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkType;
import com.google.devtools.build.lib.util.OsUtils;
import com.google.devtools.build.lib.util.StringUtilities;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.Symlinks;
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

/** Skylark API for the repository_rule's context. */
public class SkylarkRepositoryContext
    implements SkylarkRepositoryContextApi<RepositoryFunctionException> {

  private final Rule rule;
  private final Path outputDirectory;
  private final StructImpl attrObject;
  private final SkylarkOS osObject;
  private final Environment env;
  private final HttpDownloader httpDownloader;
  private final double timeoutScaling;
  private final Map<String, String> markerData;

  /**
   * Create a new context (repository_ctx) object for a skylark repository rule ({@code rule}
   * argument).
   */
  SkylarkRepositoryContext(
      Rule rule,
      Path outputDirectory,
      Environment environment,
      Map<String, String> env,
      HttpDownloader httpDownloader,
      double timeoutScaling,
      Map<String, String> markerData)
      throws EvalException {
    this.rule = rule;
    this.outputDirectory = outputDirectory;
    this.env = environment;
    this.osObject = new SkylarkOS(env);
    this.httpDownloader = httpDownloader;
    this.timeoutScaling = timeoutScaling;
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
                : SkylarkType.convertToSkylark(val,
                    (com.google.devtools.build.lib.syntax.Environment) null));
      }
    }
    attrObject = StructProvider.STRUCT.create(attrBuilder.build(), "No such attribute '%s'");
  }

  @Override
  public String getName() {
    return rule.getName();
  }

  @Override
  public StructImpl getAttr() {
    return attrObject;
  }

  @Override
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

  @Override
  public void reportProgress(String status) {
    final String message = status == null ? "" : status;
    final String id = "@" + getName();

    env.getListener()
        .post(
            new FetchProgress() {
              @Override
              public String getResourceIdentifier() {
                return id;
              }

              @Override
              public String getProgress() {
                return message;
              }

              @Override
              public boolean isFinished() {
                return false;
              }
            });
  }

  @Override
  public void symlink(Object from, Object to, Location location)
      throws RepositoryFunctionException, EvalException, InterruptedException {
    SkylarkPath fromPath = getPath("symlink()", from);
    SkylarkPath toPath = getPath("symlink()", to);
    WorkspaceRuleEvent w =
        WorkspaceRuleEvent.newSymlinkEvent(
            fromPath.toString(), toPath.toString(), rule.getLabel().toString(), location);
    env.getListener().post(w);
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

  @Override
  public void createFile(Object path, String content, Boolean executable, Location location)
      throws RepositoryFunctionException, EvalException, InterruptedException {
    SkylarkPath p = getPath("file()", path);
    WorkspaceRuleEvent w =
        WorkspaceRuleEvent.newFileEvent(
            p.toString(), content, executable, rule.getLabel().toString(), location);
    env.getListener().post(w);
    try {
      checkInOutputDirectory(p);
      makeDirectories(p.getPath());
      p.getPath().delete();
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

  @Override
  public void createFileFromTemplate(
      Object path,
      Object template,
      SkylarkDict<String, String> substitutions,
      Boolean executable,
      Location location)
      throws RepositoryFunctionException, EvalException, InterruptedException {
    SkylarkPath p = getPath("template()", path);
    SkylarkPath t = getPath("template()", template);
    WorkspaceRuleEvent w =
        WorkspaceRuleEvent.newTemplateEvent(
            p.toString(),
            t.toString(),
            substitutions,
            executable,
            rule.getLabel().toString(),
            location);
    env.getListener().post(w);
    try {
      checkInOutputDirectory(p);
      makeDirectories(p.getPath());
      String tpl = FileSystemUtils.readContent(t.getPath(), StandardCharsets.UTF_8);
      for (Map.Entry<String, String> substitution : substitutions.entrySet()) {
        tpl =
            StringUtilities.replaceAllLiteral(tpl, substitution.getKey(), substitution.getValue());
      }
      p.getPath().delete();
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
    Path parent = path.getParentDirectory();
    if (parent != null) {
      parent.createDirectoryAndParents();
    }
  }

  @Override
  public SkylarkOS getOS(Location location) {
    WorkspaceRuleEvent w = WorkspaceRuleEvent.newOsEvent(rule.getLabel().toString(), location);
    env.getListener().post(w);
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

  @Override
  public SkylarkExecutionResult execute(
      SkylarkList<Object> arguments,
      Integer timeout,
      SkylarkDict<String, String> environment,
      boolean quiet,
      String workingDirectory,
      Location location)
      throws EvalException, RepositoryFunctionException, InterruptedException {
    WorkspaceRuleEvent w =
        WorkspaceRuleEvent.newExecuteEvent(
            arguments,
            timeout,
            osObject.getEnvironmentVariables(),
            environment,
            outputDirectory.getPathString(),
            quiet,
            rule.getLabel().toString(),
            location);
    env.getListener().post(w);
    createDirectory(outputDirectory);

    Path workingDirectoryPath = outputDirectory;
    if (workingDirectory != null && !workingDirectory.isEmpty()) {
      workingDirectoryPath = getPath("execute()", workingDirectory).getPath();
    }
    createDirectory(workingDirectoryPath);
    return SkylarkExecutionResult.builder(osObject.getEnvironmentVariables())
        .addArguments(arguments)
        .setDirectory(workingDirectoryPath.getPathFile())
        .addEnvironmentVariables(environment)
        .setTimeout(Math.round(timeout.longValue() * 1000 * timeoutScaling))
        .setQuiet(quiet)
        .execute();
  }

  @Override
  public SkylarkPath which(String program, Location location) throws EvalException {
    WorkspaceRuleEvent w =
        WorkspaceRuleEvent.newWhichEvent(program, rule.getLabel().toString(), location);
    env.getListener().post(w);
    if (program.contains("/") || program.contains("\\")) {
      throw new EvalException(
          Location.BUILTIN,
          "Program argument of which() may not contains a / or a \\ ('" + program + "' given)");
    }
    try {
      SkylarkPath commandPath = findCommandOnPath(program);
      if (commandPath != null) {
        return commandPath;
      }

      if (!program.endsWith(OsUtils.executableExtension())) {
        program += OsUtils.executableExtension();
        return findCommandOnPath(program);
      }
    } catch (IOException e) {
      // IOException when checking executable file means we cannot read the file data so
      // we cannot execute it, swallow the exception.
    }
    return null;
  }

  private SkylarkPath findCommandOnPath(String program) throws IOException {
    for (String p : getPathEnvironment()) {
      PathFragment fragment = PathFragment.create(p);
      if (fragment.isAbsolute()) {
        // We ignore relative path as they don't mean much here (relative to where? the workspace
        // root?).
        Path path = outputDirectory.getFileSystem().getPath(fragment).getChild(program);
        if (path.exists() && path.isFile(Symlinks.FOLLOW) && path.isExecutable()) {
          return new SkylarkPath(path);
        }
      }
    }
    return null;
  }

  private void warnAboutSha256Error(List<URL> urls, String sha256) {
    // Inform the user immediately, even though the file will still be downloaded.
    // This cannot be done by a regular error event, as all regular events are recorded
    // and only shown once the execution of the repository rule is finished.
    // So we have to provide the information as update on the progress
    String url = "(unknown)";
    if (urls.size() > 0) {
      url = urls.get(0).toString();
    }
    reportProgress("Will fail after download of " + url + ". Invalid SHA256 '" + sha256 + "'");
  }

  @Override
  public StructImpl download(
      Object url, Object output, String sha256, Boolean executable, Location location)
      throws RepositoryFunctionException, EvalException, InterruptedException {
    List<URL> urls = getUrls(url);
    RepositoryFunctionException sha256Validation = validateSha256(sha256, location);
    if (sha256Validation != null) {
      warnAboutSha256Error(urls, sha256);
      sha256 = "";
    }
    SkylarkPath outputPath = getPath("download()", output);
    WorkspaceRuleEvent w =
        WorkspaceRuleEvent.newDownloadEvent(
            urls, output.toString(), sha256, executable, rule.getLabel().toString(), location);
    env.getListener().post(w);
    Path downloadedPath;
    try {
      checkInOutputDirectory(outputPath);
      makeDirectories(outputPath.getPath());
      downloadedPath =
          httpDownloader.download(
              urls,
              sha256,
              Optional.<String>absent(),
              outputPath.getPath(),
              env.getListener(),
              osObject.getEnvironmentVariables(),
              getName());
      if (executable) {
        outputPath.getPath().setExecutable(true);
      }
    } catch (InterruptedException e) {
      throw new RepositoryFunctionException(
          new IOException("thread interrupted"), Transience.TRANSIENT);
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }
    if (sha256Validation != null) {
      throw sha256Validation;
    }
    String finalSha256;
    try {
      finalSha256 = calculateSha256(sha256, downloadedPath);
    } catch (IOException e) {
      throw new RepositoryFunctionException(
          new IOException(
              "Couldn't hash downloaded file (" + downloadedPath.getPathString() + ")", e),
          Transience.PERSISTENT);
    }
    SkylarkDict<String, Object> dict = SkylarkDict.of(null, "sha256", finalSha256);
    return StructProvider.STRUCT.createStruct(dict, null);
  }

  @Override
  public void extract(Object archive, Object output, String stripPrefix, Location location)
      throws RepositoryFunctionException, InterruptedException, EvalException {
    SkylarkPath archivePath = getPath("extract()", archive);

    if (!archivePath.exists()) {
      throw new RepositoryFunctionException(
          new EvalException(
              Location.fromFile(archivePath.getPath()),
              String.format("Archive path '%s' does not exist.", archivePath.toString())),
          Transience.TRANSIENT);
    }

    SkylarkPath outputPath = getPath("extract()", output);
    checkInOutputDirectory(outputPath);

    WorkspaceRuleEvent w =
        WorkspaceRuleEvent.newExtractEvent(
            archive.toString(),
            output.toString(),
            stripPrefix,
            rule.getLabel().toString(),
            location);
    env.getListener().post(w);

    DecompressorValue.decompress(
        DecompressorDescriptor.builder()
            .setTargetKind(rule.getTargetKind())
            .setTargetName(rule.getName())
            .setArchivePath(archivePath.getPath())
            .setRepositoryPath(outputPath.getPath())
            .setPrefix(stripPrefix)
            .build());
  }

  @Override
  public StructImpl downloadAndExtract(
      Object url, Object output, String sha256, String type, String stripPrefix, Location location)
      throws RepositoryFunctionException, InterruptedException, EvalException {
    List<URL> urls = getUrls(url);
    RepositoryFunctionException sha256Validation = validateSha256(sha256, location);
    if (sha256Validation != null) {
      warnAboutSha256Error(urls, sha256);
      sha256 = "";
    }

    WorkspaceRuleEvent w =
        WorkspaceRuleEvent.newDownloadAndExtractEvent(
            urls,
            output.toString(),
            sha256,
            type,
            stripPrefix,
            rule.getLabel().toString(),
            location);

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
              osObject.getEnvironmentVariables(),
              getName());
    } catch (InterruptedException e) {
      env.getListener().post(w);
      throw new RepositoryFunctionException(
          new IOException("thread interrupted"), Transience.TRANSIENT);
    } catch (IOException e) {
      env.getListener().post(w);
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }
    if (sha256Validation != null) {
      throw sha256Validation;
    }
    env.getListener().post(w);
    DecompressorValue.decompress(
        DecompressorDescriptor.builder()
            .setTargetKind(rule.getTargetKind())
            .setTargetName(rule.getName())
            .setArchivePath(downloadedPath)
            .setRepositoryPath(outputPath.getPath())
            .setPrefix(stripPrefix)
            .build());
    String finalSha256 = null;
    try {
      finalSha256 = calculateSha256(sha256, downloadedPath);
    } catch (IOException e) {
      throw new RepositoryFunctionException(
          new IOException(
              "Couldn't hash downloaded file (" + downloadedPath.getPathString() + ")", e),
          Transience.PERSISTENT);
    }
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
    SkylarkDict<String, Object> dict = SkylarkDict.of(null, "sha256", finalSha256);
    return StructProvider.STRUCT.createStruct(dict, null);
  }

  private String calculateSha256(String originalSha, Path path) throws IOException {
    if (!Strings.isNullOrEmpty(originalSha)) {
      // The sha is checked on download, so if we got here, the user provided sha is good
      return originalSha;
    }
    return RepositoryCache.getChecksum(KeyType.SHA256, path);
  }

  private RepositoryFunctionException validateSha256(String sha256, Location loc) {
    if (!sha256.isEmpty() && !KeyType.SHA256.isValid(sha256)) {
      return new RepositoryFunctionException(
          new EvalException(
              loc,
              "Definition of repository "
                  + rule.getName()
                  + ": Syntactically invalid SHA256 checksum: '"
                  + sha256
                  + "' at "
                  + rule.getLocation()),
          Transience.PERSISTENT);
    }
    return null;
  }

  private static ImmutableList<String> checkAllUrls(Iterable<?> urlList) throws EvalException {
    ImmutableList.Builder<String> result = ImmutableList.builder();

    for (Object o : urlList) {
      if (!(o instanceof String)) {
        throw new EvalException(
            null,
            String.format(
                "Expected a string or sequence of strings for 'url' argument, "
                    + "but got '%s' item in the sequence",
                EvalUtils.getDataTypeName(o)));
      }
      result.add((String) o);
    }

    return result.build();
  }

  private static List<URL> getUrls(Object urlOrList)
      throws RepositoryFunctionException, EvalException {
    List<String> urlStrings;
    if (urlOrList instanceof String) {
      urlStrings = ImmutableList.of((String) urlOrList);
    } else {
      urlStrings = checkAllUrls((Iterable<?>) urlOrList);
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

  // Resolve the label given by value into a file path.
  private SkylarkPath getPathFromLabel(Label label) throws EvalException, InterruptedException {
    RootedPath rootedPath = RepositoryFunction.getRootedPathFromLabel(label, env);
    SkyKey fileSkyKey = FileValue.key(rootedPath);
    FileValue fileValue = null;
    try {
      fileValue = (FileValue) env.getValueOrThrow(fileSkyKey, IOException.class);
    } catch (IOException e) {
      throw new EvalException(Location.BUILTIN, e);
    }

    if (fileValue == null) {
      throw RepositoryFunction.restart();
    }
    if (!fileValue.isFile() || fileValue.isSpecialFile()) {
      throw new EvalException(
          Location.BUILTIN, "Not a regular file: " + rootedPath.asPath().getPathString());
    }

    // A label does not contains space so it safe to use as a key.
    try {
      markerData.put("FILE:" + label, RepositoryFunction.fileValueToMarkerValue(fileValue));
    } catch (IOException e) {
      throw new EvalException(Location.BUILTIN, e);
    }
    return new SkylarkPath(rootedPath.asPath());
  }

  /**
   * Try to compute the paths of all attibutes that are labels, including labels in list arguments.
   *
   * <p>The value is ignored, but any missing information from the environment is detected (and an
   * exception thrown). In this way, we can enforce that all arguments are evaluated before we start
   * potentially more expensive operations.
   */
  public void enforceLabelAttributes() throws EvalException, InterruptedException {
    StructImpl attr = getAttr();
    for (String name : attr.getFieldNames()) {
      Object value = attr.getValue(name);
      if (value instanceof Label) {
        getPathFromLabel((Label) value);
      }
      if (value instanceof SkylarkList) {
        for (Object entry : (SkylarkList) value) {
          if (entry instanceof Label) {
            getPathFromLabel((Label) entry);
          }
        }
      }
    }
  }
}
