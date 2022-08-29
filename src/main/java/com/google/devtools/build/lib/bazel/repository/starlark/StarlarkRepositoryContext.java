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

package com.google.devtools.build.lib.bazel.repository.starlark;

import com.github.difflib.patch.PatchFailedException;
import com.google.common.base.Ascii;
import com.google.common.base.Optional;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.bazel.debug.WorkspaceRuleEvent;
import com.google.devtools.build.lib.bazel.repository.DecompressorDescriptor;
import com.google.devtools.build.lib.bazel.repository.DecompressorValue;
import com.google.devtools.build.lib.bazel.repository.PatchUtil;
import com.google.devtools.build.lib.bazel.repository.cache.RepositoryCache;
import com.google.devtools.build.lib.bazel.repository.cache.RepositoryCache.KeyType;
import com.google.devtools.build.lib.bazel.repository.downloader.Checksum;
import com.google.devtools.build.lib.bazel.repository.downloader.DownloadManager;
import com.google.devtools.build.lib.bazel.repository.downloader.HttpUtils;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.ExtendedEventHandler.FetchProgress;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.StarlarkInfo;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.packages.StructProvider;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction.RepositoryFunctionException;
import com.google.devtools.build.lib.rules.repository.WorkspaceAttributeMapper;
import com.google.devtools.build.lib.runtime.ProcessWrapper;
import com.google.devtools.build.lib.runtime.RepositoryRemoteExecutor;
import com.google.devtools.build.lib.util.StringUtilities;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import java.io.IOException;
import java.io.OutputStream;
import java.net.MalformedURLException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkInt;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.syntax.Location;

/** Starlark API for the repository_rule's context. */
@StarlarkBuiltin(
    name = "repository_ctx",
    category = DocCategory.BUILTIN,
    doc =
        "The context of the repository rule containing"
            + " helper functions and information about attributes. You get a repository_ctx object"
            + " as an argument to the <code>implementation</code> function when you create a"
            + " repository rule.")
public class StarlarkRepositoryContext extends StarlarkBaseExternalContext {
  private final Rule rule;
  private final PathPackageLocator packageLocator;
  private final Path workspaceRoot;
  private final StructImpl attrObject;
  private final ImmutableSet<PathFragment> ignoredPatterns;
  private final SyscallCache syscallCache;

  /**
   * Create a new context (repository_ctx) object for a Starlark repository rule ({@code rule}
   * argument).
   */
  StarlarkRepositoryContext(
      Rule rule,
      PathPackageLocator packageLocator,
      Path outputDirectory,
      ImmutableSet<PathFragment> ignoredPatterns,
      Environment environment,
      ImmutableMap<String, String> env,
      DownloadManager downloadManager,
      double timeoutScaling,
      @Nullable ProcessWrapper processWrapper,
      StarlarkSemantics starlarkSemantics,
      @Nullable RepositoryRemoteExecutor remoteExecutor,
      SyscallCache syscallCache,
      Path workspaceRoot)
      throws EvalException {
    super(
        outputDirectory,
        environment,
        env,
        downloadManager,
        timeoutScaling,
        processWrapper,
        starlarkSemantics,
        remoteExecutor);
    this.rule = rule;
    this.packageLocator = packageLocator;
    this.ignoredPatterns = ignoredPatterns;
    this.syscallCache = syscallCache;
    this.workspaceRoot = workspaceRoot;
    WorkspaceAttributeMapper attrs = WorkspaceAttributeMapper.of(rule);
    ImmutableMap.Builder<String, Object> attrBuilder = new ImmutableMap.Builder<>();
    for (String name : attrs.getAttributeNames()) {
      if (!name.equals("$local")) {
        // Attribute values should be type safe
        attrBuilder.put(
            Attribute.getStarlarkName(name), Attribute.valueToStarlark(attrs.getObject(name)));
      }
    }
    attrObject = StructProvider.STRUCT.create(attrBuilder.buildOrThrow(), "No such attribute '%s'");
  }

  @Override
  protected String getIdentifyingStringForLogging() {
    return rule.getLabel().toString();
  }

  @StarlarkMethod(
      name = "name",
      structField = true,
      doc = "The name of the external repository created by this rule.")
  public String getName() {
    return rule.getName();
  }

  @StarlarkMethod(
      name = "workspace_root",
      structField = true,
      doc = "The path to the root workspace of the bazel invocation.")
  public StarlarkPath getWorkspaceRoot() {
    return new StarlarkPath(workspaceRoot);
  }

  @StarlarkMethod(
      name = "attr",
      structField = true,
      doc =
          "A struct to access the values of the attributes. The values are provided by "
              + "the user (if not, a default value is used).")
  public StructImpl getAttr() {
    return attrObject;
  }

  private StarlarkPath externalPath(String method, Object pathObject)
      throws EvalException, InterruptedException {
    StarlarkPath starlarkPath = getPath(method, pathObject);
    Path path = starlarkPath.getPath();
    if (packageLocator.getPathEntries().stream().noneMatch(root -> path.startsWith(root.asPath()))
        || path.startsWith(workingDirectory)) {
      return starlarkPath;
    }
    Path workspaceRoot = packageLocator.getWorkspaceFile(syscallCache).getParentDirectory();
    PathFragment relativePath = path.relativeTo(workspaceRoot);
    for (PathFragment ignoredPattern : ignoredPatterns) {
      if (relativePath.startsWith(ignoredPattern)) {
        return starlarkPath;
      }
    }
    throw Starlark.errorf(
        "%s can only be applied to external paths (that is, outside the workspace or ignored in"
            + " .bazelignore)",
        method);
  }

  @StarlarkMethod(
      name = "report_progress",
      doc = "Updates the progress status for the fetching of this repository",
      parameters = {
        @Param(
            name = "status",
            allowedTypes = {@ParamType(type = String.class)},
            doc = "string describing the current status of the fetch progress")
      })
  // TODO(wyv): migrate this to the base context.
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

  @StarlarkMethod(
      name = "symlink",
      doc = "Creates a symlink on the filesystem.",
      useStarlarkThread = true,
      parameters = {
        @Param(
            name = "from",
            allowedTypes = {
              @ParamType(type = String.class),
              @ParamType(type = Label.class),
              @ParamType(type = StarlarkPath.class)
            },
            doc = "path to which the created symlink should point to."),
        @Param(
            name = "to",
            allowedTypes = {
              @ParamType(type = String.class),
              @ParamType(type = Label.class),
              @ParamType(type = StarlarkPath.class)
            },
            doc = "path of the symlink to create, relative to the repository directory."),
      })
  public void symlink(Object from, Object to, StarlarkThread thread)
      throws RepositoryFunctionException, EvalException, InterruptedException {
    StarlarkPath fromPath = getPath("symlink()", from);
    StarlarkPath toPath = getPath("symlink()", to);
    WorkspaceRuleEvent w =
        WorkspaceRuleEvent.newSymlinkEvent(
            fromPath.toString(),
            toPath.toString(),
            rule.getLabel().toString(),
            thread.getCallerLocation());
    env.getListener().post(w);
    try {
      checkInOutputDirectory("write", toPath);
      makeDirectories(toPath.getPath());
      toPath.getPath().createSymbolicLink(fromPath.getPath());
    } catch (IOException e) {
      throw new RepositoryFunctionException(
          new IOException(
              "Could not create symlink from " + fromPath + " to " + toPath + ": " + e.getMessage(),
              e),
          Transience.TRANSIENT);
    } catch (InvalidPathException e) {
      throw new RepositoryFunctionException(
          Starlark.errorf("Could not create %s: %s", toPath, e.getMessage()),
          Transience.PERSISTENT);
    }
  }

  private void checkInOutputDirectory(String operation, StarlarkPath path)
      throws RepositoryFunctionException {
    if (!path.getPath().getPathString().startsWith(workingDirectory.getPathString())) {
      throw new RepositoryFunctionException(
          Starlark.errorf(
              "Cannot %s outside of the repository directory for path %s", operation, path),
          Transience.PERSISTENT);
    }
  }

  @StarlarkMethod(
      name = "file",
      doc = "Generates a file in the repository directory with the provided content.",
      useStarlarkThread = true,
      parameters = {
        @Param(
            name = "path",
            allowedTypes = {
              @ParamType(type = String.class),
              @ParamType(type = Label.class),
              @ParamType(type = StarlarkPath.class)
            },
            doc = "path of the file to create, relative to the repository directory."),
        @Param(
            name = "content",
            named = true,
            defaultValue = "''",
            doc = "the content of the file to create, empty by default."),
        @Param(
            name = "executable",
            named = true,
            defaultValue = "True",
            doc = "set the executable flag on the created file, true by default."),
        @Param(
            name = "legacy_utf8",
            named = true,
            defaultValue = "True",
            doc =
                "encode file content to UTF-8, true by default. Future versions will change"
                    + " the default and remove this parameter."),
      })
  public void createFile(
      Object path, String content, Boolean executable, Boolean legacyUtf8, StarlarkThread thread)
      throws RepositoryFunctionException, EvalException, InterruptedException {
    StarlarkPath p = getPath("file()", path);
    byte[] contentBytes;
    if (legacyUtf8) {
      contentBytes = content.getBytes(StandardCharsets.UTF_8);
    } else {
      contentBytes = content.getBytes(StandardCharsets.ISO_8859_1);
    }
    WorkspaceRuleEvent w =
        WorkspaceRuleEvent.newFileEvent(
            p.toString(),
            content,
            executable,
            rule.getLabel().toString(),
            thread.getCallerLocation());
    env.getListener().post(w);
    try {
      checkInOutputDirectory("write", p);
      makeDirectories(p.getPath());
      p.getPath().delete();
      try (OutputStream stream = p.getPath().getOutputStream()) {
        stream.write(contentBytes);
      }
      if (executable) {
        p.getPath().setExecutable(true);
      }
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    } catch (InvalidPathException e) {
      throw new RepositoryFunctionException(
          Starlark.errorf("Could not create %s: %s", p, e.getMessage()), Transience.PERSISTENT);
    }
  }

  @StarlarkMethod(
      name = "template",
      doc =
          "Generates a new file using a <code>template</code>. Every occurrence in "
              + "<code>template</code> of a key of <code>substitutions</code> will be replaced by "
              + "the corresponding value. The result is written in <code>path</code>. An optional"
              + "<code>executable</code> argument (default to true) can be set to turn on or off"
              + "the executable bit.",
      useStarlarkThread = true,
      parameters = {
        @Param(
            name = "path",
            allowedTypes = {
              @ParamType(type = String.class),
              @ParamType(type = Label.class),
              @ParamType(type = StarlarkPath.class)
            },
            doc = "path of the file to create, relative to the repository directory."),
        @Param(
            name = "template",
            allowedTypes = {
              @ParamType(type = String.class),
              @ParamType(type = Label.class),
              @ParamType(type = StarlarkPath.class)
            },
            doc = "path to the template file."),
        @Param(
            name = "substitutions",
            defaultValue = "{}",
            named = true,
            doc = "substitutions to make when expanding the template."),
        @Param(
            name = "executable",
            defaultValue = "True",
            named = true,
            doc = "set the executable flag on the created file, true by default."),
      })
  public void createFileFromTemplate(
      Object path,
      Object template,
      Dict<?, ?> substitutions, // <String, String> expected
      Boolean executable,
      StarlarkThread thread)
      throws RepositoryFunctionException, EvalException, InterruptedException {
    StarlarkPath p = getPath("template()", path);
    StarlarkPath t = getPath("template()", template);
    Map<String, String> substitutionMap =
        Dict.cast(substitutions, String.class, String.class, "substitutions");
    WorkspaceRuleEvent w =
        WorkspaceRuleEvent.newTemplateEvent(
            p.toString(),
            t.toString(),
            substitutionMap,
            executable,
            rule.getLabel().toString(),
            thread.getCallerLocation());
    env.getListener().post(w);
    try {
      checkInOutputDirectory("write", p);
      makeDirectories(p.getPath());
      String tpl = FileSystemUtils.readContent(t.getPath(), StandardCharsets.UTF_8);
      for (Map.Entry<String, String> substitution : substitutionMap.entrySet()) {
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
    } catch (InvalidPathException e) {
      throw new RepositoryFunctionException(
          Starlark.errorf("Could not create %s: %s", p, e.getMessage()), Transience.PERSISTENT);
    }
  }

  @Override
  protected boolean isRemotable() {
    Object remotable = rule.getAttr("$remotable");
    if (remotable != null) {
      return (Boolean) remotable;
    }
    return false;
  }

  @Override
  protected ImmutableMap<String, String> getRemoteExecProperties() throws EvalException {
    return ImmutableMap.copyOf(
        Dict.cast(
            getAttr().getValue("exec_properties"), String.class, String.class, "exec_properties"));
  }

  @StarlarkMethod(
      name = "delete",
      doc =
          "Deletes a file or a directory. Returns a bool, indicating whether the file or directory"
              + " was actually deleted by this call.",
      useStarlarkThread = true,
      parameters = {
        @Param(
            name = "path",
            allowedTypes = {@ParamType(type = String.class), @ParamType(type = StarlarkPath.class)},
            doc =
                "Path of the file to delete, relative to the repository directory, or absolute."
                    + " Can be a path or a string."),
      })
  public boolean delete(Object pathObject, StarlarkThread thread)
      throws EvalException, RepositoryFunctionException, InterruptedException {
    StarlarkPath starlarkPath = externalPath("delete()", pathObject);
    WorkspaceRuleEvent w =
        WorkspaceRuleEvent.newDeleteEvent(
            starlarkPath.toString(), rule.getLabel().toString(), thread.getCallerLocation());
    env.getListener().post(w);
    try {
      Path path = starlarkPath.getPath();
      path.deleteTreesBelow();
      return path.delete();
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }
  }

  @StarlarkMethod(
      name = "patch",
      doc =
          "Apply a patch file to the root directory of external repository. "
              + "The patch file should be a standard "
              + "<a href=\"https://en.wikipedia.org/wiki/Diff#Unified_format\">"
              + "unified diff format</a> file. "
              + "The Bazel-native patch implementation doesn't support fuzz match and binary patch "
              + "like the patch command line tool.",
      useStarlarkThread = true,
      parameters = {
        @Param(
            name = "patch_file",
            allowedTypes = {
              @ParamType(type = String.class),
              @ParamType(type = Label.class),
              @ParamType(type = StarlarkPath.class)
            },
            doc =
                "The patch file to apply, it can be label, relative path or absolute path. "
                    + "If it's a relative path, it will resolve to the repository directory."),
        @Param(
            name = "strip",
            named = true,
            defaultValue = "0",
            doc = "strip the specified number of leading components from file names."),
      })
  public void patch(Object patchFile, StarlarkInt stripI, StarlarkThread thread)
      throws EvalException, RepositoryFunctionException, InterruptedException {
    int strip = Starlark.toInt(stripI, "strip");
    StarlarkPath starlarkPath = getPath("patch()", patchFile);
    WorkspaceRuleEvent w =
        WorkspaceRuleEvent.newPatchEvent(
            starlarkPath.toString(), strip, rule.getLabel().toString(), thread.getCallerLocation());
    env.getListener().post(w);
    try {
      PatchUtil.apply(starlarkPath.getPath(), strip, workingDirectory);
    } catch (PatchFailedException e) {
      throw new RepositoryFunctionException(
          Starlark.errorf("Error applying patch %s: %s", starlarkPath, e.getMessage()),
          Transience.TRANSIENT);
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }
  }

  private void warnAboutChecksumError(List<URL> urls, String errorMessage) {
    // Inform the user immediately, even though the file will still be downloaded.
    // This cannot be done by a regular error event, as all regular events are recorded
    // and only shown once the execution of the repository rule is finished.
    // So we have to provide the information as update on the progress
    String url = "(unknown)";
    if (urls.size() > 0) {
      url = urls.get(0).toString();
    }
    reportProgress("Will fail after download of " + url + ". " + errorMessage);
  }

  private static Map<String, Dict<?, ?>> getAuthContents(Dict<?, ?> x, String what)
      throws EvalException {
    // Dict.cast returns Dict<String, raw Dict>.
    @SuppressWarnings({"unchecked", "rawtypes"})
    Map<String, Dict<?, ?>> res = (Map) Dict.cast(x, String.class, Dict.class, what);
    return res;
  }

  @StarlarkMethod(
      name = "download",
      doc =
          "Downloads a file to the output path for the provided url and returns a struct"
              + " containing <code>success</code>, a flag which is <code>true</code> if the"
              + " download completed successfully, and if successful, a hash of the file"
              + " with the fields <code>sha256</code> and <code>integrity</code>.",
      useStarlarkThread = true,
      parameters = {
        @Param(
            name = "url",
            allowedTypes = {
              @ParamType(type = String.class),
              @ParamType(type = Iterable.class, generic1 = String.class),
            },
            named = true,
            doc = "List of mirror URLs referencing the same file."),
        @Param(
            name = "output",
            allowedTypes = {
              @ParamType(type = String.class),
              @ParamType(type = Label.class),
              @ParamType(type = StarlarkPath.class)
            },
            defaultValue = "''",
            named = true,
            doc = "path to the output file, relative to the repository directory."),
        @Param(
            name = "sha256",
            defaultValue = "''",
            named = true,
            doc =
                "the expected SHA-256 hash of the file downloaded."
                    + " This must match the SHA-256 hash of the file downloaded. It is a security"
                    + " risk to omit the SHA-256 as remote files can change. At best omitting this"
                    + " field will make your build non-hermetic. It is optional to make development"
                    + " easier but should be set before shipping."),
        @Param(
            name = "executable",
            defaultValue = "False",
            named = true,
            doc = "set the executable flag on the created file, false by default."),
        @Param(
            name = "allow_fail",
            defaultValue = "False",
            named = true,
            doc =
                "If set, indicate the error in the return value"
                    + " instead of raising an error for failed downloads"),
        @Param(
            name = "canonical_id",
            defaultValue = "''",
            named = true,
            doc =
                "If set, restrict cache hits to those cases where the file was added to the cache"
                    + " with the same canonical id"),
        @Param(
            name = "auth",
            defaultValue = "{}",
            named = true,
            doc = "An optional dict specifying authentication information for some of the URLs."),
        @Param(
            name = "integrity",
            defaultValue = "''",
            named = true,
            positional = false,
            doc =
                "Expected checksum of the file downloaded, in Subresource Integrity format."
                    + " This must match the checksum of the file downloaded. It is a security"
                    + " risk to omit the checksum as remote files can change. At best omitting this"
                    + " field will make your build non-hermetic. It is optional to make development"
                    + " easier but should be set before shipping."),
      })
  public StructImpl download(
      Object url,
      Object output,
      String sha256,
      Boolean executable,
      Boolean allowFail,
      String canonicalId,
      Dict<?, ?> authUnchecked, // <String, Dict> expected
      String integrity,
      StarlarkThread thread)
      throws RepositoryFunctionException, EvalException, InterruptedException {
    Map<URI, Map<String, String>> authHeaders =
        getAuthHeaders(getAuthContents(authUnchecked, "auth"));

    List<URL> urls =
        getUrls(
            url,
            /* ensureNonEmpty= */ !allowFail,
            /* checksumGiven= */ !Strings.isNullOrEmpty(sha256)
                || !Strings.isNullOrEmpty(integrity));
    Optional<Checksum> checksum;
    RepositoryFunctionException checksumValidation = null;
    try {
      checksum = validateChecksum(sha256, integrity, urls);
    } catch (RepositoryFunctionException e) {
      checksum = Optional.<Checksum>absent();
      checksumValidation = e;
    }

    StarlarkPath outputPath = getPath("download()", output);
    WorkspaceRuleEvent w =
        WorkspaceRuleEvent.newDownloadEvent(
            urls,
            output.toString(),
            sha256,
            integrity,
            executable,
            rule.getLabel().toString(),
            thread.getCallerLocation());
    env.getListener().post(w);
    Path downloadedPath;
    try (SilentCloseable c =
        Profiler.instance().profile("fetching: " + rule.getLabel().toString())) {
      checkInOutputDirectory("write", outputPath);
      makeDirectories(outputPath.getPath());
      downloadedPath =
          downloadManager.download(
              urls,
              authHeaders,
              checksum,
              canonicalId,
              Optional.<String>absent(),
              outputPath.getPath(),
              env.getListener(),
              envVariables,
              getName());
      if (executable) {
        outputPath.getPath().setExecutable(true);
      }
    } catch (InterruptedException e) {
      throw new RepositoryFunctionException(
          new IOException("thread interrupted"), Transience.TRANSIENT);
    } catch (IOException e) {
      if (allowFail) {
        return StarlarkInfo.create(
            StructProvider.STRUCT, ImmutableMap.of("success", false), Location.BUILTIN);
      } else {
        throw new RepositoryFunctionException(e, Transience.TRANSIENT);
      }
    } catch (InvalidPathException e) {
      throw new RepositoryFunctionException(
          Starlark.errorf("Could not create output path %s: %s", outputPath, e.getMessage()),
          Transience.PERSISTENT);
    }
    if (checksumValidation != null) {
      throw checksumValidation;
    }

    return calculateDownloadResult(checksum, downloadedPath);
  }

  @StarlarkMethod(
      name = "extract",
      doc = "Extract an archive to the repository directory.",
      useStarlarkThread = true,
      parameters = {
        @Param(
            name = "archive",
            allowedTypes = {
              @ParamType(type = String.class),
              @ParamType(type = Label.class),
              @ParamType(type = StarlarkPath.class)
            },
            named = true,
            doc =
                "path to the archive that will be unpacked,"
                    + " relative to the repository directory."),
        @Param(
            name = "output",
            allowedTypes = {
              @ParamType(type = String.class),
              @ParamType(type = Label.class),
              @ParamType(type = StarlarkPath.class)
            },
            defaultValue = "''",
            named = true,
            doc =
                "path to the directory where the archive will be unpacked,"
                    + " relative to the repository directory."),
        @Param(
            name = "stripPrefix",
            defaultValue = "''",
            named = true,
            doc =
                "a directory prefix to strip from the extracted files."
                    + "\nMany archives contain a top-level directory that contains all files in the"
                    + " archive. Instead of needing to specify this prefix over and over in the"
                    + " <code>build_file</code>, this field can be used to strip it from extracted"
                    + " files."),
        @Param(
            name = "rename_files",
            defaultValue = "{}",
            named = true,
            positional = false,
            doc =
                "An optional dict specifying files to rename during the extraction. Archive entries"
                    + " with names exactly matching a key will be renamed to the value, prior to"
                    + " any directory prefix adjustment. This can be used to extract archives that"
                    + " contain non-Unicode filenames, or which have files that would extract to"
                    + " the same path on case-insensitive filesystems."),
      })
  public void extract(
      Object archive,
      Object output,
      String stripPrefix,
      Dict<?, ?> renameFiles, // <String, String> expected
      StarlarkThread thread)
      throws RepositoryFunctionException, InterruptedException, EvalException {
    StarlarkPath archivePath = getPath("extract()", archive);

    if (!archivePath.exists()) {
      throw new RepositoryFunctionException(
          Starlark.errorf("Archive path '%s' does not exist.", archivePath), Transience.TRANSIENT);
    }

    StarlarkPath outputPath = getPath("extract()", output);
    checkInOutputDirectory("write", outputPath);

    Map<String, String> renameFilesMap =
        Dict.cast(renameFiles, String.class, String.class, "rename_files");

    WorkspaceRuleEvent w =
        WorkspaceRuleEvent.newExtractEvent(
            archive.toString(),
            output.toString(),
            stripPrefix,
            renameFilesMap,
            rule.getLabel().toString(),
            thread.getCallerLocation());
    env.getListener().post(w);

    env.getListener()
        .post(
            new ExtractProgress(
                outputPath.getPath().toString(), "Extracting " + archivePath.getPath()));
    DecompressorValue.decompress(
        DecompressorDescriptor.builder()
            .setContext(getIdentifyingStringForLogging())
            .setArchivePath(archivePath.getPath())
            .setDestinationPath(outputPath.getPath())
            .setPrefix(stripPrefix)
            .setRenameFiles(renameFilesMap)
            .build());
    env.getListener().post(new ExtractProgress(outputPath.getPath().toString()));
  }

  @StarlarkMethod(
      name = "download_and_extract",
      doc =
          "Downloads a file to the output path for the provided url, extracts it, and returns a"
              + " struct containing <code>success</code>, a flag which is <code>true</code> if the"
              + " download completed successfully, and if successful, a hash of the file with the"
              + " fields <code>sha256</code> and <code>integrity</code>.",
      useStarlarkThread = true,
      parameters = {
        @Param(
            name = "url",
            allowedTypes = {
              @ParamType(type = String.class),
              @ParamType(type = Iterable.class, generic1 = String.class),
            },
            named = true,
            doc = "List of mirror URLs referencing the same file."),
        @Param(
            name = "output",
            allowedTypes = {
              @ParamType(type = String.class),
              @ParamType(type = Label.class),
              @ParamType(type = StarlarkPath.class)
            },
            defaultValue = "''",
            named = true,
            doc =
                "path to the directory where the archive will be unpacked,"
                    + " relative to the repository directory."),
        @Param(
            name = "sha256",
            defaultValue = "''",
            named = true,
            doc =
                "the expected SHA-256 hash of the file downloaded."
                    + " This must match the SHA-256 hash of the file downloaded. It is a security"
                    + " risk to omit the SHA-256 as remote files can change. At best omitting this"
                    + " field will make your build non-hermetic. It is optional to make development"
                    + " easier but should be set before shipping."
                    + " If provided, the repository cache will first be checked for a file with the"
                    + " given hash; a download will only be attempted if the file was not found in"
                    + " the cache. After a successful download, the file will be added to the"
                    + " cache."),
        @Param(
            name = "type",
            defaultValue = "''",
            named = true,
            doc =
                "the archive type of the downloaded file."
                    + " By default, the archive type is determined from the file extension of"
                    + " the URL."
                    + " If the file has no extension, you can explicitly specify either \"zip\","
                    + " \"jar\", \"war\", \"aar\", \"tar\", \"tar.gz\", \"tgz\", \"tar.xz\","
                    + " \"txz\", \".tar.zst\", \".tzst\", \"tar.bz2\", \".ar\", or \".deb\""
                    + " here."),
        @Param(
            name = "stripPrefix",
            defaultValue = "''",
            named = true,
            doc =
                "a directory prefix to strip from the extracted files."
                    + "\nMany archives contain a top-level directory that contains all files in the"
                    + " archive. Instead of needing to specify this prefix over and over in the"
                    + " <code>build_file</code>, this field can be used to strip it from extracted"
                    + " files."),
        @Param(
            name = "allow_fail",
            defaultValue = "False",
            named = true,
            doc =
                "If set, indicate the error in the return value"
                    + " instead of raising an error for failed downloads"),
        @Param(
            name = "canonical_id",
            defaultValue = "''",
            named = true,
            doc =
                "If set, restrict cache hits to those cases where the file was added to the cache"
                    + " with the same canonical id"),
        @Param(
            name = "auth",
            defaultValue = "{}",
            named = true,
            doc = "An optional dict specifying authentication information for some of the URLs."),
        @Param(
            name = "integrity",
            defaultValue = "''",
            named = true,
            positional = false,
            doc =
                "Expected checksum of the file downloaded, in Subresource Integrity format."
                    + " This must match the checksum of the file downloaded. It is a security"
                    + " risk to omit the checksum as remote files can change. At best omitting this"
                    + " field will make your build non-hermetic. It is optional to make development"
                    + " easier but should be set before shipping."),
        @Param(
            name = "rename_files",
            defaultValue = "{}",
            named = true,
            positional = false,
            doc =
                "An optional dict specifying files to rename during the extraction. Archive entries"
                    + " with names exactly matching a key will be renamed to the value, prior to"
                    + " any directory prefix adjustment. This can be used to extract archives that"
                    + " contain non-Unicode filenames, or which have files that would extract to"
                    + " the same path on case-insensitive filesystems."),
      })
  public StructImpl downloadAndExtract(
      Object url,
      Object output,
      String sha256,
      String type,
      String stripPrefix,
      Boolean allowFail,
      String canonicalId,
      Dict<?, ?> auth, // <String, Dict> expected
      String integrity,
      Dict<?, ?> renameFiles, // <String, String> expected
      StarlarkThread thread)
      throws RepositoryFunctionException, InterruptedException, EvalException {
    Map<URI, Map<String, String>> authHeaders = getAuthHeaders(getAuthContents(auth, "auth"));

    List<URL> urls =
        getUrls(
            url,
            /* ensureNonEmpty= */ !allowFail,
            /* checksumGiven= */ !Strings.isNullOrEmpty(sha256)
                || !Strings.isNullOrEmpty(integrity));
    Optional<Checksum> checksum;
    RepositoryFunctionException checksumValidation = null;
    try {
      checksum = validateChecksum(sha256, integrity, urls);
    } catch (RepositoryFunctionException e) {
      checksum = Optional.<Checksum>absent();
      checksumValidation = e;
    }

    Map<String, String> renameFilesMap =
        Dict.cast(renameFiles, String.class, String.class, "rename_files");

    WorkspaceRuleEvent w =
        WorkspaceRuleEvent.newDownloadAndExtractEvent(
            urls,
            output.toString(),
            sha256,
            integrity,
            type,
            stripPrefix,
            renameFilesMap,
            rule.getLabel().toString(),
            thread.getCallerLocation());

    StarlarkPath outputPath = getPath("download_and_extract()", output);
    checkInOutputDirectory("write", outputPath);
    createDirectory(outputPath.getPath());

    Path downloadedPath;
    Path downloadDirectory;
    try (SilentCloseable c =
        Profiler.instance().profile("fetching: " + rule.getLabel().toString())) {

      // Download to temp directory inside the outputDirectory and delete it after extraction
      java.nio.file.Path tempDirectory =
          Files.createTempDirectory(Paths.get(outputPath.toString()), "temp");
      downloadDirectory =
          workingDirectory.getFileSystem().getPath(tempDirectory.toFile().getAbsolutePath());

      downloadedPath =
          downloadManager.download(
              urls,
              authHeaders,
              checksum,
              canonicalId,
              Optional.of(type),
              downloadDirectory,
              env.getListener(),
              envVariables,
              getName());
    } catch (InterruptedException e) {
      env.getListener().post(w);
      throw new RepositoryFunctionException(
          new IOException("thread interrupted"), Transience.TRANSIENT);
    } catch (IOException e) {
      env.getListener().post(w);
      if (allowFail) {
        return StarlarkInfo.create(
            StructProvider.STRUCT, ImmutableMap.of("success", false), Location.BUILTIN);
      } else {
        throw new RepositoryFunctionException(e, Transience.TRANSIENT);
      }
    }
    if (checksumValidation != null) {
      throw checksumValidation;
    }
    env.getListener().post(w);
    try (SilentCloseable c =
        Profiler.instance().profile("extracting: " + rule.getLabel().toString())) {
      env.getListener()
          .post(
              new ExtractProgress(outputPath.getPath().toString(), "Extracting " + downloadedPath));
      DecompressorValue.decompress(
          DecompressorDescriptor.builder()
              .setContext(getIdentifyingStringForLogging())
              .setArchivePath(downloadedPath)
              .setDestinationPath(outputPath.getPath())
              .setPrefix(stripPrefix)
              .setRenameFiles(renameFilesMap)
              .build());
      env.getListener().post(new ExtractProgress(outputPath.getPath().toString()));
    }

    StructImpl downloadResult = calculateDownloadResult(checksum, downloadedPath);
    try {
      if (downloadDirectory.exists()) {
        downloadDirectory.deleteTree();
      }
    } catch (IOException e) {
      throw new RepositoryFunctionException(
          new IOException(
              "Couldn't delete temporary directory (" + downloadDirectory.getPathString() + ")", e),
          Transience.TRANSIENT);
    }
    return downloadResult;
  }

  private Checksum calculateChecksum(Optional<Checksum> originalChecksum, Path path)
      throws IOException, InterruptedException {
    if (originalChecksum.isPresent()) {
      // The checksum is checked on download, so if we got here, the user provided checksum is good
      return originalChecksum.get();
    }
    try {
      return Checksum.fromString(KeyType.SHA256, RepositoryCache.getChecksum(KeyType.SHA256, path));
    } catch (Checksum.InvalidChecksumException e) {
      throw new IllegalStateException(
          "Unexpected invalid checksum from internal computation of SHA-256 checksum on "
              + path.getPathString(),
          e);
    }
  }

  private Optional<Checksum> validateChecksum(String sha256, String integrity, List<URL> urls)
      throws RepositoryFunctionException, EvalException {
    if (!sha256.isEmpty()) {
      if (!integrity.isEmpty()) {
        throw Starlark.errorf("Expected either 'sha256' or 'integrity', but not both");
      }
      try {
        return Optional.of(Checksum.fromString(KeyType.SHA256, sha256));
      } catch (Checksum.InvalidChecksumException e) {
        warnAboutChecksumError(urls, e.getMessage());
        throw new RepositoryFunctionException(
            Starlark.errorf(
                "Definition of repository %s: %s at %s",
                rule.getName(), e.getMessage(), rule.getLocation()),
            Transience.PERSISTENT);
      }
    }

    if (integrity.isEmpty()) {
      return Optional.absent();
    }

    try {
      return Optional.of(Checksum.fromSubresourceIntegrity(integrity));
    } catch (Checksum.InvalidChecksumException e) {
      warnAboutChecksumError(urls, e.getMessage());
      throw new RepositoryFunctionException(
          Starlark.errorf(
              "Definition of repository %s: %s at %s",
              rule.getName(), e.getMessage(), rule.getLocation()),
          Transience.PERSISTENT);
    }
  }

  private StructImpl calculateDownloadResult(Optional<Checksum> checksum, Path downloadedPath)
      throws EvalException, InterruptedException, RepositoryFunctionException {
    Checksum finalChecksum;
    try {
      finalChecksum = calculateChecksum(checksum, downloadedPath);
    } catch (IOException e) {
      throw new RepositoryFunctionException(
          new IOException(
              "Couldn't hash downloaded file (" + downloadedPath.getPathString() + ")", e),
          Transience.PERSISTENT);
    }

    ImmutableMap.Builder<String, Object> out = ImmutableMap.builder();
    out.put("success", true);
    out.put("integrity", finalChecksum.toSubresourceIntegrity());

    // For compatibility with older Bazel versions that don't support non-SHA256 checksums.
    if (finalChecksum.getKeyType() == KeyType.SHA256) {
      out.put("sha256", finalChecksum.toString());
    }
    return StarlarkInfo.create(StructProvider.STRUCT, out.buildOrThrow(), Location.BUILTIN);
  }

  private static ImmutableList<String> checkAllUrls(Iterable<?> urlList) throws EvalException {
    ImmutableList.Builder<String> result = ImmutableList.builder();

    for (Object o : urlList) {
      if (!(o instanceof String)) {
        throw Starlark.errorf(
            "Expected a string or sequence of strings for 'url' argument, but got '%s' item in the"
                + " sequence",
            Starlark.type(o));
      }
      result.add((String) o);
    }

    return result.build();
  }

  private static List<URL> getUrls(Object urlOrList, boolean ensureNonEmpty, boolean checksumGiven)
      throws RepositoryFunctionException, EvalException, InterruptedException {
    List<String> urlStrings;
    if (urlOrList instanceof String) {
      urlStrings = ImmutableList.of((String) urlOrList);
    } else {
      urlStrings = checkAllUrls((Iterable<?>) urlOrList);
    }
    if (ensureNonEmpty && urlStrings.isEmpty()) {
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
      if (!checksumGiven) {
        if (!Ascii.equalsIgnoreCase("http", url.getProtocol())) {
          urls.add(url);
        }
      } else {
        urls.add(url);
      }
    }
    if (ensureNonEmpty && urls.isEmpty()) {
      throw new RepositoryFunctionException(
          new IOException(
              "No URLs left after removing plain http URLs due to missing checksum."
                  + " Please provide either a checksum or an https download location."),
          Transience.PERSISTENT);
    }
    return urls;
  }

  @Override
  public String toString() {
    return "repository_ctx[" + rule.getLabel() + "]";
  }

  /**
   * Try to compute the paths of all attributes that are labels, including labels in list and dict
   * arguments.
   *
   * <p>The value is ignored, but any missing information from the environment is detected (and an
   * exception thrown). In this way, we can enforce that all arguments are evaluated before we start
   * potentially more expensive operations.
   */
  // TODO(wyv): somehow migrate this to the base context too.
  public void enforceLabelAttributes() throws EvalException, InterruptedException {
    StructImpl attr = getAttr();
    for (String name : attr.getFieldNames()) {
      Object value = attr.getValue(name);
      if (value instanceof Label) {
        getPathFromLabel((Label) value);
      }
      if (value instanceof Sequence) {
        for (Object entry : (Sequence) value) {
          if (entry instanceof Label) {
            getPathFromLabel((Label) entry);
          }
        }
      }
      if (value instanceof Dict) {
        for (Object entry : ((Dict) value).keySet()) {
          if (entry instanceof Label) {
            getPathFromLabel((Label) entry);
          }
        }
      }
    }
  }

  /**
   * From an authentication dict extract a map of headers.
   *
   * <p>Given a dict as provided as "auth" argument, compute a map specifying for each URI provided
   * which additional headers (as usual, represented as a map from Strings to Strings) should
   * additionally be added to the request. For some form of authentication, in particular basic
   * authentication, adding those headers is enough; for other forms of authentication other
   * measures might be necessary.
   */
  private static Map<URI, Map<String, String>> getAuthHeaders(Map<String, Dict<?, ?>> auth)
      throws RepositoryFunctionException, EvalException {
    ImmutableMap.Builder<URI, Map<String, String>> headers = new ImmutableMap.Builder<>();
    for (Map.Entry<String, Dict<?, ?>> entry : auth.entrySet()) {
      try {
        URL url = new URL(entry.getKey());
        Dict<?, ?> authMap = entry.getValue();
        if (authMap.containsKey("type")) {
          if ("basic".equals(authMap.get("type"))) {
            if (!authMap.containsKey("login") || !authMap.containsKey("password")) {
              throw Starlark.errorf(
                  "Found request to do basic auth for %s without 'login' and 'password' being"
                      + " provided.",
                  entry.getKey());
            }
            String credentials = authMap.get("login") + ":" + authMap.get("password");
            headers.put(
                url.toURI(),
                ImmutableMap.<String, String>of(
                    "Authorization",
                    "Basic "
                        + Base64.getEncoder()
                            .encodeToString(credentials.getBytes(StandardCharsets.UTF_8))));
          } else if ("pattern".equals(authMap.get("type"))) {
            if (!authMap.containsKey("pattern")) {
              throw Starlark.errorf(
                  "Found request to do pattern auth for %s without a pattern being provided",
                  entry.getKey());
            }

            String result = (String) authMap.get("pattern");

            for (String component : Arrays.asList("password", "login")) {
              String demarcatedComponent = "<" + component + ">";

              if (result.contains(demarcatedComponent)) {
                if (!authMap.containsKey(component)) {
                  throw Starlark.errorf(
                      "Auth pattern contains %s but it was not provided in auth dict.",
                      demarcatedComponent);
                }
              } else {
                // component isn't in the pattern, ignore it
                continue;
              }

              result = result.replaceAll(demarcatedComponent, (String) authMap.get(component));
            }

            headers.put(url.toURI(), ImmutableMap.<String, String>of("Authorization", result));
          }
        }
      } catch (MalformedURLException e) {
        throw new RepositoryFunctionException(e, Transience.PERSISTENT);
      } catch (URISyntaxException e) {
        throw new EvalException(e);
      }
    }
    return headers.buildOrThrow();
  }

  private static class ExtractProgress implements FetchProgress {
    private final String repositoryPath;
    private final String progress;
    private final boolean isFinished;

    ExtractProgress(String repositoryPath, String progress) {
      this.repositoryPath = repositoryPath;
      this.progress = progress;
      this.isFinished = false;
    }

    ExtractProgress(String repositoryPath) {
      this.repositoryPath = repositoryPath;
      this.progress = "";
      this.isFinished = true;
    }

    @Override
    public String getResourceIdentifier() {
      return repositoryPath;
    }

    @Override
    public String getProgress() {
      return progress;
    }

    @Override
    public boolean isFinished() {
      return isFinished;
    }
  }
}
