// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.packages.util;

import com.google.common.primitives.Chars;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.CachingPackageLocator;
import com.google.devtools.build.lib.packages.Globber;
import com.google.devtools.build.lib.packages.Preprocessor;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.ParserInputSource;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.IOException;
import java.nio.CharBuffer;
import java.util.Arrays;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Expands subinclude() statements, and returns an error if ERROR is
 * present in the end-result.  It does not run python, and is intended
 * for testing
 */
public class SubincludePreprocessor implements Preprocessor {
  /** Creates SubincludePreprocessor factories. */
  public static class FactorySupplier implements Preprocessor.Factory.Supplier {
    private final FileSystem fileSystem;

    public FactorySupplier(FileSystem fileSystem) {
      this.fileSystem = fileSystem;
    }

    @Override
    public Factory getFactory(final CachingPackageLocator loc) {
      return new Factory() {
        @Override
        public boolean isStillValid() {
          return true;
        }

        @Override
        public boolean considersGlobs() {
          return false;
        }

        @Override
        public Preprocessor getPreprocessor() {
          return new SubincludePreprocessor(fileSystem, loc);
        }
      };
    }
  }

  private final FileSystem fileSystem;
  private final CachingPackageLocator packageLocator;

  private static final Pattern SUBINCLUDE_REGEX =
      Pattern.compile("\\bsubinclude\\(['\"]([^'\"=]*)['\"]\\)", Pattern.MULTILINE);
  public static final String TRANSIENT_ERROR = "TRANSIENT_ERROR";

  /**
   * Constructs a SubincludePreprocessor using the specified package
   * path for resolving subincludes.
   */
  public SubincludePreprocessor(FileSystem fileSystem, CachingPackageLocator packageLocator) {
    this.fileSystem = fileSystem;
    this.packageLocator = packageLocator;
  }

  // Cut & paste from PythonPreprocessor#resolveSubinclude.
  public String resolveSubinclude(String labelString) throws IOException {
    Label label;
    try {
      label = Label.parseAbsolute(labelString);
    } catch (LabelSyntaxException e) {
      throw new IOException("Cannot parse label: '" + labelString + "'");
    }

    Path buildFile = packageLocator.getBuildFileForPackage(label.getPackageIdentifier());
    if (buildFile == null) {
      return "";
    }

    Path subinclude = buildFile.getParentDirectory().getRelative(new PathFragment(label.getName()));
    return subinclude.getPathString();
  }

  @Override
  public Preprocessor.Result preprocess(
      Path buildFilePath,
      byte[] buildFileBytes,
      String packageName,
      Globber globber,
      Environment.Frame globals,
      Set<String> ruleNames)
      throws IOException, InterruptedException {
    StoredEventHandler eventHandler = new StoredEventHandler();
    char content[] = FileSystemUtils.convertFromLatin1(buildFileBytes);
    while (true) {
      Matcher matcher = SUBINCLUDE_REGEX.matcher(CharBuffer.wrap(content));
      if (!matcher.find()) {
        break;
      }
      String name = matcher.group(1);
      String path = resolveSubinclude(name);

      char subContent[];
      if (path.isEmpty()) {
        // This location is not correct, but will do for testing purposes.
        eventHandler.handle(
            Event.error(
                Location.fromFile(buildFilePath), "Cannot find subincluded file \'" + name + "\'"));
        // Emit a mocksubinclude(), so we know to preprocess again if the file becomes
        // visible. We cannot fail the preprocess here, as it would drop the content.
        subContent = new char[0];
      } else {
        // TODO(bazel-team): figure out the correct behavior for a non-existent file from an
        // existent package.
        subContent = FileSystemUtils.readContentAsLatin1(fileSystem.getPath(path));
      }

      String mock = "\nmocksubinclude('" + name + "', '" + path + "')\n";

      content =
          Chars.concat(
              Arrays.copyOf(content, matcher.start()),
              mock.toCharArray(),
              subContent,
              Arrays.copyOfRange(content, matcher.end(), content.length));
    }

    if (Chars.indexOf(content, TRANSIENT_ERROR.toCharArray()) >= 0) {
      throw new IOException("transient error requested in " + buildFilePath.asFragment());
    }

    return Preprocessor.Result.success(
        ParserInputSource.create(content, buildFilePath.asFragment()),
        eventHandler.hasErrors(),
        eventHandler.getEvents());
  }
}
