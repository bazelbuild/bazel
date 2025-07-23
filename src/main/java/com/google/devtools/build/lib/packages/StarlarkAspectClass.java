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

package com.google.devtools.build.lib.packages;

import static com.google.devtools.build.lib.util.HashCodes.hashObjects;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.BzlLoadValue;

/** {@link AspectClass} for aspects defined in Starlark. */
@Immutable
public final class StarlarkAspectClass implements AspectClass {
  private final BzlLoadValue.Key extensionKey;
  private final String exportedName;
  private final String name;

  public StarlarkAspectClass(BzlLoadValue.Key extensionKey, String exportedName) {
    this.extensionKey = extensionKey;
    this.exportedName = exportedName;
    this.name = extensionKey.getLabel() + "%" + exportedName;
  }

  BzlLoadValue.Key getExtensionKey() {
    return extensionKey;
  }

  public Label getExtensionLabel() {
    return extensionKey.getLabel();
  }

  public String getExportedName() {
    return exportedName;
  }

  @Override
  public String getName() {
    return name;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }

    if (!(o instanceof StarlarkAspectClass)) {
      return false;
    }

    StarlarkAspectClass that = (StarlarkAspectClass) o;
    return extensionKey.equals(that.extensionKey) && exportedName.equals(that.exportedName);
  }

  @Override
  public int hashCode() {
    return hashObjects(extensionKey, exportedName);
  }

  @Override
  public String toString() {
    return getName();
  }

  public static StarlarkAspectClass getAspectClassFromName(String aspect)
      throws AspectClassCreationException {
    int delimiterPosition = aspect.indexOf('%');
    if (delimiterPosition >= 0) {
      String bzlFileLoadLikeString = aspect.substring(0, delimiterPosition);
      if (!bzlFileLoadLikeString.startsWith("//") && !bzlFileLoadLikeString.startsWith("@")) {
        throw new AspectClassCreationException(
            "--exec_aspects must be specified with absolute labels, e.g."
                + " //foo/bar:baz.bzl%my_aspect, @repo//foo/bar:baz%my_aspect, or"
                + " /foo/bar:baz.bzl%my_aspect. Found: "
                + aspect);
      } else if (!bzlFileLoadLikeString.endsWith(".bzl")) {
        throw new AspectClassCreationException(
            "--exec_aspects files must end with .bzl. Found: " + aspect);
      } else {
        Label starlarkFileLabel = null;
        try {
          starlarkFileLabel = Label.parseCanonical(bzlFileLoadLikeString);
          String starlarkFunctionName = aspect.substring(delimiterPosition + 1);
          return new StarlarkAspectClass(
              BzlLoadValue.keyForBuild(starlarkFileLabel), starlarkFunctionName);
        } catch (LabelSyntaxException e) {
          throw new AspectClassCreationException(
              String.format("Invalid aspect '%s': %s", aspect, e.getMessage()));
        }
      }
    } else {
      throw new AspectClassCreationException(
          "--exec_aspects must include the aspect name, preceded by '%', e.g."
              + " //foo/bar:baz.bzl%my_aspect, @repo//foo/bar:baz%my_aspect, or"
              + " /foo/bar:baz.bzl%my_aspect. Found: "
              + aspect);
    }
  }

  /**
   * An exception indicating that there was a problem creating a {@link StarlarkAspectClass} aspect.
   */
  public static class AspectClassCreationException extends Exception {
    public AspectClassCreationException(String message) {
      super(message);
    }
  }
}
