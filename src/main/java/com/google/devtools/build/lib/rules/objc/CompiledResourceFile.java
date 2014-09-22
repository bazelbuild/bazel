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

package com.google.devtools.build.lib.rules.objc;

import static com.google.devtools.build.lib.rules.objc.ArtifactListAttribute.STRINGS;
import static com.google.devtools.build.lib.rules.objc.ArtifactListAttribute.XIBS;

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.view.RuleContext;

/**
 * Represents a strings or {@code .xib} file.
 */
public class CompiledResourceFile {
  private final Artifact original;
  private final BundleableFile bundled;

  private CompiledResourceFile(Artifact original, BundleableFile bundled) {
    this.original = Preconditions.checkNotNull(original);
    this.bundled = Preconditions.checkNotNull(bundled);
  }

  /**
   * The checked-in version of the bundled file.
   */
  public Artifact getOriginal() {
    return original;
  }

  public BundleableFile getBundled() {
    return bundled;
  }

  public static final Function<CompiledResourceFile, BundleableFile> TO_BUNDLED =
      new Function<CompiledResourceFile, BundleableFile>() {
        @Override
        public BundleableFile apply(CompiledResourceFile input) {
          return input.bundled;
        }
      };

  /**
   * Returns an instance for every file, if any, specified by the {@code strings} attribute of the
   * given rule. The value returned by {@link #getBundled()} will be the plist file in
   * binary form.
   */
  public static Iterable<CompiledResourceFile> stringsFilesFromRule(RuleContext context) {
    ImmutableList.Builder<CompiledResourceFile> result = new ImmutableList.Builder<>();
    for (Artifact originalFile : STRINGS.get(context)) {
      Artifact binaryFile = ObjcRuleClasses.artifactByAppendingToRootRelativePath(
          context, originalFile.getRootRelativePath(), ".binary");
      result.add(new CompiledResourceFile(
          originalFile, new BundleableFile(binaryFile, BundleableFile.bundlePath(originalFile))));
    }
    return result.build();
  }

  /**
   * Returns an instance for every file, if any, specified by the {@code xibs} attribute of the
   * given rule.
   */
  public static Iterable<CompiledResourceFile> xibFilesFromRule(RuleContext context) {
    ImmutableList.Builder<CompiledResourceFile> result = new ImmutableList.Builder<>();
    for (Artifact originalFile : XIBS.get(context)) {
      // Each .xib file is compiled to a single .nib file.
      Artifact nibFile = context.getRelatedArtifact(originalFile.getExecPath(), ".nib");
      result.add(new CompiledResourceFile(
          originalFile, new BundleableFile(nibFile, BundleableFile.bundlePath(nibFile))));
    }
    return result.build();
  }
}
