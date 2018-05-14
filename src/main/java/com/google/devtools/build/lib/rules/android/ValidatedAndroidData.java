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
package com.google.devtools.build.lib.rules.android;

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.RuleErrorConsumer;

/**
 * A {@link CompiledMergableAndroidData} that has been fully processed, validated, and packaged.
 *
 * <p>It contains resources and, depending on implementation, possibly assets and manifest.
 *
 * <p>TODO(b/76418178): Once resources and assets are completely decoupled and {@link
 * ResourceContainer} is removed, this interface can be replaced with {@link
 * ValidatedAndroidResources}
 */
public interface ValidatedAndroidData extends CompiledMergableAndroidData {

  Artifact getRTxt();

  Artifact getAapt2RTxt();

  Artifact getStaticLibrary();

  ValidatedAndroidData filter(
      RuleErrorConsumer errorConsumer, ResourceFilter resourceFilter, boolean isDependency)
      throws RuleErrorException;

  Artifact getJavaClassJar();

  String getJavaPackage();

  Artifact getJavaSourceJar();

  @VisibleForTesting
  Artifact getApk();

  /**
   * Gets an Artifact containing a zip of merged resources.
   *
   * <p>If assets were processed together with resources, the zip will also contain merged assets.
   *
   * @deprecated This artifact is produced by an often-expensive action and should not be used if
   *     another option is available. Furthermore, it will be replaced by flat files once we
   *     completely move to aapt2.
   */
  @Deprecated
  Artifact getMergedResources();

  ProcessedAndroidManifest getProcessedManifest();
}

