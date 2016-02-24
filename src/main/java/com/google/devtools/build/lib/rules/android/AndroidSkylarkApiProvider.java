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
package com.google.devtools.build.lib.rules.android;

import com.google.common.collect.ImmutableCollection;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.rules.SkylarkApiProvider;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider.OutputJar;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;

import javax.annotation.Nullable;

/**
 * A class that exposes the Android providers to Skylark. It is intended to provide a
 * simple and stable interface for Skylark users.
 */
@SkylarkModule(
  name = "AndroidSkylarkApiProvider",
  doc = "Provides access to information about Android rules"
)
public class AndroidSkylarkApiProvider extends SkylarkApiProvider {
  /** The name of the field in Skylark used to access this class. */
  public static final String NAME = "android";

  private final IdlInfo idlInfo = new IdlInfo();

  @SkylarkCallable(
    name = "apk",
    structField = true,
    allowReturnNones = true,
    doc = "Returns an APK produced by this target."
  )
  public Artifact getApk() {
    return getIdeInfoProvider().getSignedApk();
  }

  private AndroidIdeInfoProvider getIdeInfoProvider() {
    return getInfo().getProvider(AndroidIdeInfoProvider.class);
  }

  @SkylarkCallable(
    name = "java_package",
    structField = true,
    allowReturnNones = true,
    doc = "Returns a java package for this target."
  )
  public String getJavaPackage() {
    return getIdeInfoProvider().getJavaPackage();
  }

  @SkylarkCallable(
    name = "manifest",
    structField = true,
    allowReturnNones = true,
    doc = "Returns a manifest file for this target."
  )
  public Artifact getManifest() {
    return getIdeInfoProvider().getManifest();
  }

  @SkylarkCallable(
      name = "apks_under_test",
      structField = true,
      allowReturnNones = true,
      doc = "Returns a collection of APKs that this target tests."
  )
  public ImmutableCollection<Artifact> getApksUnderTest() {
    return getIdeInfoProvider().getApksUnderTest();
  }


  @SkylarkCallable(
      name = "idl",
      structField = true,
      doc = "Returns information about IDL files associated with this target."
  )
  public IdlInfo getIdlInfo() {
    return idlInfo;
  }

  /**
   * Helper class to provide information about IDLs related to this rule.
   */
  @SkylarkModule(
      name = "AndroidSkylarkIdlInfo",
      doc = "Provides access to information about Android rules"
  )
  public class IdlInfo {
    @SkylarkCallable(
        name = "sources",
        structField = true,
        doc = "Returns a list of IDL files."
    )
    public ImmutableCollection<Artifact> getSources() {
      return getIdeInfoProvider().getIdlSrcs();
    }
    @SkylarkCallable(
        name = "generated_java_files",
        structField = true,
        doc = "Returns a list Java files generated from IDL sources."
    )
    public ImmutableCollection<Artifact> getIdlGeneratedJavaFiles() {
      return getIdeInfoProvider().getIdlGeneratedJavaFiles();
    }

    @SkylarkCallable(
        name = "output",
        structField = true,
        allowReturnNones = true,
        doc = "Returns a jar file for classes generated from IDL sources."
    )
    @Nullable
    public JavaRuleOutputJarsProvider.OutputJar getIdlOutput() {
      if (getIdeInfoProvider().getIdlClassJar() == null) {
        return null;
      }

      return new OutputJar(
          getIdeInfoProvider().getIdlClassJar(),
          null,
          getIdeInfoProvider().getIdlSourceJar()
      );
    }
  }
}
