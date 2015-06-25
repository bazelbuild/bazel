// Copyright 2015 Google Inc. All rights reserved.
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
package com.google.devtools.build.android;

import com.android.SdkConstants;
import com.android.sdklib.AndroidTargetHash;
import com.android.sdklib.AndroidVersion;
import com.android.sdklib.BuildToolInfo;
import com.android.sdklib.IAndroidTarget;
import com.android.sdklib.ISystemImage;
import com.android.sdklib.repository.descriptors.IdDisplay;

import java.io.File;
import java.nio.file.Path;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Defines a target platform used by Bazel builds.
 */
public class BazelPlatformTarget implements IAndroidTarget {

  private final BuildToolInfo buildToolInfo;
  private final Map<Integer, Path> paths = new HashMap<>();
  private final AndroidVersion version;
  private final Path sdkRoot;

  public BazelPlatformTarget(
      Path androidJar,
      AndroidVersion version,
      BuildToolInfo buildToolInfo) {
    this.version = version;
    this.buildToolInfo = buildToolInfo;
    
    sdkRoot = new File("unused/tool/from/sdk/root").toPath();
    // pre-build the path to the platform components with default values
    // TODO(bazel-team): Allow overrides of the default values.
    paths.put(ANDROID_JAR, androidJar);
    paths.put(UI_AUTOMATOR_JAR, sdkRoot.resolve(SdkConstants.FN_UI_AUTOMATOR_LIBRARY));
    paths.put(SOURCES, sdkRoot.resolve(SdkConstants.FD_ANDROID_SOURCES));
    paths.put(ANDROID_AIDL, sdkRoot.resolve(SdkConstants.FN_FRAMEWORK_AIDL));
    paths.put(SAMPLES, sdkRoot.resolve(SdkConstants.OS_PLATFORM_SAMPLES_FOLDER));
    paths.put(SKINS, sdkRoot.resolve(SdkConstants.OS_SKINS_FOLDER));
    paths.put(TEMPLATES, sdkRoot.resolve(SdkConstants.OS_PLATFORM_TEMPLATES_FOLDER));
    paths.put(DATA, sdkRoot.resolve(SdkConstants.OS_PLATFORM_DATA_FOLDER));
    paths.put(ATTRIBUTES, sdkRoot.resolve(SdkConstants.OS_PLATFORM_ATTRS_XML));
    paths.put(MANIFEST_ATTRIBUTES, sdkRoot.resolve(SdkConstants.OS_PLATFORM_ATTRS_MANIFEST_XML));
    paths.put(RESOURCES, sdkRoot.resolve(SdkConstants.OS_PLATFORM_RESOURCES_FOLDER));
    paths.put(FONTS, sdkRoot.resolve(SdkConstants.OS_PLATFORM_FONTS_FOLDER));
    paths.put(LAYOUT_LIB,
        sdkRoot.resolve(SdkConstants.OS_PLATFORM_DATA_FOLDER + SdkConstants.FN_LAYOUTLIB_JAR));
    paths.put(WIDGETS,
        sdkRoot.resolve(SdkConstants.OS_PLATFORM_DATA_FOLDER + SdkConstants.FN_WIDGETS));
    paths.put(ACTIONS_ACTIVITY, sdkRoot.resolve(
        SdkConstants.OS_PLATFORM_DATA_FOLDER + SdkConstants.FN_INTENT_ACTIONS_ACTIVITY));
    paths.put(ACTIONS_BROADCAST, sdkRoot.resolve(
        SdkConstants.OS_PLATFORM_DATA_FOLDER + SdkConstants.FN_INTENT_ACTIONS_BROADCAST));
    paths.put(ACTIONS_SERVICE, sdkRoot.resolve(
        SdkConstants.OS_PLATFORM_DATA_FOLDER + SdkConstants.FN_INTENT_ACTIONS_SERVICE));
    paths.put(CATEGORIES,
        sdkRoot.resolve(SdkConstants.OS_PLATFORM_DATA_FOLDER + SdkConstants.FN_INTENT_CATEGORIES));
  }

  @Override
  public int compareTo(IAndroidTarget o) {
    if (o.isPlatform() == false) {
      return -1;
    }
    return version.compareTo(o.getVersion());
  }

  @Override
  public String getLocation() {
    return sdkRoot.toFile().getPath();
  }

  @Override
  public String getVendor() {
    return "Android";
  }

  @Override
  public String getName() {
    return "Android [Platform Version Name] (Bazel)";
  }

  @Override
  public String getFullName() {
    return "Android [Platform Version Name] (Bazel)";
  }

  @Override
  public String getClasspathName() {
    return "Android [Platform Version Name] (Bazel)";
  }

  @Override
  public String getShortClasspathName() {
    return "Android [Platform Version Name] (Bazel)";
  }

  @Override
  public String getDescription() {
    return String.format("Standard Android platform %s", "[Platform Version Name] (Bazel)");
  }

  @Override
  public AndroidVersion getVersion() {
    return version;
  }

  @Override
  public String getVersionName() {
    return version.getCodename();
  }

  @Override
  public int getRevision() {
    return 0;
  }

  @Override
  public boolean isPlatform() {
    return true;
  }

  @Override
  public IAndroidTarget getParent() {
    return null;
  }

  @Override
  public String getPath(int pathId) {
    return paths.get(pathId).toFile().getPath();
  }

  @Override
  public File getFile(int pathId) {
    return new File(getPath(pathId));
  }

  @Override
  public BuildToolInfo getBuildToolInfo() {
    return buildToolInfo;
  }

  @Override
  public List<String> getBootClasspath() {
    return Collections.singletonList(getPath(IAndroidTarget.ANDROID_JAR));
  }

  @Override
  public boolean hasRenderingLibrary() {
    return true;
  }

  @Override
  public File[] getSkins() {
    return new File[0];
  }

  @Override
  public File getDefaultSkin() {
    return null;
  }

  @Override
  public IOptionalLibrary[] getOptionalLibraries() {
    return new IOptionalLibrary[0];
  }

  @Override
  public String[] getPlatformLibraries() {
    return new String[] { SdkConstants.ANDROID_TEST_RUNNER_LIB };
  }

  @Override
  public String getProperty(String name) {
    return null;
  }

  @Override
  public Integer getProperty(String name, Integer defaultValue) {
    return defaultValue;
  }

  @Override
  public Boolean getProperty(String name, Boolean defaultValue) {
    return defaultValue;
  }

  @Override
  public Map<String, String> getProperties() {
    return null;
  }

  @Override
  public int getUsbVendorId() {
    return NO_USB_ID;
  }

  @Override
  public ISystemImage[] getSystemImages() {
    return new ISystemImage[0];
  }

  @Override
  public ISystemImage getSystemImage(IdDisplay tag, String abiType) {
    return null;
  }

  @Override
  public boolean canRunOn(IAndroidTarget target) {
    // TODO(bazel-team): Auto-generated method stub
    return false;
  }

  @Override
  public String hashString() {
    return AndroidTargetHash.getPlatformHashString(version);
  }
}
