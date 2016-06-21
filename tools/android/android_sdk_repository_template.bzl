"""Template for the build file used in android_sdk_repository."""
# Copyright 2016 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

def create_android_sdk_rules(
    name,
    build_tools_version,
    build_tools_directory,
    api_level):
  """Generate the contents of the android_sdk_repository.

  Args:
    name: string, the name of the repository being generated.
    build_tools_version: string, the version of Android's build tools to use.
    build_tools_directory: string, the directory name of the build tools in
        sdk's build-tools directory.
    api_level: int, the API level from which to get android.jar et al.
  """

  native.filegroup(
      name = "files",
      srcs = ["."],
  )

  native.java_import(
      name = "appcompat_v7_import",
      jars = ["extras/android/support/v7/appcompat/libs/android-support-v7-appcompat.jar"]
  )

  native.android_library(
      name = "appcompat_v7",
      custom_package = "android.support.v7.appcompat",
      manifest = "extras/android/support/v7/appcompat/AndroidManifest.xml",
      resource_files = native.glob(["extras/android/support/v7/appcompat/res/**"]),
      deps = [":appcompat_v7_import"]
  )

  native.java_import(
      name = "design_import",
      jars = ["extras/android/support/design/libs/android-support-design.jar"],
  )

  native.android_library(
      name = "design",
      custom_package = "android.support.design",
      manifest = "extras/android/support/design/AndroidManifest.xml",
      resource_files = native.glob(["extras/android/support/design/res/**"]),
      deps = [":design_import", ":appcompat_v7"]
  )

  native.java_import(
      name = "mediarouter_v7_import",
      jars = ["extras/android/support/v7/mediarouter/libs/android-support-v7-mediarouter.jar"]
  )

  native.android_library(
      name = "mediarouter_v7",
      custom_package = "android.support.v7.mediarouter",
      manifest = "extras/android/support/v7/mediarouter/AndroidManifest.xml",
      resource_files = native.glob(["extras/android/support/v7/mediarouter/res/**"]),
      deps = [
          ":appcompat_v7",
          ":mediarouter_v7_import",
      ]
  )

  native.java_import(
      name = "cardview_v7_import",
      jars = ["extras/android/support/v7/cardview/libs/android-support-v7-cardview.jar"]
  )

  native.android_library(
      name = "cardview_v7",
      custom_package = "android.support.v7.cardview",
      manifest = "extras/android/support/v7/cardview/AndroidManifest.xml",
      resource_files = native.glob(["extras/android/support/v7/cardview/res/**"]),
      deps = [":cardview_v7_import"]
  )

  native.java_import(
      name = "gridlayout_v7_import",
      jars = ["extras/android/support/v7/gridlayout/libs/android-support-v7-gridlayout.jar"]
  )

  native.android_library(
      name = "gridlayout_v7",
      custom_package = "android.support.v7.gridlayout",
      manifest = "extras/android/support/v7/gridlayout/AndroidManifest.xml",
      resource_files = native.glob(["extras/android/support/v7/gridlayout/res/**"]),
      deps = [":gridlayout_v7_import"]
  )

  native.java_import(
      name = "palette_v7_import",
      jars = ["extras/android/support/v7/palette/libs/android-support-v7-palette.jar"]
  )

  native.android_library(
      name = "palette_v7",
      custom_package = "android.support.v7.palette",
      manifest = "extras/android/support/v7/palette/AndroidManifest.xml",
      resource_files = native.glob(["extras/android/support/v7/palette/res/**"]),
      deps = [":palette_v7_import"]
  )

  native.java_import(
      name = "recyclerview_v7_import",
      jars = ["extras/android/support/v7/recyclerview/libs/android-support-v7-recyclerview.jar"]
  )

  native.android_library(
      name = "recyclerview_v7",
      custom_package = "android.support.v7.recyclerview",
      manifest = "extras/android/support/v7/recyclerview/AndroidManifest.xml",
      resource_files = native.glob(["extras/android/support/v7/recyclerview/res/**"]),
      deps = [":recyclerview_v7_import"]
  )

  if api_level >= 23:
    # Android 23 removed most of org.apache.http from android.jar and moved it
    # to a separate jar.
    native.java_import(
        name = "org_apache_http_legacy",
        jars = ["platforms/android-%d/optional/org.apache.http.legacy.jar" % api_level]
    )

  native.java_import(
      name = "appcompat_v4",
      jars = ["extras/android/support/v4/android-support-v4.jar"]
  )

  native.java_import(
      name = "appcompat_v13",
      jars = ["extras/android/support/v13/android-support-v13.jar"]
  )

  native.android_sdk(
      name = "sdk",
      build_tools_version = build_tools_version,
      proguard = ":proguard_binary",
      aapt = ":aapt_binary",
      dx = ":dx_binary",
      main_dex_list_creator = ":main_dex_list_creator",
      adb = "platform-tools/adb",
      framework_aidl = "platforms/android-%d/framework.aidl" % api_level,
      aidl = ":aidl_binary",
      android_jar = "platforms/android-%d/android.jar" % api_level,
      shrinked_android_jar = "platforms/android-%d/android.jar" % api_level,
      annotations_jar = "tools/support/annotations.jar",
      main_dex_classes = "build-tools/%s/mainDexClasses.rules" % build_tools_directory,
      apkbuilder = "@bazel_tools//third_party/java/apkbuilder:embedded_apkbuilder",
      zipalign = ":zipalign_binary",
      jack = ":fail",
      jill = ":fail",
      resource_extractor = ":fail"
  )

  native.java_import(
      name = "proguard_import",
      jars = ["tools/proguard/lib/proguard.jar"]
  )

  native.java_binary(
      name = "proguard_binary",
      main_class = "proguard.ProGuard",
      runtime_deps = [":proguard_import"]
  )

  native.filegroup(
      name = "build_tools_libs",
      srcs = native.glob([
          "build-tools/%s/lib/**" % build_tools_directory,
          # Build tools version 24.0.0 added a lib64 folder.
          "build-tools/%s/lib64/**" % build_tools_directory,
      ])
  )

  for tool in ["aapt", "aidl", "zipalign"]:
    native.genrule(
        name = tool + "_runner",
        outs = [tool + "_runner.sh"],
        srcs = [],
        cmd  = "\n".join([
            "cat > $@ << 'EOF'",
            "#!/bin/bash",
            "set -eu",
            # The tools under build-tools/VERSION require the libraries under build-tools/VERSION/lib,
            # so we can't simply depend on them as a file like we do with aapt.
            "SDK=$${0}.runfiles/%s" % name,
            "exec $${SDK}/build-tools/%s/%s $$*" % (build_tools_directory, tool),
            "EOF\n"]),
    )

    native.sh_binary(
        name = tool + "_binary",
        srcs = [tool + "_runner.sh"],
        data = [
            ":build_tools_libs",
            "build-tools/%s/%s" % (build_tools_directory, tool)
        ],
    )

  native.sh_binary(
      name = "fail",
      srcs = ["fail.sh"])

  native.genrule(
      name = "generate_fail_sh",
      srcs = [],
      outs = ["fail.sh"],
      cmd = "echo -e '#!/bin/bash\\nexit 1' >> $@; chmod +x $@",
  )


  native.genrule(
      name = "main_dex_list_creator_source",
      srcs = [],
      outs = ["main_dex_list_creator.sh"],
      cmd = "\n".join(["cat > $@ <<'EOF'",
            "#!/bin/bash",
            "",
            "MAIN_DEX_LIST=$$1",
            "STRIPPED_JAR=$$2",
            "JAR=$$3",
            "" +
            "DIRNAME=$$(dirname $$0)",
            "JAVA_BINARY=TBD/main_dex_list_creator_java",  # Proper runfiles path comes here
            "$$JAVA_BINARY $$STRIPPED_JAR $$JAR > $$MAIN_DEX_LIST",
            "exit $$?",
            "",
            "EOF\n"]),
  )

  native.sh_binary(
      name = "main_dex_list_creator",
      srcs = ["main_dex_list_creator.sh"],
      data = [":main_dex_list_creator_java"],
  )

  native.java_binary(
      name = "main_dex_list_creator_java",
      main_class = "com.android.multidex.ClassReferenceListBuilder",
      runtime_deps = [":dx_jar_import"],
  )

  native.java_binary(
      name = "dx_binary",
      main_class = "com.android.dx.command.Main",
      runtime_deps = [":dx_jar_import"],
  )

  native.filegroup(
      name = "dx_jar",
      srcs = ["build-tools/%s/lib/dx.jar" % build_tools_directory],
  )

  native.java_import(
      name = "dx_jar_import",
      jars = [":dx_jar"],
  )

  GOOGLE_PLAY_SERVICES_DIR = "extras/google/google_play_services/libproject/google-play-services_lib"

  native.java_import(
      name = "google_play_services_lib",
      jars = [GOOGLE_PLAY_SERVICES_DIR + "/libs/google-play-services.jar"])

  native.android_library(
      name = "google_play_services",
      custom_package = "com.google.android.gms",
      manifest = GOOGLE_PLAY_SERVICES_DIR + "/AndroidManifest.xml",
      exports_manifest = 1,
      resource_files = native.glob([GOOGLE_PLAY_SERVICES_DIR + "/res/**"]),
      proguard_specs = [GOOGLE_PLAY_SERVICES_DIR + "/proguard.txt"],
      deps = [":google_play_services_lib"])
