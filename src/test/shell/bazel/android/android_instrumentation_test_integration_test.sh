#!/bin/bash
#
# Copyright 2017 The Bazel Authors. All rights reserved.
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

# For these tests to run do the following:
#
#   1. Install an Android SDK from https://developer.android.com
#   2. Set the $ANDROID_HOME environment variable
#   3. Uncomment the line in WORKSPACE containing android_sdk_repository

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/android_helper.sh" \
  || { echo "android_helper.sh not found!" >&2; exit 1; }
fail_if_no_android_sdk

source "${CURRENT_DIR}/../../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

# TODO(#8169): Make this test compatible with Python toolchains. Blocked on the
# fact that there's no PY3 environment on our Mac workers
# (bazelbuild/continuous-integration#578).
add_to_bazelrc "build --incompatible_use_python_toolchains=false"

function setup_android_instrumentation_test_env() {
  mkdir -p java/com/bin/res/values
  mkdir -p javatests/com/bin

  # Targets for android_binary application under test
  cat > java/com/bin/BUILD <<EOF
android_binary(
  name = 'target',
  manifest = 'AndroidManifest.xml',
  deps = [':lib'],
  visibility = ["//visibility:public"],
)
android_library(
  name = 'lib',
  manifest = 'AndroidManifest.xml',
  exports_manifest = 0,
  resource_files = ['res/values/values.xml'],
  srcs = ['Bar.java'],
  visibility = ["//visibility:public"],
)
EOF
  cat > java/com/bin/AndroidManifest.xml <<EOF
<?xml version="1.0" encoding="utf-8"?>
<manifest package='com.bin' xmlns:android="http://schemas.android.com/apk/res/android" />
EOF
  cat > java/com/bin/Bar.java <<EOF
package com.bin;
public class Bar { }
EOF
  cat > java/com/bin/res/values/values.xml <<EOF
<?xml version="1.0" encoding="utf-8"?>
<resources xmlns:android="http://schemas.android.com/apk/res/android">
</resources>
EOF

  # Targets for instrumentation android_binary
  cat > javatests/com/bin/BUILD <<EOF
android_binary(
  name = 'instr',
  srcs = ['BarTest.java'],
  manifest = 'AndroidManifest.xml',
  instruments = '//java/com/bin:target',
  deps = ['//java/com/bin:lib'],
)
EOF
  cat > javatests/com/bin/AndroidManifest.xml <<EOF
<?xml version="1.0" encoding="utf-8"?>
<manifest package='com.bin' xmlns:android='http://schemas.android.com/apk/res/android'>
  <instrumentation android:targetPackage='com.bin' android:name='some.test.runner' />
</manifest>
EOF
  cat > javatests/com/bin/BarTest.java <<EOF
package com.bin;
public class BarTest {
  public Bar getBar() {
    return new Bar();
  }
}
EOF
}

function test_correct_target_package_build_succeed() {
  create_new_workspace
  setup_android_sdk_support
  setup_android_instrumentation_test_env
  assert_build //javatests/com/bin:instr
}

function test_incorrect_target_package_build_failure() {
  create_new_workspace
  setup_android_sdk_support
  setup_android_instrumentation_test_env

  cat > javatests/com/bin/AndroidManifest.xml <<EOF
<?xml version="1.0" encoding="utf-8"?>
<manifest package='com.bin' xmlns:android='http://schemas.android.com/apk/res/android'>
  <instrumentation android:targetPackage='not.com.bin' android:name='some.test.runner' />
</manifest>
EOF

  assert_build_fails //javatests/com/bin:instr \
    "does not match the package name of"
}

function test_multiple_instrumentations_with_different_package_names_build_failure() {
  create_new_workspace
  setup_android_sdk_support
  setup_android_instrumentation_test_env

  cat > javatests/com/bin/AndroidManifest.xml <<EOF
<?xml version="1.0" encoding="utf-8"?>
<manifest package='com.bin' xmlns:android='http://schemas.android.com/apk/res/android'>
  <instrumentation android:targetPackage='com.bin' android:name='some.test.runner' />
  <instrumentation android:targetPackage='not.com.bin' android:name='some.test.runner' />
</manifest>
EOF

  assert_build_fails //javatests/com/bin:instr \
    "do not reference the same target package"
}

function test_android_instrumentation_binary_class_filtering() {
  create_new_workspace
  setup_android_sdk_support
  mkdir -p java/com/bin
  cat > java/com/bin/BUILD <<EOF
android_binary(
  name = 'instr',
  srcs = ['Foo.java'],
  manifest = 'TestAndroidManifest.xml',
  instruments = ':target',
  deps = [':lib'],
)
android_binary(
  name = 'target',
  manifest = 'AndroidManifest.xml',
  deps = [':lib'],
)
android_library(
  name = 'lib',
  manifest = 'AndroidManifest.xml',
  resource_files = ['res/values/values.xml'],
  srcs = ['Bar.java', 'Baz.java'],
)
EOF
  cat > java/com/bin/AndroidManifest.xml <<EOF
<manifest package='com.bin' />
EOF
  cat > java/com/bin/TestAndroidManifest.xml <<EOF
<?xml version="1.0" encoding="utf-8"?>
<manifest package='com.bin' xmlns:android='http://schemas.android.com/apk/res/android'>
  <instrumentation android:targetPackage='com.bin' android:name='some.test.runner' />
</manifest>
EOF
  cat > java/com/bin/Foo.java <<EOF
package com.bin;
public class Foo {
  public Bar getBar() {
    return new Bar();
  }
  public Baz getBaz() {
    return new Baz();
  }
}
EOF
  cat > java/com/bin/Bar.java <<EOF
package com.bin;
public class Bar {
  public Baz getBaz() {
    return new Baz();
  }
}
EOF
  cat > java/com/bin/Baz.java <<EOF
package com.bin;
public class Baz {}
EOF
  mkdir -p java/com/bin/res/values
  cat > java/com/bin/res/values/values.xml <<EOF
<?xml version="1.0" encoding="utf-8"?>
<resources xmlns:android="http://schemas.android.com/apk/res/android">
</resources>
EOF
  assert_build //java/com/bin:instr
  output_classes=$(zipinfo -1 bazel-bin/java/com/bin/instr_filtered.jar)
  assert_one_of $output_classes "META-INF/MANIFEST.MF"
  assert_one_of $output_classes "com/bin/Foo.class"
  assert_not_one_of $output_classes "com/bin/R.class"
  assert_not_one_of $output_classes "com/bin/Bar.class"
  assert_not_one_of $output_classes "com/bin/Baz.class"
}

run_suite "android_instrumentation_test integration tests"

