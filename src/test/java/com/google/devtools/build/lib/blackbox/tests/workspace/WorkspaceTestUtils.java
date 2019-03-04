// Copyright 2019 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.blackbox.tests.workspace;

import com.google.devtools.build.lib.util.OS;
import java.util.Arrays;
import java.util.stream.Collectors;

public class WorkspaceTestUtils {
  public static String removeMsysFromPath(String path) {
    // if (!OS.WINDOWS.equals(OS.getCurrent())) return path;
    if (path.indexOf(';') == -1) {
      return path;
    }
    String[] parts = path.split(";");
    System.out.println("size: " + parts.length);
    return Arrays.stream(parts)
        .filter(s -> !s.contains("msys")).collect(Collectors.joining(";"));
  }

  public static void main(String[] args) {
    System.out.println("Filtered: " +
        removeMsysFromPath("C:\\python3\\Scripts\\;C:\\python3\\;C:\\Windows\\system32;C:\\Windows;C:\\Windows\\System32\\Wbem;C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\;C:\\Windows\\System32\\OpenSSH\\;C:\\ProgramData\\GooGet;C:\\Program Files\\Google\\Compute Engine\\metadata_scripts;C:\\Program Files (x86)\\Google\\Cloud SDK\\google-cloud-sdk\\bin;C:\\Program Files\\Google\\Compute Engine\\sysprep;C:\\ProgramData\\chocolatey\\bin;C:\\Program Files\\Git\\cmd;C:\\tools\\msys64\\usr\\bin;c:\\openjdk\\bin;C:\\Program Files (x86)\\Windows Kits\\8.1\\Windows Performance Toolkit\\;C:\\Program Files\\CMake\\bin;c:\\ninja;c:\\bazel;C:\\Users\\ichern\\AppData\\Local\\Microsoft\\WindowsApps;"));
  }
}
