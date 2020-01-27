# Copyright 2019 The Bazel Authors. All rights reserved.
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

# This module creates a Windows Installer (.msi) package for a Bazel release.
#
# Example usage:
#   C:\src\bazel> powershell -NonInteractive scripts\packages\msi\make_msi.ps1 outputs\bazel-0.28.0rc5-windows-x86_64.exe
#
# Mandatory arguments:
#   -BazelExe <path>: path to the Bazel binary to package. Its name must
#                     conform to "bazel-<version>-windows-x86_64.exe"
#
# Optional arguments:
#    -OutMsi: path of the output file (default: same path as -BazelExe, with
#             ".msi" extension)
#    -WorkDir: directory for temp files (default: ".\_make_msi_tmp_")

# Command line arguments.
param (
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidatePattern("bazel-[0-9]+\.[0-9]+\.[0-9]+(rc[1-9]|rc[1-9][0-9]+)?-windows-x86_64.exe$")]
    [string]$BazelExe,

    [string]$WorkDir,
    [string]$OutMsi
)

# Exit immediately when a Cmdlet fails.
$ErrorActionPreference = 'Stop'

Import-Module .\scripts\packages\msi\make_msi_lib.ps1

# Ensure all paths are Windows-style.
$BazelExe = "$(Replace-Slashes($BazelExe))"

if ($WorkDir) {
    $WorkDir = "$(Replace-Slashes($WorkDir))"
}
else {
    $WorkDir = '.\_make_msi_tmp_'
}

if ($OutMsi) {
    $OutMsi = "$(Replace-Slashes($OutMsi))"
}
else {
    $OutMsi = $BazelExe.Substring(0, $BazelExe.Length - 3) + 'msi'
}

Make-Msi -BazelExe $BazelExe -WorkDir $WorkDir -OutMsi $OutMsi
