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

# Set path constants. These are relative to the Bazel source tree root.
$Icon = "site\images\favicon.ico"
$ArtworkBig = "scripts\packages\wix\dialog.bmp"
$ArtworkSmall = "scripts\packages\wix\banner.bmp"
$Licence = "scripts\packages\wix\license.rtf"
$Guids = "scripts\packages\wix\guids.txt"
$Wxs = "scripts\packages\wix\bazelmsi.wxs"

# Exit immediately when a Cmdlet fails.
$ErrorActionPreference = "Stop"

# Logs a message to stdout.
function Log-Info {
    Param([string]$msg)
    $basename = $PSCommandPath.Substring($PSCommandPath.LastIndexOfAny("/\") + 1)
    $time = (Get-Date).ToString("HH:mm:ss")
    Write-Host "INFO[$basename $time] $msg"
}

function Replace-Slashes {
    Param([string]$s)
    return $s.Replace("/", "\")
}

# Computes the release name from the 'BazelExe'.
# 'BazelExe' must already have been validated to match this regex.
# 'ReleaseName' is like "0.28.0rc5"
# 'Version' is the SemVer part of 'ReleaseName', e.g. "0.28.0"
function Compute-RelaseNameAndVersion {
    Param([Parameter(Mandatory=$true)][string]$BazelExe)

    $rel = [regex]::Match($BazelExe, "bazel-([^-]+)-windows.*\.exe$").captures.groups[1].value
    $ver = [regex]::Match($rel, "^([0-9]+\.[0-9]+\.[0-9]+)(rc[0-9]+)?$").captures.groups[1].value
    return $rel, $ver
}

# Generates a new GUID, prints it as uppercase.
# The WiX Toolkit expects uppercase GUIDs in the .wxs file.
function Generate-Guid {
    return [guid]::NewGuid().ToString().ToUpper()
}

# Returns the UpgradeGuid for this release.
function Get-UpgradeGuid {
    Param([Parameter(Mandatory=$true)][string]$release_name)

    $d = @{}
    $result = $null
    foreach ($line in Get-Content -Path $Guids) {
        $line = "$line".Split('#')[0]
        if ($line) {
            $k, $v = $line.Split(' ', 2)
            # Ensure all version prefixes in the file are unique.
            if ($d.ContainsKey($k)) {
                Throw "Duplicate relase prefix $k in $Guids"
            } else {
                $d[$k] = $null
            }
            # Ensure all GUIDs in the file are unique. We can use the same
            # hashtable because the value domains are distinct.
            if ($d.ContainsKey($v)) {
                Throw "Duplicate GUID $v in $Guids"
            } else {
                $d[$v] = $null
            }
            if (! $result -and $release_name.StartsWith($k)) {
                $result = "$v"
                # Do not return yet, so we check that all GUIDs are unique.
            }
        }
    }
    if ($result) {
        return "$result"
    } else {
        Throw "UpgradeGuid for $release_name not found in $Guids"
    }
}

# Returns the Bazel version (as a SemVer string) from the release name.
function Get-BazelVersion {
    if ("$ReleaseName" -match "rc[0-9]+$") {
        return "$ReleaseName".Substring(0, "$ReleaseName".LastIndexOf("rc"))
    } else {
        return "$ReleaseName"
    }
}

# Downloads and extracts the WiX Toolkit.
# Returns the path where the tools are (e.g. "candle.exe").
function Download-Wix {
    # Create a Bazel workspace just for the Wix tools.
    #
    # Downloading the zip with Start-BitsTransfer seems to fail on
    # permission errors, but Bazel can download and extract it for us.
    $wix_dir = "${WorkDir}\_wix"
    New-Item -Path "$wix_dir" -ItemType Directory -Force | Out-Null
    @"
load('@bazel_tools//tools/build_defs/repo:http.bzl', 'http_archive')

http_archive(
    name = "wix_toolset",
    url = "https://github.com/wixtoolset/wix3/releases/download/wix3111rtm/wix311-binaries.zip",
    sha256 = "37f0a533b0978a454efb5dc3bd3598becf9660aaf4287e55bf68ca6b527d051d",
    build_file_content = "exports_files(['candle.exe', 'light.exe'])",
)
"@ | Out-File -FilePath "${wix_dir}\WORKSPACE" -Force -Encoding ascii

    $owd = Get-Location
    $out_base = try {
            & {
            Set-Location -Path "$wix_dir"
            $output = & bazel --max_idle_secs=1 build `
                            @wix_toolset//:candle.exe @wix_toolset//:light.exe
            if (! $?) {
                # $? is a boolean: whether the last native program or Cmdlet was successful.
                Throw "$output"
            }

            $out_base = & bazel --max_idle_secs=1 info output_base
            if ($LASTEXITCODE -ne 0) {
                # $LASTEXITCODE is an integer, stores the exit code of the last native program.
                Throw "bazel info failed"
            }
            $out_base = Replace-Slashes($out_base)
            return "$out_base"
        }
    } finally {
        Set-Location $owd
    }

    return "${out_base}\external\wix_toolset"
}

# Creates the .wixobj file using candle.exe (the "compiler").
function Run-Candle {
    Param([Parameter(Mandatory=$true)][string]$wix_root)

    $out="${WorkDir}\bazelmsi.wixobj"

    $output = & "${wix_root}\candle.exe" `
        -nologo `
        -arch x64 `
        "-dBAZEL_EXE=$BazelExe" `
        "-dICON=$Icon" `
        "-dUPGRADE_GUID=$(Get-UpgradeGuid $ReleaseName)" `
        "-dRELEASE_NAME=$ReleaseName" `
        "-dVERSION=$(Get-BazelVersion)" `
        "-dRANDOM_GUID_1=$(Generate-Guid)" `
        "-dRANDOM_GUID_2=$(Generate-Guid)" `
        -o "$out" `
        "$Wxs"
    if (! $?) {
        Throw "$output"
    }
    return "$out"
}

# Creates the .msi file using light.exe (the "linker").
function Run-Light {
    Param(
        [Parameter(Mandatory=$true)][string]$wix_root,
        [Parameter(Mandatory=$true)][string]$wixobj
    )

    $output = & "${wix_root}\light.exe" `
        -nologo `
        -ext WixUIExtension `
        "-cultures:en-us" `
        "-dWixUILicenseRtf=$Licence" `
        "-dWixUIDialogBmp=$ArtworkBig" `
        "-dWixUIBannerBmp=$ArtworkSmall" `
        -o "$OutMsi" `
        "$wixobj"
    if (! $?) {
        Throw "$output"
    }
}

function Main {
    Param(
        [Parameter(Mandatory=$true)]
        [ValidatePattern("bazel-[0-9]+\.[0-9]+\.[0-9]+(rc[1-9]|rc[1-9][0-9]+)?-windows-x86_64.exe$")]
        [string]
        $BazelExe,

        [Parameter(Mandatory=$true)][string]$WorkDir,
        [Parameter(Mandatory=$true)][string]$OutMsi
    )

    $ReleaseName, $Version = Compute-RelaseNameAndVersion $BazelExe

    Log-Info "Building packaging tools"
    $wix_root = Download-Wix

    Log-Info "Creating wixobj"
    $wixobj = Run-Candle -wix_root "$wix_root"

    Log-Info "Creating msi"
    Run-Light -wix_root "$wix_root" -wixobj "$wixobj"

    Log-Info "Done: $OutMsi"
}
