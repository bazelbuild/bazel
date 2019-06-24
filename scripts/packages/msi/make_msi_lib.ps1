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
$Icon = 'site\images\favicon.ico'
$ArtworkBig = 'scripts\packages\msi\dialog.bmp'
$ArtworkSmall = 'scripts\packages\msi\banner.bmp'
$Licence = 'scripts\packages\msi\license.rtf'
$Guids = 'scripts\packages\msi\guids.txt'
$Wxs = 'scripts\packages\msi\bazelmsi.wxs'

# Logs a message to stdout.
function Log-Info {
    param (
      [string]$msg
    )
    $basename = $PSCommandPath.Substring($PSCommandPath.LastIndexOfAny('/\') + 1)
    $time = (Get-Date).ToString(.HH:mm:ss.)
    Write-Host 'INFO[$basename $time] $msg'
}

function Replace-Slashes {
    param (
      [string]$s
    )
    return $s.Replace('/', '\')
}

# Computes the release name from the 'BazelExe'.
# 'BazelExe' must already have been validated to match this regex.
# 'ReleaseName' is like "0.28.0rc5"
# 'Version' is the SemVer part of 'ReleaseName', e.g. "0.28.0"
function Compute-RelaseNameAndVersion {
    param (
      [Parameter(Mandatory=$true)][string]$BazelExe
    )

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
    param (
      [Parameter(Mandatory=$true)][string]$release_name
    )

    $d = @{}
    $result = $null
    foreach ($line in Get-Content -Path $Guids) {
        $line = '$line'.Split('#')[0]
        if ($line) {
            $k, $v = $line.Split(' ', 2)
            # Ensure all version prefixes in the file are unique.
            if ($d.ContainsKey($k)) {
                throw 'Duplicate relase prefix $k in $Guids'
            }
            else {
                $d[$k] = $null
            }
            # Ensure all GUIDs in the file are unique. We can use the same
            # hashtable because the value domains are distinct.
            if ($d.ContainsKey($v)) {
                throw 'Duplicate GUID $v in $Guids'
            }
            else {
                $d[$v] = $null
            }
            if (! $result -and $release_name.StartsWith($k)) {
                $result = '$v'
                # Do not return yet, so we check that all GUIDs are unique.
            }
        }
    }
    if ($result) {
        return $result
    }
    else {
        throw "UpgradeGuid for $release_name not found in $Guids"
    }
}

# Returns the Bazel version (as a SemVer string) from the release name.
function Get-BazelVersion {
    if ($ReleaseName -match "rc[0-9]+$") {
        return $ReleaseName.Substring(0, $ReleaseName.LastIndexOf('rc'))
    }
    else {
        return $ReleaseName
    }
}

# Downloads and extracts the WiX Toolkit.
# Returns the path where the tools are (e.g. "candle.exe").
function Download-Wix {
    # Use TLS1.2 for HTTPS (fixes an issue where later steps can't connect to github.com).
    # See https://github.com/bazelbuild/continuous-integration/blob/master/buildkite/setup-windows.ps1
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

    $wix_dir = $(New-Item -Path "${WorkDir}\_wix" -ItemType Directory -Force).FullName
    $zip_file = "$wix_dir\wix311.zip"
    (New-Object Net.WebClient).DownloadFile(
        'https://github.com/wixtoolset/wix3/releases/download/wix3111rtm/wix311-binaries.zip',
        $zip_file)

    $actual_sha = (Get-FileHash -Path $zip_file -Algorithm SHA256).Hash.ToString()
    $expected_sha = '37F0A533B0978A454EFB5DC3BD3598BECF9660AAF4287E55BF68CA6B527D051D'
    if ($actual_sha -ne $expected_sha) {
        throw "Bad checksum: wix311-binaries.zip SHA256 is $actual_sha, expected $expected_sha"
    }

    Expand-Archive -Path $zip_file -DestinationPath $wix_dir -Force

    return $wix_dir
}

# Creates the .wixobj file using candle.exe (the "compiler").
function Run-Candle {
    param (
      [Parameter(Mandatory=$true)][string]$wix_root
    )

    $out = "${WorkDir}\bazelmsi.wixobj"

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
        -o $out `
        $Wxs
    if (! $?) {
        throw $output
    }
    return $out
}

# Creates the .msi file using light.exe (the "linker").
function Run-Light {
    param (
        [Parameter(Mandatory=$true)][string]$wix_root,
        [Parameter(Mandatory=$true)][string]$wixobj
    )

    $output = & "${wix_root}\light.exe" `
        -nologo `
        -ext WixUIExtension `
        '-cultures:en-us' `
        "-dWixUILicenseRtf=$Licence" `
        "-dWixUIDialogBmp=$ArtworkBig" `
        "-dWixUIBannerBmp=$ArtworkSmall" `
        -o $OutMsi `
        $wixobj
    if (! $?) {
        throw $output
    }
}

function Make-Msi {
    param (
        [Parameter(Mandatory=$true)]
        [ValidatePattern("bazel-[0-9]+\.[0-9]+\.[0-9]+(rc[1-9]|rc[1-9][0-9]+)?-windows-x86_64.exe$")]
        [string]
        $BazelExe,

        [Parameter(Mandatory=$true)][string]$WorkDir,
        [Parameter(Mandatory=$true)][string]$OutMsi
    )

    $ReleaseName, $Version = Compute-RelaseNameAndVersion $BazelExe

    Log-Info 'Downloading WiX Toolkit'
    $wix_root = Download-Wix

    Log-Info 'Creating wixobj'
    $wixobj = Run-Candle -wix_root $wix_root

    Log-Info 'Creating msi'
    Run-Light -wix_root $wix_root -wixobj $wixobj

    Log-Info "Done: $OutMsi"
}
