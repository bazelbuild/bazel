param(
  [string] $version = "0.3.1",
  [switch] $isRelease
)

$tvVersion = $version
$tvFilename = "bazel_$($version)_windows_x86_64.zip"
if ($isRelease) {
  $tvUri = "https://github.com/bazelbuild/bazel/releases/download/$($version)/$($tvFilename)"
} else {
  $tvUri = "http://localhost:8000/$($tvFilename)"
}
write-host "download uri: $($tvUri)"

choco uninstall bazel --force -y
rm -force ./*.nupkg
rm -force ./*.zip
rm -force ./bazel.nuspec
rm -force ./tools/chocolateyinstall.ps1

Add-Type -A System.IO.Compression.FileSystem
$zipDir = "$pwd/../../../output"
$zipFile = "$pwd/$($tvFilename)"
write-host "Creating zip package from directory: $zipDir to file: $zipFile"
[IO.Compression.ZipFile]::CreateFromDirectory($zipDir, $zipFile)
$tvChecksum = (get-filehash $zipFile -algorithm sha256).Hash
write-host "zip sha256: $tvChecksum"

$nuspecTemplate = get-content "bazel.nuspec.template" | out-string
$installerScriptTemplate = get-content "chocolateyinstall.ps1.template" | out-string
$nuspecExpanded = $ExecutionContext.InvokeCommand.ExpandString($nuspecTemplate)
add-content -value $nuspecExpanded -path bazel.nuspec
$installerScriptExpanded = $ExecutionContext.InvokeCommand.ExpandString($installerScriptTemplate)
$installerScriptExpanded = $installerScriptExpanded -replace "ps_var_","$"
$installerScriptExpanded = $installerScriptExpanded -replace "escape_char","``"
add-content -value $installerScriptExpanded -path ./tools/chocolateyinstall.ps1

write-host "Copying LICENSE.txt from repo-root to tools directory"
$licenseHeader = @"
From: https://github.com/bazelbuild/bazel/blob/master/LICENSE.txt

"@
add-content -value $licenseHeader -path "./tools/LICENSE.txt"
add-content -value (get-content "../../../LICENSE.txt") -path "./tools/LICENSE.txt"

choco pack ./bazel.nuspec
$pkg = get-childitem ./bazel*.nupkg
choco install $pkg.FullName --verbose --debug --force -y
