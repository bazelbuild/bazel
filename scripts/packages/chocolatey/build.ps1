param(
  [string] $version = "0.3.2",
  [switch] $isRelease
)

$tvVersion = $version
$tvFilename = "bazel-$($version)-windows-x86_64.zip"
if ($isRelease) {
  $tvUri = "https://github.com/bazelbuild/bazel/releases/download/$($version)/$($tvFilename)"
} else {
  $tvUri = "http://localhost:8000/$($tvFilename)"
}
write-host "download uri: $($tvUri)"

rm -force -ErrorAction SilentlyContinue ./*.nupkg
rm -force -ErrorAction SilentlyContinue ./*.zip
rm -force -ErrorAction SilentlyContinue ./bazel.nuspec
rm -force -ErrorAction SilentlyContinue ./tools/chocolateyinstall.ps1
rm -force -ErrorAction SilentlyContinue ./tools/chocolateyuninstall.ps1
rm -force -ErrorAction SilentlyContinue ./tools/LICENSE.txt

if ($isRelease) {
  Invoke-WebRequest "$($tvUri).sha256" -UseBasicParsing -passthru -outfile sha256.txt
  $tvChecksum = (gc sha256.txt).split(' ')[0]
  rm sha256.txt
} else {
  Add-Type -A System.IO.Compression.FileSystem
  $outputDir = "$pwd/../../../output"
  $zipFile = "$pwd/$($tvFilename)"
  write-host "Creating zip package with $outputDir/bazel.exe: $zipFile"
  Compress-Archive -Path "$outputDir/bazel.exe" -DestinationPath $zipFile
  $tvChecksum = (get-filehash $zipFile -algorithm sha256).Hash
  write-host "zip sha256: $tvChecksum"
}
$nuspecTemplate = get-content "bazel.nuspec.template" | out-string
$nuspecExpanded = $ExecutionContext.InvokeCommand.ExpandString($nuspecTemplate)
add-content -value $nuspecExpanded -path bazel.nuspec

$installerScriptTemplate = get-content "chocolateyinstall.ps1.template" | out-string
$installerScriptExpanded = $ExecutionContext.InvokeCommand.ExpandString($installerScriptTemplate)
$installerScriptExpanded = $installerScriptExpanded -replace "ps_var_","$"
$installerScriptExpanded = $installerScriptExpanded -replace "escape_char","``"
add-content -value $installerScriptExpanded -path ./tools/chocolateyinstall.ps1

$uninstallerScriptTemplate = get-content "chocolateyuninstall.ps1.template" | out-string
$uninstallerScriptExpanded = $ExecutionContext.InvokeCommand.ExpandString($uninstallerScriptTemplate)
$uninstallerScriptExpanded = $uninstallerScriptExpanded -replace "ps_var_","$"
$uninstallerScriptExpanded = $uninstallerScriptExpanded -replace "escape_char","``"
add-content -value $uninstallerScriptExpanded -path ./tools/chocolateyuninstall.ps1

write-host "Copying LICENSE.txt from repo-root to tools directory"
$licenseHeader = @"
From: https://github.com/bazelbuild/bazel/blob/master/LICENSE.txt

"@
add-content -value $licenseHeader -path "./tools/LICENSE.txt"
add-content -value (get-content "../../../LICENSE.txt") -path "./tools/LICENSE.txt"

choco pack ./bazel.nuspec
