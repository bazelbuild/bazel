param(
  [string] $version = "0.3.2",
  [int] $rc = 0,
  [switch] $fixPackage,
  [string] $mode = "local",
  [string] $checksum = ""
)

$ErrorActionPreference = 'Stop'

write-host "mode: $mode"
if ($mode -eq "release") {
  $tvVersion = $version
  $tvFilename = "bazel-$($tvVersion)-windows-x86_64.zip"
  $tvUri = "https://github.com/bazelbuild/bazel/releases/download/$($tvVersion)/$($tvFilename)"
  $tvReleaseNotesUri = "https://github.com/bazelbuild/bazel/releases/tag/$tvVersion"
} elseif ($mode -eq "rc") {
  $tvVersion = "$($version)-rc$($rc)"
  $tvFilename = "bazel-$($version)rc$($rc)-windows-x86_64.zip"
  $tvUri = "https://releases.bazel.build/$($version)/rc$($rc)/$($tvFilename)"
  $tvReleaseNotesUri = "https://releases.bazel.build/$($version)/rc$($rc)/index.html"
} elseif ($mode -eq "local") {
  $tvVersion = $version
  $tvFilename = "bazel-$($tvVersion)-windows-x86_64.zip"
  $tvUri = "http://localhost:8000/$($tvFilename)"
  $tvReleaseNotesUri = "http://localhost:8000/dummy"
} else {
  throw "mode parameter '$mode' unsupported. Please use local, rc, or release."
}

if ($fixPackage -eq $true) {
  $prefix = "-"
  if ($mode -eq "release") {
    $prefix = "."
  }
  $tvPackageFixVersion = "$($prefix)$((get-date).tostring("yyyyMMdd_hhmmss"))"
}
remove-item -force -ErrorAction SilentlyContinue "./*.nupkg"
remove-item -force -ErrorAction SilentlyContinue "./bazel.nuspec"
remove-item -force -ErrorAction SilentlyContinue "./tools/LICENSE*"
remove-item -force -ErrorAction SilentlyContinue "./tools/*.orig"
remove-item -force -ErrorAction SilentlyContinue "./tools/params.*"
if ($checksum -eq "") {
  remove-item -force -ErrorAction SilentlyContinue ./*.zip
}
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
if (($mode -eq "release") -or ($mode -eq "rc")) {
  Invoke-WebRequest "$($tvUri).sha256" -UseBasicParsing -passthru -outfile sha256.txt
  $tvChecksum = (gc sha256.txt).split(' ')[0]
  remove-item sha256.txt
} elseif ($mode -eq "local") {
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

write-host "Copying LICENSE from repo-root to tools directory"
$licenseHeader = @"
From: https://github.com/bazelbuild/bazel/blob/master/LICENSE

"@
add-content -value $licenseHeader -path "./tools/LICENSE"
add-content -value (get-content "../../../LICENSE") -path "./tools/LICENSE"

$params = @"
$tvUri
$tvChecksum
"@
add-content -value $params -path "./tools/params.txt"

choco pack ./bazel.nuspec
