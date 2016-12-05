param(
  [string] $version = "0.3.2",
  [int] $rc = 0,
  [switch] $fixPackage,
  [string] $mode = "local",
  [string] $checksum = ""
)

write-host "mode: $mode"
if ($mode -eq "release") {
  $tvVersion = $version
  $tvFilename = "bazel-$($tvVersion)-windows-x86_64.zip"
  $tvUri = "https://github.com/bazelbuild/bazel/releases/download/$($tvVersion)/$($tvFilename)"
  $tvReleaseNotesUri = "https://github.com/bazelbuild/bazel/releases/tag/$tvVersion"
} elseif ($mode -eq "rc") {
  $tvVersion = "$($version)-rc$($rc)"
  $tvFilename = "bazel-$($version)rc$($rc)-windows-x86_64.zip"
  $tvUri = "https://storage.googleapis.com/bazel/$($version)/rc$($rc)/$($tvFilename)"
  $tvReleaseNotesUri = "https://storage.googleapis.com/bazel/$($version)/rc$($rc)/index.html"
} elseif ($mode -eq "local") {
  $tvVersion = $version
  $tvFilename = "bazel-$($tvVersion)-windows-x86_64.zip"
  $tvUri = "http://localhost:8000/$($tvFilename)"
  $tvReleaseNotesUri = "http://localhost:8000/dummy"
} else {
  throw "mode parameter '$mode' unsupported. Please use local, rc, or release."
}

if ($fixPackage -eq $true) {
  $tvPackageFixVersion = "-$((get-date).tostring("yyyyMMdd-hhmmss"))"
}
rm -force -ErrorAction SilentlyContinue ./*.nupkg
rm -force -ErrorAction SilentlyContinue ./bazel.nuspec
rm -force -ErrorAction SilentlyContinue ./tools/LICENSE.txt
rm -force -ErrorAction SilentlyContinue ./tools/params.json
rm -force -ErrorAction SilentlyContinue ./tools/*.orig
if ($checksum -eq "") {
  rm -force -ErrorAction SilentlyContinue ./*.zip
}

if (($mode -eq "release") -or ($mode -eq "rc")) {
  Invoke-WebRequest "$($tvUri).sha256" -UseBasicParsing -passthru -outfile sha256.txt
  $tvChecksum = (gc sha256.txt).split(' ')[0]
  rm sha256.txt
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

write-host "Copying LICENSE.txt from repo-root to tools directory"
$licenseHeader = @"
From: https://github.com/bazelbuild/bazel/blob/master/LICENSE.txt

"@
add-content -value $licenseHeader -path "./tools/LICENSE.txt"
add-content -value (get-content "../../../LICENSE.txt") -path "./tools/LICENSE.txt"

$params = @{
  package = @{
    uri = $tvUri;
    checksum = $tvChecksum;
    checksumType = "sha256";
  }
}
add-content -value (ConvertTo-Json $params) -path "./tools/params.json"

choco pack ./bazel.nuspec
