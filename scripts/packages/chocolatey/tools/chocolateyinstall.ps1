$ErrorActionPreference = 'Stop'; # stop on all errors
$packageName = 'bazel'

$toolsDir = Split-Path -parent $MyInvocation.MyCommand.Definition
$paramsText = get-content "$($toolsDir)\params.txt"
write-host "Content of $($toolsDir)\params.txt:"
write-host $paramsText
write-host "url:  $($paramsText[0])"
write-host "hash: $($paramsText[1])"
write-host "Type: $($paramsText.GetType())"

$packageDir = Split-Path -parent $toolsDir

Install-ChocolateyZipPackage -PackageName "$packageName" `
  -Url64bit "$($paramsText[0])" `
  -Checksum64 "$($paramsText[1])" `
  -ChecksumType64 "sha256" `
  -UnzipLocation "$packageDir"

write-host @"
bazel installed to $packageDir

You also need, in your environment variables (adjust paths for your system):
  BAZEL_SH=c:\tools\msys64\usr\bin\bash.exe
  BAZEL_PYTHON=c:\tools\python2\python.exe

See also https://bazel.build/docs/windows.html
"@

