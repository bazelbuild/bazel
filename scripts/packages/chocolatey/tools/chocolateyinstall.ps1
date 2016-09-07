$ErrorActionPreference = 'Stop'; # stop on all errors
$packageName= 'bazel'

$toolsDir = Split-Path -parent $MyInvocation.MyCommand.Definition
$packageDir = Split-Path -parent $toolsDir
$binRoot = (Get-ToolsLocation) -replace "\\", "/"

$destDir = "$binRoot/$packageName/"
if (-not(test-path "$binRoot/$packageName")) {
  new-item $destDir -type directory
}
cp "$packageDir/tools/$($packageName).exe" $destDir -force

write-host @"
bazel installed to $destDir

To use it, you should add that to your msys2 PATH:
  export PATH=$($destDir):`$PATH

You also need, in your msys2 environment:
  export JAVA_HOME=`"`$(ls -d C:/Program\ Files/Java/jdk* | sort | tail -n 1)`"
  export BAZEL_SH=c:/tools/msys64/usr/bin/bash.exe
  export BAZEL_PYTHON=c:/tools/python2/python.exe

See also https://bazel.io/docs/windows.html
"@
