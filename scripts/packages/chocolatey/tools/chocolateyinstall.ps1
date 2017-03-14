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
  -Checksum64Type "sha256" `
  -UnzipLocation "$packageDir"

write-host "Ensure that msys2 dll is present in PATH to allow bazel to be run from non-msys2 shells"

# from docs: https://github.com/chocolatey/choco/wiki/How-To-Parse-PackageParameters-Argument
$msys2Path = "c:\tools\msys64"
if ($packageParameters)
{
  $match_pattern = "\/(?<option>([a-zA-Z]+)):(?<value>([`"'])?([a-zA-Z0-9- _\\:\.]+)([`"'])?)|\/(?<option>([a-zA-Z]+))"
  $option_name = 'option'
  $value_name = 'value'

  if ($packageParameters -match $match_pattern)
  {
    $results = $packageParameters | Select-String $match_pattern -AllMatches
    $results.matches | % {
      $arguments.Add(
        $_.Groups[$option_name].Value.Trim(),
        $_.Groups[$value_name].Value.Trim())
    }
  }
  else
  {
    Throw "Package Parameters were found but were invalid (REGEX Failure)"
  }

  if ($arguments.ContainsKey("msys2Path")) {
    $msys2Path = $arguments["msys2Path"]
    Write-Host "msys2Path Argument Found: $msys2Path"
  }
}
Install-ChocolateyPath -PathToInstall "$msys2Path\usr\bin" -PathType "Machine"

$addToMsysPath = ($packageDir -replace '^([a-zA-Z]):\\(.*)','/$1/$2') -replace '\\','/'
write-host @"
bazel installed to $packageDir

To use it in powershell or cmd, you should ensure your PATH environment variable contains
  $($msys2Path)\usr\bin
BEFORE both
  c:\windows\system32 (because bash-on-windows' bash.exe will be found here, if it's installed)
  any references to msysgit (like c:\program files (x86)\git\bin or c:\program files (x86)\git\cmd) (because git's vendored version of msys2 will interfere with the real msys2)

To use it in msys2, you should add that to your msys2 PATH:
  export PATH=$($addToMsysPath):`$PATH

You also need, in your msys2 environment (adjust paths for your system):
  export JAVA_HOME="`$(ls -d C:/Program\ Files/Java/jdk* | sort | tail -n 1)`"
  export BAZEL_SH=c:/tools/msys64/usr/bin/bash.exe
  export BAZEL_PYTHON=c:/tools/python2/python.exe

See also https://bazel.build/docs/windows.html
"@

