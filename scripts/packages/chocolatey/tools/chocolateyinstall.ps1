# IMPORTANT: Before releasing this package, copy/paste the next 2 lines into PowerShell to remove all comments from this file:
#   $f='c:\path\to\thisFile.ps1'
#   gc $f | ? {$_ -notmatch "^\s*#"} | % {$_ -replace '(^.*?)\s*?[^``]#.*','$1'} | Out-File $f+".~" -en utf8; mv -fo $f+".~" $f

$ErrorActionPreference = 'Stop'; # stop on all errors

#Items that could be replaced based on what you call chocopkgup.exe with
#{{PackageName}} - Package Name (should be same as nuspec file and folder) |/p
#{{PackageVersion}} - The updated version | /v
#{{DownloadUrl}} - The url for the native file | /u
#{{PackageFilePath}} - Downloaded file if including it in package | /pp
#{{PackageGuid}} - This will be used later | /pg
#{{DownloadUrlx64}} - The 64-bit url for the native file | /u64
#{{Checksum}} - The checksum for the url | /c
#{{Checksumx64}} - The checksum for the 64-bit url | /c64

$packageName= 'bazel' # arbitrary name for the package, used in messages
$toolsDir   = "$(Split-Path -parent $MyInvocation.MyCommand.Definition)"
$url        = '{{DownloadUrl}}' # download url, HTTPS preferred
$url64      = '{{DownloadUrlx64}}' # 64bit URL here (HTTPS preferred) or remove - if installer contains both (very rare), use $url
#$fileLocation = Join-Path $toolsDir 'NAME_OF_EMBEDDED_INSTALLER_FILE'
#$fileLocation = '\\SHARE_LOCATION\to\INSTALLER_FILE'

$packageArgs = @{
  packageName   = $packageName
  unzipLocation = $toolsDir
  fileType      = 'EXE_MSI_OR_MSU' #only one of these: exe, msi, msu
  url           = $url
  url64bit      = $url64
  #file         = $fileLocation

  softwareName  = 'bazel*' #part or all of the Display Name as you see it in Programs and Features. It should be enough to be unique

  # Checksums are now required as of 0.10.0.
  # To determine checksums, you can get that from the original site if provided. 
  # You can also use checksum.exe (choco install checksum) and use it 
  # e.g. checksum -t sha256 -f path\to\file
  checksum      = '{{Checksum}}'
  checksumType  = 'md5' #default is md5, can also be sha1, sha256 or sha512
  checksum64    = '{{Checksumx64}}'
  checksumType64= 'md5' #default is checksumType

  #MSI
  silentArgs    = "/qn /norestart /l*v `"$($env:TEMP)\$($packageName).$($env:chocolateyPackageVersion).MsiInstall.log`"" # ALLUSERS=1 DISABLEDESKTOPSHORTCUT=1 ADDDESKTOPICON=0 ADDSTARTMENU=0
  validExitCodes= @(0, 3010, 1641)
  #OTHERS
  # Uncomment matching EXE type (sorted by most to least common)
  #silentArgs   = '/S'           # NSIS
  #silentArgs   = '/VERYSILENT /SUPPRESSMSGBOXES /NORESTART /SP-' # Inno Setup
  #silentArgs   = '/s'           # InstallShield
  #silentArgs   = '/s /v"/qn"' # InstallShield with MSI
  #silentArgs   = '/s'           # Wise InstallMaster
  #silentArgs   = '-s'           # Squirrel
  #silentArgs   = '-q'           # Install4j
  #silentArgs   = '-s -u'        # Ghost
  # Note that some installers, in addition to the silentArgs above, may also need assistance of AHK to achieve silence.
  #silentArgs   = ''             # none; make silent with input macro script like AutoHotKey (AHK)
                                 #       https://chocolatey.org/packages/autohotkey.portable
  #validExitCodes= @(0) #please insert other valid exit codes here
}

Install-ChocolateyPackage @packageArgs # https://chocolatey.org/docs/helpers-install-chocolatey-package
#Install-ChocolateyZipPackage @packageArgs # https://chocolatey.org/docs/helpers-install-chocolatey-zip-package
## If you are making your own internal packages (organizations), you can embed the installer or 
## put on internal file share and use the following instead (you'll need to add $file to the above)
#Install-ChocolateyInstallPackage @packageArgs # https://chocolatey.org/docs/helpers-install-chocolatey-install-package

## Main helper functions - these have error handling tucked into them already
## see https://chocolatey.org/docs/helpers-reference

## Install an application, will assert administrative rights
## - https://chocolatey.org/docs/helpers-install-chocolatey-package
## - https://chocolatey.org/docs/helpers-install-chocolatey-install-package
## add additional optional arguments as necessary
##Install-ChocolateyPackage $packageName $fileType $silentArgs $url [$url64 -validExitCodes $validExitCodes -checksum $checksum -checksumType $checksumType -checksum64 $checksum64 -checksumType64 $checksumType64]

## Download and unpack a zip file - https://chocolatey.org/docs/helpers-install-chocolatey-zip-package
##Install-ChocolateyZipPackage $packageName $url $toolsDir [$url64 -checksum $checksum -checksumType $checksumType -checksum64 $checksum64 -checksumType64 $checksumType64]

## Install Visual Studio Package - https://chocolatey.org/docs/helpers-install-chocolatey-vsix-package
#Install-ChocolateyVsixPackage $packageName $url [$vsVersion] [-checksum $checksum -checksumType $checksumType]
#Install-ChocolateyVsixPackage @packageArgs

## see the full list at https://chocolatey.org/docs/helpers-reference

## downloader that the main helpers use to download items
## if removing $url64, please remove from here
## - https://chocolatey.org/docs/helpers-get-chocolatey-web-file
#Get-ChocolateyWebFile $packageName 'DOWNLOAD_TO_FILE_FULL_PATH' $url $url64

## Installer, will assert administrative rights - used by Install-ChocolateyPackage
## use this for embedding installers in the package when not going to community feed or when you have distribution rights
## - https://chocolatey.org/docs/helpers-install-chocolatey-install-package
#Install-ChocolateyInstallPackage $packageName $fileType $silentArgs '_FULLFILEPATH_' -validExitCodes $validExitCodes

## Unzips a file to the specified location - auto overwrites existing content
## - https://chocolatey.org/docs/helpers-get-chocolatey-unzip
#Get-ChocolateyUnzip "FULL_LOCATION_TO_ZIP.zip" $toolsDir

## Runs processes asserting UAC, will assert administrative rights - used by Install-ChocolateyInstallPackage
## - https://chocolatey.org/docs/helpers-start-chocolatey-process-as-admin
#Start-ChocolateyProcessAsAdmin 'STATEMENTS_TO_RUN' 'Optional_Application_If_Not_PowerShell' -validExitCodes $validExitCodes

## add specific folders to the path - any executables found in the chocolatey package 
## folder will already be on the path. This is used in addition to that or for cases 
## when a native installer doesn't add things to the path.
## - https://chocolatey.org/docs/helpers-install-chocolatey-path
#Install-ChocolateyPath 'LOCATION_TO_ADD_TO_PATH' 'User_OR_Machine' # Machine will assert administrative rights

## Add specific files as shortcuts to the desktop
## - https://chocolatey.org/docs/helpers-install-chocolatey-shortcut
#$target = Join-Path $toolsDir "$($packageName).exe"
# Install-ChocolateyShortcut -shortcutFilePath "<path>" -targetPath "<path>" [-workDirectory "C:\" -arguments "C:\test.txt" -iconLocation "C:\test.ico" -description "This is the description"]

## Outputs the bitness of the OS (either "32" or "64")
## - https://chocolatey.org/docs/helpers-get-o-s-architecture-width
#$osBitness = Get-ProcessorBits

## Set persistent Environment variables
## - https://chocolatey.org/docs/helpers-install-chocolatey-environment-variable
#Install-ChocolateyEnvironmentVariable -variableName "SOMEVAR" -variableValue "value" [-variableType = 'Machine' #Defaults to 'User']

## Set up a file association
## - https://chocolatey.org/docs/helpers-install-chocolatey-file-association
#Install-ChocolateyFileAssociation 

## Adding a shim when not automatically found - Cocolatey automatically shims exe files found in package directory.
## - https://chocolatey.org/docs/helpers-install-bin-file
## - https://chocolatey.org/docs/create-packages#how-do-i-exclude-executables-from-getting-shims
#Install-BinFile

##PORTABLE EXAMPLE
#$toolsDir = "$(Split-Path -parent $MyInvocation.MyCommand.Definition)"
# despite the name "Install-ChocolateyZipPackage" this also works with 7z archives
#Install-ChocolateyZipPackage $packageName $url $toolsDir $url64
## END PORTABLE EXAMPLE

## [DEPRECATING] PORTABLE EXAMPLE
#$binRoot = Get-BinRoot
#$installDir = Join-Path $binRoot "$packageName"
#Write-Host "Adding `'$installDir`' to the path and the current shell path"
#Install-ChocolateyPath "$installDir"
#$env:Path = "$($env:Path);$installDir"

# if removing $url64, please remove from here
# despite the name "Install-ChocolateyZipPackage" this also works with 7z archives
#Install-ChocolateyZipPackage "$packageName" "$url" "$installDir" "$url64"
## END PORTABLE EXAMPLE
