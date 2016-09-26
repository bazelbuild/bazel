## Summary
How do I create packages? See https://chocolatey.org/docs/create-packages

If you are submitting packages to the community feed (https://chocolatey.org)
always try to ensure you have read, understood and adhere to the create
packages wiki link above.

## Automatic Packaging Updates?
Consider making this package an automatic package, for the best 
maintainability over time. Read up at https://chocolatey.org/docs/automatic-packages

## Shim Generation
Any executables you include in the package or download (but don't call 
install against using the built-in functions) will be automatically shimmed.

This means those executables will automatically be included on the path.
Shim generation runs whether the package is self-contained or uses automation 
scripts. 

By default, these are considered console applications.

If the application is a GUI, you should create an empty file next to the exe 
named 'name.exe.gui' e.g. 'bob.exe' would need a file named 'bob.exe.gui'.
See https://chocolatey.org/docs/create-packages#how-do-i-set-up-shims-for-applications-that-have-a-gui

If you want to ignore the executable, create an empty file next to the exe 
named 'name.exe.ignore' e.g. 'bob.exe' would need a file named 
'bob.exe.ignore'. 
See https://chocolatey.org/docs/create-packages#how-do-i-exclude-executables-from-getting-shims

## Self-Contained? 
If you have a self-contained package, you can remove the automation scripts 
entirely and just include the executables, they will automatically get shimmed, 
which puts them on the path. Ensure you have the legal right to distribute 
the application though. See https://chocolatey.org/docs/legal. 

You should read up on the Shim Generation section to familiarize yourself 
on what to do with GUI applications and/or ignoring shims.

## Automation Scripts
You have a powerful use of Chocolatey, as you are using PowerShell. So you
can do just about anything you need. Choco has some very handy built-in 
functions that you can use, these are sometimes called the helpers.

### Built-In Functions
https://chocolatey.org/docs/helpers-reference

A note about a couple:
* Get-BinRoot - this is a horribly named function that doesn't do what new folks think it does. It gets you the 'tools' root, which by default is set to 'c:\tools', not the chocolateyInstall bin folder - see https://chocolatey.org/docs/helpers-get-tools-location
* Install-BinFile - used for non-exe files - executables are automatically shimmed... - see https://chocolatey.org/docs/helpers-install-bin-file
* Uninstall-BinFile - used for non-exe files - executables are automatically shimmed - see https://chocolatey.org/docs/helpers-uninstall-bin-file

### Getting package specific information
Use the package parameters pattern - see https://chocolatey.org/docs/how-to-parse-package-parameters-argument

### Need to mount an ISO?
https://chocolatey.org/docs/how-to-mount-an-iso-in-chocolatey-package


### Environment Variables
Chocolatey makes a number of environment variables available (You can access any of these with $env:TheVariableNameBelow):

 * TEMP = Overridden to the CacheLocation, but may be the same as the original TEMP folder
 * ChocolateyInstall = Top level folder where Chocolatey is installed
 * chocolateyPackageName = The name of the package, equivalent to the id in the nuspec (0.9.9+)
 * chocolateyPackageVersion = The version of the package, equivalent to the version in the nuspec (0.9.9+)
 * chocolateyPackageFolder = The top level location of the package folder

#### Advanced Environment Variables
The following are more advanced settings:

 * chocolateyPackageParameters = (0.9.8.22+)
 * CHOCOLATEY_VERSION = The version of Choco you normally see. Use if you are 'lighting' things up based on choco version. (0.9.9+)
    - Otherwise take a dependency on the specific version you need. 
 * chocolateyForceX86 = If available and set to 'true', then user has requested 32bit version. (0.9.9+)
    - Automatically handled in built in Choco functions. 
 * OS_PLATFORM = Like Windows, OSX, Linux. (0.9.9+)
 * OS_VERSION = The version of OS, like 6.1 something something for Windows. (0.9.9+)
 * OS_NAME = The reported name of the OS. (0.9.9+)
 * IS_PROCESSELEVATED = Is the process elevated? (0.9.9+)
 
#### Experimental Environment Variables
The following are experimental or use not recommended:

 * OS_IS64BIT = This may not return correctly - it may depend on the process the app is running under (0.9.9+)
 * CHOCOLATEY_VERSION_PRODUCT = the version of Choco that may match CHOCOLATEY_VERSION but may be different (0.9.9+)
    - it's based on git describe
 * IS_ADMIN = Is the user an administrator? But doesn't tell you if the process is elevated. (0.9.9+)
 * chocolateyInstallOverride = Not for use in package automation scripts. (0.9.9+)
 * chocolateyInstallArguments = the installer arguments meant for the native installer. You should use chocolateyPackageParameters intead. (0.9.9+)

