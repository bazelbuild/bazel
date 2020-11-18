**ReTrace** is a companion tool for **ProGuard** that 'de-obfuscates'
stack traces.

When an obfuscated program throws an exception, the resulting stack
trace typically isn't very informative. Class names and method names
have been replaced by short meaningless strings. Source file names and
line numbers are missing altogether. While this may be intentional, it
can also be inconvenient when debugging problems.

<table class="diagram" align="center">
  <tr>
    <td><div class="lightgreen box">Original code</div></td>
    <td><div class="right arrow">ProGuard</div></td>
    <td><div class="darkgreen box">Obfuscated code</div></td>
  </tr>
  <tr>
    <td/>
    <td><div class="overlap"><div class="down arrow"></div></div>
        <div class="overlap"><div class="green box">Mapping file</div></div></td>
    <td><div class="down arrow">Crash!</div></td>
  </tr>
  <tr>
    <td><div class="lightgreen box">Readable stack trace</div></td>
    <td><div class="left arrow">ReTrace</div></td>
    <td><div class="darkgreen box">Obfuscated stack trace</div></td>
  </tr>
</table>

ReTrace can read an obfuscated stack trace and restore it to what it
would look like without obfuscation. The restoration is based on the
mapping file that ProGuard can write out while obfuscating. The mapping
file links the original class names and class member names to their
obfuscated names.

!!! note ""
    ![android](../../android_small.png){: .icon} The standard Android build
    process automatically writes out mapping files to their traditional
    locations in your Android project, e.g.
    `build/outputs/release/mapping/mapping.txt`.
