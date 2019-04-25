-- A solution contains projects, and defines the available configurations
solution "brotli"
configurations { "Release", "Debug" }
platforms { "x64", "x86" }
targetdir "bin"
location "buildfiles"
flags "RelativeLinks"
includedirs { "c/include" }

filter "configurations:Release"
  optimize "Speed"
  flags { "StaticRuntime" }

filter "configurations:Debug"
  flags { "Symbols" }

filter { "platforms:x64" }
   architecture "x86_64"

filter { "platforms:x86" }
   architecture "x86"

configuration { "gmake" }
  buildoptions { "-Wall -fno-omit-frame-pointer" }
  location "buildfiles/gmake"

configuration { "xcode4" }
  location "buildfiles/xcode4"

configuration "linux"
  links "m"

configuration { "macosx" }
  defines { "OS_MACOSX" }

project "brotlicommon"
  kind "SharedLib"
  language "C"
  files { "c/common/**.h", "c/common/**.c" }

project "brotlicommon_static"
  kind "StaticLib"
  targetname "brotlicommon"
  language "C"
  files { "c/common/**.h", "c/common/**.c" }

project "brotlidec"
  kind "SharedLib"
  language "C"
  files { "c/dec/**.h", "c/dec/**.c" }
  links "brotlicommon"

project "brotlidec_static"
  kind "StaticLib"
  targetname "brotlidec"
  language "C"
  files { "c/dec/**.h", "c/dec/**.c" }
  links "brotlicommon_static"

project "brotlienc"
  kind "SharedLib"
  language "C"
  files { "c/enc/**.h", "c/enc/**.c" }
  links "brotlicommon"

project "brotlienc_static"
  kind "StaticLib"
  targetname "brotlienc"
  language "C"
  files { "c/enc/**.h", "c/enc/**.c" }
  links "brotlicommon_static"

project "brotli"
  kind "ConsoleApp"
  language "C"
  linkoptions "-static"
  files { "c/tools/brotli.c" }
  links { "brotlicommon_static", "brotlidec_static", "brotlienc_static" }
