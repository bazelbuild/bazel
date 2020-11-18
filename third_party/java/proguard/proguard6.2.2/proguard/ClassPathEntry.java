/*
 * ProGuard -- shrinking, optimization, obfuscation, and preverification
 *             of Java bytecode.
 *
 * Copyright (c) 2002-2019 Guardsquare NV
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */
package proguard;

import proguard.util.ListUtil;

import java.io.*;
import java.util.List;


/**
 * This class represents an entry from a class path: an apk, a jar, an aar, a
 * war, a zip, an ear, or a directory, with a name and a flag to indicates
 * whether the entry is an input entry or an output entry. Optional filters can
 * be specified for the names of the contained resource/classes, apks, jars,
 * aars, wars, ears, and zips.
 *
 * @author Eric Lafortune
 */
public class ClassPathEntry
{
    private File    file;
    private boolean output;
    private List    filter;
    private List    apkFilter;
    private List    jarFilter;
    private List    aarFilter;
    private List    warFilter;
    private List    earFilter;
    private List    jmodFilter;
    private List    zipFilter;

    private String cachedName;


    /**
     * Creates a new ClassPathEntry with the given file and output flag.
     */
    public ClassPathEntry(File file, boolean isOutput)
    {
        this.file   = file;
        this.output = isOutput;
    }


    /**
     * Returns the path name of the entry.
     */
    public String getName()
    {
        if (cachedName == null)
        {
            cachedName = getUncachedName();
        }

        return cachedName;
    }


    /**
     * Returns the uncached path name of the entry.
     */
    private String getUncachedName()
    {
        try
        {
            return file.getCanonicalPath();
        }
        catch (IOException ex)
        {
            return file.getPath();
        }
    }


    /**
     * Returns the file.
     */
    public File getFile()
    {
        return file;
    }


    /**
     * Sets the file.
     */
    public void setFile(File file)
    {
        this.file       = file;
        this.cachedName = null;
    }


    /**
     * Returns whether this data entry is an output entry.
     */
    public boolean isOutput()
    {
        return output;
    }


    /**
     * Specifies whether this data entry is an output entry.
     */
    public void setOutput(boolean output)
    {
        this.output = output;
    }


    /**
     * Returns whether this data entry is a dex file.
     */
    public boolean isDex()
    {
        return hasExtension(".dex");
    }


    /**
     * Returns whether this data entry is an apk file.
     */
    public boolean isApk()
    {
        return hasExtension(".apk") ||
               hasExtension(".ap_");
    }


    /**
     * Returns whether this data entry is a jar file.
     */
    public boolean isJar()
    {
        return hasExtension(".jar");
    }


    /**
     * Returns whether this data entry is an aar file.
     */
    public boolean isAar()
    {
        return hasExtension(".aar");
    }


    /**
     * Returns whether this data entry is a war file.
     */
    public boolean isWar()
    {
        return hasExtension(".war");
    }


    /**
     * Returns whether this data entry is a ear file.
     */
    public boolean isEar()
    {
        return hasExtension(".ear");
    }


    /**
     * Returns whether this data entry is a jmod file.
     */
    public boolean isJmod()
    {
        return hasExtension(".jmod");
    }


    /**
     * Returns whether this data entry is a zip file.
     */
    public boolean isZip()
    {
        return hasExtension(".zip");
    }


    /**
     * Returns whether this data entry has the given extension.
     */
    private boolean hasExtension(String extension)
    {
        return endsWithIgnoreCase(file.getPath(), extension);
    }


    /**
     * Returns whether the given string ends with the given suffix, ignoring
     * its case.
     */
    private static boolean endsWithIgnoreCase(String string, String suffix)
    {
        int stringLength = string.length();
        int suffixLength = suffix.length();

        return string.regionMatches(true, stringLength -
                                          suffixLength, suffix, 0, suffixLength);
    }


    /**
     * Returns whether this data entry has any kind of filter.
     */
    public boolean isFiltered()
    {
        return filter     != null ||
               apkFilter  != null ||
               jarFilter  != null ||
               aarFilter  != null ||
               warFilter  != null ||
               earFilter  != null ||
               jmodFilter != null ||
               zipFilter  != null;
    }


    /**
     * Returns the name filter that is applied to bottom-level files in this entry.
     */
    public List getFilter()
    {
        return filter;
    }

    /**
     * Sets the name filter that is applied to bottom-level files in this entry.
     */
    public void setFilter(List filter)
    {
        this.filter = filter == null || filter.size() == 0 ? null : filter;
    }


    /**
     * Returns the name filter that is applied to apk files in this entry, if any.
     */
    public List getApkFilter()
    {
        return apkFilter;
    }

    /**
     * Sets the name filter that is applied to apk files in this entry, if any.
     */
    public void setApkFilter(List filter)
    {
        this.apkFilter = filter == null || filter.size() == 0 ? null : filter;
    }


    /**
     * Returns the name filter that is applied to jar files in this entry, if any.
     */
    public List getJarFilter()
    {
        return jarFilter;
    }

    /**
     * Sets the name filter that is applied to jar files in this entry, if any.
     */
    public void setJarFilter(List filter)
    {
        this.jarFilter = filter == null || filter.size() == 0 ? null : filter;
    }


    /**
     * Returns the name filter that is applied to aar files in this entry, if any.
     */
    public List getAarFilter()
    {
        return aarFilter;
    }

    /**
     * Sets the name filter that is applied to aar files in this entry, if any.
     */
    public void setAarFilter(List filter)
    {
        this.aarFilter = filter == null || filter.size() == 0 ? null : filter;
    }


    /**
     * Returns the name filter that is applied to war files in this entry, if any.
     */
    public List getWarFilter()
    {
        return warFilter;
    }

    /**
     * Sets the name filter that is applied to war files in this entry, if any.
     */
    public void setWarFilter(List filter)
    {
        this.warFilter = filter == null || filter.size() == 0 ? null : filter;
    }


    /**
     * Returns the name filter that is applied to ear files in this entry, if any.
     */
    public List getEarFilter()
    {
        return earFilter;
    }

    /**
     * Sets the name filter that is applied to ear files in this entry, if any.
     */
    public void setEarFilter(List filter)
    {
        this.earFilter = filter == null || filter.size() == 0 ? null : filter;
    }


    /**
     * Returns the name filter that is applied to jmod files in this entry, if any.
     */
    public List getJmodFilter()
    {
        return jmodFilter;
    }

    /**
     * Sets the name filter that is applied to jmod files in this entry, if any.
     */
    public void setJmodFilter(List filter)
    {
        this.jmodFilter = filter == null || filter.size() == 0 ? null : jmodFilter;
    }

    /**
     * Returns the name filter that is applied to zip files in this entry, if any.
     */
    public List getZipFilter()
    {
        return zipFilter;
    }

    /**
     * Sets the name filter that is applied to zip files in this entry, if any.
     */
    public void setZipFilter(List filter)
    {
        this.zipFilter = filter == null || filter.size() == 0 ? null : filter;
    }


    // Implementations for Object.

    public String toString()
    {
        String string = getName();

        if (filter     != null ||
            jarFilter  != null ||
            aarFilter  != null ||
            warFilter  != null ||
            earFilter  != null ||
            jmodFilter != null ||
            zipFilter  != null)
        {
            string +=
                ConfigurationConstants.OPEN_ARGUMENTS_KEYWORD +
                (aarFilter  != null ? ListUtil.commaSeparatedString(aarFilter, true)  : "") +
                ConfigurationConstants.SEPARATOR_KEYWORD +
                (apkFilter  != null ? ListUtil.commaSeparatedString(apkFilter, true)  : "") +
                ConfigurationConstants.SEPARATOR_KEYWORD +
                (zipFilter  != null ? ListUtil.commaSeparatedString(zipFilter, true)  : "") +
                ConfigurationConstants.SEPARATOR_KEYWORD +
                (jmodFilter != null ? ListUtil.commaSeparatedString(jmodFilter, true) : "") +
                ConfigurationConstants.SEPARATOR_KEYWORD +
                (earFilter  != null ? ListUtil.commaSeparatedString(earFilter, true)  : "") +
                ConfigurationConstants.SEPARATOR_KEYWORD +
                (warFilter  != null ? ListUtil.commaSeparatedString(warFilter, true)  : "") +
                ConfigurationConstants.SEPARATOR_KEYWORD +
                (jarFilter  != null ? ListUtil.commaSeparatedString(jarFilter, true)  : "") +
                ConfigurationConstants.SEPARATOR_KEYWORD +
                (filter     != null ? ListUtil.commaSeparatedString(filter, true)     : "") +
                ConfigurationConstants.CLOSE_ARGUMENTS_KEYWORD;
        }

        return string;
    }
}
