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

import java.io.*;
import java.util.*;

/**
 * This class checks and prints out information about the GPL.
 *
 * @author Eric Lafortune
 */
public class GPL
{
    /**
     * Prints out a note about the GPL if ProGuard is linked against unknown
     * code.
     */
    public static void check()
    {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        new Exception().printStackTrace(new PrintStream(out));
        LineNumberReader reader = new LineNumberReader(
                                  new InputStreamReader(
                                  new ByteArrayInputStream(out.toByteArray())));

        Set unknownPackageNames = unknownPackageNames(reader);

        if (unknownPackageNames.size() > 0)
        {
            String uniquePackageNames = uniquePackageNames(unknownPackageNames);

            System.out.println("ProGuard is released under the GNU General Public License. You therefore");
            System.out.println("must ensure that programs that link to it ("+uniquePackageNames+"...)");
            System.out.println("carry the GNU General Public License as well. Alternatively, you can");
            System.out.println("apply for an exception with the author of ProGuard.");
        }
    }


    /**
     * Returns a set of package names from the given stack trace.
     */
    private static Set unknownPackageNames(LineNumberReader reader)
    {
        Set packageNames = new HashSet();

        try
        {
            while (true)
            {
                String line = reader.readLine();
                if (line == null)
                {
                    break;
                }

                line = line.trim();
                if (line.startsWith("at "))
                {
                    line = line.substring(2).trim();
                    line = trimSuffix(line, '(');
                    line = trimSuffix(line, '.');
                    line = trimSuffix(line, '.');

                    if (line.length() > 0 && !isKnown(line))
                    {
                        packageNames.add(line);
                    }
                }
            }
        }
        catch (IOException ex)
        {
            // We'll just stop looking for more names.
        }

        return packageNames;
    }


    /**
     * Returns a comma-separated list of package names from the set, excluding
     * any subpackages of packages in the set.
     */
    private static String uniquePackageNames(Set packageNames)
    {
        StringBuffer buffer = new StringBuffer();

        Iterator iterator = packageNames.iterator();
        while (iterator.hasNext())
        {
            String packageName = (String)iterator.next();
            if (!containsPrefix(packageNames, packageName))
            {
                buffer.append(packageName).append(", ");
            }
        }

        return buffer.toString();
    }


    /**
     * Returns a given string without the suffix, as defined by the given
     * separator.
     */
    private static String trimSuffix(String string, char separator)
    {
        int index = string.lastIndexOf(separator);
        return index < 0 ? "" : string.substring(0, index);
    }


    /**
     * Returns whether the given set contains a prefix of the given name.
     */
    private static boolean containsPrefix(Set set, String name)
    {
        int index = 0;

        while (!set.contains(name.substring(0, index)))
        {
            index = name.indexOf('.', index + 1);
            if (index < 0)
            {
                return false;
            }
        }

        return true;
    }


    /**
     * Returns whether the given package name has been granted an exception
     * against the GPL linking clause, by the copyright holder of ProGuard.
     * This method is not legally binding, but of course the actual license is.
     * Please contact the copyright holder if you would like an exception for
     * your code as well.
     */
    private static boolean isKnown(String packageName)
    {
        return packageName.startsWith("java")                   ||
               packageName.startsWith("sun.reflect")            ||
               packageName.startsWith("proguard")               ||
               packageName.startsWith("org.apache.tools.ant")   ||
               packageName.startsWith("org.apache.tools.maven") ||
               packageName.startsWith("org.gradle")             ||
               packageName.startsWith("org.codehaus.groovy")    ||
               packageName.startsWith("org.eclipse")            ||
               packageName.startsWith("org.netbeans")           ||
               packageName.startsWith("com.android")            ||
               packageName.startsWith("com.intel")              ||
               packageName.startsWith("com.sun.kvem")           ||
               packageName.startsWith("net.certiv.proguarddt")  ||
               packageName.startsWith("groovy")                 ||
               packageName.startsWith("scala")                  ||
               packageName.startsWith("sbt")                    ||
               packageName.startsWith("xsbt")                   ||
               packageName.startsWith("eclipseme");
    }


    public static void main(String[] args)
    {
        LineNumberReader reader = new LineNumberReader(
                                  new InputStreamReader(System.in));

        Set unknownPackageNames = unknownPackageNames(reader);

        if (unknownPackageNames.size() > 0)
        {
            String uniquePackageNames = uniquePackageNames(unknownPackageNames);

            System.out.println(uniquePackageNames);
        }
    }
}
