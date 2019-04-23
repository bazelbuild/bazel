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
package proguard.gui.splash;

/**
 * This VariableString produces a String that grows linearly with respect to its
 * Timing, as if it is being written on a typewriter. A cursor at the end
 * precedes the typed characters.
 *
 * @author Eric Lafortune
 */
public class TypeWriterString implements VariableString
{
    private final String string;
    private final Timing timing;

    private int    cachedLength = -1;
    private String cachedString;


    /**
     * Creates a new TypeWriterString.
     * @param string the basic String.
     * @param timing the applied timing.
     */
    public TypeWriterString(String string, Timing timing)
    {
        this.string = string;
        this.timing = timing;
    }


    // Implementation for VariableString.

    public String getString(long time)
    {
        double t = timing.getTiming(time);

        int stringLength = string.length();
        int length = (int)(stringLength * t + 0.5);
        if (length != cachedLength)
        {
            cachedLength = length;
            cachedString = string.substring(0, length);
            if (t > 0.0 && length < stringLength)
            {
                cachedString += "_";
            }
        }

        return cachedString;
    }
}
